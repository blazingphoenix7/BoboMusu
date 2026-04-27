"""
preprocess_fp.py — surgically extract a fingerprint from a finger photo
and emit a clean heightmap PNG cropped tight to the ridge area.

Extraction strategy:
  1. YCbCr+saturation skin segmentation -> finger mask
  2. Structure-tensor orientation coherence inside the finger
       - Ridges have strong oriented structure -> high coherence
       - Smooth skin (cuticle, sides), nail, fabric -> low coherence
  3. Otsu threshold on coherence -> binary ridge mask
  4. Largest connected component, modest morphology
  5. TIGHT bbox crop (no margin) so every output pixel is a ridge pixel

Output (in designs/output/):
  <name>.png            — the heightmap to sample (white=ridge crest, black=nothing)
  <name>_preview.png    — original | mask overlay | heightmap (sanity check)
  <name>_mask.png       — binary mask of the extracted print (debug)

Heightmap convention downstream:
  pixel value 0   = no engrave
  pixel value 255 = max engrave depth (ridge crest)

Usage:
    python preprocess_fp.py <input_image> <output_basename>
"""

import sys
import os
import numpy as np
import cv2


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────
def _largest_component(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8) * 255


def detect_skin(bgr: np.ndarray) -> np.ndarray:
    """YCbCr range AND saturation gate. Skin clusters tightly in Cr/Cb."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    ycc_mask = cv2.inRange(ycc,
                            np.array([0,   140, 90],  dtype=np.uint8),
                            np.array([255, 175, 130], dtype=np.uint8))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat_mask = (hsv[:, :, 1] >= 40).astype(np.uint8) * 255
    skin = cv2.bitwise_and(ycc_mask, sat_mask)

    # close ridge-shadow / specular holes; open to drop fabric specks
    se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    se_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, se_close)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, se_open)
    return _largest_component(skin)


def coherence_map(gray: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """
    Per-pixel orientation coherence from the structure tensor.
    Returns a float32 array in [0, 1]. High = strong oriented structure (ridges).
    """
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

    # Structure tensor components, smoothed at fingerprint scale
    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    Jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)

    # eigenvalues of [[Jxx, Jxy], [Jxy, Jyy]]
    trace = Jxx + Jyy
    discr = np.sqrt(np.maximum((Jxx - Jyy) ** 2 + 4.0 * Jxy * Jxy, 0.0))
    lam1 = 0.5 * (trace + discr)
    lam2 = 0.5 * (trace - discr)

    coh = np.where(trace > 1e-6, (lam1 - lam2) / (trace + 1e-9), 0.0)
    return np.clip(coh, 0.0, 1.0).astype(np.float32)


def detect_ridge_region(bgr: np.ndarray, gray: np.ndarray,
                         debug_dir: str | None = None) -> np.ndarray:
    """
    Stage 1  YCbCr+sat skin mask  -> finger silhouette (fabric rejected)
    Stage 2  CLAHE on grayscale   -> ridges sharpened locally
    Stage 3  Per-pixel local std-dev of CLAHE (kernel=3) -> ridge-edge map
              (high where ridges alternate bright/dark; near-zero on smooth skin)
    Stage 4  Smooth std-dev with sigma=15 -> continuous "ridginess" field
    Stage 5  Otsu threshold on field values inside the (eroded) skin
    Stage 6  Morphology + largest CC + boundary smoothing
    """
    skin = detect_skin(bgr)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "_dbg_skin.png"), skin)
    if skin.max() == 0:
        return skin

    # Erode skin so we don't sample at the skin/fabric edge (any spillover there
    # would otherwise raise the std-dev threshold)
    er_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    skin_inner = cv2.erode(skin, er_kernel)

    # CLAHE for crisp ridges
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "_dbg_clahe.png"), enhanced)

    # Local std-dev — captures alternating bright/dark of ridges
    g = enhanced.astype(np.float32)
    k = 3
    mean = cv2.boxFilter(g, -1, (k, k))
    sq = cv2.boxFilter(g * g, -1, (k, k))
    var = np.maximum(sq - mean * mean, 0.0)
    sd = np.sqrt(var)

    # Mask to inner skin BEFORE smoothing (so smooth doesn't pull in fabric stddev)
    sd_in = np.where(skin_inner > 0, sd, 0).astype(np.float32)

    # Smooth into a continuous ridginess field
    field = cv2.GaussianBlur(sd_in, (0, 0), sigmaX=15, sigmaY=15)
    # Bring back inside non-eroded skin (boundary smoothing pulls field slightly outward)
    field = np.where(skin > 0, field, 0).astype(np.float32)
    if field.max() > 0:
        field_norm_u8 = np.clip(field * 255.0 / field.max(), 0, 255).astype(np.uint8)
    else:
        field_norm_u8 = np.zeros_like(sd_in, dtype=np.uint8)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "_dbg_ridginess.png"), field_norm_u8)

    # Otsu inside skin
    skin_vals = field_norm_u8[skin > 0]
    if skin_vals.size == 0:
        return skin
    otsu_t, _ = cv2.threshold(skin_vals, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    region = ((field_norm_u8 >= otsu_t) & (skin > 0)).astype(np.uint8) * 255
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "_dbg_ridge_raw.png"), region)

    # Modest morphology
    se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    se_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, se_close)
    region = cv2.morphologyEx(region, cv2.MORPH_OPEN, se_open)

    largest = _largest_component(region)
    if largest.max() == 0:
        return largest

    # Convex hull of the ridge area — closes the internal gaps where ridge
    # contrast was locally weak, and gives a smooth print outline.
    contours, _ = cv2.findContours(largest, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(biggest)
        hull_mask = np.zeros_like(largest)
        cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)
        # Clip to skin so we don't extend past the actual finger boundary
        largest = cv2.bitwise_and(hull_mask, skin)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "_dbg_ridge_hull.png"), largest)

    # Final boundary smoothing
    largest = cv2.GaussianBlur(largest.astype(np.float32), (0, 0),
                                sigmaX=3, sigmaY=3)
    largest = (largest >= 128).astype(np.uint8) * 255
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "_dbg_ridge_final.png"), largest)
    return largest


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def main(in_path: str, out_basename: str) -> int:
    if not os.path.exists(in_path):
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    bgr = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"ERROR: could not decode image: {in_path}", file=sys.stderr)
        return 2

    out_dir = os.path.dirname(out_basename) or "."
    os.makedirs(out_dir, exist_ok=True)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"loaded: {w}x{h}")

    ridge_mask = detect_ridge_region(bgr, gray, debug_dir=out_dir)
    if ridge_mask.max() == 0:
        print("ERROR: no ridge region detected", file=sys.stderr)
        return 3

    # TIGHT bbox crop — every pixel inside output is an in-mask ridge pixel
    ys, xs = np.where(ridge_mask > 0)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    print(f"ridge bbox: x[{x0}..{x1}] y[{y0}..{y1}] -> {x1 - x0}x{y1 - y0}")

    gray_c = gray[y0:y1, x0:x1]
    mask_c = ridge_mask[y0:y1, x0:x1]
    bgr_c  = bgr[y0:y1, x0:x1]

    # CLAHE for crisp local contrast on ridges
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_c)

    # Outside-mask pixels: drive to 0 in heightmap regardless of CLAHE.
    # Inside-mask pixels: invert so dark photo ridges -> bright heightmap.
    inverted = 255 - enhanced
    inverted[mask_c == 0] = 0

    # ── INPAINTING ──────────────────────────────────────────────────────
    # Fill the blank corners of the cropped rectangle with extrapolated print
    # content so the heightmap has FP-like ridges everywhere. This kills the
    # "FP missing at edges" problem when COVER-mapping onto the pendant — the
    # user explicitly accepted computer-extended ridges over blank areas.
    fill_mask = (mask_c == 0).astype(np.uint8) * 255
    if fill_mask.max() > 0:
        # TELEA inpaint with a generous radius extends ridge pattern outward
        # from the mask boundary into the blank corners.
        inpainted = cv2.inpaint(inverted, fill_mask, 12, cv2.INPAINT_TELEA)
        # The TELEA result can plateau in big blank corners; do a second pass
        # at smaller radius using the same "to-fill" mask to sharpen any
        # smoothed-out areas.
        # (skip second pass — single pass is generally good enough)
    else:
        inpainted = inverted.copy()

    heightmap_path = out_basename + ".png"
    mask_path      = out_basename + "_mask.png"
    preview_path   = out_basename + "_preview.png"
    cv2.imwrite(heightmap_path, inpainted)
    cv2.imwrite(mask_path, mask_c)

    # 4-panel preview: original w/ bbox | mask overlay | heightmap (pre-inpaint) | heightmap (post-inpaint)
    panel_h = max(y1 - y0, h)
    panel_w_total = w + (x1 - x0) * 3 + 60
    preview = np.full((panel_h, panel_w_total, 3), 240, dtype=np.uint8)
    # Panel 1: original w/ bbox
    preview[:h, :w] = bgr
    cv2.rectangle(preview[:h, :w], (x0, y0), (x1, y1), (0, 255, 0), 3)
    # Panel 2: cropped image with mask overlay (red over the kept ridge area)
    overlay = bgr_c.copy()
    red = np.zeros_like(overlay); red[:, :, 2] = 255
    overlay = np.where(mask_c[..., None] > 0,
                        cv2.addWeighted(overlay, 0.55, red, 0.45, 0),
                        overlay)
    p2_x = w + 20
    preview[:overlay.shape[0], p2_x:p2_x + overlay.shape[1]] = overlay
    # Panel 3: heightmap pre-inpaint
    hm_pre = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
    p3_x = p2_x + overlay.shape[1] + 20
    preview[:hm_pre.shape[0], p3_x:p3_x + hm_pre.shape[1]] = hm_pre
    # Panel 4: heightmap post-inpaint (the actual file Rhino consumes)
    hm_post = cv2.cvtColor(inpainted, cv2.COLOR_GRAY2BGR)
    p4_x = p3_x + hm_pre.shape[1] + 20
    preview[:hm_post.shape[0], p4_x:p4_x + hm_post.shape[1]] = hm_post
    cv2.imwrite(preview_path, preview)

    coverage_in_crop = float((mask_c > 0).sum()) / mask_c.size * 100
    coverage_in_orig = float((ridge_mask > 0).sum()) / ridge_mask.size * 100
    nz_inside_mask = inverted[mask_c > 0]
    print(f"saved heightmap:  {heightmap_path}  ({inpainted.shape[1]}x{inpainted.shape[0]})  [post-inpaint]")
    print(f"saved mask:       {mask_path}")
    print(f"saved preview:    {preview_path}")
    print(f"ridge area: {(ridge_mask > 0).sum()} px  "
          f"({coverage_in_orig:.1f}% of original, {coverage_in_crop:.1f}% of crop)")
    if nz_inside_mask.size > 0:
        print(f"heightmap intensity inside mask: "
              f"min={int(nz_inside_mask.min())}  "
              f"mean={nz_inside_mask.mean():.1f}  "
              f"max={int(nz_inside_mask.max())}  / 255")
    nonzero_pre = float((inverted > 0).sum()) / inverted.size * 100
    nonzero_post = float((inpainted > 0).sum()) / inpainted.size * 100
    print(f"non-zero coverage of crop:  pre-inpaint={nonzero_pre:.1f}%  post-inpaint={nonzero_post:.1f}%")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2]))
