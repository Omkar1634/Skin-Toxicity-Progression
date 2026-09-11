"""
ita.py -- Individual Typology Angle for FFHQ-UV albedo faces.

ITA is the standard colourimetric measure of skin tone:
    ITA = atan2(L* - 50, b*) * 180/pi        (degrees)
Higher = lighter skin, lower = darker skin.

Two public functions:
    build_butterfly_mask(H, W, mask_paths)  -> the feathered mask (once, reused)
    face_ita(albedo_path, mask)             -> {ita, band, n_pixels} for one face

Assumptions locked in earlier:
    - albedo is a raw FFHQ-UV PNG (8-bit, sRGB gamma-encoded)
    - loaded with cv2  -> BGR order
    - mask is float32 in [0,1], same H x W as the albedo
"""

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# mask  (same logic as your mask.py, but takes paths as arguments so it can be
#        imported without pulling in argparse/config)
# ----------------------------------------------------------------------------
def build_butterfly_mask(H, W, mask_path, mask_mouth_path, mask_nostril_path):
    mask         = cv2.imread(mask_path,         cv2.IMREAD_GRAYSCALE)
    mask_mouth   = cv2.imread(mask_mouth_path,   cv2.IMREAD_GRAYSCALE)
    mask_nostril = cv2.imread(mask_nostril_path, cv2.IMREAD_GRAYSCALE)

    mask         = cv2.resize(mask,         (W, H), interpolation=cv2.INTER_NEAREST)
    mask_mouth   = cv2.resize(mask_mouth,   (W, H), interpolation=cv2.INTER_NEAREST)
    mask_nostril = cv2.resize(mask_nostril, (W, H), interpolation=cv2.INTER_NEAREST)

    combined  = cv2.subtract(mask, cv2.add(mask_nostril, mask_mouth))
    feathered = cv2.GaussianBlur(combined, (31, 31), 8)
    return feathered.astype(np.float32) / 255.0


# ----------------------------------------------------------------------------
# colour math
# ----------------------------------------------------------------------------
_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_D65 = np.array([0.95047, 1.0, 1.08883])


def _band(ita):
    """Del Bino tone label. NOTE: report as ITA bands, NOT Fitzpatrick I-VI."""
    if ita > 55:  return "very_light"
    if ita > 41:  return "light"
    if ita > 28:  return "intermediate"
    if ita > 10:  return "tan"
    if ita > -30: return "brown"
    return "dark"


def face_ita(albedo_path, mask, mask_threshold=0.99):
    """ITA of the in-mask skin pixels of one FFHQ-UV albedo PNG."""
    albedo = cv2.imread(albedo_path, cv2.IMREAD_COLOR)   # BGR uint8
    if albedo is None:
        raise FileNotFoundError(albedo_path)
    
    H, W = mask.shape[:2]                                 # <-- add
    if albedo.shape[:2] != (H, W):                        # <-- add
        albedo = cv2.resize(albedo, (W, H), interpolation=cv2.INTER_AREA)

    rgb = albedo[..., ::-1].astype(np.float64) / 255.0   # BGR->RGB, [0,1]
    # sRGB gamma -> linear light
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

    px = rgb[mask >= mask_threshold]                     # solid-in-mask only
    if px.shape[0] < 256:
        raise ValueError(f"only {px.shape[0]} in-mask pixels for {albedo_path}")

    # linear RGB -> XYZ -> CIELAB
    xyz = (px @ _M.T) / _D65
    d = 6.0 / 29.0
    f = np.where(xyz > d**3, np.cbrt(xyz), xyz / (3 * d**2) + 4.0 / 29.0)
    L = 116 * f[:, 1] - 16
    b = 200 * (f[:, 1] - f[:, 2])

    ita_px = np.degrees(np.arctan2(L - 50.0, b))
    ita = float(np.median(ita_px))
    return {"ita": ita, "band": _band(ita), "n_pixels": int(px.shape[0])}


# ----------------------------------------------------------------------------
# standalone runner: all five faces -> table
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # ---- FILL THESE IN ----
    MASK_PATH    = "path/to/mask.png"
    MASK_MOUTH   = "path/to/mask_mouth.png"
    MASK_NOSTRIL = "path/to/mask_nostril.png"

    # face id -> albedo path  (add/adjust as needed)
    FACES = {
        "000000": "path/to/000000_albedo.png",
        "000037": "path/to/000037_albedo.png",
        "000091": "path/to/000091_albedo.png",
        "000148": "path/to/000148_albedo.png",
        "000190": "path/to/000190_albedo.png",
    }
    # -----------------------

    # build the mask once, from the size of the first face
    first = cv2.imread(next(iter(FACES.values())), cv2.IMREAD_COLOR)
    H, W = first.shape[:2]
    mask = build_butterfly_mask(H, W, MASK_PATH, MASK_MOUTH, MASK_NOSTRIL)

    print(f"{'face':10s} {'ITA':>8s}   band")
    print("-" * 34)
    for fid, path in FACES.items():
        r = face_ita(path, mask)
        print(f"{fid:10s} {r['ita']:8.2f}   {r['band']}")