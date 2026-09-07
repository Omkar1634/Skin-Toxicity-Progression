import os
import sys
import csv
import numpy as np
import torch
import cv2 
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--resume", type=str, default=None)
args = parser.parse_args()


with open(args.config, "r",encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

P = cfg["paths"]




def build_gaussian_mask(H, W, cx_frac, cy_frac, rad_frac, feather=1.0):
    cy, cx = cy_frac * H, cx_frac * W
    radius = rad_frac * min(H, W)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    sigma = max(radius * feather, 1.0)
    mask = np.exp(-d2 / (2.0 * sigma ** 2)).astype(np.float32)
    mask[mask < 0.01] = 0.0
    return mask

def butterfly_mask(H, W):
    """
    Build a feathered centrofacial ("butterfly") mask in the FFHQ-UV texture layout.

    Loads the FFHQ-UV region masks, carves the nostrils and mouth out of the
    valid-face region, feathers the result, and returns it as a soft weight in
    [0, 1] suitable for multiplying against a chromophore residual.

    Because the mask is defined in the fixed FFHQ-UV unwrap, it is identity-
    independent: build it once and reuse it for every albedo in that layout.

    Steps
    -----
    1. Load the whole-face valid mask and the mouth / nostril exclusion masks from the paths in the config dict ``P`` (grayscale).
    2. Resize each to the working resolution (W, H) with nearest-neighbour, so the binary masks stay crisp (no interpolated grey).
    3. Subtract the mouth+nostril exclusions from the face region using saturating ops (``cv2.add`` / ``cv2.subtract``) to avoid uint8 wrap-around, Gaussian-blur the binary result to soften every border, then normalise to float [0, 1].

    Parameters
    ----------
    H : int [Target mask height in pixels (match the albedo).]
    W : int [Target mask width in pixels (match the albedo). ]

    Returns
    -------
    numpy.ndarray
        Feathered mask of shape (H, W), dtype float32, values in [0, 1]:
        ~1.0 over the cheeks/forehead/nose bridge, 0.0 outside the face and in
        the excluded mouth/nostril regions, with a smooth ramp at every edge.

    Notes
    -----
    - Feathering happens on the *combined* binary mask (blur last, once), so the outer face edge and the exclusion holes soften together.
    - Kernel (31, 31) with sigma 8 is tuned to soften borders while keeping the nostril and mouth holes open; reduce sigma if they start filling in.
    - Reshape row-major (mask.reshape(-1)) when flattening to match the pixel order used by the encoder input.
    """
    print(f"[info] building butterfly mask for {W}x{H}...")
    # 1. Load the mask from the config path 
    mask = cv2.imread(P["mask"], cv2.IMREAD_GRAYSCALE)
    mask_mouth = cv2.imread(P["mask_mouth"], cv2.IMREAD_GRAYSCALE)
    mask_nostril = cv2.imread(P["mask_nostril"], cv2.IMREAD_GRAYSCALE)
    
    # 2. Resize the masks to match the target size
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_mouth = cv2.resize(mask_mouth, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_nostril = cv2.resize(mask_nostril, (W, H), interpolation=cv2.INTER_NEAREST)
    
    #3. Create the combined mask using cv2.subtract and cv2.add
    mask_combined = cv2.subtract(mask, cv2.add(mask_nostril, mask_mouth))      
    feathered_mask_combinedG = cv2.GaussianBlur(mask_combined, (31, 31), 8)
    feathered_mask_float = feathered_mask_combinedG.astype(np.float32) / 255.
    print(f"[info] butterfly mask built completed ...")
    return feathered_mask_float