#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
redness_sanity_check.py
=======================
Two-part test to confirm which channel order gives correct redness for your pipeline.

Part A: Synthetic test
    Creates a pure-red and pure-blue patch in memory, computes redness both ways.
    Tells you which formula is correct without needing any real image.

Part B: Real-image test
    Loads one of your actual flush frames (EXR) and computes redness both ways
    over the mask region. The positive one is correct.

Run this from your erythema environment:
    python redness_sanity_check.py

You only need to fill in FLUSH_FRAME_PATH and MASK_PATH at the top.
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"

import numpy as np
import cv2

# =============================================================================
# CONFIG — fill in one of your flush frames and the mask
# =============================================================================

# A frame where the face is visibly pink/flushed (any amp > 0.05 frame)
FLUSH_FRAME_PATH = r"D:\Github\PhD Code\Erythema-Progression\output\progression\2026-09-08_12-38-41\frame_08_amp0.15.exr"

# Your feathered mask (saved as mask_feathered.exr) — to restrict the check
# to the actual flushed region. If you can't find it, leave as None and the
# test uses the whole image.
MASK_PATH = r"D:\Github\PhD Code\Erythema-Progression\output\progression\2026-09-08_12-38-41\mask_feathered.exr"


# =============================================================================
# PART A — Synthetic patch test (no real image needed)
# =============================================================================

def part_a_synthetic():
    print("=" * 60)
    print("PART A — Synthetic colour patch test")
    print("=" * 60)

    # Pure red patch: if the image were RGB, channel 0=R=1.0, G=0, B=0
    red_rgb  = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)   # shape (1,1,3)
    blue_rgb = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)

    # Pure red patch: if the image were BGR (OpenCV default), channel 0=B=0, R=1.0
    red_bgr  = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)   # B=0,G=0,R=1
    blue_bgr = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)   # B=1,G=0,R=0

    def redness_rgb(patch):
        """Formula assuming index 0=R, 1=G, 2=B"""
        r, g, b = patch[..., 0], patch[..., 1], patch[..., 2]
        return float((r - 0.5 * (g + b)).mean())

    def redness_bgr(patch):
        """Formula assuming index 0=B, 1=G, 2=R  (OpenCV default)"""
        b, g, r = patch[..., 0], patch[..., 1], patch[..., 2]
        return float((r - 0.5 * (b + g)).mean())

    print("\n  Testing on a PURE RED patch:")
    print(f"    redness_rgb  (expects +1.0 if RGB) : {redness_rgb(red_rgb):+.4f}")
    print(f"    redness_bgr  (expects +1.0 if BGR) : {redness_bgr(red_bgr):+.4f}")

    print("\n  Testing on a PURE BLUE patch:")
    print(f"    redness_rgb  (expects -0.5 if RGB) : {redness_rgb(blue_rgb):+.4f}")
    print(f"    redness_bgr  (expects -0.5 if BGR) : {redness_bgr(blue_bgr):+.4f}")

    print()
    print("  -> redness_rgb and redness_bgr will both print +1.0 / -0.5 above")
    print("     because the patches are defined to match their own formula.")
    print("     Part B (real image) is what tells you which formula YOUR pipeline uses.")


# =============================================================================
# PART B — Real flush-frame test
# =============================================================================

def part_b_real():
    print()
    print("=" * 60)
    print("PART B — Real flush-frame test")
    print("=" * 60)

    # Load flush frame
    frame = cv2.imread(FLUSH_FRAME_PATH,
                       cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if frame is None:
        print(f"  ERROR: could not load frame from:\n    {FLUSH_FRAME_PATH}")
        print("  -> Check FLUSH_FRAME_PATH at the top of this script.")
        return
    frame = frame.astype(np.float32)
    print(f"  Frame loaded: shape={frame.shape}  dtype={frame.dtype}")
    print(f"  Value range: min={frame.min():.4f}  max={frame.max():.4f}")

    # Load mask (optional)
    mask_arr = None
    if MASK_PATH and os.path.exists(MASK_PATH):
        m = cv2.imread(MASK_PATH, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if m is not None:
            if m.ndim == 3:
                m = m[:, :, 0]
            mask_arr = m > 0.5
            print(f"  Mask loaded: {mask_arr.sum()} pixels inside mask "
                  f"({100 * mask_arr.mean():.1f}% of image)")
        else:
            print("  Mask could not be loaded — using whole image.")
    else:
        print("  No mask path set or file not found — using whole image.")

    def apply_mask(arr2d):
        return arr2d[mask_arr] if mask_arr is not None else arr2d.reshape(-1)

    # -- Formula 1: assume channel order is RGB (index 0=R) --
    # This is what BioSkin tensors use (model output, not cv2.imread)
    ch0, ch1, ch2 = frame[:,:,0], frame[:,:,1], frame[:,:,2]
    redness_assuming_rgb = float(
        apply_mask(ch0 - 0.5 * (ch1 + ch2)).mean())

    # -- Formula 2: assume channel order is BGR (index 0=B, index 2=R) --
    # This is what cv2.imread returns by default
    redness_assuming_bgr = float(
        apply_mask(ch2 - 0.5 * (ch0 + ch1)).mean())

    print()
    print(f"  redness assuming RGB order  (idx 0=R):  {redness_assuming_rgb:+.4f}")
    print(f"  redness assuming BGR order  (idx 0=B):  {redness_assuming_bgr:+.4f}")
    print()

    # Verdict
    if redness_assuming_rgb > 0 and redness_assuming_bgr <= 0:
        print("  VERDICT: your data is RGB order.")
        print("  Use:  r - 0.5*(g+b)  where  r=img[...,0], g=img[...,1], b=img[...,2]")
    elif redness_assuming_bgr > 0 and redness_assuming_rgb <= 0:
        print("  VERDICT: your data is BGR order (OpenCV default).")
        print("  Use:  r - 0.5*(b+g)  where  b=img[...,0], g=img[...,1], r=img[...,2]")
    elif redness_assuming_rgb > 0 and redness_assuming_bgr > 0:
        print("  VERDICT: both positive — the image is dominated by warm tones overall.")
        print("  The LARGER value is the correct formula.")
        if redness_assuming_rgb > redness_assuming_bgr:
            print("  -> Use RGB formula (idx 0=R).")
        else:
            print("  -> Use BGR formula (idx 2=R).")
    else:
        print("  VERDICT: both negative — unusual. Possible causes:")
        print("    - The frame is not flushed (try a higher amplitude frame)")
        print("    - The mask is too small or misaligned")
        print("    - The image values are not in [0,1] (check the value range above)")

    # Extra: print per-channel mean inside the mask so you can see which is largest
    print()
    print("  Per-channel mean inside mask (tells you which channel is 'red'):")
    print(f"    channel 0 mean: {apply_mask(ch0).mean():.4f}")
    print(f"    channel 1 mean: {apply_mask(ch1).mean():.4f}")
    print(f"    channel 2 mean: {apply_mask(ch2).mean():.4f}")
    print("  The channel with the HIGHEST mean on a flushed face is the red channel.")
    print("  If that is channel 0 -> RGB.  If channel 2 -> BGR.")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    part_a_synthetic()
    part_b_real()
    print()
    print("=" * 60)
    print("Done. Use the verdict above to fix _in_mask_redness() before Level B.")
    print("=" * 60)
