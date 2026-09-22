import os 
import sys
import cv2
import numpy as np
import torch
import argparse
import yaml
from helper import compute_a_star, in_mask_mean, make_chromophore_composite, PARAM_NAMES, PARAM_COLORMAPS, save_chromophore_column

    
    
def apply_reciep(skin_props, mask_flat, bio_skin, A,C,amp):
    HEMO = C["BLOOD_VOLUME_INDEX"]
    MEL  = C["MELANIN_INDEX"]
    OXY  = C["HAEMO_TYPE_INDEX"]
    EU   = C["MELANIN_TYPE_INDEX"]

    sp = skin_props.clone()                       # fresh baseline EVERY frame

    # 1) Hemoglobin flush 
    sp[:, HEMO] = torch.clamp(sp[:, HEMO] + float(amp) * mask_flat, 0.0, 1.0)

    sp[:, OXY] = torch.clamp(sp[:, OXY] + float(A["OXY_Boost"]) * mask_flat, 0.0, 1.0)

    sp[:, EU] = torch.clamp(sp[:, EU] + float(A["EU_Boost"]) * mask_flat, 0.0, 1.0)
    
    _, flush_rgb, _, _ = bio_skin.skin_props_to_reflectance(sp)

    return  flush_rgb
    
    
def calibrate_grade(skin_props, mask_flat, bio_skin, A, C,
                    redness_target, inside, clean_a_star,high=0.40,
                    tolerance=0.005, max_iterations=6):
    
    print(f"[calibrate_grade] Starting calibration with target redness {redness_target:.4f}, tolerance {tolerance:.4f}, max_iterations {max_iterations}")
    
    low = 0.0
    high = high
    
    for _ in range(max_iterations):
        amp_mid = (low + high) / 2.0
        flush_rgb = apply_reciep(skin_props, mask_flat, bio_skin, A,C,amp_mid)
        flush_a_star  = compute_a_star(flush_rgb, inside)
        redness_measured = flush_a_star - clean_a_star   # this is now Δa*
        if abs(redness_measured - redness_target) <= tolerance:
            break
        
        if redness_measured < redness_target:
            low = amp_mid
        else:
            high = amp_mid
            
    if redness_measured < redness_target - tolerance:
        print(f"[calibrate_grade] WARNING: target redness {redness_target:.4f} not reached, measured {redness_measured:.4f}")    
    return amp_mid, redness_measured


    