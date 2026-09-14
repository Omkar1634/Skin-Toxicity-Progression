import os 
import sys
import cv2
import numpy as np
import torch
import argparse
import yaml


    
    
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
    
    
def calibrate_grade(skin_props, mask_flat, 
                    bio_skin, A,C,redness_target: float,
                    inside,
                    tolerance: float = 0.005,
                    max_iterations: int = 6):
    
    print(f"[calibrate_grade] Starting calibration with target redness {redness_target:.4f}, tolerance {tolerance:.4f}, max_iterations {max_iterations}")
    
    low = 0.0
    high = 0.40
    
    for _ in range(max_iterations):
        amp_mid = (low + high) / 2.0
        flush_rgb = apply_reciep(skin_props, mask_flat, bio_skin, A,C,amp_mid)
        redness_measured = (flush_rgb[inside,2] - 0.5 * (flush_rgb[inside,0] + flush_rgb[inside,1])).mean().item()
        if abs(redness_measured - redness_target) <= tolerance:
            break
        
        if redness_measured < redness_target:
            low = amp_mid
        else:
            high = amp_mid
            
    if redness_measured < redness_target - tolerance:
        print(f"[calibrate_grade] WARNING: target redness {redness_target:.4f} not reached, measured {redness_measured:.4f}")    
    return amp_mid, redness_measured


    