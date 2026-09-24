#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
erythema_sweep.py
=================================================================================
Erythema pipeline with a SEVERITY SWEEP + a CHROMOPHORE-MAP COMPOSITE.
Built entirely on BioSkin's own I/O + inference wrappers. Edits the HEMOGLOBIN
CHROMOPHORE (skin_props col 1), never RGB — the decoder synthesizes the colour.

Outputs:
    progression/frame_00..NN      severity frames (00 = clean)
    erythema_progression_montage.jpeg
    erythema_sweep.csv
    erythema_control_curve.png
    erythema_chromophore_composite.png   <-- NEW: clean vs edited vs delta, all params

skin_props cols: 0 melanin | 1 hemoglobin | 2 epi_thickness | 3 eumelanin_ratio
                 4 oxygenation | 5 occlusion(AO, keep, never edit)
=================================================================================
"""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # 3090 only; hides 5070 Ti; kills DataParallel
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import csv
import numpy as np
import torch
import cv2
import argparse
import datetime 
from utils.helper import  compute_a_star, in_mask_mean, make_chromophore_composite, PARAM_NAMES, PARAM_COLORMAPS, save_chromophore_column, save_montage, save_control_curve, save_control_allcurve, save_metadata_csv,ita_to_fitzpatrick, save_chromophore_maps
from Inflammatory.oxy_direction_test import run_oxy_direction_test
from Inflammatory.hemoglobin_direction import run_hemoglobin_direction
from Inflammatory.mixes_sweep import run_hemoglobin_oxy_direction
from utils.mask import butterfly_mask, build_gaussian_mask
from utils.skin_tone import face_ita
import yaml 
from Inflammatory.Binary_search_calibration_system import calibrate_grade, apply_reciep
BIOSKIN_REPO = None

if BIOSKIN_REPO and BIOSKIN_REPO not in sys.path:
    sys.path.insert(0, BIOSKIN_REPO)

from bioskin.bioskin import BioSkinInference
import bioskin.utils.io as io

# =============================================================================
# CONFIG
# =========================================================================

MAX_WIDTH    = 800
BATCH_SIZE   = 512000
COMPOSITE_AMP = None  # None = last level, or float = specific level for composite

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Config.yaml", help="YAML config path")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def create_output_directory(output_root):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(output_root, "progression", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir



# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    chromophore = config["chromophore_parameter"]
    amplitude = config["amplitude_parameter"]
    mask_config = config["mask_parameter"]
    face_id = os.path.splitext(os.path.basename(paths["ALBEDO_PATH"]))[0]


    bioskin_repo = paths.get("BIOSKIN_REPO")
    if bioskin_repo and bioskin_repo not in sys.path:
        sys.path.insert(0, bioskin_repo)

    from bioskin.bioskin import BioSkinInference
    import bioskin.utils.io as bioskin_io

    output_dir = create_output_directory(paths["OUTPUT_DIR"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cpu = device.type == "cpu"
    print(f"[init] device = {device}")

    bio_skin = BioSkinInference(paths["MODEL_PREFIX"], device=device,batch_size=BATCH_SIZE)
    image = bioskin_io.load_image(paths["ALBEDO_PATH"], max_width=MAX_WIDTH)
    if image is None:
        raise FileNotFoundError(paths["ALBEDO_PATH"])

    height, width = image.shape[:2]
    shape = image.shape
    reflectance = bioskin_io.vectorize_image(image, device=device)
    skin_props, _, reference_rgb, _, _, reconstruction_error = bio_skin.reconstruct(reflectance)
    print(f"[p1] skin_props.shape = {tuple(skin_props.shape)}")
    print(f"[p1] reconstruction error = {reconstruction_error.mean().item():.6f}")

    bioskin_io.save_tensor_to_image(os.path.join(output_dir, face_id + ".png"),reference_rgb, shape, channels=3, cpu=use_cpu)
    

    mask = butterfly_mask(height, width)
    mask_flat = torch.from_numpy(mask.reshape(-1)).to(device)
    inside = mask.reshape(-1) > 0.5
    outside = ~inside
    
    
    # print(f"Computing the a* value for the flush_rgb image using OpenCV's LAB conversion.")
    clean_a_star = compute_a_star(reference_rgb, inside)
    print(f"[p1] clean a* = {clean_a_star:.4f}")
    
    # ── PUT THE FLOOR BLOCK HERE ────────────────────────────────────
    sp_floor = skin_props.clone()
    sp_floor[:, chromophore["MELANIN_TYPE_INDEX"]] = torch.clamp(
        sp_floor[:, chromophore["MELANIN_TYPE_INDEX"]] + float(amplitude["EU_Boost"]) * mask_flat, 0, 1)
    sp_floor[:, chromophore["HAEMO_TYPE_INDEX"]] = torch.clamp(
        sp_floor[:, chromophore["HAEMO_TYPE_INDEX"]] + float(amplitude["OXY_Boost"]) * mask_flat, 0, 1)
    _, floor_rgb, _, _ = bio_skin.skin_props_to_reflectance(sp_floor)
    floor_a_star = compute_a_star(floor_rgb, inside)
    print(f"[p1] recipe floor a* = {floor_a_star:.4f}  floor Δa* = {floor_a_star - clean_a_star:.4f}")
    
    
    probe_amps   = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35]
    probe_deltas = []
    for a in probe_amps:
        flush = apply_reciep(skin_props, mask_flat, bio_skin, amplitude, chromophore, a)
        probe_deltas.append(compute_a_star(flush, inside) - floor_a_star)
    peak_idx  = int(np.argmax(probe_deltas))
    face_high = probe_amps[peak_idx]
    max_delta = probe_deltas[peak_idx]
    print(f"[p1] face ceiling: amp={face_high:.3f}  max Δa*={max_delta:.3f}")

    
        # ── calibration pre-pass ──────────────────────────────────────
    severity   = config["severity_parameter"]
    calibrated = {}
    grade_names = list(severity.keys())

    for i, (grade_name, target) in enumerate(severity.items()):
        fraction      = (i + 1) / len(severity)
        capped_target = min(target, max_delta * fraction)
        amp, redness = calibrate_grade(
            skin_props, mask_flat, bio_skin,
            amplitude, chromophore,
            redness_target=capped_target,
            inside=inside,
            clean_a_star=floor_a_star,
            high=face_high,
            tolerance=0.005,
            max_iterations=6
        )
        calibrated[grade_name] = {"amp": float(amp), "redness": float(redness)}
        print(f"[calib] {grade_name}: amp={amp:.4f}  redness={redness:.4f}")

    calibrated_levels = [calibrated[g]["amp"] for g in calibrated]
    calibrated_grades = [
        {
            "amp":          calibrated[g]["amp"],
            "delta_a_star": calibrated[g]["redness"]
        }
        for g in calibrated
    ]

    # ── save mask / chromo BEFORE metadata (no dependencies) ──────
    bioskin_io.save_tensor_to_image(os.path.join(output_dir, "mask_feathered"), mask_flat.float(), shape, channels=1, cpu=use_cpu)
    # save_chromophore_column(skin_props, skin_props, shape, os.path.join(output_dir, "original.png"))
    save_chromophore_maps(skin_props, skin_props, shape, output_dir, suffix=f"{face_id}")


    # ── ITA + clean chromophores (must precede metadata) ──────────
    ita_result = face_ita(paths["ALBEDO_PATH"], mask)
    print(f"[p1] Fitzpatrick skin tone: {ita_result['ita']:.2f} ({ita_result['band']})")

    BLOOD_VOLUME_INDEX = chromophore["BLOOD_VOLUME_INDEX"]
    melanin_index      = chromophore["MELANIN_INDEX"]
    HAEMO_TYPE_INDEX   = chromophore["HAEMO_TYPE_INDEX"]
    eumelanin_index    = chromophore["MELANIN_TYPE_INDEX"]
    clean_hemoglobin   = in_mask_mean(skin_props[:, BLOOD_VOLUME_INDEX], inside)
    clean_melanin      = in_mask_mean(skin_props[:, melanin_index],      inside)
    clean_oxygenation  = in_mask_mean(skin_props[:, HAEMO_TYPE_INDEX],   inside)
    clean_eumelanin    = in_mask_mean(skin_props[:, eumelanin_index],    inside)
    print(f"[p1] Original Hemoglobin = {clean_hemoglobin:.4f}")
    print(f"[p1] Original Melanin    = {clean_melanin:.4f}")
    print(f"[p1] Original Oxygenation = {clean_oxygenation:.4f}")
    print(f"[p1] Original Eumelanin  = {clean_eumelanin:.4f}")

    # ── metadata CSV (all variables now defined) ──────────────────
    clean_chromophores = {
        "melanin":    clean_melanin,
        "hemoglobin": clean_hemoglobin,
        "oxygenation":clean_oxygenation,
        "eumelanin":  clean_eumelanin,
    }
    save_metadata_csv(
        output_dir         = output_dir,
        face_id            = face_id,
        ita_result         = ita_result,
        clean_chromophores = clean_chromophores,
        clean_a_star       = clean_a_star,
        floor_a_star       = floor_a_star,
        face_ceiling_delta = max_delta,
        calibrated_grades  = calibrated_grades,
        mask_type          = "butterfly",
        pipeline_version   = "v1.0"
    )

    # ── amplitude sweep ───────────────────────────────────────────
    levels = np.round(np.arange(amplitude["AMP_START"],
                                amplitude["AMP_STOP"] + amplitude["AMP_STEP"] / 2.0,
                                amplitude["AMP_STEP"]), 3)
    target_amplitude = float(levels[-1]) if COMPOSITE_AMP is None else float(COMPOSITE_AMP)
    print(f" amplitude start: {amplitude['AMP_START']:.3f}, stop: {amplitude['AMP_STOP']:.3f}, step: {amplitude['AMP_STEP']:.3f}")
    print(f"\n[sweep] amplitude levels: {list(levels)}")
    print(f"[sweep] target amplitude: {target_amplitude:.2f}")
    sp_composite, amp_composite = None, None

    rows = run_hemoglobin_oxy_direction(
        skin_props=skin_props, mask_flat=mask_flat, mask=mask,
        inside=inside, outside=outside,
        bio_skin=bio_skin, ref_vis_rgb=reference_rgb,
        shape=shape, prog_dir=output_dir,
        io=bioskin_io, C=chromophore, A=amplitude, P=paths,
        use_cpu=use_cpu, target_amp=calibrated_levels[-1],
        hemo_levels=calibrated_levels, clean_a_star=clean_a_star,
    )

    # build lookup by amp from the returned rows (no CSV needed)
    sweep_rows = {round(float(r["residual_amp"]), 6): r for r in rows}

    for grade in calibrated_grades:
        sr = sweep_rows.get(round(grade["amp"], 6), {})
        grade["contrast"]      = float(sr.get("in_mask_contrast",  0))
        grade["hemo_after"]    = float(sr.get("hemo_in_after",     0))
        grade["mel_drift"]     = float(sr.get("melanin_drift_in",  0))
        grade["out_drift"]     = float(sr.get("hemo_drift_out",    0))
        grade["oxy_out_drift"] = float(sr.get("oxy_drift_out",     0))
        grade["eu_out_drift"]  = float(sr.get("eu_drift_out",      0))

    # ── metadata CSV (all values now populated) ───────────────────
    save_metadata_csv(
        output_dir         = output_dir,
        face_id            = face_id,
        ita_result         = ita_result,
        clean_chromophores = clean_chromophores,
        clean_a_star       = clean_a_star,
        floor_a_star       = floor_a_star,
        face_ceiling_delta = max_delta,   # kept internally, just not in CSV
        calibrated_grades  = calibrated_grades,
        mask_type          = "butterfly",
        pipeline_version   = "v1.0"
    )
    


    print("\nDONE. Outputs in:", paths["OUTPUT_DIR"])


if __name__ == "__main__":
    main()





  # run_oxy_direction_test(
    #     skin_props=skin_props, mask_flat=mask_flat, inside=inside, outside=outside,
    #     bio_skin=bio_skin, ref_vis_rgb=reference_rgb, shape=shape, prog_dir=output_dir,
    #     io=io, C=chromophore, use_cpu=use_cpu,
    #     hemo_fixed=0.10,                                  # constant flush
    #     oxy_levels=np.round(np.arange(-0.15, 0.15+1e-9, 0.02), 3),  # ± sweep
    # )

    # run_hemoglobin_direction(
    #     skin_props=skin_props, mask_flat=mask_flat, mask=mask, inside=inside, outside=outside,
    #     bio_skin=bio_skin, ref_vis_rgb=reference_rgb, shape=shape, prog_dir=output_dir,
    #     io=io, C=chromophore, A=amplitude,
    #     use_cpu=use_cpu,target_amp=target_amplitude,
    #     hemo_levels= levels, 
    # )


















