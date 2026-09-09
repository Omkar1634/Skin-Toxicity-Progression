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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # 3090 only; hides 5070 Ti; kills DataParallel
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"

import sys
import csv
import numpy as np
import torch
import cv2
import argparse
import datetime 
from helper import  in_mask_mean, make_chromophore_composite, PARAM_NAMES, PARAM_COLORMAPS
from oxy_direction_test import run_oxy_direction_test
from hemoglobin_direction import run_hemoglobin_direction
from mixes_sweep import run_hemoglobin_oxy_direction
from mask import butterfly_mask, build_gaussian_mask
import yaml 

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

    bioskin_io.save_tensor_to_image(os.path.join(output_dir, "frame_00_amp0.00"),reference_rgb, shape, channels=3, cpu=use_cpu)

    mask = butterfly_mask(height, width)
    mask_flat = torch.from_numpy(mask.reshape(-1)).to(device)
    inside = mask.reshape(-1) > 0.5
    outside = ~inside
    bioskin_io.save_tensor_to_image(os.path.join(output_dir, "mask_feathered"), mask_flat.float(), shape, channels=1, cpu=use_cpu)

    BLOOD_VOLUME_INDEX = chromophore["BLOOD_VOLUME_INDEX"]
    melanin_index = chromophore["MELANIN_INDEX"]
    HAEMO_TYPE_INDEX = chromophore["HAEMO_TYPE_INDEX"]
    eumelanin_index = chromophore["MELANIN_TYPE_INDEX"]
    clean_hemoglobin = in_mask_mean(skin_props[:, BLOOD_VOLUME_INDEX], inside)
    clean_melanin = in_mask_mean(skin_props[:, melanin_index], inside)
    clean_oxygenation = in_mask_mean(skin_props[:, HAEMO_TYPE_INDEX], inside)
    clean_eumelanin = in_mask_mean(skin_props[:, eumelanin_index], inside)
    print(f"[p1] Original Hemoglobin = {clean_hemoglobin:.4f}")
    print(f"[p1] Original Melanin = {clean_melanin:.4f}")
    print(f"[p1] Original Oxygenation = {clean_oxygenation:.4f}")
    print(f"[p1] Original Eumelanin = {clean_eumelanin:.4f}")

    levels = np.round(np.arange(amplitude["AMP_START"],
                                amplitude["AMP_STOP"] + amplitude["AMP_STEP"] / 2.0,
                                amplitude["AMP_STEP"]), 3)
    
    target_amplitude = float(levels[-1]) if COMPOSITE_AMP is None else float(COMPOSITE_AMP)

    # target_amplitude = mask_config.get("COMPOSITE_AMP")
    # target_amplitude = float(levels[-1] if target_amplitude is None else target_amplitude)
    print(f" amplitude start: {amplitude['AMP_START']:.3f}, stop: {amplitude['AMP_STOP']:.3f}, step: {amplitude['AMP_STEP']:.3f}")
    print(f"\n[sweep] amplitude levels: {list(levels)}")
    print(f"[sweep] target amplitude: {target_amplitude:.2f}")
    sp_composite, amp_composite = None, None


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
    
    run_hemoglobin_oxy_direction(
        skin_props=skin_props, mask_flat=mask_flat, mask=mask, inside=inside, outside=outside,
        bio_skin=bio_skin, ref_vis_rgb=reference_rgb, shape=shape, prog_dir=output_dir,
        io=io, C=chromophore, A=amplitude,
        use_cpu=use_cpu,target_amp=target_amplitude,
        hemo_levels= levels, 
    )

    

    print("\nDONE. Outputs in:", paths["OUTPUT_DIR"])


if __name__ == "__main__":
    main()



























# rows = []
#     print(f"\n  {'idx':>3} {'amp':>6} {'contrast':>9} "
#           f"{'hemoIn->':>9} {'after':>7} {'melDrift':>9} {'outDrift':>9} ")
#     print("  " + "-" * 60)

#     for i, amp in enumerate(levels, start=1):
#         residual = float(amp) * mask_flat
#         sp = skin_props.clone()
        
#         sp[:, C["BLOOD_VOLUME_INDEX"]] = torch.clamp(sp[:, C["BLOOD_VOLUME_INDEX"]] + residual, 0.0, 1.0)

#         _, flush_rgb, _, _ = bio_skin.skin_props_to_reflectance(sp)

#         io.save_tensor_to_image(os.path.join(prog_dir, f"frame_{i:02d}_amp{amp:.2f}"),
#                                 flush_rgb, shape, channels=3, cpu=use_cpu)

#         diff = (flush_rgb - ref_vis_rgb)[inside]
#         contrast = float(torch.sqrt((diff ** 2).sum(dim=1)).mean().detach())

#         sp_check = bio_skin.reflectance_to_skin_props(flush_rgb.float())
#         hemo_after_in = in_mask_mean(sp_check[:, C["BLOOD_VOLUME_INDEX"]], inside)
#         mel_after_in  = in_mask_mean(sp_check[:, C["MELANIN_INDEX"]], inside)
#         hemo_out_before = in_mask_mean(skin_props[:, C["BLOOD_VOLUME_INDEX"]], outside)
#         hemo_out_after  = in_mask_mean(sp_check[:, C["BLOOD_VOLUME_INDEX"]], outside)
#         mel_drift = mel_after_in - melanin_clean_in
#         out_drift = hemo_out_after - hemo_out_before

#         print(f"  {i:>3} {amp:>6.2f} {contrast:>9.4f} " f"{hemo_clean_in:>9.4f} {hemo_after_in:>7.4f} " 
#               f"{mel_drift:>+9.4f} {out_drift:>+9.4f} ")

#         rows.append({
#             'index': i, 'residual_amp': float(amp), 'in_mask_contrast': contrast,
#             'hemo_in_clean': hemo_clean_in, 'hemo_in_after': hemo_after_in,
#             'melanin_drift_in': mel_drift, 'hemo_drift_out': out_drift,
#         })

#         # capture edited params for the composite at the chosen level
#         if abs(float(amp) - target_amp) < A["AMP_STEP"] / 2.0:
#             sp_composite, amp_composite = sp.clone(), float(amp)

#     # -- CSV --
#     csv_path = os.path.join(prog_dir, "erythema_sweep.csv")
#     with open(csv_path, 'w', newline='') as f:
#         w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
#         w.writeheader(); w.writerows(rows)
#     print(f"\n[out] CSV -> {csv_path}")

#     # -- Progression montage --
#     frame_files = sorted([f for f in os.listdir(prog_dir)
#                           if f.startswith("frame_") and f.endswith(".jpeg")])
#     panels, panel_w = [], 240
#     for f in frame_files:
#         im = cv2.imread(os.path.join(prog_dir, f))
#         if im is None:
#             continue
#         h = int(im.shape[0] * panel_w / im.shape[1])
#         im = cv2.resize(im, (panel_w, h), interpolation=cv2.INTER_AREA)
#         label = f.split("_amp")[-1].replace(".jpeg", "")
#         cv2.putText(im, f"amp {label}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6, (255, 255, 255), 2, cv2.LINE_AA)
#         panels.append(im)
#     if panels:
#         cv2.imwrite(os.path.join(prog_dir, "erythema_progression_montage.jpeg"),
#                     cv2.hconcat(panels))
#         print("[out] montage -> erythema_progression_montage.jpeg")

#     # -- Control curve --
#     try:
#         import matplotlib
#         matplotlib.use('Agg')
#         import matplotlib.pyplot as plt
#         amps  = [r['residual_amp'] for r in rows]
#         after = [r['hemo_in_after'] for r in rows]
#         fig, ax = plt.subplots(figsize=(7, 4))
#         ax.axhline(hemo_clean_in, ls='--', c='gray', label='clean in-mask hemoglobin')
#         ax.plot(amps, after, marker='o', label='recovered in-mask hemoglobin')
#         ax.set_xlabel('residual amplitude (input)')
#         ax.set_ylabel('in-mask hemoglobin (re-encoded)')
#         ax.set_title('Erythema control curve - should rise monotonically')
#         ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
#         plt.savefig(os.path.join(prog_dir, "erythema_control_curve.png"), dpi=150)
#         plt.close(fig)
#         print("[out] control curve -> erythema_control_curve.png")
#     except Exception as e:
#         print(f"[out] (skipped control curve: {e})")

#     # -- Chromophore composite (clean vs edited vs delta) --
#     if sp_composite is not None:
#         try:
#             comp_path = os.path.join(prog_dir, "erythema_chromophore_composite.png")
#             deltas = make_chromophore_composite(
#                 skin_props, sp_composite, shape, mask, comp_path, f"{amp_composite:.2f}")
#             print(f"[out] chromophore composite -> {comp_path}")
#             print("\n[composite] in-mask mean delta per chromophore (edited - clean):")
#             for name in PARAM_NAMES:
#                 flag = "  <-- target" if name == 'hemoglobin' else \
#                        ("  (should be ~0)" if name == 'melanin' else "")
#                 print(f"    {name:22s} {deltas[name]:+.4f}{flag}")
#         except Exception as e:
#             print(f"[out] (skipped composite: {e})")