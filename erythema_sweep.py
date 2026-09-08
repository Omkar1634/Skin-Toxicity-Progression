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

# ── Argument Parser ────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--resume", type=str, default=None)
args = parser.parse_args()



with open(args.config, "r",encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

P = cfg["paths"]
C = cfg["chromophore_parameter"]
A = cfg["amplitude_parameter"]
MA = cfg["mask_parameter"]


COMPOSITE_AMP = None

MAX_WIDTH    = 800
BATCH_SIZE   = 512000





# =============================================================================
# MAIN
# =============================================================================

def main():
    prog_dir = os.path.join(P["OUTPUT_DIR"], "progression")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prog_dir = os.path.join(prog_dir, timestamp)
    os.makedirs(prog_dir, exist_ok=True)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cpu = (device.type == 'cpu')
    print(f"[init] device = {device}")

    bio_skin = BioSkinInference(P["MODEL_PREFIX"], device=device, batch_size=BATCH_SIZE)

    # -- Encode ONCE --
    img = io.load_image(P["ALBEDO_PATH"], max_width=MAX_WIDTH)
    if img is None:
        raise FileNotFoundError(P["ALBEDO_PATH"])
    H, W  = img.shape[:2]
    shape = img.shape
    refl  = io.vectorize_image(img, device=device)

    (skin_props, ref_vis, ref_vis_rgb,ref_ir, ref_ir_avg, recon_err) = bio_skin.reconstruct(refl)
    print(f"[p1] skin_props.shape = {tuple(skin_props.shape)}")
    print(f"[p1] reconstruction error = {recon_err.mean().item():.6f}")

    io.save_tensor_to_image(os.path.join(prog_dir, "frame_00_amp0.00"),
                            ref_vis_rgb, shape, channels=3, cpu=use_cpu)

    # -- Mask ONCE --
    #mask = build_gaussian_mask(H, W, MA["MASK_CX_FRAC"], MA["MASK_CY_FRAC"],MA["MASK_RAD_FRAC"], MA["MASK_FEATHER"])
    mask = butterfly_mask(H, W)
    mask_flat = torch.from_numpy(mask.reshape(-1)).to(device)
    inside = mask.reshape(-1) > 0.5
    outside = ~inside
    # io.save_tensor_to_image(os.path.join(prog_dir, "mask_binary"), (mask_flat > 0.5).float(), shape, channels=1, cpu=use_cpu)
    
    # keep the binary for reference if you like, but also save the feathered one:
    io.save_tensor_to_image(os.path.join(prog_dir, "mask_feathered"),
                        mask_flat.float(), shape, channels=1, cpu=use_cpu)

    hemo_clean_in    = in_mask_mean(skin_props[:, C["HEMOGLOBIN_INDEX"]], inside)
    print(f"[p1] Original Hemoglobin = {hemo_clean_in:.4f}")
    melanin_clean_in = in_mask_mean(skin_props[:, C["MELANIN_INDEX"]], inside)
    print(f"[p1] Original Melanin = {melanin_clean_in:.4f}")
    oxy_clean_in    = in_mask_mean(skin_props[:, C["OXYGENATION_INDEX"]], inside)
    print(f"[p1] Original Oxygenation = {oxy_clean_in:.4f}")

    

    # -- Severity sweep --
    levels = np.round(np.arange(A["AMP_START"], A["AMP_STOP"] + A["AMP_STEP"] / 2.0, A["AMP_STEP"]), 3)
    print(f"\n[sweep] amplitude levels: {list(levels)}")

    target_amp = float(levels[-1]) if COMPOSITE_AMP is None else float(COMPOSITE_AMP)
    sp_composite, amp_composite = None, None


    # run_oxy_direction_test(
    #     skin_props=skin_props, mask_flat=mask_flat, inside=inside, outside=outside,
    #     bio_skin=bio_skin, ref_vis_rgb=ref_vis_rgb, shape=shape, prog_dir=prog_dir,
    #     io=io, C=C, use_cpu=use_cpu,
    #     hemo_fixed=0.10,                                  # constant flush
    #     oxy_levels=np.round(np.arange(-0.15, 0.15+1e-9, 0.02), 3),  # ± sweep
    # )

    run_hemoglobin_direction(
        skin_props=skin_props, mask_flat=mask_flat, inside=inside, outside=outside,
        bio_skin=bio_skin, ref_vis_rgb=ref_vis_rgb, shape=shape, prog_dir=prog_dir,
        io=io, C=C, use_cpu=use_cpu,target_amp=target_amp,
        hemo_levels= levels, 
    )

    

    print("\nDONE. Outputs in:", P["OUTPUT_DIR"])


if __name__ == "__main__":
    main()



























# rows = []
#     print(f"\n  {'idx':>3} {'amp':>6} {'contrast':>9} "
#           f"{'hemoIn->':>9} {'after':>7} {'melDrift':>9} {'outDrift':>9} ")
#     print("  " + "-" * 60)

#     for i, amp in enumerate(levels, start=1):
#         residual = float(amp) * mask_flat
#         sp = skin_props.clone()
        
#         sp[:, C["HEMOGLOBIN_INDEX"]] = torch.clamp(sp[:, C["HEMOGLOBIN_INDEX"]] + residual, 0.0, 1.0)

#         _, flush_rgb, _, _ = bio_skin.skin_props_to_reflectance(sp)

#         io.save_tensor_to_image(os.path.join(prog_dir, f"frame_{i:02d}_amp{amp:.2f}"),
#                                 flush_rgb, shape, channels=3, cpu=use_cpu)

#         diff = (flush_rgb - ref_vis_rgb)[inside]
#         contrast = float(torch.sqrt((diff ** 2).sum(dim=1)).mean().detach())

#         sp_check = bio_skin.reflectance_to_skin_props(flush_rgb.float())
#         hemo_after_in = in_mask_mean(sp_check[:, C["HEMOGLOBIN_INDEX"]], inside)
#         mel_after_in  = in_mask_mean(sp_check[:, C["MELANIN_INDEX"]], inside)
#         hemo_out_before = in_mask_mean(skin_props[:, C["HEMOGLOBIN_INDEX"]], outside)
#         hemo_out_after  = in_mask_mean(sp_check[:, C["HEMOGLOBIN_INDEX"]], outside)
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