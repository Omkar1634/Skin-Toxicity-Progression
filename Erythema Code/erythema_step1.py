#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
erythema_sweep.py
=================================================================================
Erythema pipeline with a SEVERITY SWEEP + a CHROMOPHORE PROGRESSION grid.
Built entirely on BioSkin's own I/O + inference wrappers. Edits the HEMOGLOBIN
CHROMOPHORE (skin_props col 1), never RGB — the decoder synthesizes the colour.

Outputs:
    progression/frame_00..NN                 severity RGB frames (00 = clean)
    erythema_progression_montage.jpeg
    erythema_sweep.csv
    erythema_control_curve.png
    erythema_chromophore_progression.png     <-- chromophore maps across ALL levels

skin_props cols: 0 melanin | 1 hemoglobin | 2 epi_thickness | 3 eumelanin_ratio
                 4 oxygenation | 5 occlusion(AO, keep, never edit)
=================================================================================
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # 3090 only; hides 5070 Ti; kills DataParallel
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"

import sys
import csv
import numpy as np
import torch
import cv2


# =============================================================================
# CONFIG
# =============================================================================

BIOSKIN_REPO = None
MODEL_PREFIX = r"D:\Github\PhD Code\Biophysical-LDM\Pretrain_Model\BioSkinAO"
ALBEDO_PATH  = r"D:\Github\PhD Code\Biophysical-LDM\dataset\Albedo-UV\000011.png"
OUTPUT_DIR   = r"D:\Github\PhD Code\Biophysical-LDM\erythema_out\experiments\erythema_sweepstep0.12"  # <-- change this to your desired output folder

MAX_WIDTH    = 800
BATCH_SIZE   = 512000

MELANIN_INDEX = 0
HEMO_INDEX    = 1

MASK_CX_FRAC  = 0.50
MASK_CY_FRAC  = 0.50
MASK_RAD_FRAC = 0.12
MASK_FEATHER  = 1.5

AMP_START = 0.12
AMP_STOP  = 0.50
AMP_STEP  = 0.12

# The 5 chromophores (col 5 = AO omitted) and their display colormaps.
PARAM_NAMES = ['melanin', 'hemoglobin', 'epidermal_thickness',
               'eumelanin_ratio', 'oxygenation']
PARAM_COLORMAPS = {
    'melanin':             'YlOrBr',
    'hemoglobin':          'Reds',
    'epidermal_thickness': 'Blues',
    'eumelanin_ratio':     'Oranges',
    'oxygenation':         'RdYlGn_r',
}


# =============================================================================
# IMPORTS
# =============================================================================

if BIOSKIN_REPO and BIOSKIN_REPO not in sys.path:
    sys.path.insert(0, BIOSKIN_REPO)

from bioskin.bioskin import BioSkinInference
import bioskin.utils.io as io


# =============================================================================
# HELPERS
# =============================================================================

def build_gaussian_mask(H, W, cx_frac, cy_frac, rad_frac, feather=1.0):
    cy, cx = cy_frac * H, cx_frac * W
    radius = rad_frac * min(H, W)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    sigma = max(radius * feather, 1.0)
    mask = np.exp(-d2 / (2.0 * sigma ** 2)).astype(np.float32)
    mask[mask < 0.01] = 0.0
    return mask


def in_mask_mean(tensor_col, inside_bool):
    return float(tensor_col.detach().cpu().numpy()[inside_bool].mean())


def make_chromophore_progression(clean_maps, levels_maps, shape, mask2d, out_path):
    """
    Rows = the 5 chromophores; columns = clean then each amplitude level.
    Read severity left->right, chromophore top->bottom. Colour scale is shared
    per-row (per-chromophore) across all columns so the progression is
    comparable. Only the hemoglobin row should intensify; the rest stay flat.

    clean_maps  : (N, 5) numpy
    levels_maps : list of (amp: float, maps: (N, 5) numpy)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    H, W = shape[0], shape[1]
    nrows = len(PARAM_NAMES)
    row_maps = [clean_maps] + [m for _, m in levels_maps]     # column order
    col_labels = ['clean'] + [f'amp {a:.2f}' for a, _ in levels_maps]
    ncols = len(row_maps)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.35 * ncols, 2.5 * nrows), squeeze=False)

    for j, name in enumerate(PARAM_NAMES):
        cm = PARAM_COLORMAPS[name]
        # shared scale for this chromophore across every column
        col = np.concatenate([M[:, j] for M in row_maps])
        vmin, vmax = float(col.min()), float(col.max())
        for k, M in enumerate(row_maps):
            ax = axes[j][k]
            ax.imshow(M[:, j].reshape(H, W), cmap=cm, vmin=vmin, vmax=vmax)
            if name == 'hemoglobin':
                ax.contour(mask2d, levels=[0.5], colors='k', linewidths=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_title(col_labels[k], fontsize=10)
        axes[j][0].set_ylabel(name, fontsize=11, rotation=90)

    fig.suptitle('Chromophore progression - each column is a severity level '
                 '(only hemoglobin should intensify)', fontsize=13)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.01,
                        wspace=0.05, hspace=0.05)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    prog_dir = os.path.join(OUTPUT_DIR, "progression")
    os.makedirs(prog_dir, exist_ok=True)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cpu = (device.type == 'cpu')
    print(f"[init] device = {device}")

    bio_skin = BioSkinInference(MODEL_PREFIX, device=device, batch_size=BATCH_SIZE)

    # -- Encode ONCE --
    img = io.load_image(ALBEDO_PATH, max_width=MAX_WIDTH)
    if img is None:
        raise FileNotFoundError(ALBEDO_PATH)
    H, W  = img.shape[:2]
    shape = img.shape
    refl  = io.vectorize_image(img, device=device)

    (skin_props, ref_vis, ref_vis_rgb,
     ref_ir, ref_ir_avg, recon_err) = bio_skin.reconstruct(refl)
    print(f"[p1] skin_props.shape = {tuple(skin_props.shape)}")
    print(f"[p1] reconstruction error = {recon_err.mean().item():.6f}")

    io.save_tensor_to_image(os.path.join(prog_dir, "frame_00_amp0.00"),
                            ref_vis_rgb, shape, channels=3, cpu=use_cpu)

    # -- Mask ONCE --
    mask = build_gaussian_mask(H, W, MASK_CX_FRAC, MASK_CY_FRAC,
                               MASK_RAD_FRAC, MASK_FEATHER)
    mask_flat = torch.from_numpy(mask.reshape(-1)).to(device)
    inside = mask.reshape(-1) > 0.5
    outside = ~inside
    io.save_tensor_to_image(os.path.join(prog_dir, "mask_binary"),
                            (mask_flat > 0.5).float(), shape, channels=1, cpu=use_cpu)

    hemo_clean_in    = in_mask_mean(skin_props[:, HEMO_INDEX], inside)
    melanin_clean_in = in_mask_mean(skin_props[:, MELANIN_INDEX], inside)

    # -- Severity sweep --
    levels = np.round(np.arange(AMP_START, AMP_STOP + AMP_STEP / 2.0, AMP_STEP), 3)
    print(f"\n[sweep] amplitude levels: {list(levels)}")

    levels_maps = []          # (amp, (N,5) numpy) for the progression grid
    rows = []
    print(f"\n  {'idx':>3} {'amp':>6} {'contrast':>9} "
          f"{'hemoIn->':>9} {'after':>7} {'melDrift':>9} {'outDrift':>9}")
    print("  " + "-" * 60)

    for i, amp in enumerate(levels, start=1):
        residual = float(amp) * mask_flat
        sp = skin_props.clone()
        sp[:, HEMO_INDEX] = torch.clamp(sp[:, HEMO_INDEX] + residual, 0.0, 1.0)

        _, flush_rgb, _, _ = bio_skin.skin_props_to_reflectance(sp)

        io.save_tensor_to_image(os.path.join(prog_dir, f"frame_{i:02d}_amp{amp:.2f}"),
                                flush_rgb, shape, channels=3, cpu=use_cpu)

        diff = (flush_rgb - ref_vis_rgb)[inside]
        contrast = float(torch.sqrt((diff ** 2).sum(dim=1)).mean())

        sp_check = bio_skin.reflectance_to_skin_props(flush_rgb.float())
        hemo_after_in = in_mask_mean(sp_check[:, HEMO_INDEX], inside)
        mel_after_in  = in_mask_mean(sp_check[:, MELANIN_INDEX], inside)
        hemo_out_before = in_mask_mean(skin_props[:, HEMO_INDEX], outside)
        hemo_out_after  = in_mask_mean(sp_check[:, HEMO_INDEX], outside)
        mel_drift = mel_after_in - melanin_clean_in
        out_drift = hemo_out_after - hemo_out_before

        print(f"  {i:>3} {amp:>6.2f} {contrast:>9.4f} "
              f"{hemo_clean_in:>9.4f} {hemo_after_in:>7.4f} "
              f"{mel_drift:>+9.4f} {out_drift:>+9.4f}")

        rows.append({
            'index': i, 'residual_amp': float(amp), 'in_mask_contrast': contrast,
            'hemo_in_clean': hemo_clean_in, 'hemo_in_after': hemo_after_in,
            'melanin_drift_in': mel_drift, 'hemo_drift_out': out_drift,
        })

        # collect the 5 param maps for the progression grid (every level)
        levels_maps.append((float(amp), sp[:, :5].detach().cpu().numpy()))

    # -- CSV --
    csv_path = os.path.join(OUTPUT_DIR, "erythema_sweep.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[out] CSV -> {csv_path}")

    # -- Progression montage (RGB) --
    frame_files = sorted([f for f in os.listdir(prog_dir)
                          if f.startswith("frame_") and f.endswith(".jpeg")])
    panels, panel_w = [], 240
    for f in frame_files:
        im = cv2.imread(os.path.join(prog_dir, f))
        if im is None:
            continue
        h = int(im.shape[0] * panel_w / im.shape[1])
        im = cv2.resize(im, (panel_w, h), interpolation=cv2.INTER_AREA)
        label = f.split("_amp")[-1].replace(".jpeg", "")
        cv2.putText(im, f"amp {label}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(im)
    if panels:
        cv2.imwrite(os.path.join(OUTPUT_DIR, "erythema_progression_montage.jpeg"),
                    cv2.hconcat(panels))
        print("[out] montage -> erythema_progression_montage.jpeg")

    # -- Control curve --
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        amps  = [r['residual_amp'] for r in rows]
        after = [r['hemo_in_after'] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axhline(hemo_clean_in, ls='--', c='gray', label='clean in-mask hemoglobin')
        ax.plot(amps, after, marker='o', label='recovered in-mask hemoglobin')
        ax.set_xlabel('residual amplitude (input)')
        ax.set_ylabel('in-mask hemoglobin (re-encoded)')
        ax.set_title('Erythema control curve - should rise monotonically')
        ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "erythema_control_curve.png"), dpi=150)
        plt.close(fig)
        print("[out] control curve -> erythema_control_curve.png")
    except Exception as e:
        print(f"[out] (skipped control curve: {e})")

    # -- Chromophore progression grid (all levels) --
    try:
        comp_path = os.path.join(OUTPUT_DIR, "erythema_chromophore_progression.png")
        clean_maps = skin_props[:, :5].detach().cpu().numpy()
        make_chromophore_progression(clean_maps, levels_maps, shape, mask, comp_path)
        print(f"[out] chromophore progression -> {comp_path}")
    except Exception as e:
        print(f"[out] (skipped chromophore progression: {e})")

    print("\nDONE. Outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()