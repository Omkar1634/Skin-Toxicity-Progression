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
MODEL_PREFIX = r"D:\Github\PhD Code\Biophysical-LDM\Bioskin_pretrain_model\BioSkinAO"
ALBEDO_PATH  = r"D:\Github\PhD Code\Biophysical-LDM\dataset\Albedo-UV\000011.png"
OUTPUT_DIR   = r"D:\Github\PhD Code\Biophysical-LDM\Erythema Code\experiments\erythema_sweepstep0.08"

MAX_WIDTH    = 800
BATCH_SIZE   = 512000

MELANIN_INDEX = 0
HEMO_INDEX    = 1

MASK_CX_FRAC  = 0.50
MASK_CY_FRAC  = 0.50
MASK_RAD_FRAC = 0.12
MASK_FEATHER  = 1.5

AMP_START = 0.20
AMP_STOP  = 0.50
AMP_STEP  = 0.08

# Which sweep level to use for the chromophore composite.
# None -> use the strongest (last) level, where changes are clearest.
COMPOSITE_AMP = None

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


def make_chromophore_composite(sp_clean, sp_edited, shape, mask2d, out_path, amp_label):
    """
    3 rows (clean / edited / delta) x 5 chromophores. The delta row uses a
    diverging colormap centred at 0, with the mask outline overlaid — so you can
    SEE hemoglobin rise inside the mask while melanin & others stay flat.
    Returns {param: in-mask mean delta} for a numeric companion to the figure.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    H, W = shape[0], shape[1]
    clean = sp_clean.detach().cpu().numpy()
    edit  = sp_edited.detach().cpu().numpy()
    inside = mask2d.reshape(-1) > 0.5
    ncol = len(PARAM_NAMES)

    fig, axes = plt.subplots(3, ncol, figsize=(3.1 * ncol, 9.2))
    deltas = {}
    for j, name in enumerate(PARAM_NAMES):
        cm = PARAM_COLORMAPS[name]
        c = clean[:, j].reshape(H, W)
        e = edit[:, j].reshape(H, W)
        d = e - c
        vmin, vmax = min(c.min(), e.min()), max(c.max(), e.max())

        axes[0, j].imshow(c, cmap=cm, vmin=vmin, vmax=vmax)
        axes[0, j].set_title(name, fontsize=11)
        axes[1, j].imshow(e, cmap=cm, vmin=vmin, vmax=vmax)

        amax = float(np.abs(d).max()) + 1e-8
        imd = axes[2, j].imshow(d, cmap='RdBu_r', vmin=-amax, vmax=amax)
        axes[2, j].contour(mask2d, levels=[0.5], colors='k', linewidths=0.6)
        fig.colorbar(imd, ax=axes[2, j], fraction=0.046, pad=0.04)

        for r in range(3):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])

        deltas[name] = float(d.reshape(-1)[inside].mean())

    for y, txt in zip((0.80, 0.50, 0.20),
                      ('clean', 'erythema (amp ' + amp_label + ')', 'delta (edited - clean)')):
        fig.text(0.015, y, txt, rotation=90, va='center', fontsize=12, weight='bold')

    fig.suptitle('Chromophore maps - only hemoglobin should rise inside the mask',
                 fontsize=13)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.02,
                        wspace=0.15, hspace=0.08)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return deltas


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

    target_amp = float(levels[-1]) if COMPOSITE_AMP is None else float(COMPOSITE_AMP)
    sp_composite, amp_composite = None, None

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

        # capture edited params for the composite at the chosen level
        if abs(float(amp) - target_amp) < AMP_STEP / 2.0:
            sp_composite, amp_composite = sp.clone(), float(amp)

    # -- CSV --
    csv_path = os.path.join(OUTPUT_DIR, "erythema_sweep.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[out] CSV -> {csv_path}")

    # -- Progression montage --
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

    # -- Chromophore composite (clean vs edited vs delta) --
    if sp_composite is not None:
        try:
            comp_path = os.path.join(OUTPUT_DIR, "erythema_chromophore_composite.png")
            deltas = make_chromophore_composite(
                skin_props, sp_composite, shape, mask, comp_path, f"{amp_composite:.2f}")
            print(f"[out] chromophore composite -> {comp_path}")
            print("\n[composite] in-mask mean delta per chromophore (edited - clean):")
            for name in PARAM_NAMES:
                flag = "  <-- target" if name == 'hemoglobin' else \
                       ("  (should be ~0)" if name == 'melanin' else "")
                print(f"    {name:22s} {deltas[name]:+.4f}{flag}")
        except Exception as e:
            print(f"[out] (skipped composite: {e})")

    print("\nDONE. Outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
