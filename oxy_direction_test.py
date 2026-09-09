# =============================================================================
# oxy_direction_test.py  -- Level A: isolate oxygenation direction
#
# Hemoglobin is HELD FIXED (a constant flush, same every frame).
# Oxygenation is the ONLY thing swept, across +/- levels.
# So the montage shows a constant flush whose hue shifts purely with oxygenation
# -> you can finally read which direction (if any) warms it toward red.
#
# This replaces the sweep loop in your main() for this one experiment. Everything
# up to `skin_props` (encode once) and `mask_flat` (butterfly mask) stays as you
# already have it. Call run_oxy_direction_test(...) instead of your current loop.
# =============================================================================

import os
import csv
import numpy as np
import torch
import cv2


def _in_mask_mean(col, inside_bool):
    return float(col.detach().cpu().numpy()[inside_bool].mean())


def _in_mask_redness(flush_rgb, inside_bool):
    """R - (G+B)/2 over masked pixels. BioSkin output is RGB order (0=R,1=G,2=B)."""
    x = flush_rgb.detach().cpu().numpy()
    # CORRECT — BGR order (cv2 EXR output)
    b, g, r = x[:, 0], x[:, 1], x[:, 2]
    return float((r - 0.5 * (b + g))[inside_bool].mean())


def run_oxy_direction_test(skin_props, mask_flat, inside, outside,
                           bio_skin, ref_vis_rgb, shape, prog_dir,
                           io, C, use_cpu=False,
                           hemo_fixed=0.10,
                           oxy_levels=None):
    """
    Fixed hemoglobin background + swept oxygenation (the clean direction test).

    hemo_fixed : constant hemoglobin bump applied EVERY frame (not swept).
    oxy_levels : symmetric list of oxygenation amplitudes, e.g.
                 np.round(np.arange(-0.15, 0.15 + 1e-9, 0.02), 3)
    """
    if oxy_levels is None:
        oxy_levels = np.round(np.arange(-0.15, 0.15 + 1e-9, 0.02), 3)

    HEMO = C["BLOOD_VOLUME_INDEX"]
    OXY  = C["HAEMO_TYPE_INDEX"]
    MEL  = C["MELANIN_INDEX"]

    hemo_clean_in = _in_mask_mean(skin_props[:, HEMO], inside)
    oxy_clean_in  = _in_mask_mean(skin_props[:, OXY],  inside)
    mel_clean_in  = _in_mask_mean(skin_props[:, MEL],  inside)

    print(f"\n[oxyA] hemoglobin HELD FIXED at +{hemo_fixed:.2f} (constant every frame)")
    print(f"[oxyA] sweeping oxygenation over {list(oxy_levels)}")
    print(f"\n  {'idx':>3} {'oxyAmp':>7} {'contrast':>9} {'hemoAft':>8} "
          f"{'oxyAft':>8} {'oxyDrift':>9} {'melDrift':>9} {'redness':>8}")
    print("  " + "-" * 72)

    rows = []
    for i, oxy_amp in enumerate(oxy_levels, start=1):
        sp = skin_props.clone()                       # fresh baseline EVERY frame

        # 1) FIXED hemoglobin flush -- same constant, never uses oxy_amp
        sp[:, HEMO] = torch.clamp(sp[:, HEMO] + float(hemo_fixed) * mask_flat, 0.0, 1.0)

        # 2) SWEPT oxygenation -- the ONLY thing that changes frame to frame
        sp[:, OXY] = torch.clamp(sp[:, OXY] + float(oxy_amp) * mask_flat, 0.0, 1.0)

        # decode edited maps -> albedo
        _, flush_rgb, _, _ = bio_skin.skin_props_to_reflectance(sp)

        # signed filename; montage sorts by the frame index, so order stays monotonic
        io.save_tensor_to_image(
            os.path.join(prog_dir, f"frame_{i:02d}_oxy{oxy_amp:+.2f}"),
            flush_rgb, shape, channels=3, cpu=use_cpu)

        # metrics
        diff = (flush_rgb - ref_vis_rgb)[inside]
        contrast = float(torch.sqrt((diff ** 2).sum(dim=1)).mean().detach())

        sp_check = bio_skin.reflectance_to_skin_props(flush_rgb.float())
        hemo_after = _in_mask_mean(sp_check[:, HEMO], inside)
        oxy_after  = _in_mask_mean(sp_check[:, OXY],  inside)
        mel_after  = _in_mask_mean(sp_check[:, MEL],  inside)
        redness    = _in_mask_redness(flush_rgb, inside)

        oxy_drift = oxy_after - oxy_clean_in
        mel_drift = mel_after - mel_clean_in

        print(f"  {i:>3} {oxy_amp:>+7.2f} {contrast:>9.4f} {hemo_after:>8.4f} "
              f"{oxy_after:>8.4f} {oxy_drift:>+9.4f} {mel_drift:>+9.4f} {redness:>+8.4f}")

        rows.append({
            'index': i, 'oxy_amp': float(oxy_amp), 'hemo_fixed': float(hemo_fixed),
            'contrast': contrast, 'hemo_after': hemo_after, 'oxy_after': oxy_after,
            'oxy_drift': oxy_drift, 'melanin_drift': mel_drift,
            'in_mask_redness': redness,
        })

    # CSV
    csv_path = os.path.join(prog_dir, "oxy_direction_test.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[oxyA] CSV -> {csv_path}")

    # montage (sorted by frame index -> -0.15 ... +0.15 in order)
    frame_files = sorted([f for f in os.listdir(prog_dir)
                          if f.startswith("frame_") and f.endswith(".jpeg")])
    panels, panel_w = [], 240
    for f in frame_files:
        im = cv2.imread(os.path.join(prog_dir, f))
        if im is None:
            continue
        h = int(im.shape[0] * panel_w / im.shape[1])
        im = cv2.resize(im, (panel_w, h), interpolation=cv2.INTER_AREA)
        label = f.split("_oxy")[-1].replace(".jpeg", "")
        cv2.putText(im, f"oxy {label}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(im)
    if panels:
        out = os.path.join(prog_dir, "oxy_direction_montage.jpeg")
        cv2.imwrite(out, cv2.hconcat(panels))
        print(f"[oxyA] montage -> {out}")

    # redness-vs-oxygenation plot (the number that answers 'which way warms it')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        xs = [r['oxy_amp'] for r in rows]
        ys = [r['in_mask_redness'] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axvline(0.0, ls='--', c='gray')
        ax.plot(xs, ys, marker='o')
        ax.set_xlabel('oxygenation amplitude (hemoglobin held fixed)')
        ax.set_ylabel('in-mask redness  R-(G+B)/2')
        ax.set_title('Does oxygenation warm the flush?  (up = redder)')
        ax.grid(alpha=0.3); plt.tight_layout()
        p = os.path.join(prog_dir, "oxy_redness_curve.png")
        plt.savefig(p, dpi=150); plt.close(fig)
        print(f"[oxyA] redness curve -> {p}")
    except Exception as e:
        print(f"[oxyA] (skipped redness curve: {e})")

    return rows
