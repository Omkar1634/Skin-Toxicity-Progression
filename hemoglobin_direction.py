import os
import csv
import numpy as np
import torch
import cv2


def in_mask_mean(col, inside_bool):
    return float(col.detach().cpu().numpy()[inside_bool].mean())

def run_hemoglobin_direction(skin_props, mask_flat, inside, outside,
                               bio_skin, ref_vis_rgb, shape, prog_dir,
                               io, C, A,use_cpu=False, target_amp=None,
                               hemo_levels=None):
    """

    hemo_levels : symmetric list of hemoglobin amplitudes, e.g.
                  np.round(np.arange(-0.15, 0.15 + 1e-9, 0.02), 3)
    """

    HEMO = C["HEMOGLOBIN_INDEX"]
    MEL  = C["MELANIN_INDEX"]
    Amplitude= A


    hemo_clean_in = in_mask_mean(skin_props[:, HEMO], inside)
    mel_clean_in  = in_mask_mean(skin_props[:, MEL],  inside)

    print(f"\n[hemoA] hemoglobin at {hemo_clean_in:.2f} (constant every frame)")
    print(f"[hemoA] sweeping hemoglobin over {list(hemo_levels)}")
    print(f"\n  {'idx':>3} {'amp':>6} {'contrast':>9} "
              f"{'hemoIn->':>9} {'after':>7} {'melDrift':>9} {'outDrift':>9} ")
    print("  " + "-" * 72)

    rows = []
    for i, amp in enumerate(hemo_levels, start=1):
        sp = skin_props.clone()                       # fresh baseline EVERY frame

        # 1) Hemoglobin flush 
        sp[:, HEMO] = torch.clamp(sp[:, HEMO] + float(hemo_clean_in) * mask_flat, 0.0, 1.0)
        
        _, flush_rgb, _, _ = bio_skin.skin_props_to_reflectance(sp)

        
        io.save_tensor_to_image(os.path.join(prog_dir, f"frame_{i:02d}_amp{amp:.2f}"),
                                        flush_rgb, shape, channels=3, cpu=use_cpu)
        
        diff = (flush_rgb - ref_vis_rgb)[inside]
        contrast = float(torch.sqrt((diff ** 2).sum(dim=1)).mean().detach())

        sp_check = bio_skin.reflectance_to_skin_props(flush_rgb.float())
        hemo_after_in = in_mask_mean(sp_check[:, C["HEMOGLOBIN_INDEX"]], inside)
        mel_after_in  = in_mask_mean(sp_check[:, C["MELANIN_INDEX"]], inside)
        hemo_out_before = in_mask_mean(skin_props[:, C["HEMOGLOBIN_INDEX"]], outside)
        hemo_out_after  = in_mask_mean(sp_check[:, C["HEMOGLOBIN_INDEX"]], outside)
        mel_drift = mel_after_in - mel_clean_in
        out_drift = hemo_out_after - hemo_out_before
        
        print(f"  {i:>3} {amp:>6.2f} {contrast:>9.4f} " f"{hemo_clean_in:>9.4f} {hemo_after_in:>7.4f} " 
                      f"{mel_drift:>+9.4f} {out_drift:>+9.4f} ")
        
        rows.append({
            'index': i, 'residual_amp': float(amp), 'in_mask_contrast': contrast,
            'hemo_in_clean': hemo_clean_in, 'hemo_in_after': hemo_after_in,
            'melanin_drift_in': mel_drift, 'hemo_drift_out': out_drift,
        })

        # capture edited params for the composite at the chosen level
        if abs(float(amp) - target_amp) < Amplitude["AMP_STEP"] / 2.0:
            sp_composite, amp_composite = sp.clone(), float(amp)

    # -- CSV --
    csv_path = os.path.join(prog_dir, "erythema_sweep.csv")
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
        cv2.imwrite(os.path.join(prog_dir, "erythema_progression_montage.jpeg"),
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
        plt.savefig(os.path.join(prog_dir, "erythema_control_curve.png"), dpi=150)
        plt.close(fig)
        print("[out] control curve -> erythema_control_curve.png")
    except Exception as e:
        print(f"[out] (skipped control curve: {e})")

    # -- Chromophore composite (clean vs edited vs delta) --
    if sp_composite is not None:
        try:
            comp_path = os.path.join(prog_dir, "erythema_chromophore_composite.png")
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
            
    return rows        

    