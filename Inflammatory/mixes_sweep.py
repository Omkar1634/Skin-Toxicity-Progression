import os
import csv
import numpy as np
import torch
import cv2
from utils.helper import save_montage, save_control_curve,save_chromophore_maps,  print_composite_deltas,make_chromophore_composite, save_control_allcurve,save_chromophore_column, compute_a_star


def in_mask_mean(col, inside_bool):
    return float(col.detach().cpu().numpy()[inside_bool].mean())

def _in_mask_redness(flush_rgb, inside_bool):
    """R - (G+B)/2 over masked pixels. BioSkin output is RGB order (0=R,1=G,2=B)."""
    x = flush_rgb.detach().cpu().numpy()
    # CORRECT — BGR order (cv2 EXR output)
    b, g, r = x[:, 0], x[:, 1], x[:, 2]
    return float((r - 0.5 * (b + g))[inside_bool].mean())

def run_hemoglobin_oxy_direction(skin_props, mask_flat, mask, inside, outside,
                                 bio_skin, ref_vis_rgb, shape, prog_dir,
                                 io, C, A, P, use_cpu=False, target_amp=None,
                                 hemo_levels=None, clean_a_star=0.0):
    """

    hemo_levels : symmetric list of hemoglobin amplitudes, e.g.
                  np.round(np.arange(-0.15, 0.15 + 1e-9, 0.02), 3)
    """

    HEMO = C["BLOOD_VOLUME_INDEX"]
    MEL  = C["MELANIN_INDEX"]
    OXY  = C["HAEMO_TYPE_INDEX"]
    EU   = C["MELANIN_TYPE_INDEX"]
    
    paths = P["ALBEDO_PATH"]
    
    face_id = os.path.splitext(os.path.basename(paths))[0]


    Amplitude= A
    print(F"\n OXY boost at {Amplitude['OXY_Boost']:.3f}")
    print(F"\n EU boost at {Amplitude['EU_Boost']:.3f}")


    hemo_clean_in = in_mask_mean(skin_props[:, HEMO], inside)
    mel_clean_in  = in_mask_mean(skin_props[:, MEL],  inside)
    oxy_clean_in  = in_mask_mean(skin_props[:, OXY],  inside)
    eu_clean_in   = in_mask_mean(skin_props[:, EU],   inside)
    print(f"\n[hemoA] hemoglobin at {hemo_clean_in:.2f} (constant every frame)")
    print(f"[hemoA] sweeping hemoglobin over {list(hemo_levels)}")
    print(f"\n  {'idx':>3} {'amp':>6} {'contrast':>9} "
              f"{'hemoIn->':>9} {'after':>7} {'melDrift':>9} {'outDrift':>9} {'oxyOutDrift':>9} {'euOutDrift':>9} {'redness':>9}")
    print("  " + "-" * 72)
    sp_composite, amp_composite = None, None   # add these TWO lines before the loop
    rows = []
    for i, amp in enumerate(hemo_levels, start=1):
        sp = skin_props.clone()                       # fresh baseline EVERY frame

        # 1) Hemoglobin flush 
        sp[:, HEMO] = torch.clamp(sp[:, HEMO] + float(amp) * mask_flat, 0.0, 1.0)
    
        sp[:, OXY] = torch.clamp(sp[:, OXY] + float(Amplitude["OXY_Boost"]) * mask_flat, 0.0, 1.0)
        
        sp[:, EU] = torch.clamp(sp[:, EU] + float(Amplitude["EU_Boost"]) * mask_flat, 0.0, 1.0)
        
        _, flush_rgb, _, _ = bio_skin.skin_props_to_reflectance(sp)
        
        cea_labels = {1: "1", 2: "2", 3: "3", 4: "4"}
        grade_label = cea_labels.get(i, f"grade_{i}")
        io.save_tensor_to_image(os.path.join(prog_dir, f"cea_{face_id}_{grade_label}"),
                        flush_rgb, shape, channels=3, cpu=use_cpu)
        
        save_chromophore_maps(skin_props, sp, shape, prog_dir, suffix=f"cea_{face_id}_{grade_label}")
        
        diff = (flush_rgb - ref_vis_rgb)[inside]
        contrast = float(torch.sqrt((diff ** 2).sum(dim=1)).mean().detach())

        sp_check = bio_skin.reflectance_to_skin_props(flush_rgb.float())
        hemo_after_in = in_mask_mean(sp_check[:,HEMO], inside)
        mel_after_in  = in_mask_mean(sp_check[:, MEL], inside)
        oxy_after_in  = in_mask_mean(sp_check[:, OXY], inside)
        eu_after_in   = in_mask_mean(sp_check[:, EU], inside)
        hemo_out_before = in_mask_mean(skin_props[:, HEMO], outside)
        hemo_out_after  = in_mask_mean(sp_check[:, HEMO], outside)
        oxy_out_before = in_mask_mean(skin_props[:, OXY], outside)
        oxy_out_after  = in_mask_mean(sp_check[:, OXY], outside)
        eu_out_before = in_mask_mean(skin_props[:, EU], outside)
        eu_out_after  = in_mask_mean(sp_check[:, EU], outside)
        eu_drift = eu_out_after - eu_out_before
        mel_drift = mel_after_in - mel_clean_in
        out_drift = hemo_out_after - hemo_out_before
        oxy_out_drift = oxy_out_after - oxy_out_before
        flush_a_star = compute_a_star(flush_rgb, inside)
        redness      = flush_a_star - clean_a_star

        
        print(f"  {i:>3} {amp:>6.2f} {contrast:>9.4f} " f"{hemo_clean_in:>9.4f} {hemo_after_in:>7.4f} " 
                      f"{mel_drift:>+9.4f} {out_drift:>+9.4f} {oxy_out_drift:>+9.4f}  {eu_drift:>+9.4f} {redness:>+9.4f} ")
        
        rows.append({
            'index': i, 'residual_amp': float(amp), 'in_mask_contrast': contrast,
            'hemo_in_clean': hemo_clean_in, 'hemo_in_after': hemo_after_in,
            'eu_in_after':  eu_after_in,     # add this
            'oxy_in_after': oxy_after_in,    # add this
            'melanin_drift_in': mel_drift, 'hemo_drift_out': out_drift,
            'oxy_drift_out': oxy_out_drift, 'in_mask_redness': redness,
            'eu_drift_out': eu_drift,
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
    
    save_montage(prog_dir)
    save_control_curve(prog_dir, rows, hemo_clean_in)
    save_control_allcurve(prog_dir, rows, hemo_clean_in, eu_clean_in, oxy_clean_in)
    
    
    if sp_composite is not None:
        composite_path = os.path.join(prog_dir, "erythema_chromophore_composite.png")
        try:
            deltas = make_chromophore_composite(
                skin_props, sp_composite, shape, mask, composite_path,
                f"{amp_composite:.2f}")
            print(f"[out] chromophore composite -> {composite_path}")
            print_composite_deltas(deltas)
        except Exception as error:
            print(f"[out] skipped composite: {error}")

    
            
    return rows        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    