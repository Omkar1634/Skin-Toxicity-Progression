"""
run_h_bruise_sweep.py
=====================
Run H — open-loop monotonicity check for BRUISE (contusion) edits.

WHY THIS SCRIPT EXISTS
----------------------
Before trusting dE00 as a calibration target (the job calibrate_grade will do),
we must confirm two things on the existing 20-face quadrant set:

  (1) dE00 (colour distance from each face's own clean baseline) rises
      MONOTONICALLY as the bruise edit strengthens, and
  (2) the colour actually travels DARKER + BLUER (dL* < 0, db* < 0),
      i.e. the deoxygenated signature, not redder.

If either fails, we catch it here — exactly the failure mode that killed the
old linear-RGB erythema metric (it fell as Vb rose). No point calibrating a
metric that isn't monotonic.

RECIPE (mirrors erythema Runs A-F, signs changed for bruising)
--------------------------------------------------------------
  - Vb (channel 1) swept UP as the severity driver.
  - phi_h (channel 4) held at a FIXED POSITIVE offset per row  -> deoxygenated.
      NOTE: deoxygenated = phi_h toward 1 = POSITIVE offset. This is the
      OPPOSITE of erythema, which used phi_h -0.15 (toward oxygenated/red).
  - phi_m is NOT touched (erythema used +0.10 as a red-hue correction; a bruise
    must not warm toward red).
  - phi_h offset 0.0 row = pure-Vb control (oxygenation unchanged), kept for
    contrast so the phi_h contribution is visible in the CSV.

Output: one combined CSV, plus a printed monotonicity verdict per series.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # BioSkin runs on the 3090 only

import numpy as np
import pandas as pd
import torch
from skimage.color import deltaE_ciede2000  # pip install scikit-image if missing

# ---------------------------------------------------------------------------
# BioSkin channel indices (confirmed from the paper)
# 0=Vm melanin | 1=Vb blood volume | 2=epidermal thickness
# 3=phi_m melanin-type ratio | 4=phi_h haemoglobin-type ratio (0=oxy/red, 1=deoxy/purple)
# ---------------------------------------------------------------------------
VM, VB, THICK, PHI_M, PHI_H = 0, 1, 2, 3, 4

# ---------------------------------------------------------------------------
# Sweep config — loaded from Config.yaml : bruise_parameter
# (indices above already match Config.yaml : chromophore_parameter)
# ---------------------------------------------------------------------------
import yaml
CONFIG_PATH = os.environ.get("CONFIG_PATH", "Config.yaml")
with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)
_bp = CFG["bruise_parameter"]
AMP_START, AMP_STOP, AMP_STEP = _bp["VB_AMP_START"], _bp["VB_AMP_STOP"], _bp["VB_AMP_STEP"]
PHI_H_OFFSETS = _bp["PHI_H_OFFSETS"]   # positive = deoxygenated (bruise)
MONO_TOL      = _bp["MONO_TOL"]
OUT_CSV       = _bp["OUT_CSV"]
MASK_KEY      = _bp["MASK_KEY"]        # e.g. "mask_cen" -> CFG["paths"][MASK_KEY]

# ===========================================================================
# >>> WIRE TO YOUR CODE — 4 hooks. Names are my best guess from your repo;
#     fix the bodies, leave the rest of the file alone.
# ===========================================================================
# from bioskin import BioSkin, color   # your import path; color.linear_to_sRGB used below

def load_model():
    """Return your loaded BioSkin model. Must expose:
         sp   = model.reflectance_to_skin_props(albedo)   # -> skin props, layout [C,H,W]
         refl = model.skin_props_to_reflectance(sp)        # -> LINEAR reflectance, BGR (ch2=R)
    """
    raise NotImplementedError

def load_faces():
    """Return [(face_id:str, albedo), ...] for the 20-face quadrant set."""
    raise NotImplementedError

def load_mask(face_id, hw):
    """Return a soft 0..1 mask of shape hw=(H,W).
    For Run H the mask SHAPE doesn't matter (monotonicity is shape-invariant),
    so reuse the butterfly or a plain central patch. Don't block on the new
    irregular bruise mask here."""
    raise NotImplementedError

def reflectance_to_lab(refl):
    """LINEAR reflectance (BGR, ch2=R) -> per-pixel CIELAB array [H,W,3].
    Generalise your existing compute_a_star: it already does
    linear -> sRGB (via color.linear_to_sRGB) -> LAB; just return all three
    channels instead of a* only. Kept as a hook so Run H uses the SAME colour
    pipeline you validated for erythema (no second conversion to drift against).
    """
    raise NotImplementedError
# ===========================================================================


def inmask_lab(refl, mask):
    """In-mask mean (L*, a*, b*)."""
    lab = reflectance_to_lab(refl)
    m = mask > 0.5
    return (float(lab[..., 0][m].mean()),
            float(lab[..., 1][m].mean()),
            float(lab[..., 2][m].mean()))


def apply_bruise_edit(sp, mask_t, amp, phi_h_off):
    """Return (sp_edit, clamp_count). sp layout assumed [C,H,W].
    Vb += amp and phi_h += phi_h_off, inside the mask only, then clamp to [0,1].
    clamp_count = in-mask pixels pushed out of range on either edited channel."""
    sp_e = sp.clone()
    sp_e[VB]    = sp_e[VB]    + amp       * mask_t
    sp_e[PHI_H] = sp_e[PHI_H] + phi_h_off * mask_t

    m = mask_t > 0.5
    pre = torch.stack([sp_e[VB], sp_e[PHI_H]])
    clamped = ((pre < 0) | (pre > 1)) & m.unsqueeze(0)
    clamp_count = int(clamped.sum().item())

    sp_e.clamp_(0.0, 1.0)
    return sp_e, clamp_count


def main():
    model = load_model()
    faces = load_faces()
    amps = np.round(np.arange(AMP_START, AMP_STOP + 1e-9, AMP_STEP), 4)
    rows = []

    for face_id, albedo in faces:
        sp = model.reflectance_to_skin_props(albedo)          # [C,H,W]
        H, W = sp.shape[-2], sp.shape[-1]
        mask = load_mask(face_id, (H, W))
        mask_t = torch.as_tensor(mask, dtype=sp.dtype, device=sp.device)

        # clean baseline (decode clean sp through the same path)
        refl_clean = model.skin_props_to_reflectance(sp)
        L0, a0, b0 = inmask_lab(refl_clean, mask)

        for phi_h_off in PHI_H_OFFSETS:
            for amp in amps:
                sp_e, clamp_count = apply_bruise_edit(sp, mask_t, amp, phi_h_off)
                refl_e = model.skin_props_to_reflectance(sp_e)
                L, a, b = inmask_lab(refl_e, mask)

                dL, da, db = L - L0, a - a0, b - b0
                dE76 = float(np.sqrt(dL*dL + da*da + db*db))
                dE00 = float(deltaE_ciede2000([L0, a0, b0], [L, a, b]))

                # round-trip: re-encode edited reflectance, read back in-mask Vb/phi_h
                sp_rt = model.reflectance_to_skin_props(refl_e)
                m = mask_t > 0.5
                vb_rt   = float(sp_rt[VB][m].mean().item())
                phih_rt = float(sp_rt[PHI_H][m].mean().item())

                rows.append(dict(
                    face_id=face_id, phi_h_off=phi_h_off, amp=float(amp),
                    L0=L0, a0=a0, b0=b0, L=L, a=a, b=b,
                    dL=dL, da=da, db=db, dE76=dE76, dE00=dE00,
                    clamp_count=clamp_count, vb_decoded=vb_rt, phih_decoded=phih_rt,
                ))
        print(f"[done] {face_id}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(df)} rows)")

    # ---- the verdict Run H exists to deliver ----
    print("\n=== monotonicity & direction check (per face x phi_h series) ===")
    for (fid, ph), g in df.groupby(["face_id", "phi_h_off"]):
        g = g.sort_values("amp")
        mono = bool((g["dE00"].diff().dropna() >= -MONO_TOL).all())
        print(f"{fid}  phi_h={ph:>4}:  dE00 monotonic={mono}  "
              f"final dE00={g['dE00'].iloc[-1]:6.2f}  "
              f"final dL={g['dL'].iloc[-1]:+5.2f}  final db={g['db'].iloc[-1]:+5.2f}")

    sub = df[df["phi_h_off"] > 0]
    frac_mono = (df[df["phi_h_off"] > 0]
                 .sort_values(["face_id", "phi_h_off", "amp"])
                 .groupby(["face_id", "phi_h_off"])["dE00"]
                 .apply(lambda s: (s.diff().dropna() >= -MONO_TOL).all()).mean())
    print(f"\nPASS if ~all series monotonic AND dL<0, db<0 under phi_h>0.")
    print(f"  fraction of phi_h>0 series monotonic : {frac_mono:.2f}")
    print(f"  mean final dL (phi_h>0)              : {sub['dL'].mean():+.2f}  (want < 0)")
    print(f"  mean final db (phi_h>0)              : {sub['db'].mean():+.2f}  (want < 0)")


if __name__ == "__main__":
    main()