"""
sweep_curve.py — dense open-loop single-knob sweeps.

Diagnostic only. No calibration, no binary search, no frame/EXR output.

Sweeps each chromophore knob independently over its OWN available headroom,
per face, and records in-mask CIELAB a*/L*/b*, dE2000 and the legacy redness
metric, all relative to that face's clean albedo.

Key change from the first version: the sweep variable is `t` in [0, 1],
meaning FRACTION OF AVAILABLE TRAVEL for that knob on that face -- not a raw
delta. A raw 0->0.40 sweep is not comparable across knobs (different
baselines) or across faces (same knob, different headroom).

Usage:
    python sweep_curve.py
    python sweep_curve.py --runs A_vb F_thick --faces 000000
    python sweep_curve.py --grid --faces 000190
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # 3090 only; hides 5070 Ti
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"

import sys
import csv
import argparse
import datetime

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mask import butterfly_mask

BIOSKIN_REPO = None
if BIOSKIN_REPO and BIOSKIN_REPO not in sys.path:
    sys.path.insert(0, BIOSKIN_REPO)

from bioskin.bioskin import BioSkinInference
import bioskin.utils.io as io


# ----------------------------------------------------------------------------
# Sweep configuration
# ----------------------------------------------------------------------------

MAX_WIDTH = 800
T_MIN, T_MAX, T_STEP = 0.0, 0.5, 0.025     # fraction of available travel
MASK_THRESHOLD = 0.5
GRID_N = 6                                 # Vb x thickness grid resolution

# Each run maps a knob name -> direction (+1 = toward 1.0, -1 = toward 0.0).
# The swept delta is  t * headroom_in_that_direction, computed per face.
# "fixed" entries are absolute deltas applied at every step (legacy recipe).
RUNS = {
    "A_vb":        {"sweep": {"hemo": +1}},
    "B_phim":      {"sweep": {"eu": +1}},
    "C_phih":      {"sweep": {"oxy": -1}},
    "D_phim_phih": {"sweep": {"eu": +1, "oxy": -1}},
    "E_recipe":    {"sweep": {"hemo": +1},
                    "fixed": {"eu": +0.10, "oxy": -0.15}},
    "F_thick":     {"sweep": {"thick": -1}},
}

KNOBS = ("melanin", "hemo", "thick", "eu", "oxy")   # display order


def channel_index(C):
    """Knob name -> skin_props column, from the YAML chromophore section."""
    return {
        "melanin": C["MELANIN_INDEX"],
        "hemo":    C["BLOOD_VOLUME_INDEX"],
        "thick":   C["EPIDERMIS_THICKNESS_INDEX"],
        "eu":      C["MELANIN_TYPE_INDEX"],
        "oxy":     C["HAEMO_TYPE_INDEX"],
    }


# ----------------------------------------------------------------------------
# Linear RGB -> CIELAB (D65, 2 degree observer)
#
# BioSkin outputs LINEAR reflectance, so go straight to XYZ. Do NOT
# sRGB-encode first and then call a library rgb2lab() -- it would linearise
# it again.
# ----------------------------------------------------------------------------

_M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

_WHITE_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
_DELTA = 6.0 / 29.0


def _f_lab(t):
    t = np.asarray(t, dtype=np.float64)
    return np.where(
        t > _DELTA ** 3,
        np.cbrt(np.clip(t, 0.0, None)),
        t / (3.0 * _DELTA ** 2) + 4.0 / 29.0,
    )


def linear_rgb_to_lab(rgb_linear):
    """(M,3) linear RGB in [0,1], R first -> (M,3) L*a*b*."""
    rgb = np.clip(np.asarray(rgb_linear, dtype=np.float64), 0.0, 1.0)
    xyz_n = (rgb @ _M_RGB2XYZ.T) / _WHITE_D65[None, :]
    fx, fy, fz = _f_lab(xyz_n[:, 0]), _f_lab(xyz_n[:, 1]), _f_lab(xyz_n[:, 2])
    return np.stack([116.0 * fy - 16.0,
                     500.0 * (fx - fy),
                     200.0 * (fy - fz)], axis=1)


def ciede2000(lab1, lab2):
    """CIEDE2000 between two L*a*b* triples. kL = kC = kH = 1."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    C1 = float(np.hypot(a1, b1))
    C2 = float(np.hypot(a2, b2))
    Cbar = 0.5 * (C1 + C2)
    G = 0.5 * (1.0 - np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7))) if Cbar > 0 else 0.5

    a1p, a2p = (1.0 + G) * a1, (1.0 + G) * a2
    C1p, C2p = float(np.hypot(a1p, b1)), float(np.hypot(a2p, b2))
    h1p = float(np.degrees(np.arctan2(b1, a1p)) % 360.0)
    h2p = float(np.degrees(np.arctan2(b2, a2p)) % 360.0)

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0.0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180.0:
        dhp = h2p - h1p
    elif h2p - h1p > 180.0:
        dhp = h2p - h1p - 360.0
    else:
        dhp = h2p - h1p + 360.0
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    Lbp = 0.5 * (L1 + L2)
    Cbp = 0.5 * (C1p + C2p)

    if C1p * C2p == 0.0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180.0:
        hbp = 0.5 * (h1p + h2p)
    elif h1p + h2p < 360.0:
        hbp = 0.5 * (h1p + h2p + 360.0)
    else:
        hbp = 0.5 * (h1p + h2p - 360.0)

    T = (1.0
         - 0.17 * np.cos(np.radians(hbp - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hbp))
         + 0.32 * np.cos(np.radians(3.0 * hbp + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hbp - 63.0)))

    dtheta = 30.0 * np.exp(-(((hbp - 275.0) / 25.0) ** 2))
    RC = 2.0 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7)) if Cbp > 0 else 0.0
    SL = 1.0 + 0.015 * (Lbp - 50.0) ** 2 / np.sqrt(20.0 + (Lbp - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cbp
    SH = 1.0 + 0.015 * Cbp * T
    RT = -np.sin(np.radians(2.0 * dtheta)) * RC

    return float(np.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2
                         + RT * (dCp / SC) * (dHp / SH)))


# ----------------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------------

def measure(rgb_bgr, inside):
    """rgb_bgr: (N,3) BioSkin tensor, BGR, linear. inside: boolean (N,)."""
    px = rgb_bgr[inside.to(rgb_bgr.device)]

    redness = (px[:, 2] - 0.5 * (px[:, 0] + px[:, 1])).mean().item()

    rgb = px[:, [2, 1, 0]].detach().float().cpu().numpy()
    lab = linear_rgb_to_lab(rgb)

    return {
        "redness": float(redness),
        "L": float(lab[:, 0].mean()),
        "a": float(lab[:, 1].mean()),
        "b": float(lab[:, 2].mean()),
    }


def baselines(skin_props, inside, C):
    """In-mask mean of every chromophore channel, keyed by knob name."""
    idx = channel_index(C)
    sel = inside.to(skin_props.device)
    return {k: float(skin_props[sel, i].mean().item()) for k, i in idx.items()}


def headroom(base, direction):
    """Available travel for a knob from its baseline, in the given direction."""
    return (1.0 - base) if direction > 0 else base


# ----------------------------------------------------------------------------
# Recipe
# ----------------------------------------------------------------------------

def apply_recipe(skin_props, mask_flat, bio_skin, C, deltas):
    """
    deltas : dict knob_name -> absolute delta (may be negative).
             Applied as  channel += delta * mask, clamped to [0, 1].
             Any knob not present is untouched.

    Returns (flush_rgb, clamp_counts) where clamp_counts[knob] is the number
    of pixels pushed outside [0, 1] and clipped back.
    """
    idx = channel_index(C)
    sp = skin_props.clone()
    clamp_counts = {}

    for knob, d in deltas.items():
        if d == 0.0:
            continue
        col = idx[knob]
        raw = sp[:, col] + float(d) * mask_flat
        clamp_counts[knob] = int(((raw < 0.0) | (raw > 1.0)).sum().item())
        sp[:, col] = torch.clamp(raw, 0.0, 1.0)

    _, flush_rgb, _, _ = bio_skin.skin_props_to_reflectance(sp)
    return flush_rgb, clamp_counts


# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------

def load_face(face_id, config, bio_skin, device):
    paths = config["paths"]
    albedo_path = paths["ALBEDO_PATH"]
    albedo_dir = albedo_path if os.path.isdir(albedo_path) \
        else os.path.dirname(albedo_path)
    img_path = os.path.join(albedo_dir, f"{face_id}.png")

    print(f"[{face_id}] loading {img_path}")

    # ONE load, with max_width. The earlier version loaded twice and the
    # second load had no max_width, so the mask was built at 1024 while the
    # skin_props were 800 -- that is what pushed in-mask to 22.56%.
    input_image = io.load_image(img_path, max_width=MAX_WIDTH)
    if input_image is None:
        raise FileNotFoundError(f"Could not load albedo: {img_path}")

    height, width = input_image.shape[:2]
    print(f"[{face_id}] working resolution = {width}x{height}")

    input_reflectance = io.vectorize_image(input_image, device=device)
    skin_props = bio_skin.reflectance_to_skin_props(input_reflectance)
    _, reference_rgb, _, _ = bio_skin.skin_props_to_reflectance(skin_props)

    mask = butterfly_mask(height, width)
    mask_flat = torch.from_numpy(mask.reshape(-1)).to(device).float()

    skin_props = skin_props.to(device)
    reference_rgb = reference_rgb.to(device)

    if mask_flat.numel() != skin_props.shape[0]:
        raise ValueError(
            f"Mask/image size mismatch for {face_id}: mask has "
            f"{mask_flat.numel()} pixels, image has {skin_props.shape[0]}")

    inside = mask_flat > MASK_THRESHOLD
    return skin_props, reference_rgb, mask_flat, inside


# ----------------------------------------------------------------------------
# Row assembly
# ----------------------------------------------------------------------------

FIELDS = (["run", "face", "t", "t_thick", "is_clean"]
          + [f"d_{k}" for k in KNOBS]
          + ["redness", "L", "a", "b",
             "delta_redness", "delta_L", "delta_a", "delta_b", "delta_e2000"]
          + [f"clamp_{k}" for k in KNOBS])


def make_row(run, face_id, t, m, clean, deltas, clamp_counts, is_clean=0):
    row = {
        "run": run, "face": face_id, "t": t, "t_thick": "", "is_clean": is_clean,
        "redness": m["redness"], "L": m["L"], "a": m["a"], "b": m["b"],
        "delta_redness": m["redness"] - clean["redness"],
        "delta_L": m["L"] - clean["L"],
        "delta_a": m["a"] - clean["a"],
        "delta_b": m["b"] - clean["b"],
        "delta_e2000": ciede2000((clean["L"], clean["a"], clean["b"]),
                                 (m["L"], m["a"], m["b"])),
    }
    for k in KNOBS:
        row[f"d_{k}"] = float(deltas.get(k, 0.0))
        row[f"clamp_{k}"] = int(clamp_counts.get(k, 0))
    return row


# ----------------------------------------------------------------------------
# Sweeps
# ----------------------------------------------------------------------------

def sweep_face(face_id, run_names, config, bio_skin, device, face_data):
    C = config["chromophore_parameter"]
    skin_props, reference_rgb, mask_flat, inside = face_data

    n_inside = int(inside.sum().item())
    print(f"[{face_id}] in-mask pixels = {n_inside} "
          f"({100.0 * n_inside / inside.numel():.2f}%)")

    base = baselines(skin_props, inside, C)
    print(f"[{face_id}] baselines: " +
          "  ".join(f"{k}={base[k]:.4f}" for k in KNOBS))

    clean = measure(reference_rgb, inside)
    print(f"[{face_id}] clean: redness={clean['redness']:.4f}  "
          f"L*={clean['L']:.2f}  a*={clean['a']:.2f}  b*={clean['b']:.2f}")

    rows = [make_row("clean", face_id, -1.0, clean, clean, {}, {}, is_clean=1)]
    ts = np.arange(T_MIN, T_MAX + 1e-9, T_STEP)

    for run in run_names:
        spec = RUNS[run]
        sweep, fixed = spec["sweep"], spec.get("fixed", {})
        room = {k: headroom(base[k], d) for k, d in sweep.items()}

        print(f"\n[{face_id}] === {run} ===  travel: " +
              "  ".join(f"{k}: {base[k]:.4f} -> "
                        f"{base[k] + np.sign(d) * room[k]:.4f}"
                        for k, d in sweep.items()))

        for t in ts:
            deltas = dict(fixed)
            for k, d in sweep.items():
                deltas[k] = deltas.get(k, 0.0) + np.sign(d) * float(t) * room[k]

            flush_rgb, clamps = apply_recipe(
                skin_props, mask_flat, bio_skin, C, deltas)
            m = measure(flush_rgb, inside)
            rows.append(make_row(run, face_id, float(t), m, clean,
                                 deltas, clamps))

            print(f"[{face_id}] {run} t={t:.2f}  "
                  f"d_a*={rows[-1]['delta_a']:+.3f}  "
                  f"d_L*={rows[-1]['delta_L']:+.3f}  "
                  f"dE00={rows[-1]['delta_e2000']:6.3f}  "
                  f"clamped={sum(clamps.values())}")

    return rows, base, clean


def grid_face(face_id, config, bio_skin, base, clean, face_data):
    """Run G: Vb x epidermal thickness. Does Vb regain response when the
    melanin path length is shortened?"""
    C = config["chromophore_parameter"]
    skin_props, _, mask_flat, inside = face_data
    rows = []

    room_hemo = headroom(base["hemo"], +1)
    room_thick = headroom(base["thick"], -1)
    grid = np.linspace(0.0, 1.0, GRID_N)

    print(f"\n[{face_id}] === G_vb_x_thick ===  {GRID_N}x{GRID_N}")
    for tv in grid:
        for tt in grid:
            deltas = {"hemo": float(tv) * room_hemo,
                      "thick": -float(tt) * room_thick}
            flush_rgb, clamps = apply_recipe(
                skin_props, mask_flat, bio_skin, C, deltas)
            m = measure(flush_rgb, inside)
            row = make_row("G_vb_x_thick", face_id, float(tv), m, clean,
                           deltas, clamps)
            row["t_thick"] = float(tt)
            rows.append(row)
            print(f"[{face_id}] G  t_vb={tv:.2f} t_thick={tt:.2f}  "
                  f"d_a*={row['delta_a']:+.3f}  d_L*={row['delta_L']:+.3f}")
    return rows


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def plot_runs(face_id, rows, out_dir):
    """Per face: delta a* and delta L* vs fraction of travel, one line per run."""
    runs = sorted({x["run"] for x in rows} - {"clean", "G_vb_x_thick"})
    if not runs:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for run in runs:
        r = sorted([x for x in rows if x["run"] == run], key=lambda x: x["t"])
        ax1.plot([x["t"] for x in r], [x["delta_a"] for x in r],
                 marker="o", ms=3, label=run)
        ax2.plot([x["t"] for x in r], [x["delta_L"] for x in r],
                 marker="o", ms=3, label=run)

    ax1.set_ylabel(r"$\Delta a^*$")
    ax1.set_title(f"Face {face_id} — single-knob response")
    ax2.set_ylabel(r"$\Delta L^*$")
    ax2.set_xlabel("fraction of available travel")
    for ax in (ax1, ax2):
        ax.axhline(0.0, lw=0.8, color="k", alpha=0.4)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, f"runs_{face_id}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[{face_id}] plot -> {path}")


def plot_overlay(all_rows, out_dir):
    """Per run, delta a* for every face on one axes."""
    runs = sorted({x["run"] for rows in all_rows.values() for x in rows}
                  - {"clean", "G_vb_x_thick"})
    if not runs:
        return

    fig, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 4),
                             sharey=True, squeeze=False)
    for ax, run in zip(axes[0], runs):
        for face_id, rows in all_rows.items():
            r = sorted([x for x in rows if x["run"] == run],
                       key=lambda x: x["t"])
            if r:
                ax.plot([x["t"] for x in r], [x["delta_a"] for x in r],
                        marker="o", ms=3, label=face_id)
        ax.axhline(0.0, lw=0.8, color="k", alpha=0.4)
        ax.set_title(run, fontsize=10)
        ax.set_xlabel("fraction of travel")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel(r"$\Delta a^*$")
    axes[0][-1].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "overlay_by_run.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[overlay] plot -> {path}")


def plot_grid(face_id, rows, out_dir):
    g = [x for x in rows if x["run"] == "G_vb_x_thick"]
    if not g:
        return
    tv = sorted({x["t"] for x in g})
    tt = sorted({x["t_thick"] for x in g})
    Z = np.full((len(tt), len(tv)), np.nan)
    for x in g:
        Z[tt.index(x["t_thick"]), tv.index(x["t"])] = x["delta_a"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(Z, origin="lower", aspect="auto",
                   extent=[tv[0], tv[-1], tt[0], tt[-1]])
    fig.colorbar(im, ax=ax, label=r"$\Delta a^*$")
    ax.set_xlabel("Vb (fraction of travel)")
    ax.set_ylabel("epidermal thinning (fraction of travel)")
    ax.set_title(f"Face {face_id} — Vb x thickness")

    fig.tight_layout()
    path = os.path.join(out_dir, f"grid_{face_id}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[{face_id}] grid plot -> {path}")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--faces", nargs="+", default=["000000", "000190"])
    ap.add_argument("--runs", nargs="+", default=list(RUNS.keys()),
                    choices=list(RUNS.keys()))
    ap.add_argument("--grid", action="store_true",
                    help="also run the Vb x thickness grid (Run G)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device = {device}")

    bio_skin = BioSkinInference(config["paths"]["MODEL_PREFIX"])

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args.out or os.path.join(
        config["paths"]["OUTPUT_DIR"], "sweep_curve", stamp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[init] out_dir = {out_dir}")

    all_rows = {}
    for face_id in args.faces:
        # Load ONCE per face; the grid reuses the same tensors.
        face_data = load_face(face_id, config, bio_skin, device)

        rows, base, clean = sweep_face(
            face_id, args.runs, config, bio_skin, device, face_data)

        if args.grid:
            rows += grid_face(face_id, config, bio_skin, base, clean, face_data)

        all_rows[face_id] = rows
        plot_runs(face_id, rows, out_dir)
        plot_grid(face_id, rows, out_dir)

    # One CSV for everything -- run and face are columns.
    csv_path = os.path.join(out_dir, "sweep_curve.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rows in all_rows.values():
            writer.writerows(rows)
    print(f"\n[out] CSV -> {csv_path}")

    if len(all_rows) > 1:
        plot_overlay(all_rows, out_dir)

    print("DONE.")


if __name__ == "__main__":
    main()