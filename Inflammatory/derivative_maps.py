"""
derivative_maps.py
==================
Generates spatial ΔE* heatmaps (pixel-wise colour-change maps) for
representative faces from each Fitzpatrick phototype, at each grade step.

Layout: columns = [Original Grade-0 face] + [Grade 0→1, 1→2, 2→3, 3→4]
        rows    = one per Fitzpatrick type (I–VI)

Produces three figures:
  fig_deriv_A_raw.png        — shared ΔE* colour scale (shows magnitude)
  fig_deriv_B_normalised.png — each map normalised 0→1 (shows spatial pattern only)
  fig_deriv_C_grayscale.png  — ΔL* diverging map (Alessio's suggestion)

Set the three paths at the top, then run:
    python derivative_maps.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from skimage import io, color
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURE THESE PATHS
# ─────────────────────────────────────────────
OUTPUT_FOLDER = r"D:\Github\PhD Code\Erythema-Progression\output\1200-identities"
METRICS_CSV   = r"D:\Github\PhD Code\Erythema-Progression\output\derivative_analysis\multichannel_metrics.csv"
SAVE_FOLDER   = r"D:\Github\PhD Code\Erythema-Progression\output\derivative_maps"
# ─────────────────────────────────────────────

os.makedirs(SAVE_FOLDER, exist_ok=True)
 
GRADE_STEPS = [(0, 1), (1, 2), (2, 3), (3, 4)]
FITZ_ORDER  = [1, 2, 3, 4, 5, 6]
N_REPS      = 1   # representative faces per Fitzpatrick type
 
# Full descriptive labels for rows and figure text
FITZ_FULL = {
    1: "Fitzpatrick I\nVery Light",
    2: "Fitzpatrick II\nLight",
    3: "Fitzpatrick III\nIntermediate",
    4: "Fitzpatrick IV\nTan / Olive",
    5: "Fitzpatrick V\nBrown",
    6: "Fitzpatrick VI\nDark Brown / Black",
}
FITZ_SHORT = {1: "Fitz I", 2: "Fitz II", 3: "Fitz III",
              4: "Fitz IV", 5: "Fitz V",  6: "Fitz VI"}
 
# Column layout: first column = original face, then one per grade step
COL_LABELS = ["Original\n(Grade 0 baseline)"] + \
             [f"Grade {a}→{b}\nΔE* map" for (a, b) in GRADE_STEPS]
 
# ─── helpers ──────────────────────────────────────────────────────────────────
 
def list_identity_files(identity_folder: Path) -> list[str]:
    """Return sorted list of filenames inside an identity folder (for diagnostics)."""
    if not identity_folder.exists():
        return ["<folder does not exist>"]
    return sorted(p.name for p in identity_folder.iterdir() if p.is_file())
 
 
def load_mask(identity_folder: Path):
    """Load feathered mask; returns float32 array H×W in [0,1], or None."""
    for name in ("mask_feathered.png", "mask_feathered.jpg", "mask_feathered.jpeg"):
        p = identity_folder / name
        if p.exists():
            m = io.imread(str(p))
            if m.ndim == 3:
                m = m[..., 0]
            return m.astype(np.float32) / 255.0
    return None
 
 
def _grade0_candidates(identity_id: str) -> list[Path]:
    """All plausible disk locations for the grade-0 baseline image.
    BioSkin outputs '{id}.png.jpeg' (double extension) — include that first."""
    subfolder = Path(OUTPUT_FOLDER) / identity_id
    root      = Path(OUTPUT_FOLDER)
    paths = []
    # double-extension variants first (BioSkin default output naming)
    for ext in (".png.jpeg", ".png.jpg", ".png.exr"):
        paths.append(subfolder / f"{identity_id}{ext}")
        paths.append(root      / f"{identity_id}{ext}")
    # standard single-extension fallbacks
    for ext in (".jpeg", ".jpg", ".png"):
        paths.append(subfolder / f"{identity_id}{ext}")
        paths.append(root      / f"{identity_id}{ext}")
    return paths
 
 
def has_grade0(identity_id: str) -> bool:
    return any(p.exists() for p in _grade0_candidates(identity_id))
 
 
def load_grade_image(identity_folder: Path, identity_id: str, grade: int):
    """Load one grade image; return (float32 RGB [0,1], Path used) or raise."""
    if grade == 0:
        search = _grade0_candidates(identity_id)
    else:
        search = [
            identity_folder / f"cea_{identity_id}_{grade}.png",
            identity_folder / f"cea_{identity_id}_{grade}.jpg",
            identity_folder / f"cea_{identity_id}_{grade}.jpeg",
        ]
    for p in search:
        if p.exists():
            img = io.imread(str(p))
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :3]
            return img, p
    raise FileNotFoundError(
        f"No grade-{grade} image for {identity_id} in {identity_folder}"
    )
 
 
def rgb_to_lab(img_rgb):
    return color.rgb2lab(img_rgb)
 
 
def compute_delta_e_map(lab_a, lab_b):
    """Per-pixel ΔE* = √(ΔL*² + Δa*² + Δb*²)."""
    return np.sqrt(np.sum((lab_b - lab_a) ** 2, axis=-1))
 
 
def parse_fitz(v):
    ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
    s = str(v).strip().upper()
    if s in ROMAN:
        return ROMAN[s]
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None
 
 
def pick_representative(df, fitz, n=1):
    """Face closest to median ITA for a Fitzpatrick type.
    Prefers identities with grade-0 on disk; falls back otherwise."""
    sub    = df[df[FITZ_COL] == fitz].copy()
    unique = sub.drop_duplicates(ID_COL).copy()
    with_g0 = unique[unique[ID_COL].apply(has_grade0)].copy()
    pool = with_g0 if not with_g0.empty else unique
    if pool.empty:
        return []
    if with_g0.empty:
        print(f"  WARNING (Fitz {fitz}): no grade-0 found for any identity "
              f"— picking closest-to-median regardless.")
    median_ita = pool[ITA_COL].median()
    pool = pool.copy()
    pool["_dist"] = (pool[ITA_COL] - median_ita).abs()
    return pool.nsmallest(n, "_dist")[ID_COL].tolist()
 
 
# ─── load CSV ─────────────────────────────────────────────────────────────────
 
print("Loading metrics CSV …")
df = pd.read_csv(METRICS_CSV)
print(f"  Shape: {df.shape}  |  Columns: {list(df.columns)}\n")
 
ID_COL   = "face_id"   if "face_id"   in df.columns else "identity_id"
FITZ_COL = "fitzpatrick" if "fitzpatrick" in df.columns else "fitzpatrick_type"
ITA_COL  = "ita"         if "ita"         in df.columns else "ITA"
 
df[ID_COL]   = df[ID_COL].astype(str).str.zfill(6)
df[FITZ_COL] = df[FITZ_COL].apply(parse_fitz)
 
print(f"  Columns used: id='{ID_COL}', fitz='{FITZ_COL}', ita='{ITA_COL}'")
print(f"  Fitzpatrick counts:\n{df[FITZ_COL].value_counts().sort_index()}\n")
 
# ─── pick representatives ─────────────────────────────────────────────────────
 
print("Selecting representative face per Fitzpatrick type …")
representatives = {}
ita_of = {}   # fitz → ITA of selected face
for fitz in FITZ_ORDER:
    reps = pick_representative(df, fitz, N_REPS)
    if reps:
        iid = reps[0]
        representatives[fitz] = reps
        row = df[df[ID_COL] == iid].iloc[0]
        ita_of[fitz] = round(float(row[ITA_COL]), 1)
        print(f"  Fitz {fitz}: ID={iid}  ITA={ita_of[fitz]}°")
    else:
        print(f"  Fitz {fitz}: NO DATA — skipped")
 
# ─── DIAGNOSTIC: list files in each identity folder ───────────────────────────
 
print("\nDiagnostic — files found in each identity folder:")
for fitz in [f for f in FITZ_ORDER if f in representatives]:
    iid    = representatives[fitz][0]
    folder = Path(OUTPUT_FOLDER) / iid
    files  = list_identity_files(folder)
    # flag which one we expect to be grade-0
    print(f"  Fitz {fitz} ({iid}): {files}")
print()
 
fitz_present = [f for f in FITZ_ORDER if f in representatives]
n_rows = len(fitz_present)
n_cols = 1 + len(GRADE_STEPS)   # original + 4 derivative columns
 
# ─── cache images (RGB + Lab) per identity ────────────────────────────────────
 
print("Loading images …")
rgb_cache = {}   # fitz → {grade: ndarray | None}
lab_cache = {}   # fitz → {grade: ndarray | None}
mask_cache = {}  # fitz → mask | None
 
for fitz in fitz_present:
    iid    = representatives[fitz][0]
    folder = Path(OUTPUT_FOLDER) / iid
    mask_cache[fitz] = load_mask(folder)
    rgb_cache[fitz]  = {}
    lab_cache[fitz]  = {}
    for g in range(5):
        try:
            rgb, used = load_grade_image(folder, iid, g)
            rgb_cache[fitz][g] = rgb
            lab_cache[fitz][g] = rgb_to_lab(rgb)
            if g == 0:
                print(f"  Fitz {fitz} ({iid}) grade 0 → found at: {used.name}")
        except FileNotFoundError as e:
            rgb_cache[fitz][g] = None
            lab_cache[fitz][g] = None
            if g == 0:
                print(f"  Fitz {fitz} ({iid}) grade 0 → NOT FOUND ({e})")
 
# ─── compute ΔE* maps ─────────────────────────────────────────────────────────
 
print("\nComputing pixel-wise ΔE* and ΔL* maps …")
de_maps = {}   # fitz → {step: ndarray | None}
dl_maps = {}   # fitz → {step: ndarray | None}
 
for fitz in fitz_present:
    de_maps[fitz] = {}
    dl_maps[fitz] = {}
    for (g_from, g_to) in GRADE_STEPS:
        la = lab_cache[fitz].get(g_from)
        lb = lab_cache[fitz].get(g_to)
        if la is not None and lb is not None:
            de_maps[fitz][(g_from, g_to)] = compute_delta_e_map(la, lb)
            dl_maps[fitz][(g_from, g_to)] = lb[..., 0] - la[..., 0]
        else:
            de_maps[fitz][(g_from, g_to)] = None
            dl_maps[fitz][(g_from, g_to)] = None
 
# ─── shared colourscale for ΔE* (raw) ────────────────────────────────────────
 
all_de_vals = []
for fitz in fitz_present:
    mask = mask_cache[fitz]
    for step in GRADE_STEPS:
        de = de_maps[fitz][step]
        if de is None:
            continue
        vals = de[mask > 0.5].ravel() if mask is not None else de.ravel()
        all_de_vals.extend(vals.tolist())
 
vmax_de = float(np.percentile(all_de_vals, 99)) if all_de_vals else 10.0
vmax_de = max(vmax_de, 1.0)
 
# ─── symmetric scale for ΔL* ─────────────────────────────────────────────────
 
all_dl_vals = []
for fitz in fitz_present:
    mask = mask_cache[fitz]
    for step in GRADE_STEPS:
        dl = dl_maps[fitz][step]
        if dl is None:
            continue
        vals = dl[mask > 0.5].ravel() if mask is not None else dl.ravel()
        all_dl_vals.extend(vals.tolist())
 
dl_abs_max = float(np.percentile(np.abs(all_dl_vals), 99)) if all_dl_vals else 5.0
dl_abs_max = max(dl_abs_max, 1.0)
 
 
# ─── helper: build row ylabel ──────────────────────────────────────────────────
 
def row_label(fitz):
    iid  = representatives[fitz][0]
    ita  = ita_of.get(fitz, "?")
    name = FITZ_FULL[fitz]
    return f"{name}\nITA: {ita}°  |  ID: {iid}"
 
 
# ─── helper: render one "no data" cell ────────────────────────────────────────
 
def no_data_cell(ax, label="No grade-0\nimage found"):
    ax.set_facecolor("#1a1a1a")
    ax.text(0.5, 0.5, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=8, color="#888888",
            style="italic")
    ax.axis("off")
 
 
# ──────────────────────────────────────────────────────────────────────────────
# FIGURE A — Raw ΔE* maps (shared colour scale) + original face column
# ──────────────────────────────────────────────────────────────────────────────
 
print("\nRendering Figure A (raw ΔE*, shared scale + original face) …")
 
CMAP_DE = "inferno"
fig_a, axes_a = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3.2, n_rows * 3.4),
                              constrained_layout=True)
if n_rows == 1:
    axes_a = axes_a[np.newaxis, :]
 
for r, fitz in enumerate(fitz_present):
    mask = mask_cache[fitz]
    iid  = representatives[fitz][0]
 
    # ── col 0: original baseline face ─────────────────
    ax0 = axes_a[r, 0]
    rgb0 = rgb_cache[fitz].get(0)
    if rgb0 is not None:
        ax0.imshow(np.clip(rgb0, 0, 1))
        ax0.set_title("Original\n(Grade 0)", fontsize=9, fontweight="bold",
                      color="#222222") if r == 0 else None
    else:
        no_data_cell(ax0)
        if r == 0:
            ax0.set_title("Original\n(Grade 0)", fontsize=9,
                          fontweight="bold", color="#222222")
    ax0.axis("off")
 
    # ── cols 1–4: ΔE* derivative maps ─────────────────
    for c, step in enumerate(GRADE_STEPS):
        ax = axes_a[r, c + 1]
        de = de_maps[fitz][step]
 
        if de is None:
            no_data_cell(ax, f"Grade {step[0]}→{step[1]}\nMissing")
            if r == 0:
                ax.set_title(f"Grade {step[0]}→{step[1]}\nΔE* map",
                             fontsize=9, fontweight="bold")
            continue
 
        display = de.copy()
        if mask is not None:
            display = display * mask
 
        ax.imshow(display, cmap=CMAP_DE, vmin=0, vmax=vmax_de)
        ax.axis("off")
 
        if r == 0:
            ax.set_title(f"Grade {step[0]}→{step[1]}\nΔE* map",
                         fontsize=9, fontweight="bold")
 
    # ── row label (left side) ──────────────────────────
    axes_a[r, 0].set_ylabel(row_label(fitz), fontsize=8.5, rotation=0,
                             labelpad=130, va="center", ma="right")
 
# shared ΔE* colourbar
sm_a = plt.cm.ScalarMappable(cmap=CMAP_DE,
                              norm=mcolors.Normalize(vmin=0, vmax=vmax_de))
sm_a.set_array([])
cbar_a = fig_a.colorbar(sm_a, ax=axes_a, shrink=0.55, pad=0.02)
cbar_a.set_label("ΔE* — perceptual colour change magnitude", fontsize=9)
 
fig_a.suptitle(
    "Figure A — Pixel-wise ΔE* maps (raw, shared scale)\n",
    fontsize=11, fontweight="bold", y=1.01
)
 
path_a = os.path.join(SAVE_FOLDER, "fig_deriv_A_raw.png")
fig_a.savefig(path_a, dpi=150, bbox_inches="tight")
print(f"  Saved: {path_a}")
plt.close(fig_a)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# FIGURE B — Normalised ΔE* maps (each map 0→1) + original face column
# ──────────────────────────────────────────────────────────────────────────────
 
print("Rendering Figure B (normalised ΔE* + original face) …")
 
fig_b, axes_b = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3.2, n_rows * 3.4),
                              constrained_layout=True)
if n_rows == 1:
    axes_b = axes_b[np.newaxis, :]
 
for r, fitz in enumerate(fitz_present):
    mask = mask_cache[fitz]
    iid  = representatives[fitz][0]
 
    # ── col 0: original ───────────────────────────────
    ax0 = axes_b[r, 0]
    rgb0 = rgb_cache[fitz].get(0)
    if rgb0 is not None:
        ax0.imshow(np.clip(rgb0, 0, 1))
        if r == 0:
            ax0.set_title("Original\n(Grade 0)", fontsize=9, fontweight="bold")
    else:
        no_data_cell(ax0)
        if r == 0:
            ax0.set_title("Original\n(Grade 0)", fontsize=9, fontweight="bold")
    ax0.axis("off")
 
    # ── cols 1–4: normalised ΔE* ──────────────────────
    for c, step in enumerate(GRADE_STEPS):
        ax = axes_b[r, c + 1]
        de = de_maps[fitz][step]
 
        if de is None:
            no_data_cell(ax, f"Grade {step[0]}→{step[1]}\nMissing")
            if r == 0:
                ax.set_title(f"Grade {step[0]}→{step[1]}\nΔE* normalised",
                             fontsize=9, fontweight="bold")
            continue
 
        display = de.copy()
        if mask is not None:
            display = display * mask
 
        local_max = display.max()
        if local_max > 0:
            display = display / local_max
 
        ax.imshow(display, cmap=CMAP_DE, vmin=0, vmax=1)
        ax.axis("off")
 
        if r == 0:
            ax.set_title(f"Grade {step[0]}→{step[1]}\nΔE* normalised",
                         fontsize=9, fontweight="bold")
 
    axes_b[r, 0].set_ylabel(row_label(fitz), fontsize=8.5, rotation=0,
                             labelpad=130, va="center", ma="right")
 
fig_b.suptitle(
    "Figure B — Pixel-wise ΔE* maps (normalised, each map 0→1)\n",
    fontsize=11, fontweight="bold", y=1.01
)
 
path_b = os.path.join(SAVE_FOLDER, "fig_deriv_B_normalised.png")
fig_b.savefig(path_b, dpi=150, bbox_inches="tight")
print(f"  Saved: {path_b}")
plt.close(fig_b)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# FIGURE C — ΔL* diverging map + original face column
# ──────────────────────────────────────────────────────────────────────────────
 
print("Rendering Figure C (ΔL* grayscale + original face) …")
 
CMAP_DL = "RdBu_r"   # blue = skin got darker (negative ΔL*), red = lighter
 
fig_c, axes_c = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3.2, n_rows * 3.4),
                              constrained_layout=True)
if n_rows == 1:
    axes_c = axes_c[np.newaxis, :]
 
for r, fitz in enumerate(fitz_present):
    mask = mask_cache[fitz]
    iid  = representatives[fitz][0]
 
    # ── col 0: original ───────────────────────────────
    ax0 = axes_c[r, 0]
    rgb0 = rgb_cache[fitz].get(0)
    if rgb0 is not None:
        ax0.imshow(np.clip(rgb0, 0, 1))
        if r == 0:
            ax0.set_title("Original\n(Grade 0)", fontsize=9, fontweight="bold")
    else:
        no_data_cell(ax0)
        if r == 0:
            ax0.set_title("Original\n(Grade 0)", fontsize=9, fontweight="bold")
    ax0.axis("off")
 
    # ── cols 1–4: ΔL* diverging ───────────────────────
    for c, step in enumerate(GRADE_STEPS):
        ax = axes_c[r, c + 1]
        dl = dl_maps[fitz][step]
 
        if dl is None:
            no_data_cell(ax, f"Grade {step[0]}→{step[1]}\nMissing")
            if r == 0:
                ax.set_title(f"Grade {step[0]}→{step[1]}\nΔL* map",
                             fontsize=9, fontweight="bold")
            continue
 
        display = dl.copy()
        if mask is not None:
            display = display * mask
 
        ax.imshow(display, cmap=CMAP_DL,
                  vmin=-dl_abs_max, vmax=dl_abs_max)
        ax.axis("off")
 
        if r == 0:
            ax.set_title(f"Grade {step[0]}→{step[1]}\nΔL* map",
                         fontsize=9, fontweight="bold")
 
    axes_c[r, 0].set_ylabel(row_label(fitz), fontsize=8.5, rotation=0,
                             labelpad=130, va="center", ma="right")
 
sm_c = plt.cm.ScalarMappable(cmap=CMAP_DL,
                              norm=mcolors.Normalize(vmin=-dl_abs_max,
                                                     vmax=dl_abs_max))
sm_c.set_array([])
cbar_c = fig_c.colorbar(sm_c, ax=axes_c, shrink=0.55, pad=0.02)
cbar_c.set_label("ΔL*  (blue = skin got darker  |  red = skin got lighter)",
                 fontsize=9)
 
fig_c.suptitle(
    "Figure C — ΔL* maps (lightness change only, diverging scale)\n",
    fontsize=11, fontweight="bold", y=1.01
)
 
path_c = os.path.join(SAVE_FOLDER, "fig_deriv_C_grayscale.png")
fig_c.savefig(path_c, dpi=150, bbox_inches="tight")
print(f"  Saved: {path_c}")
plt.close(fig_c)
 
print(f"\nDone — three figures saved to: {SAVE_FOLDER}")
 