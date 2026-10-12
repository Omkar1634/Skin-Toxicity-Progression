"""
image_derivative_analysis_v2.py
────────────────────────────────────────────────────────────────────────────────
Extended image derivative analysis — all three CIELab channels + ΔE*.

Computes pixel-wise derivatives between consecutive CEA grades (0→1 … 3→4)
for every face identity, now across:
    ΔL*  — lightness change        (hyperpigmentation / darkening signal)
    Δa*  — red–green axis          (classic erythema redness)
    Δb*  — yellow–blue axis        (violaceous / dusky shift in dark skin)
    ΔE*  — full CIELab distance    √(ΔL*² + Δa*² + Δb*²)

Folder structure expected per face:
    <INPUT_FOLDER>/
        000180/
            000180.png              ← grade 0 (baseline)
            cea_000180_1.png        ← grade 1
            cea_000180_2.png        ← grade 2
            cea_000180_3.png        ← grade 3
            cea_000180_4.png        ← grade 4
            mask_feathered          ← butterfly mask (JPEG, no extension)

Outputs:
    <OUTPUT_FOLDER>/
        000180/
            derivative_0to1.png          ← 4-panel heatmap (L*, a*, b*, E*)
            derivative_1to2.png
            derivative_2to3.png
            derivative_3to4.png
            derivative_composite.png     ← all 4 grade-steps side by side
        multichannel_metrics.csv         ← one row per face × grade step
    Console: per-Fitzpatrick mean table + Pearson r for all four channels
"""

import os
import warnings
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURE PATHS HERE
# ══════════════════════════════════════════════════════════════════════════════

INPUT_FOLDER    = r"D:\Github\PhD Code\Erythema-Progression\output\1200-identities"
OUTPUT_FOLDER   = r"D:\Github\PhD Code\Erythema-Progression\output\derivative_analysis"
SELECTED_CSV    = r"D:\Github\PhD Code\Erythema-Progression\Inflammatory\selected_identities_stratified.csv"
REPLACEMENTS_CSV= r"D:\Github\PhD Code\Erythema-Progression\Inflammatory\replacements_fitz6.csv"

FACE_IDS = None   # None = all faces; or e.g. ["000180", "000063"] for a quick test

# ══════════════════════════════════════════════════════════════════════════════

EPS = 1e-8
GRADE_PAIRS = [(0,1,"0to1","0→1 (mild)"),
               (1,2,"1to2","1→2 (moderate)"),
               (2,3,"2to3","2→3 (severe)"),
               (3,4,"3to4","3→4 (very severe)")]


# ─────────────────────────────────────────────────────────────────────────────
# Image / mask loaders  (identical logic to original)
# ─────────────────────────────────────────────────────────────────────────────

def find_image(face_dir: Path, stem: str):
    """Return path string for first matching extension, or None."""
    for ext in [".jpeg", ".jpg", ".png", ".png.jpeg", ".png.jpg"]:
        p = face_dir / (stem + ext)
        if p.exists():
            return str(p)
    return None


def load_image_lab(path: str):
    """Load image → float32 CIELab (H, W, 3)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def load_mask(face_dir: Path, target_shape: tuple):
    """
    Load butterfly mask as float32 [0,1].
    Tries: mask_feathered → .png → .jpg → .jpeg
    Falls back to all-ones if not found.
    """
    base = str(face_dir / "mask_feathered")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for suffix in ["", ".png", ".jpg", ".jpeg"]:
            m = cv2.imread(base + suffix, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (target_shape[1], target_shape[0]))
                return m.astype(np.float32) / 255.0
    print(f"  [WARN] Mask not found in {face_dir.name} — using full-image mask")
    return np.ones(target_shape[:2], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CIELab channel extraction
# ─────────────────────────────────────────────────────────────────────────────

def decode_lab(lab_enc: np.ndarray):
    """
    Split encoded float32 LAB (values 0-255) into true CIELab channels.
        L*  : 0–100   (lab_enc[:,:,0] * 100/255)
        a*  : -128–127 (lab_enc[:,:,1] - 128)
        b*  : -128–127 (lab_enc[:,:,2] - 128)
    """
    L = lab_enc[:, :, 0] * (100.0 / 255.0)
    a = lab_enc[:, :, 1] - 128.0
    b = lab_enc[:, :, 2] - 128.0
    return L, a, b


def channel_deltas(lab_prev: np.ndarray, lab_next: np.ndarray):
    """Return (dL, da, db, dE) pixel arrays."""
    L0, a0, b0 = decode_lab(lab_prev)
    L1, a1, b1 = decode_lab(lab_next)
    dL = L1 - L0
    da = a1 - a0
    db = b1 - b0
    dE = np.sqrt(dL**2 + da**2 + db**2)
    return dL, da, db, dE


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(delta: np.ndarray, mask: np.ndarray, prefix: str):
    """
    Compute inside/outside mask stats for one channel delta.
    Uses hard threshold (mask > 0.5) matching original behaviour.
    Returns dict with keys prefixed by `prefix`.
    """
    inside  = mask > 0.5
    outside = ~inside

    mean_in  = float(np.mean(delta[inside]))   if inside.any()  else 0.0
    std_in   = float(np.std(delta[inside]))    if inside.any()  else 0.0
    mean_out = float(np.mean(delta[outside]))  if outside.any() else 0.0
    std_out  = float(np.std(delta[outside]))   if outside.any() else 0.0
    max_in   = float(np.max(delta[inside]))    if inside.any()  else 0.0
    max_out  = float(np.max(delta[outside]))   if outside.any() else 0.0

    abs_in   = np.sum(np.abs(delta[inside]))
    cr       = float(abs_in / (np.sum(np.abs(delta)) + EPS))
    leakage  = bool(abs(mean_out) > 0.1 * abs(mean_in + EPS))

    return {
        f"{prefix}_mean_inside":       mean_in,
        f"{prefix}_std_inside":        std_in,
        f"{prefix}_mean_outside":      mean_out,
        f"{prefix}_std_outside":       std_out,
        f"{prefix}_max_inside":        max_in,
        f"{prefix}_max_outside":       max_out,
        f"{prefix}_containment_ratio": cr,
        f"{prefix}_leakage_flag":      leakage,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

CHANNEL_META = [
    ("dL", "ΔL* (lightness)",   "RdGy",   -5,  5),
    ("da", "Δa* (red–green)",   "RdBu_r", -3,  3),
    ("db", "Δb* (yellow–blue)", "PuOr_r", -3,  3),
    ("dE", "ΔE* (colour dist)", "YlOrRd",  0,  6),
]


def save_grade_heatmap(deltas: dict, mask: np.ndarray, save_path: str, title: str):
    """
    4-panel heatmap (one column per channel: L*, a*, b*, E*),
    2 rows: full image | inside-mask only.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for col, (key, label, cmap_name, vmin, vmax) in enumerate(CHANNEL_META):
        ch = deltas[key]
        cmap = plt.get_cmap(cmap_name)

        # top: full image
        im = axes[0, col].imshow(ch, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"{label}\n(full image)", fontsize=9)
        axes[0, col].axis('off')
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        # bottom: masked
        im2 = axes[1, col].imshow(ch * mask, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, col].set_title("inside mask", fontsize=9)
        axes[1, col].axis('off')
        plt.colorbar(im2, ax=axes[1, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()


def save_composite(all_deltas: list, mask: np.ndarray, save_path: str, face_id: str):
    """
    Composite: rows = grade steps (4), cols = channels (4).
    Shows inside-mask only for compactness.
    """
    grade_labels = ["0→1 (mild)", "1→2 (moderate)", "2→3 (severe)", "3→4 (very severe)"]
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    fig.suptitle(f"Multi-channel Derivatives — Face {face_id}", fontsize=14, fontweight='bold')

    for row, (deltas, glabel) in enumerate(zip(all_deltas, grade_labels)):
        for col, (key, label, cmap_name, vmin, vmax) in enumerate(CHANNEL_META):
            ch = deltas[key] * mask
            im = axes[row, col].imshow(ch, cmap=plt.get_cmap(cmap_name),
                                       vmin=vmin, vmax=vmax)
            col_title = label if row == 0 else ""
            row_title = glabel if col == 0 else ""
            axes[row, col].set_title(f"{col_title}", fontsize=9)
            axes[row, col].set_ylabel(row_title, fontsize=9)
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Per-face processing
# ─────────────────────────────────────────────────────────────────────────────

def process_face(face_dir: Path, output_dir: Path):
    """Process one face. Returns list of metric dicts (one per grade step)."""
    face_id  = face_dir.name
    face_out = output_dir / face_id
    face_out.mkdir(parents=True, exist_ok=True)

    # locate grade images
    grade0 = find_image(face_dir, face_id)
    gradeN = [find_image(face_dir, f"cea_{face_id}_{g}") for g in range(1, 5)]
    all_paths = [grade0] + gradeN

    missing = [i for i, p in enumerate(all_paths) if p is None]
    if missing:
        stems = [face_id] + [f"cea_{face_id}_{g}" for g in range(1,5)]
        print(f"  [SKIP] {face_id} — missing: {[stems[i] for i in missing]}")
        return []

    labs  = [load_image_lab(p) for p in all_paths]
    mask  = load_mask(face_dir, labs[0].shape)

    rows       = []
    all_deltas = []

    for ga, gb, name, label in GRADE_PAIRS:
        dL, da, db, dE = channel_deltas(labs[ga], labs[gb])
        deltas = {"dL": dL, "da": da, "db": db, "dE": dE}
        all_deltas.append(deltas)

        row = {"face_id": face_id,
               "grade_step": f"grade_{ga}_to_{gb}"}
        for ch_key, prefix in [("dL","L"), ("da","a"), ("db","b"), ("dE","E")]:
            row.update(compute_metrics(deltas[ch_key], mask, prefix))
        rows.append(row)

        save_grade_heatmap(
            deltas, mask,
            save_path=str(face_out / f"derivative_{name}.png"),
            title=f"Face {face_id} | Grade {label}",
        )

    save_composite(
        all_deltas, mask,
        save_path=str(face_out / "derivative_composite.png"),
        face_id=face_id,
    )

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Statistical summary
# ─────────────────────────────────────────────────────────────────────────────

FITZ_ORDER = ['I','II','III','IV','V','VI']

def print_summary(merged: pd.DataFrame):
    CHANNELS = [("L","ΔL*"), ("a","Δa*"), ("b","Δb*"), ("E","ΔE*")]

    print("\n" + "═"*80)
    print("PER-FITZPATRICK MEAN inside-mask values")
    print("═"*80)

    for step_key, step_label in [
        ("grade_0_to_1","0→1"), ("grade_1_to_2","1→2"),
        ("grade_2_to_3","2→3"), ("grade_3_to_4","3→4")
    ]:
        sub = merged[merged["grade_step"] == step_key]
        print(f"\n── Grade {step_label} ──")
        header = f"{'Fitzpatrick':>14}" + "".join(f"  {lbl:>10}" for _, lbl in CHANNELS)
        print(header)
        for ftype in FITZ_ORDER:
            fsub = sub[sub["fitzpatrick"] == ftype]
            if fsub.empty:
                continue
            vals = "".join(
                f"  {fsub[f'{ch}_mean_inside'].mean():>10.4f}"
                for ch, _ in CHANNELS
            )
            print(f"{ftype:>14}{vals}")

    print("\n" + "═"*80)
    print("PEARSON r — ITA vs mean_inside (n per step shown in header)")
    print("═"*80)
    print(f"{'Grade step':>20}" + "".join(f"  {lbl:>10}" for _, lbl in CHANNELS))

    for step_key, step_label in [
        ("grade_0_to_1","0→1"), ("grade_1_to_2","1→2"),
        ("grade_2_to_3","2→3"), ("grade_3_to_4","3→4")
    ]:
        sub = merged[merged["grade_step"] == step_key]
        n   = len(sub)
        row_str = f"{'grade_'+step_label+f' (n={n})':>20}"
        for ch, _ in CHANNELS:
            r, p = stats.pearsonr(sub["ita"], sub[f"{ch}_mean_inside"])
            row_str += f"  {r:>10.4f}"
        print(row_str)

    # attenuation summary (cumulative post-onset)
    print("\n" + "═"*80)
    print("ATTENUATION — cumulative mean_inside (grades 1→4 summed) per Fitzpatrick")
    print("═"*80)
    post_onset = merged[merged["grade_step"].isin(
        ["grade_1_to_2","grade_2_to_3","grade_3_to_4"]
    )]
    agg = post_onset.groupby("fitzpatrick")[
        [f"{ch}_mean_inside" for ch, _ in CHANNELS]
    ].sum().reindex(FITZ_ORDER)
    header = f"{'Fitzpatrick':>14}" + "".join(f"  {lbl:>10}" for _, lbl in CHANNELS)
    print(header)
    for ftype in FITZ_ORDER:
        if ftype not in agg.index:
            continue
        vals = "".join(
            f"  {agg.loc[ftype, f'{ch}_mean_inside']:>10.4f}"
            for ch, _ in CHANNELS
        )
        print(f"{ftype:>14}{vals}")

    # ratio I vs VI
    if 'I' in agg.index and 'VI' in agg.index:
        print("\nAttenuation ratio Fitz I / Fitz VI (cumulative post-onset):")
        for ch, lbl in CHANNELS:
            vi  = agg.loc['VI', f'{ch}_mean_inside']
            fi  = agg.loc['I',  f'{ch}_mean_inside']
            ratio = fi / (vi + EPS)
            print(f"  {lbl:>5}: {fi:.4f} / {vi:.4f} = {ratio:.1f}×")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    input_root  = Path(INPUT_FOLDER)
    output_root = Path(OUTPUT_FOLDER)
    output_root.mkdir(parents=True, exist_ok=True)

    face_dirs = (
        [input_root / fid for fid in FACE_IDS]
        if FACE_IDS
        else sorted([d for d in input_root.iterdir() if d.is_dir()])
    )

    print(f"\n{'─'*60}")
    print(f"  Multi-channel Image Derivative Analysis (v2)")
    print(f"  Input  : {input_root}")
    print(f"  Output : {output_root}")
    print(f"  Faces  : {len(face_dirs)}")
    print(f"{'─'*60}\n")

    all_rows = []
    for face_dir in tqdm(face_dirs, desc="Processing faces"):
        all_rows.extend(process_face(face_dir, output_root))

    if not all_rows:
        print("No faces processed successfully.")
        return

    metrics = pd.DataFrame(all_rows)

    # ── Metadata merge ────────────────────────────────────────────────────────
    selected     = pd.read_csv(SELECTED_CSV)
    replacements = pd.read_csv(REPLACEMENTS_CSV)

    metrics["face_id_int"] = (
        metrics["face_id"].astype(str).str.lstrip("0")
        .apply(lambda x: int(x) if x else 0)
    )
    processed_ids = set(metrics["face_id_int"].unique())

    selected["face_id_int"]     = selected["face_id"].astype(int)
    replacements["face_id_int"] = replacements["face_id"].astype(int)

    lookup = pd.concat([
        selected[selected["face_id_int"].isin(processed_ids)][["face_id_int","ita","fitzpatrick"]],
        replacements[replacements["face_id_int"].isin(processed_ids)][["face_id_int","ita","fitzpatrick"]],
    ], ignore_index=True).drop_duplicates(subset="face_id_int")

    merged = metrics.merge(lookup, on="face_id_int", how="inner")
    print(f"\nMerged: {merged['face_id_int'].nunique()} faces, {len(merged)} rows")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = output_root / "multichannel_metrics.csv"
    merged.to_csv(csv_path, index=False)
    print(f"✓ Saved → {csv_path}")

    # ── Print statistical summary ─────────────────────────────────────────────
    print_summary(merged)


if __name__ == "__main__":
    main()
