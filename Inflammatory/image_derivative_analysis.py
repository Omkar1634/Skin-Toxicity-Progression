
"""
image_derivative_analysis.py
─────────────────────────────────────────────────────────────────────────────
Computes image derivatives between consecutive CEA grades (0→1, 1→2, 2→3, 3→4)
for each face identity in the erythema output folder.

Usage:
    python image_derivative_analysis.py \
        --input_folder  "D:/erythema_out" \
        --output_folder "D:/derivative_analysis"

Folder structure expected per face:
    <input_folder>/
        000180/
            000180.png              ← grade 0 (baseline)
            cea_000180_1.png        ← grade 1
            cea_000180_2.png        ← grade 2
            cea_000180_3.png        ← grade 3
            cea_000180_4.png        ← grade 4
            mask_feathered          ← butterfly mask (JPEG, no extension)

Outputs per face:
    <output_folder>/
        000180/
            derivative_0to1.png     ← heatmap of a* change grade 0→1
            derivative_1to2.png
            derivative_2to3.png
            derivative_3to4.png
            derivative_composite.png ← all 4 panels side by side
        derivative_metrics.csv      ← summary table across all faces
"""

import os
import sys
import glob
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_image_lab(path: str) -> np.ndarray:
    """Load image and return float32 LAB (H, W, 3)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return img


def load_mask(mask_path: str, target_shape: tuple) -> np.ndarray:
    """
    Load butterfly mask. Returns float32 (H, W) in [0, 1].
    Handles both JPEG (no extension) and .exr files.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        # Try adding common extensions
        for ext in ['.png', '.jpg', '.jpeg']:
            mask = cv2.imread(mask_path + ext, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                break
    if mask is None:
        # Fallback: all-ones mask (whole image)
        print(f"  [WARN] Mask not found at {mask_path} — using full image")
        return np.ones(target_shape[:2], dtype=np.float32)

    mask = cv2.resize(mask, (target_shape[1], target_shape[0]))
    mask = mask.astype(np.float32) / 255.0
    return mask


def a_star_channel(lab: np.ndarray) -> np.ndarray:
    """Extract a* channel (index 1) from LAB image."""
    # OpenCV LAB: a* is stored as uint8 offset by 128 → convert to true a*
    # When loaded as float32 from LAB conversion, values are in [0, 255]
    # True a* = value - 128
    return lab[:, :, 1] - 128.0


def compute_derivative(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Pixel-wise difference in a* channel: img_b - img_a."""
    return a_star_channel(img_b) - a_star_channel(img_a)


def derivative_metrics(deriv: np.ndarray, mask: np.ndarray) -> dict:
    """
    Compute spatial containment metrics from a derivative map.

    inside_mask  : pixels where mask > 0.5
    outside_mask : pixels where mask <= 0.5
    """
    inside  = mask > 0.5
    outside = ~inside

    metrics = {
        "mean_inside":   float(np.mean(deriv[inside]))   if inside.any()  else 0.0,
        "std_inside":    float(np.std(deriv[inside]))    if inside.any()  else 0.0,
        "mean_outside":  float(np.mean(deriv[outside]))  if outside.any() else 0.0,
        "std_outside":   float(np.std(deriv[outside]))   if outside.any() else 0.0,
        "max_inside":    float(np.max(deriv[inside]))    if inside.any()  else 0.0,
        "max_outside":   float(np.max(deriv[outside]))   if outside.any() else 0.0,
        # containment ratio: inside activation vs total activation
        "containment_ratio": (
            float(np.sum(np.abs(deriv[inside])) /
                  (np.sum(np.abs(deriv)) + 1e-8))
        ),
        # leakage flag: outside mean > 10% of inside mean
        "leakage_flag": bool(
            abs(float(np.mean(deriv[outside]))) >
            0.1 * abs(float(np.mean(deriv[inside])) + 1e-8)
        ),
    }
    return metrics


def save_derivative_heatmap(
    deriv: np.ndarray,
    mask: np.ndarray,
    save_path: str,
    title: str,
    vmin: float = -3.0,
    vmax: float = 3.0,
):
    """Save a single derivative heatmap with mask overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    cmap = plt.cm.RdBu_r  # red = positive (redder), blue = negative

    # Panel 1: raw derivative
    im = axes[0].imshow(deriv, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("a* Derivative (full image)")
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: derivative × mask (inside mask only)
    masked_deriv = deriv * mask
    im2 = axes[1].imshow(masked_deriv, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("a* Derivative (inside mask)")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: outside-mask leakage
    leakage = deriv * (1.0 - mask)
    im3 = axes[2].imshow(leakage, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title("a* Leakage (outside mask)")
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_composite(
    derivs: list,
    mask: np.ndarray,
    save_path: str,
    face_id: str,
    vmin: float = -3.0,
    vmax: float = 3.0,
):
    """Save all 4 grade-step derivatives as a 2×4 composite."""
    labels = ["0→1 (mild)", "1→2 (moderate)", "2→3 (severe)", "3→4 (very severe)"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(f"Image Derivatives — Face {face_id}", fontsize=14, fontweight='bold')

    cmap = plt.cm.RdBu_r

    for col, (deriv, label) in enumerate(zip(derivs, labels)):
        # Top row: full derivative
        im = axes[0, col].imshow(deriv, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"Grade {label}\n(full)", fontsize=10)
        axes[0, col].axis('off')
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        # Bottom row: masked derivative
        masked = deriv * mask
        im2 = axes[1, col].imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, col].set_title("(inside mask)", fontsize=10)
        axes[1, col].axis('off')
        plt.colorbar(im2, ax=axes[1, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────
# Main per-face processing
# ─────────────────────────────────────────────

def process_face(face_dir: Path, output_dir: Path) -> list[dict]:
    """
    Process one face folder. Returns list of metric dicts (one per grade step).
    """
    face_id = face_dir.name  # e.g. "000180"
    face_out = output_dir / face_id
    face_out.mkdir(parents=True, exist_ok=True)

    # ── Locate grade images ──────────────────
    # Try candidate extensions in order; take first match
    def find_image(stem: str) -> str | None:
        for ext in [".jpeg", ".jpg", ".png", ".png.jpeg", ".png.jpg"]:
            p = face_dir / (stem + ext)
            if p.exists():
                return str(p)
        return None

    grade0_path  = find_image(face_id)
    grade_paths  = [find_image(f"cea_{face_id}_{g}") for g in range(1, 5)]

    # Verify all grade images exist
    all_found    = [grade0_path] + grade_paths
    missing_idx  = [i for i, p in enumerate(all_found) if p is None]
    if missing_idx:
        missing_stems = (
            [face_id] + [f"cea_{face_id}_{g}" for g in range(1, 5)]
        )
        print(f"  [SKIP] {face_id} — missing: "
              f"{[missing_stems[i] for i in missing_idx]}")
        return []

    all_paths = all_found

    # ── Load images ──────────────────────────
    labs = [load_image_lab(p) for p in all_paths]  # grades 0,1,2,3,4

    # ── Load mask ────────────────────────────
    mask_path = str(face_dir / "mask_feathered")
    mask = load_mask(mask_path, labs[0].shape)

    # ── Compute derivatives ──────────────────
    # grade pairs: (0,1), (1,2), (2,3), (3,4)
    pairs      = [(0,1), (1,2), (2,3), (3,4)]
    pair_names = ["0to1", "1to2", "2to3", "3to4"]
    grade_labels = ["0→1 (mild)", "1→2 (moderate)", "2→3 (severe)", "3→4 (very severe)"]

    derivs   = []
    rows     = []
    vmin, vmax = -3.0, 3.0  # fixed scale across all maps

    for (ga, gb), name, label in zip(pairs, pair_names, grade_labels):
        deriv = compute_derivative(labs[ga], labs[gb])
        derivs.append(deriv)

        # Metrics
        m = derivative_metrics(deriv, mask)
        m["face_id"]    = face_id
        m["grade_step"] = f"grade_{ga}_to_{gb}"
        rows.append(m)

        # Per-step heatmap
        save_derivative_heatmap(
            deriv, mask,
            save_path=str(face_out / f"derivative_{name}.png"),
            title=f"Face {face_id} | Grade {label} | a* Derivative",
            vmin=vmin, vmax=vmax,
        )

    # Composite
    save_composite(
        derivs, mask,
        save_path=str(face_out / "derivative_composite.png"),
        face_id=face_id,
        vmin=vmin, vmax=vmax,
    )

    return rows


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    # ── Configure paths here ─────────────────────────────────────────────────
    INPUT_FOLDER  = r"D:\Github\PhD Code\Erythema-Progression\output\1000-identities"
    OUTPUT_FOLDER = r"D:\Github\PhD Code\Erythema-Progression\output\derivative_analysis\052609"
    FACE_IDS      = ["052609"]   # None = process ALL faces
                           # or e.g. ["000180", "000063"] to test a subset first
    # ─────────────────────────────────────────────────────────────────────────

    input_root  = Path(INPUT_FOLDER)
    output_root = Path(OUTPUT_FOLDER)
    output_root.mkdir(parents=True, exist_ok=True)

    # Discover face folders
    if FACE_IDS:
        face_dirs = [input_root / fid for fid in FACE_IDS]
    else:
        face_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])

    print(f"\n{'─'*60}")
    print(f"  Image Derivative Analysis")
    print(f"  Input  : {input_root}")
    print(f"  Output : {output_root}")
    print(f"  Faces  : {len(face_dirs)}")
    print(f"{'─'*60}\n")

    all_rows = []

    for face_dir in tqdm(face_dirs, desc="Processing faces"):
        rows = process_face(face_dir, output_root)
        all_rows.extend(rows)

    # ── Save metrics CSV ─────────────────────
    if all_rows:
        df = pd.DataFrame(all_rows)

        # Reorder columns nicely
        col_order = [
            "face_id", "grade_step",
            "mean_inside", "std_inside",
            "mean_outside", "std_outside",
            "max_inside", "max_outside",
            "containment_ratio", "leakage_flag"
        ]
        df = df[[c for c in col_order if c in df.columns]]

        csv_path = output_root / "derivative_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Metrics saved → {csv_path}")

        # ── Print summary ────────────────────
        print(f"\n{'─'*60}")
        print("SUMMARY — Mean inside-mask derivative per grade step:")
        summary = df.groupby("grade_step")["mean_inside"].agg(["mean","std"]).round(4)
        print(summary.to_string())

        print(f"\nLeakage flags (outside > 10% of inside):")
        leaking = df[df["leakage_flag"] == True]
        if leaking.empty:
            print("  ✓ No leakage detected across all faces.")
        else:
            print(f"  ⚠ {len(leaking)} face-grade pairs show leakage:")
            print(leaking[["face_id","grade_step","mean_inside","mean_outside"]].to_string(index=False))

        print(f"\nMean containment ratio: {df['containment_ratio'].mean():.3f}")
        print(f"{'─'*60}\n")
    else:
        print("No faces processed successfully.")


if __name__ == "__main__":
    main()