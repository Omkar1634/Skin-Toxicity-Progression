"""
sobel_derivative_maps.py
========================
Sobel spatial derivative maps for each face × grade (0–4).

Layout per figure:
  rows    = [Sobel X | Sobel Y | Magnitude]
  columns = [Grade 0 | Grade 1 | Grade 2 | Grade 3 | Grade 4]

Image naming follows the existing pipeline convention:
  Grade 0 : {id}.png.jpeg / {id}.png.jpg / {id}.jpeg / {id}.jpg / {id}.png
  Grade 1–4: cea_{id}_{grade}.png / .jpg / .jpeg
  Mask    : mask_feathered.png / .jpg / .jpeg

One figure saved per face to SAVE_FOLDER.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io
from scipy.ndimage import sobel

# ── Configure ─────────────────────────────────────────────────────────────────

OUTPUT_FOLDER = r"D:\Github\PhD Code\Erythema-Progression\output\1200-identities"
METRICS_CSV   = r"D:\Github\PhD Code\Erythema-Progression\output\derivative_analysis\multichannel_metrics.csv"
SAVE_FOLDER   = r"D:\Github\PhD Code\Erythema-Progression\output\derivative_maps_sobel"

os.makedirs(SAVE_FOLDER, exist_ok=True)

GRADES = [0, 1, 2, 3, 4]

# ── Image loading (matches existing pipeline naming) ──────────────────────────

def _grade0_candidates(identity_folder: Path, identity_id: str) -> list[Path]:
    """Grade-0 candidates: double-extension first (BioSkin default), then standard."""
    root = Path(OUTPUT_FOLDER)
    paths = []
    for ext in (".png.jpeg", ".png.jpg", ".png.exr"):
        paths.append(identity_folder / f"{identity_id}{ext}")
        paths.append(root / f"{identity_id}{ext}")
    for ext in (".jpeg", ".jpg", ".png"):
        paths.append(identity_folder / f"{identity_id}{ext}")
        paths.append(root / f"{identity_id}{ext}")
    return paths


def load_grade_image(identity_folder: Path, identity_id: str, grade: int):
    """Load one grade image as float32 RGB [0, 1]. Returns (array, path) or raises."""
    if grade == 0:
        search = _grade0_candidates(identity_folder, identity_id)
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


def load_mask(identity_folder: Path) -> np.ndarray | None:
    """Load feathered mask; returns float32 H×W in [0, 1], or None."""
    for name in ("mask_feathered.png", "mask_feathered.jpg", "mask_feathered.jpeg"):
        p = identity_folder / name
        if p.exists():
            m = io.imread(str(p))
            if m.ndim == 3:
                m = m[..., 0]
            return m.astype(np.float32) / 255.0
    return None


# ── Sobel helpers ─────────────────────────────────────────────────────────────

def to_greyscale(rgb: np.ndarray) -> np.ndarray:
    """Standard luminance weights (ITU-R BT.601)."""
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def compute_sobel(grey: np.ndarray):
    """Returns sx (x-gradient), sy (y-gradient), magnitude — all float32."""
    sx  = sobel(grey, axis=1).astype(np.float32)
    sy  = sobel(grey, axis=0).astype(np.float32)
    mag = np.hypot(sx, sy).astype(np.float32)
    return sx, sy, mag


def apply_mask(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return arr
    return arr * mask


def normalise(arr: np.ndarray) -> np.ndarray:
    a, b = arr.min(), arr.max()
    if b - a < 1e-8:
        return np.zeros_like(arr)
    return (arr - a) / (b - a)


# ── Per-face plot ─────────────────────────────────────────────────────────────

def plot_face(identity_id: str):
    identity_folder = Path(OUTPUT_FOLDER) / identity_id
    mask = load_mask(identity_folder)

    row_labels = ["Sobel X", "Sobel Y", "Magnitude"]
    cmaps      = ["gray", "gray", "gray"]
    n_rows, n_cols = 3, len(GRADES)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 3.2 * n_rows),
        constrained_layout=True
    )
    fig.suptitle(f"Sobel Derivative Maps — Face {identity_id}", fontsize=13, weight="bold")

    for col, grade in enumerate(GRADES):
        try:
            rgb, used = load_grade_image(identity_folder, identity_id, grade)
            print(f"  grade {grade} → {used.name}")
        except FileNotFoundError as e:
            print(f"  grade {grade} → NOT FOUND: {e}")
            for row in range(n_rows):
                axes[row, col].set_facecolor("#1a1a1a")
                axes[row, col].text(0.5, 0.5, f"Grade {grade}\nMissing",
                                    transform=axes[row, col].transAxes,
                                    ha="center", va="center", fontsize=8,
                                    color="#888", style="italic")
                axes[row, col].axis("off")
            continue

        grey         = to_greyscale(rgb)
        sx, sy, mag  = compute_sobel(grey)

        # Apply mask AFTER derivative (preserves edge accuracy at mask boundary)
        maps = [
            apply_mask(np.abs(sx),  mask),   # abs so negative gradients are visible
            apply_mask(np.abs(sy),  mask),
            apply_mask(mag, mask),
        ]

        for row, (data, cmap) in enumerate(zip(maps, cmaps)):
            ax = axes[row, col]
            ax.imshow(normalise(data), cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=10, labelpad=6)
            if row == 0:
                ax.set_title(f"Grade {grade}", fontsize=10)

    save_path = os.path.join(SAVE_FOLDER, f"sobel_{identity_id}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading metrics CSV …")
    df = pd.read_csv(METRICS_CSV)

    ID_COL = "face_id" if "face_id" in df.columns else "identity_id"
    df[ID_COL] = df[ID_COL].astype(str).str.zfill(6)

    face_ids = df[ID_COL].unique()
    print(f"Found {len(face_ids)} faces\n")

    for fid in sorted(face_ids):
        print(f"[face] {fid}")
        plot_face(fid)

    print(f"[done] All maps saved to {SAVE_FOLDER}")


if __name__ == "__main__":
    main()