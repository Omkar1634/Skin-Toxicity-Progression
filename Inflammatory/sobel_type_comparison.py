"""
sobel_type_comparison.py
========================
Cross-skin-tone Sobel comparison figures — one representative face per
Fitzpatrick type (I–VI), laid out like the pixel-wise dE* figure.

Figure A — absolute Sobel magnitude
    rows    = Fitz I … Fitz VI
    columns = [Original (Grade 0) | Sobel G0 | G1 | G2 | G3 | G4]
    Answers: does the gradient structure change as erythema progresses?

Figure B — delta-Sobel between consecutive grades
    rows    = Fitz I … Fitz VI
    columns = [Original (Grade 0) | d(0>1) | d(1>2) | d(2>3) | d(3>4)]
    Answers: how much new spatial structure does each grade step add?

Both figures share one global intensity scale per figure, so brightness is
directly comparable across skin tones (this is the whole point).

Image naming follows the existing pipeline convention:
    Grade 0  : {id}.png.jpeg / {id}.png.jpg / {id}.png.exr / {id}.jpeg / {id}.jpg / {id}.png
    Grade 1-4: cea_{id}_{grade}.png / .jpg / .jpeg
    Mask     : mask_feathered.png / .jpg / .jpeg
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
from skimage import io
from scipy.ndimage import sobel

# ── Configure ─────────────────────────────────────────────────────────────────

OUTPUT_FOLDER = r"D:\Github\PhD Code\Erythema-Progression\output\1200-identities"
METRICS_CSV   = r"D:\Github\PhD Code\Erythema-Progression\output\derivative_analysis\multichannel_metrics.csv"
SAVE_FOLDER   = r"D:\Github\PhD Code\Erythema-Progression\output\derivative_maps_sobel\type_comparison"

os.makedirs(SAVE_FOLDER, exist_ok=True)

GRADES = [0, 1, 2, 3, 4]

SOBEL_CMAP   = "gray"       # greyscale, OpenCV-style
ORIGINAL_BG  = "#000000"    # background behind masked-out regions
CLIP_PCT     = 99.5         # robust vmax percentile (ignores a few hot outliers)
SIGNED_DELTA = False        # True -> signed d-Sobel on a diverging map

# Fitzpatrick bands by ITA (Chardon et al. 1991 boundaries).
# `centre` is the target used to pick the most representative face;
# open-ended bands use a nominal interior value.
FITZ_BANDS = [
    ("I",   55.0,   np.inf,  65.0),
    ("II",  41.0,   55.0,    48.0),
    ("III", 28.0,   41.0,    34.5),
    ("IV",  10.0,   28.0,    19.0),
    ("V",  -30.0,   10.0,   -10.0),
    ("VI", -np.inf, -30.0,  -50.0),
]


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
    """Load one grade image as float32 RGB in [0, 1]. Returns array or None."""
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
            img = img.astype(np.float32)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :3]
            return np.clip(img, 0.0, 1.0)
    return None


def load_mask(identity_folder: Path):
    """Load feathered mask as float32 H*W in [0, 1], or None."""
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
    """ITU-R BT.601 luminance weights."""
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def sobel_magnitude(grey: np.ndarray) -> np.ndarray:
    sx = sobel(grey, axis=1).astype(np.float32)
    sy = sobel(grey, axis=0).astype(np.float32)
    return np.hypot(sx, sy).astype(np.float32)


def robust_vmax(arrays: list[np.ndarray], pct: float = CLIP_PCT) -> float:
    """Shared upper limit from pooled non-zero values across every panel."""
    pool = np.concatenate([a[a > 0].ravel() for a in arrays if np.any(a > 0)])
    if pool.size == 0:
        return 1.0
    return float(np.percentile(pool, pct))


# ── Face selection ────────────────────────────────────────────────────────────

def find_column(df: pd.DataFrame, candidates: list[str]):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def ita_from_lab(df: pd.DataFrame):
    """Fallback: compute ITA = atan((L*-50)/b*) * 180/pi from baseline L*, b*."""
    l_col = find_column(df, ["l_star", "lstar", "baseline_l_star", "l_baseline", "L"])
    b_col = find_column(df, ["b_star", "bstar", "baseline_b_star", "b_baseline", "B"])
    if l_col is None or b_col is None:
        return None
    b = df[b_col].replace(0, np.nan)
    return np.degrees(np.arctan((df[l_col] - 50.0) / b))


def select_faces(df: pd.DataFrame, id_col: str):
    """One face per Fitz band: the one whose ITA sits closest to the band centre."""
    ita_col = find_column(df, ["ita", "ita_deg", "ita_degrees", "baseline_ita",
                               "ita_baseline", "ita_angle"])
    if ita_col is not None:
        df = df.assign(_ita=df[ita_col])
    else:
        derived = ita_from_lab(df)
        if derived is None:
            raise SystemExit(
                "No ITA column found and L*/b* not available to derive it.\n"
                f"Columns present: {list(df.columns)}"
            )
        print("[info] No ITA column — derived from L*/b*.")
        df = df.assign(_ita=derived)

    per_face = (df.groupby(id_col)["_ita"]
                  .mean()
                  .dropna()
                  .reset_index()
                  .rename(columns={"_ita": "ita"}))

    chosen = []
    for name, lo, hi, centre in FITZ_BANDS:
        band = per_face[(per_face["ita"] > lo) & (per_face["ita"] <= hi)]
        if band.empty:
            print(f"[warn] Fitz {name}: no face with ITA in ({lo}, {hi}] — skipped")
            chosen.append((name, None, np.nan))
            continue
        pick = band.iloc[(band["ita"] - centre).abs().argsort().iloc[0]]
        chosen.append((name, str(pick[id_col]).zfill(6), float(pick["ita"])))
        print(f"[pick] Fitz {name:<3} -> face {str(pick[id_col]).zfill(6)}  "
              f"(ITA = {pick['ita']:.2f})")
    return chosen


# ── Data assembly ─────────────────────────────────────────────────────────────

def build_face_data(identity_id: str):
    """Returns dict with original RGB, per-grade Sobel magnitude, and mask."""
    folder = Path(OUTPUT_FOLDER) / identity_id
    mask = load_mask(folder)

    originals, mags = {}, {}
    for g in GRADES:
        rgb = load_grade_image(folder, identity_id, g)
        if rgb is None:
            print(f"  [miss] {identity_id} grade {g}")
            continue
        originals[g] = rgb
        m = sobel_magnitude(to_greyscale(rgb))
        mags[g] = m * mask if mask is not None else m

    deltas = {}
    for g in GRADES[:-1]:
        if g in mags and (g + 1) in mags:
            d = mags[g + 1] - mags[g]
            deltas[(g, g + 1)] = d if SIGNED_DELTA else np.abs(d)

    return {"original": originals.get(0), "mag": mags, "delta": deltas, "mask": mask}


# ── Plotting ──────────────────────────────────────────────────────────────────

def _blank(ax, text):
    ax.set_facecolor("#141414")
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="#888", style="italic")
    ax.set_xticks([]); ax.set_yticks([])


def _row_label(ax, fitz, ita, fid):
    ita_txt = "n/a" if np.isnan(ita) else f"{ita:.1f}"
    ax.set_ylabel(f"Fitz {fitz}\nITA {ita_txt}\n{fid or '—'}",
                  fontsize=9, labelpad=8)


def plot_figure_a(rows, data, save_path):
    """Absolute Sobel magnitude per grade."""
    panels = [v for d in data.values() for v in d["mag"].values()]
    vmax = robust_vmax(panels)

    n_rows, n_cols = len(rows), 1 + len(GRADES)
    fig = plt.figure(figsize=(3.0 * n_cols + 1.2, 3.0 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols + 1, figure=fig,
                           width_ratios=[1] * n_cols + [0.06],
                           wspace=0.04, hspace=0.06)

    im = None
    for r, (fitz, fid, ita) in enumerate(rows):
        d = data.get(fid)
        ax0 = fig.add_subplot(gs[r, 0])
        if d and d["original"] is not None:
            ax0.imshow(d["original"])
        else:
            _blank(ax0, "missing")
        ax0.set_xticks([]); ax0.set_yticks([])
        _row_label(ax0, fitz, ita, fid)
        if r == 0:
            ax0.set_title("Original\n(Grade 0)", fontsize=10, weight="bold")

        for c, g in enumerate(GRADES, start=1):
            ax = fig.add_subplot(gs[r, c])
            if d and g in d["mag"]:
                im = ax.imshow(d["mag"][g], cmap=SOBEL_CMAP, vmin=0, vmax=vmax)
            else:
                _blank(ax, f"Grade {g}\nmissing")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"Grade {g}\nSobel magnitude", fontsize=10, weight="bold")

    if im is not None:
        cax = fig.add_subplot(gs[:, -1])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Sobel gradient magnitude (shared scale)",
                     fontsize=9, rotation=270, labelpad=16)

    fig.suptitle("Figure A — Sobel spatial derivative magnitude by Fitzpatrick type "
                 "(shared scale)", fontsize=13, weight="bold")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {save_path}")


def plot_figure_b(rows, data, save_path):
    """Change in Sobel magnitude between consecutive grades."""
    panels = [v for d in data.values() for v in d["delta"].values()]
    if SIGNED_DELTA:
        lim = robust_vmax([np.abs(p) for p in panels])
        kw, cmap_label = dict(cmap="RdBu_r", vmin=-lim, vmax=lim), "signed"
    else:
        lim = robust_vmax(panels)
        kw, cmap_label = dict(cmap=SOBEL_CMAP, vmin=0, vmax=lim), "absolute"

    steps = [(g, g + 1) for g in GRADES[:-1]]
    n_rows, n_cols = len(rows), 1 + len(steps)
    fig = plt.figure(figsize=(3.0 * n_cols + 1.2, 3.0 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols + 1, figure=fig,
                           width_ratios=[1] * n_cols + [0.06],
                           wspace=0.04, hspace=0.06)

    im = None
    for r, (fitz, fid, ita) in enumerate(rows):
        d = data.get(fid)
        ax0 = fig.add_subplot(gs[r, 0])
        if d and d["original"] is not None:
            ax0.imshow(d["original"])
        else:
            _blank(ax0, "missing")
        ax0.set_xticks([]); ax0.set_yticks([])
        _row_label(ax0, fitz, ita, fid)
        if r == 0:
            ax0.set_title("Original\n(Grade 0)", fontsize=10, weight="bold")

        for c, step in enumerate(steps, start=1):
            ax = fig.add_subplot(gs[r, c])
            if d and step in d["delta"]:
                im = ax.imshow(d["delta"][step], **kw)
            else:
                _blank(ax, f"{step[0]}>{step[1]}\nmissing")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"Grade {step[0]}\u2192{step[1]}\n\u0394 Sobel",
                             fontsize=10, weight="bold")

    if im is not None:
        cax = fig.add_subplot(gs[:, -1])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(f"\u0394 Sobel gradient magnitude ({cmap_label}, shared scale)",
                     fontsize=9, rotation=270, labelpad=16)

    fig.suptitle("Figure B — Change in Sobel magnitude per grade step by "
                 "Fitzpatrick type (shared scale)", fontsize=13, weight="bold")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading metrics CSV …")
    df = pd.read_csv(METRICS_CSV)

    id_col = find_column(df, ["face_id", "identity_id", "id"])
    if id_col is None:
        raise SystemExit(f"No id column found. Columns: {list(df.columns)}")
    df[id_col] = df[id_col].astype(str).str.zfill(6)

    print(f"{df[id_col].nunique()} faces in CSV\n")
    rows = select_faces(df, id_col)

    print("\nBuilding Sobel maps …")
    data = {}
    for fitz, fid, _ in rows:
        if fid is None:
            continue
        print(f"[face] {fid}  (Fitz {fitz})")
        data[fid] = build_face_data(fid)

    print()
    plot_figure_a(rows, data, os.path.join(SAVE_FOLDER, "figureA_sobel_magnitude_by_fitz.png"))
    plot_figure_b(rows, data, os.path.join(SAVE_FOLDER, "figureB_delta_sobel_by_fitz.png"))
    print("\n[done]")


if __name__ == "__main__":
    main()
