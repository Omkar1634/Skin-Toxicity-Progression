# =============================================================================
# HELPERS
# =============================================================================
import os
import sys
import csv
import numpy as np
import torch
import cv2
import argparse
BIOSKIN_REPO = None

if BIOSKIN_REPO and BIOSKIN_REPO not in sys.path:
    sys.path.insert(0, BIOSKIN_REPO)

from bioskin.bioskin import BioSkinInference
import bioskin.utils.io as io
import bioskin.spectrum.color_spectrum as color

PARAM_NAMES = ['melanin', 'hemoglobin', 'epidermal_thickness',
               'eumelanin_ratio', 'oxygenation']

PARAM_COLORMAPS = {
    'melanin':             'YlOrBr',
    'hemoglobin':          'Reds',
    'epidermal_thickness': 'Blues',
    'eumelanin_ratio':     'Oranges',
    'oxygenation':         'RdYlGn_r',
}

def compute_a_star(flush_rgb, inside):
    """
    Compute mean a* over in-mask pixels from BioSkin flush_rgb.
    flush_rgb: (N, 3) tensor, BGR order (ch0=B, ch1=G, ch2=R), linear float.
    inside: (N,) bool array — True for in-mask pixels.
    Returns: float — mean a* over mask.
    
    """
    
    x = flush_rgb.detach().cpu().numpy()
    # BioSkin BGR → stack as RGB for cv2.COLOR_RGB2LAB
    b, g, r = x[:, 0], x[:, 1], x[:, 2]
    rgb = np.stack([r, g, b], axis=1)             # (N, 3) RGB

    # apply BioSkin's linear→sRGB before LAB conversion (matches save_jpeg)
    rgb = color.linear_to_sRGB(rgb)               # color = bioskin.spectrum.color_spectrum
    rgb = np.clip(rgb, 0.0, 1.0)

    # scale to uint8 for cv2
    rgb_uint8 = (rgb * 255.0).astype(np.uint8)    # (N, 3)

    # cvtColor needs (H, W, 3) — reshape to (N, 1, 3) then back
    lab = cv2.cvtColor(rgb_uint8.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)

    # channel 1 is a* in OpenCV LAB, stored unsigned (0=−128, 128=0, 255=+127)
    a_star = lab[:, 1].astype(np.float32) - 128.0

    return float(a_star[inside].mean())

def save_control_curve(output_dir, rows, clean_hemoglobin):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        amplitudes = [row["residual_amp"] for row in rows]
        recovered = [row["hemo_in_after"] for row in rows]
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.axhline(clean_hemoglobin, linestyle="--", color="gray",
                     label="clean in-mask hemoglobin")
        axis.plot(amplitudes, recovered, marker="o",
                  label="recovered in-mask hemoglobin")
        axis.set_xlabel("residual amplitude (input)")
        axis.set_ylabel("in-mask hemoglobin (re-encoded)")
        axis.set_title("Erythema control curve")
        axis.legend()
        axis.grid(alpha=0.3)
        figure.tight_layout()
        curve_path = os.path.join(output_dir, "erythema_control_curve.png")
        figure.savefig(curve_path, dpi=150)
        plt.close(figure)
        print(f"[out] control curve -> {curve_path}")
    except Exception as error:
        print(f"[out] skipped control curve: {error}")
        

def save_control_allcurve(output_dir, rows, clean_hemoglobin, clean_eumelanin, clean_oxygenation):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        amplitudes  = [row["residual_amp"]   for row in rows]
        hemo_after  = [row["hemo_in_after"]  for row in rows]
        eu_after    = [row["eu_in_after"]    for row in rows]   # new
        oxy_after   = [row["oxy_in_after"]   for row in rows]   # new

        figure, axis = plt.subplots(figsize=(8, 4))

        # clean baselines as dashed horizontal lines
        axis.axhline(clean_hemoglobin,  ls="--", color="red",    alpha=0.5, label="clean hemoglobin")
        axis.axhline(clean_eumelanin,   ls="--", color="orange",  alpha=0.5, label="clean eumelanin")
        axis.axhline(clean_oxygenation, ls="--", color="green",  alpha=0.5, label="clean oxygenation")

        # recovered values per amplitude
        axis.plot(amplitudes, hemo_after, marker="o", color="red",    label="hemoglobin (re-encoded)")
        axis.plot(amplitudes, eu_after,   marker="s", color="orange",  label="eumelanin  (re-encoded)")
        axis.plot(amplitudes, oxy_after,  marker="^", color="green",  label="oxygenation (re-encoded)")

        axis.set_xlabel("residual amplitude (input)")
        axis.set_ylabel("in-mask value (re-encoded)")
        axis.set_title("Erythema control curve — hemoglobin, eumelanin, oxygenation")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
        figure.tight_layout()
        curve_path = os.path.join(output_dir, "erythema_control_allcurve.png")
        figure.savefig(curve_path, dpi=150)
        plt.close(figure)
        print(f"[out] control curve -> {curve_path}")
    except Exception as error:
        print(f"[out] skipped control curve: {error}")


def print_composite_deltas(deltas):
    print("\n[composite] in-mask mean delta per chromophore (edited - clean):")
    for name in PARAM_NAMES:
        if name == "hemoglobin":
            marker = "  <-- target"
        elif name == "melanin":
            marker = "  (should be ~0)"
        else:
            marker = ""
        print(f"    {name:22s} {deltas[name]:+.4f}{marker}")



def in_mask_mean(tensor_col, inside_bool):
    return float(tensor_col.detach().cpu().numpy()[inside_bool].mean())


def make_chromophore_composite(sp_clean, sp_edited, shape, mask2d, out_path, amp_label):
    """
    3 rows (clean / edited / delta) x 5 chromophores. The delta row uses a
    diverging colormap centred at 0, with the mask outline overlaid — so you can
    SEE hemoglobin rise inside the mask while melanin & others stay flat.
    Returns {param: in-mask mean delta} for a numeric companion to the figure.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    H, W = shape[0], shape[1]
    clean = sp_clean.detach().cpu().numpy()
    edit  = sp_edited.detach().cpu().numpy()
    inside = mask2d.reshape(-1) > 0.5
    ncol = len(PARAM_NAMES)

    fig, axes = plt.subplots(3, ncol, figsize=(3.1 * ncol, 9.2))
    deltas = {}
    for j, name in enumerate(PARAM_NAMES):
        cm = PARAM_COLORMAPS[name]
        c = clean[:, j].reshape(H, W)
        e = edit[:, j].reshape(H, W)
        d = e - c
        vmin, vmax = min(c.min(), e.min()), max(c.max(), e.max())

        axes[0, j].imshow(c, cmap=cm, vmin=vmin, vmax=vmax)
        axes[0, j].set_title(name, fontsize=11)
        axes[1, j].imshow(e, cmap=cm, vmin=vmin, vmax=vmax)

        amax = float(np.abs(d).max()) + 1e-8
        imd = axes[2, j].imshow(d, cmap='RdBu_r', vmin=-amax, vmax=amax)
        axes[2, j].contour(mask2d, levels=[0.5], colors='k', linewidths=0.6)
        fig.colorbar(imd, ax=axes[2, j], fraction=0.046, pad=0.04)

        for r in range(3):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])

        deltas[name] = float(d.reshape(-1)[inside].mean())

    for y, txt in zip((0.80, 0.50, 0.20),
                      ('clean', 'erythema (amp ' + amp_label + ')', 'delta (edited - clean)')):
        fig.text(0.015, y, txt, rotation=90, va='center', fontsize=12, weight='bold')

    fig.suptitle('Chromophore maps - only hemoglobin should rise inside the mask',
                 fontsize=13)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.02,
                        wspace=0.15, hspace=0.08)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return deltas



def _chromo_vranges(sp_clean, shape):
    arr = sp_clean.detach().cpu().numpy()
    H, W = shape[0], shape[1]
    return {name: (float(arr[:, j].reshape(H, W).min()),
                   float(arr[:, j].reshape(H, W).max()))
            for j, name in enumerate(PARAM_NAMES)}


def save_chromophore_column(sp_clean, sp_edited, shape, out_path,
                            panel_width=240, label=True):
    """One amp's five chromophore maps stacked VERTICALLY into one image,
    on the SAME colour scale as the clean baseline so every amp column in the
    montage is comparable. No delta."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    vranges = _chromo_vranges(sp_clean, shape)
    arr = sp_edited.detach().cpu().numpy()
    H, W = shape[0], shape[1]
    tiles = []
    for j, name in enumerate(PARAM_NAMES):
        m = arr[:, j].reshape(H, W)
        vmin, vmax = vranges[name]
        cmap = plt.get_cmap(PARAM_COLORMAPS[name])
        rgba = cmap(Normalize(vmin, vmax)(m))
        bgr = (rgba[..., :3] * 255).astype(np.uint8)[..., ::-1]   # RGB->BGR
        h = int(bgr.shape[0] * panel_width / bgr.shape[1])
        bgr = cv2.resize(bgr, (panel_width, h), interpolation=cv2.INTER_AREA)
        if label:
            cv2.putText(bgr, name, (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(bgr)
    cv2.imwrite(out_path, cv2.vconcat(tiles))


def save_montage(output_dir):
    """Each amp column = face frame on top + its 5 chromophore maps below."""
    frame_files = sorted(f for f in os.listdir(output_dir)
                         if f.startswith("frame_") and f.endswith(".jpeg"))
    panel_width = 240
    columns = []
    for filename in frame_files:
        face = cv2.imread(os.path.join(output_dir, filename))
        if face is None:
            continue
        h = int(face.shape[0] * panel_width / face.shape[1])
        face = cv2.resize(face, (panel_width, h), interpolation=cv2.INTER_AREA)
        amp = filename.split("_amp")[-1].replace(".jpeg", "")
        try:
            is_clean = abs(float(amp)) < 1e-9
        except ValueError:
            is_clean = False
        label = "original" if is_clean else f"amp {amp}"
        cv2.putText(face, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)        

        chromo_path = os.path.join(output_dir, f"chromo_amp{amp}.png")
        column = face
        if os.path.exists(chromo_path):
            chromo = cv2.imread(chromo_path)
            if chromo is not None:
                if chromo.shape[1] != panel_width:
                    ch = int(chromo.shape[0] * panel_width / chromo.shape[1])
                    chromo = cv2.resize(chromo, (panel_width, ch),
                                        interpolation=cv2.INTER_AREA)
                column = cv2.vconcat([face, chromo])
        columns.append(column)

    if not columns:
        return
    max_h = max(c.shape[0] for c in columns)
    padded = []
    for c in columns:
        if c.shape[0] < max_h:
            pad = np.full((max_h - c.shape[0], c.shape[1], 3), 255, np.uint8)
            c = cv2.vconcat([c, pad])
        padded.append(c)

    montage_path = os.path.join(output_dir, "erythema_progression_montage.jpeg")
    cv2.imwrite(montage_path, cv2.hconcat(padded))
    print(f"[out] montage -> {montage_path}")