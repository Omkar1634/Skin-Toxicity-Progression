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


PARAM_NAMES = ['melanin', 'hemoglobin', 'epidermal_thickness',
               'eumelanin_ratio', 'oxygenation']

PARAM_COLORMAPS = {
    'melanin':             'YlOrBr',
    'hemoglobin':          'Reds',
    'epidermal_thickness': 'Blues',
    'eumelanin_ratio':     'Oranges',
    'oxygenation':         'RdYlGn_r',
}


def save_montage(output_dir):
    frame_files = sorted(
        filename for filename in os.listdir(output_dir)
        if filename.startswith("frame_") and filename.endswith(".jpeg")
    )
    panels = []
    panel_width = 240

    for filename in frame_files:
        image = cv2.imread(os.path.join(output_dir, filename))
        if image is None:
            continue
        height = int(image.shape[0] * panel_width / image.shape[1])
        image = cv2.resize(image, (panel_width, height), interpolation=cv2.INTER_AREA)
        amplitude = filename.split("_amp")[-1].replace(".jpeg", "")
        cv2.putText(image, f"amp {amplitude}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(image)

    if panels:
        montage_path = os.path.join(output_dir, "erythema_progression_montage.jpeg")
        cv2.imwrite(montage_path, cv2.hconcat(panels))
        print(f"[out] montage -> {montage_path}")


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