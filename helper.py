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