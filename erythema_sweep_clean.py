#!/usr/bin/env python
"""Run an erythema severity sweep with BioSkin."""

import argparse
import csv
import datetime
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "true")

import cv2
import numpy as np
import torch
import yaml

from helper import PARAM_NAMES, in_mask_mean, make_chromophore_composite
from mask import butterfly_mask


MAX_WIDTH = 800
BATCH_SIZE = 512000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Config.yaml", help="YAML config path")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def create_output_directory(output_root):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(output_root, "progression", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


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


def main():
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    chromophore = config["chromophore_parameter"]
    amplitude = config["amplitude_parameter"]
    mask_config = config["mask_parameter"]

    bioskin_repo = paths.get("BIOSKIN_REPO")
    if bioskin_repo and bioskin_repo not in sys.path:
        sys.path.insert(0, bioskin_repo)

    from bioskin.bioskin import BioSkinInference
    import bioskin.utils.io as bioskin_io

    output_dir = create_output_directory(paths["OUTPUT_DIR"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cpu = device.type == "cpu"
    print(f"[init] device = {device}")

    bio_skin = BioSkinInference(paths["MODEL_PREFIX"], device=device,batch_size=BATCH_SIZE)
    image = bioskin_io.load_image(paths["ALBEDO_PATH"], max_width=MAX_WIDTH)
    if image is None:
        raise FileNotFoundError(paths["ALBEDO_PATH"])

    height, width = image.shape[:2]
    shape = image.shape
    reflectance = bioskin_io.vectorize_image(image, device=device)
    skin_props, _, reference_rgb, _, _, reconstruction_error = bio_skin.reconstruct(reflectance)
    print(f"[p1] skin_props.shape = {tuple(skin_props.shape)}")
    print(f"[p1] reconstruction error = {reconstruction_error.mean().item():.6f}")

    bioskin_io.save_tensor_to_image(os.path.join(output_dir, "frame_00_amp0.00"),reference_rgb, shape, channels=3, cpu=use_cpu)

    mask = butterfly_mask(height, width)
    mask_flat = torch.from_numpy(mask.reshape(-1)).to(device)
    inside = mask.reshape(-1) > 0.5
    outside = ~inside
    bioskin_io.save_tensor_to_image(os.path.join(output_dir, "mask_feathered"), mask_flat.float(), shape, channels=1, cpu=use_cpu)

    BLOOD_VOLUME_INDEX = chromophore["BLOOD_VOLUME_INDEX"]
    melanin_index = chromophore["MELANIN_INDEX"]
    HAEMO_TYPE_INDEX = chromophore["HAEMO_TYPE_INDEX"]
    clean_hemoglobin = in_mask_mean(skin_props[:, BLOOD_VOLUME_INDEX], inside)
    clean_melanin = in_mask_mean(skin_props[:, melanin_index], inside)
    clean_oxygenation = in_mask_mean(skin_props[:, HAEMO_TYPE_INDEX], inside)
    print(f"[p1] Original Hemoglobin = {clean_hemoglobin:.4f}")
    print(f"[p1] Original Melanin = {clean_melanin:.4f}")
    print(f"[p1] Original Oxygenation = {clean_oxygenation:.4f}")

    levels = np.round(np.arange(amplitude["AMP_START"],
                                amplitude["AMP_STOP"] + amplitude["AMP_STEP"] / 2.0,
                                amplitude["AMP_STEP"]), 3)
    target_amplitude = mask_config.get("COMPOSITE_AMP")
    target_amplitude = float(levels[-1] if target_amplitude is None else target_amplitude)
    print(f"\n[sweep] amplitude levels: {list(levels)}")

    rows = []
    composite_props = None
    composite_amplitude = None
    for index, level in enumerate(levels, start=1):
        residual = float(level) * mask_flat
        edited_props = skin_props.clone()
        edited_props[:, BLOOD_VOLUME_INDEX] = torch.clamp(edited_props[:, BLOOD_VOLUME_INDEX] + residual, 0.0, 1.0)
        edited_props[:, HAEMO_TYPE_INDEX] = torch.clamp(edited_props[:, HAEMO_TYPE_INDEX] + residual, 0.0, 1.0)

        _, edited_rgb, _, _ = bio_skin.skin_props_to_reflectance(edited_props)
        bioskin_io.save_tensor_to_image( os.path.join(output_dir, f"frame_{index:02d}_amp{level:.2f}"),
            edited_rgb, shape, channels=3, cpu=use_cpu)

        difference = (edited_rgb - reference_rgb)[inside]
        contrast = float(torch.linalg.vector_norm(difference, dim=1).mean().detach())
        checked_props = bio_skin.reflectance_to_skin_props(edited_rgb.float())
        hemoglobin_after = in_mask_mean(checked_props[:, BLOOD_VOLUME_INDEX], inside)
        melanin_after = in_mask_mean(checked_props[:, melanin_index], inside)
        hemoglobin_out_before = in_mask_mean(skin_props[:, BLOOD_VOLUME_INDEX], outside)
        hemoglobin_out_after = in_mask_mean(checked_props[:, BLOOD_VOLUME_INDEX], outside)
        oxygenation_after = in_mask_mean(checked_props[:, HAEMO_TYPE_INDEX], inside)

        rows.append({
            "index": index,
            "residual_amp": float(level),
            "in_mask_contrast": contrast,
            "hemo_in_clean": clean_hemoglobin,
            "hemo_in_after": hemoglobin_after,
            "melanin_drift_in": melanin_after - clean_melanin,
            "hemo_drift_out": hemoglobin_out_after - hemoglobin_out_before,
            "oxy_drift": oxygenation_after - clean_oxygenation,
        })

        if abs(float(level) - target_amplitude) < amplitude["AMP_STEP"] / 2.0:
            composite_props = edited_props.clone()
            composite_amplitude = float(level)

    csv_path = os.path.join(output_dir, "erythema_sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[out] CSV -> {csv_path}")

    save_montage(output_dir)
    save_control_curve(output_dir, rows, clean_hemoglobin)

    if composite_props is not None:
        composite_path = os.path.join(output_dir, "erythema_chromophore_composite.png")
        try:
            deltas = make_chromophore_composite(
                skin_props, composite_props, shape, mask, composite_path,
                f"{composite_amplitude:.2f}")
            print(f"[out] chromophore composite -> {composite_path}")
            print_composite_deltas(deltas)
        except Exception as error:
            print(f"[out] skipped composite: {error}")

    print(f"\nDONE. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()