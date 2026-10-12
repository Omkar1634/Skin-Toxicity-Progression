# import pandas as pd

# metrics = pd.read_csv(r"D:\Github\PhD Code\Erythema-Progression\output\derivative_analysis\derivative_metrics.csv")
# selected = pd.read_csv(r"D:\Github\PhD Code\Erythema-Progression\Inflammatory\selected_identities_stratified.csv")

# # Normalise face_id to match
# metrics['face_id_int'] = metrics['face_id'].astype(str).str.zfill(6).astype(int) if metrics['face_id'].dtype == object else metrics['face_id'].astype(int)
# selected['face_id_int'] = selected['face_id'].astype(int)

# merged = metrics.merge(selected[['face_id_int','fitzpatrick','ita']], on='face_id_int', how='left')

# # Table 1: Mean ± std of mean_inside per grade step × Fitzpatrick type
# table1 = merged.groupby(['fitzpatrick','grade_step'])['mean_inside'].agg(['mean','std']).round(4)
# print(table1.to_string())
# table1.to_csv(r"D:\Github\PhD Code\Erythema-Progression\output\derivative_analysis\table1_fitz_grade.csv")


import pandas as pd
from scipy import stats

metrics  = pd.read_csv(r"D:\Github\PhD Code\Erythema-Progression\output\derivative_analysis\derivative_metrics.csv")
selected = pd.read_csv(r"D:\Github\PhD Code\Erythema-Progression\Inflammatory\selected_identities_stratified.csv")
replacements = pd.read_csv(r"D:\Github\PhD Code\Erythema-Progression\Inflammatory\replacements_fitz6.csv")

# Which Fitz VI IDs from selected are actually in metrics?
metrics['face_id_int'] = metrics['face_id'].astype(str).str.lstrip('0').apply(lambda x: int(x) if x else 0)
processed_ids = set(metrics['face_id_int'].unique())

# Keep original selected rows that were processed, replace missing Fitz VI with replacements
selected['face_id_int'] = selected['face_id'].astype(int)
selected_processed = selected[selected['face_id_int'].isin(processed_ids)]

replacements['face_id_int'] = replacements['face_id'].astype(int)
replacements_processed = replacements[replacements['face_id_int'].isin(processed_ids)][['face_id_int','ita','fitzpatrick']]

lookup = pd.concat([
    selected_processed[['face_id_int','ita','fitzpatrick']],
    replacements_processed
], ignore_index=True).drop_duplicates(subset='face_id_int')

merged = metrics.merge(lookup, on='face_id_int', how='inner')

print(f"Total faces in merged: {merged['face_id_int'].nunique()}")
print("\nPearson r (ITA vs mean_inside Δa*) per grade step:")
for step in sorted(merged['grade_step'].unique()):
    sub = merged[merged['grade_step'] == step].dropna(subset=['ita','mean_inside'])
    r, p = stats.pearsonr(sub['ita'], sub['mean_inside'])
    print(f"  {step}: r = {r:.4f}, p = {p:.4e}, n = {len(sub)}")

print("\nPer-Fitzpatrick table:")
table = merged.groupby(['fitzpatrick','grade_step'])['mean_inside'].agg(['mean','std']).round(4)
print(table.to_string())




# import pandas as pd
# from scipy import stats

# metrics = pd.read_csv(r"D:\Github\PhD Code\Erythema-Progression\output\derivative_analysis\derivative_metrics.csv")
# selected = pd.read_csv(r"D:\Github\PhD Code\Erythema-Progression\Inflammatory\selected_identities_stratified.csv")

# metrics['face_id_int'] = metrics['face_id'].astype(str).str.lstrip('0').apply(lambda x: int(x) if x else 0)
# selected['face_id_int'] = selected['face_id'].astype(int)
# merged = metrics.merge(selected[['face_id_int','ita']], on='face_id_int', how='left')

# print("Pearson r (ITA vs mean_inside Δa*) per grade step:")
# for step in sorted(merged['grade_step'].unique()):
#     sub = merged[merged['grade_step'] == step].dropna(subset=['ita','mean_inside'])
#     r, p = stats.pearsonr(sub['ita'], sub['mean_inside'])
#     print(f"  {step}: r = {r:.4f}, p = {p:.4e}, n = {len(sub)}")







# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
# import numpy as np
# from scipy import stats

# # ── Load data ────────────────────────────────────────────────────────────────
# metrics  = pd.read_csv(r"D:\Github\PhD Code\Erythema-Progression\output\derivative_analysis\derivative_metrics.csv")
# selected = pd.read_csv(r"D:\Github\PhD Code\Erythema-Progression\Inflammatory\selected_identities_stratified.csv")

# metrics['face_id_int']  = metrics['face_id'].astype(str).str.lstrip('0').apply(lambda x: int(x) if x else 0)
# selected['face_id_int'] = selected['face_id'].astype(int)
# merged = metrics.merge(selected[['face_id_int','ita','fitzpatrick']], on='face_id_int', how='left')

# # ── Colour palette: light → dark skin tone ───────────────────────────────────
# fitz_order  = ['I','II','III','IV','V','VI']
# fitz_labels = ['I\n(very light)','II\n(light)','III\n(interm.)','IV\n(tan)','V\n(brown)','VI\n(dark)']
# fitz_colors = ['#F4C8A0','#E8A87C','#C98060','#A0522D','#6B3A2A','#3B1A0E']
# grade_steps = ['grade_0_to_1','grade_1_to_2','grade_2_to_3','grade_3_to_4']
# grade_labels = ['0→1','1→2','2→3','3→4']

# # ── Panel A: line plot per Fitzpatrick ───────────────────────────────────────
# table = merged.groupby(['fitzpatrick','grade_step'])['mean_inside'].mean().unstack('grade_step')
# table = table.reindex(fitz_order)[grade_steps]

# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# fig.suptitle("Erythema Signal Attenuation Across Skin Tones (n = 1200)", fontsize=13, fontweight='bold', y=1.01)

# ax = axes[0]
# for i, (fitz, color) in enumerate(zip(fitz_order, fitz_colors)):
#     vals = table.loc[fitz].values
#     ax.plot(grade_labels, vals, marker='o', color=color, linewidth=2.2, markersize=7, label=f'Fitz {fitz}')
# ax.set_xlabel("CEA Grade Step", fontsize=11)
# ax.set_ylabel("Mean Δa* (inside butterfly mask)", fontsize=11)
# ax.set_title("(A) Erythema Progression by Fitzpatrick Type", fontsize=11)
# ax.legend(title="Fitzpatrick", fontsize=9, title_fontsize=9)
# ax.set_ylim(bottom=0)
# ax.grid(axis='y', alpha=0.3)
# ax.spines[['top','right']].set_visible(False)

# # ── Panel B: scatter ITA vs Δa* (grade 1→2) with regression ─────────────────
# ax2 = axes[1]
# sub = merged[merged['grade_step'] == 'grade_1_to_2'].dropna(subset=['ita','mean_inside','fitzpatrick'])

# for fitz, color in zip(fitz_order, fitz_colors):
#     s = sub[sub['fitzpatrick'] == fitz]
#     ax2.scatter(s['ita'], s['mean_inside'], c=color, alpha=0.35, s=12, label=f'Fitz {fitz}')

# # Regression line
# x = sub['ita'].values
# y = sub['mean_inside'].values
# slope, intercept, r, p, _ = stats.linregress(x, y)
# xr = np.linspace(x.min(), x.max(), 200)
# ax2.plot(xr, slope * xr + intercept, 'k--', linewidth=1.8, label=f'r = {r:.3f}, p < 10⁻⁶⁴')

# ax2.set_xlabel("ITA (Individual Typology Angle)", fontsize=11)
# ax2.set_ylabel("Mean Δa* grade 1→2", fontsize=11)
# ax2.set_title("(B) ITA vs Erythema Signal (Grade 1→2)", fontsize=11)
# ax2.legend(fontsize=8.5, markerscale=1.5)
# ax2.grid(alpha=0.3)
# ax2.spines[['top','right']].set_visible(False)

# plt.tight_layout()
# out = r"D:\Github\PhD Code\Erythema-Progression\output\derivative_analysis\fig_attenuation.png"
# plt.savefig(out, dpi=200, bbox_inches='tight')
# plt.show()
# print(f"Saved → {out}")





































































































# """
# ita_one_face.py -- load ONE face, build the butterfly mask, print its ITA.

# Self-contained: no imports from your other files, nothing gets modified.
# Just fill in the four paths at the top and run it.
# """

# import cv2
# import numpy as np
# import yaml 
# import argparse

# def parse_args():
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument("--config", default="Config.yaml", help="YAML config path")
#     return parser.parse_args()

# def load_config(config_path):
#     with open(config_path, "r", encoding="utf-8") as config_file:
#         return yaml.safe_load(config_file)

# args = parse_args()
# config = load_config(args.config)
# paths = config["paths"]
# chromophore = config["chromophore_parameter"]
# amplitude = config["amplitude_parameter"]
# mask_config = config["mask_parameter"]


# # ----------------------------------------------------------------------------
# # 1. butterfly mask  (copied from your mask.py, unchanged logic)
# # ----------------------------------------------------------------------------
# def butterfly_mask(H, W):
#     mask         = cv2.imread(paths["mask"],    cv2.IMREAD_GRAYSCALE)
#     mask_mouth   = cv2.imread(paths["mask_mouth"],   cv2.IMREAD_GRAYSCALE)
#     mask_nostril = cv2.imread(paths["mask_nostril"], cv2.IMREAD_GRAYSCALE)

#     mask         = cv2.resize(mask,         (W, H), interpolation=cv2.INTER_NEAREST)
#     mask_mouth   = cv2.resize(mask_mouth,   (W, H), interpolation=cv2.INTER_NEAREST)
#     mask_nostril = cv2.resize(mask_nostril, (W, H), interpolation=cv2.INTER_NEAREST)

#     combined  = cv2.subtract(mask, cv2.add(mask_nostril, mask_mouth))
#     feathered = cv2.GaussianBlur(combined, (31, 31), 8)
#     return feathered.astype(np.float32) / 255.0


# # ----------------------------------------------------------------------------
# # 2. ITA from the in-mask skin pixels
# # ----------------------------------------------------------------------------
# def compute_ita(albedo_bgr_uint8, mask):
#     # BGR uint8 -> RGB float [0,1]
#     rgb = albedo_bgr_uint8[..., ::-1].astype(np.float64) / 255.0

#     # PNG is gamma-encoded sRGB -> convert to linear light
#     rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

#     # keep only solidly-in-mask pixels (drop the feathered edge)
#     sel = mask >= 0.99
#     px = rgb[sel]

#     # linear RGB -> XYZ (sRGB primaries, D65) -> CIELAB
#     M = np.array([[0.4124564, 0.3575761, 0.1804375],
#                   [0.2126729, 0.7151522, 0.0721750],
#                   [0.0193339, 0.1191920, 0.9503041]])
#     xyz = px @ M.T
#     xyz /= np.array([0.95047, 1.0, 1.08883])
#     d = 6.0 / 29.0
#     f = np.where(xyz > d**3, np.cbrt(xyz), xyz / (3 * d**2) + 4.0 / 29.0)
#     L = 116 * f[:, 1] - 16
#     b = 200 * (f[:, 1] - f[:, 2])

#     # ITA per pixel, then take the median
#     ita_px = np.degrees(np.arctan2(L - 50.0, b))
#     return float(np.median(ita_px)), int(px.shape[0])


# # ----------------------------------------------------------------------------
# # run it
# # ----------------------------------------------------------------------------
# if __name__ == "__main__":
#     config = load_config("Config.yaml")
#     paths = config["paths"]
#     chromophore = config["chromophore_parameter"]
#     amplitude = config["amplitude_parameter"]
#     mask_config = config["mask_parameter"]

#     albedo = cv2.imread(paths["ALBEDO_PATH"], cv2.IMREAD_COLOR)   # BGR, uint8
#     if albedo is None:
#         raise FileNotFoundError(f"could not read {paths['ALBEDO_PATH']}")

#     H, W = albedo.shape[:2]
#     mask = butterfly_mask(H, W)

#     ita, n = compute_ita(albedo, mask)
#     print(f"ITA = {ita:.2f} degrees   (over {n} in-mask pixels)")
#     print("higher = lighter skin, lower = darker skin")


















# # import cv2 as cv
# # import numpy as np

# # # 1. Load image and masks
# # img = cv.imread(r"D:\Github\PhD Code\Erythema-Progression\data\000000.png")
# # mask = cv.imread(r"D:\Github\PhD Code\Erythema-Progression\FFHQ-UV\topo_assets\major_valid_whole_mask.png", cv.IMREAD_GRAYSCALE)
# # mask_cen = cv.imread(r"D:\Github\PhD Code\Erythema-Progression\FFHQ-UV\topo_assets\center_face_mask.png", cv.IMREAD_GRAYSCALE)
# # mask_left = cv.imread(r"D:\Github\PhD Code\Erythema-Progression\FFHQ-UV\topo_assets\major_valid_left_mask.png", cv.IMREAD_GRAYSCALE)
# # mask_right = cv.imread(r"D:\Github\PhD Code\Erythema-Progression\FFHQ-UV\topo_assets\major_valid_right_mask.png", cv.IMREAD_GRAYSCALE)
# # mask_front = cv.imread(r"D:\Github\PhD Code\Erythema-Progression\FFHQ-UV\topo_assets\major_valid_front_mask.png", cv.IMREAD_GRAYSCALE)
# # mask_mouth = cv.imread(r"D:\Github\PhD Code\Erythema-Progression\FFHQ-UV\topo_assets\mouth_constract_mask.png", cv.IMREAD_GRAYSCALE)
# # mask_nosal = cv.imread(r"D:\Github\PhD Code\Erythema-Progression\FFHQ-UV\topo_assets\nosal_base_mask.png", cv.IMREAD_GRAYSCALE)
# # mask_nostril = cv.imread(r"D:\Github\PhD Code\Erythema-Progression\FFHQ-UV\topo_assets\nostril_mask.png", cv.IMREAD_GRAYSCALE)



# # # Safety check for image loading
# # if img is None:
# #     raise FileNotFoundError("Base image could not be loaded. Check path.")

# # target_size = (img.shape[1], img.shape[0])

# # # 2. Resize masks
# # mask = cv.resize(mask, target_size, interpolation=cv.INTER_NEAREST)
# # mask_cen = cv.resize(mask_cen, target_size, interpolation=cv.INTER_NEAREST)
# # mask_left = cv.resize(mask_left, target_size, interpolation=cv.INTER_NEAREST)
# # mask_right = cv.resize(mask_right, target_size, interpolation=cv.INTER_NEAREST)
# # mask_front = cv.resize(mask_front, target_size, interpolation=cv.INTER_NEAREST)
# # mask_mouth = cv.resize(mask_mouth, target_size, interpolation=cv.INTER_NEAREST)
# # mask_nosal = cv.resize(mask_nosal, target_size, interpolation=cv.INTER_NEAREST)
# # mask_nostril = cv.resize(mask_nostril, target_size, interpolation=cv.INTER_NEAREST)



# # # 3. Create mask using cv.subtract (saturates at 0, preventing byte overflow)
# # # mask_combined = mask - (mask_nostril +  mask_mouth)
# # # mask_combinedG = mask - (mask_nostril +  mask_mouth)
# # # mask_combinedGs = mask - (mask_nostril +  mask_mouth)


# # # mask_mouthG = cv.GaussianBlur(mask_combinedG, (21,21), 8)
# # # mask_mouthGs = cv.GaussianBlur(mask_combinedGs, (31,31),8)

# # # mask_combined = mask_combined.astype(np.float32) / 255.0
# # # mask_combinedG = mask_mouthG.astype(np.float32) / 255.0      
# # # mask_combinedGs = mask_mouthGs.astype(np.float32) / 255.0

# # mask_combined = cv.subtract(mask, cv.add(mask_nostril, mask_mouth))
# # feathered_mask_combinedG = cv.GaussianBlur(mask_combined, (31, 31), 8)
# # feathered_mask_float = feathered_mask_combinedG.astype(np.float32) / 255.0


# # # single_channel_mask = cv.subtract(mask, mask_combined)
# # # single_channel_maskG = cv.subtract(mask, mask_combinedG)
# # # single_channel_maskGs = cv.subtract(mask, mask_combinedGs)



# # # 4. Perform bitwise operations using the 2D single-channel mask
# # # masked_img = cv.bitwise_and(img, img, mask=mask_combined)
# # # masked_imgG = cv.bitwise_and(img, img, mask=mask_combinedG)
# # # masked_imgGs = cv.bitwise_and(img, img, mask=mask_combinedGs)






# # # 5. Display results
# # cv.imshow('Masked Image  ', feathered_mask_float)
# # # cv.imshow('Masked Image with gaussian sigma 21*21*8', mask_combinedG)
# # # cv.imshow('Masked Image with gaussian sigma 31*31*8', mask_combinedGs)

# # #cv.imshow('mouth', mask_mouth)
# # # cv.imshow('bitXor', bitXor)
# # # cv.imshow('bitNot', bitNot)

# # cv.waitKey(0)
# # cv.destroyAllWindows()

# import os, glob, re, csv, shutil

# albedo_dir = r"D:\Github\PhD Code\Biophysical-LDM\dataset\Albedo-UV"
# out_csv    = r"D:\Github\PhD Code\Biophysical-LDM\dataset\variant_id_mapping.csv"
# VARIANT_RE = re.compile(r"^\d{6}_cof_a\d+_final_uv$")
# START_ID   = 53751

# variant_files = sorted([
#     p for p in glob.glob(os.path.join(albedo_dir, "*.png"))
#     if VARIANT_RE.fullmatch(os.path.splitext(os.path.basename(p))[0])
# ])

# print(f"Found {len(variant_files)} variant files — renaming and writing CSV...")

# rows = []
# for i, src in enumerate(variant_files):
#     new_id       = START_ID + i
#     old_name     = os.path.basename(src)                        # e.g. 000061_cof_a8_final_uv.png
#     new_name     = f"{new_id:06d}.png"                          # e.g. 053751.png
#     dst          = os.path.join(albedo_dir, new_name)
#     shutil.copy2(src, dst)                                      # copy; delete src manually if you want
#     rows.append({"new_id": f"{new_id:06d}",
#                  "new_filename": new_name,
#                  "old_filename": old_name,
#                  "original_path": src})

# with open(out_csv, "w", newline="") as f:
#     w = csv.DictWriter(f, fieldnames=["new_id","new_filename","old_filename","original_path"])
#     w.writeheader()
#     w.writerows(rows)

# print(f"Done. {len(rows)} files copied and mapped → {out_csv}")






# import os, glob, re

# albedo_dir = r"D:\Github\PhD Code\Biophysical-LDM\dataset\Albedo-UV"
# VARIANT_RE = re.compile(r"^\d{6}_cof_a\d+_final_uv$")

# to_delete = [
#     p for p in glob.glob(os.path.join(albedo_dir, "*.png"))
#     if VARIANT_RE.fullmatch(os.path.splitext(os.path.basename(p))[0])
# ]

# print(f"Deleting {len(to_delete)} files...")
# for p in to_delete:
#     os.remove(p)
# print("Done.")

# from pathlib import Path
# import shutil

# # Path to your 1000-identities folder
# base_dir = Path(r"D:\Github\PhD Code\Erythema-Progression\output\1000-identities")

# expected_count = 43
# deleted_count = 0

# # Iterate through all subfolders
# for folder in base_dir.iterdir():
#   if folder.is_dir():
#     # Count files
#     file_count = sum(1 for f in folder.iterdir() if f.is_file())

#     if file_count != expected_count:
#       print(
#           f"Deleting folder {folder.name}: contained {file_count} files"
#           f" (Expected {expected_count})"
#       )
#       # Completely remove the folder and its contents
#       shutil.rmtree(folder)
#       deleted_count += 1

# print("-" * 40)
# print(f"Cleanup complete. Successfully deleted {deleted_count} folders.")