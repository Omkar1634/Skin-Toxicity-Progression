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

from pathlib import Path
import shutil

# Path to your 1000-identities folder
base_dir = Path(r"D:\Github\PhD Code\Erythema-Progression\output\1000-identities")

expected_count = 43
deleted_count = 0

# Iterate through all subfolders
for folder in base_dir.iterdir():
  if folder.is_dir():
    # Count files
    file_count = sum(1 for f in folder.iterdir() if f.is_file())

    if file_count != expected_count:
      print(
          f"Deleting folder {folder.name}: contained {file_count} files"
          f" (Expected {expected_count})"
      )
      # Completely remove the folder and its contents
      shutil.rmtree(folder)
      deleted_count += 1

print("-" * 40)
print(f"Cleanup complete. Successfully deleted {deleted_count} folders.")