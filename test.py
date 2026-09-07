import cv2 as cv
import numpy as np

# 1. Load image and masks
img = cv.imread(r"D:\Github\PhD Code\Erythema-Progression\data\000000.png")
mask = cv.imread(r"D:\Github\PhD Code\Erythema-Progression\FFHQ-UV\topo_assets\major_valid_whole_mask.png", cv.IMREAD_GRAYSCALE)
mask_cen = cv.imread(r"D:\Github\PhD Code\Erythema-Progression\FFHQ-UV\topo_assets\center_face_mask.png", cv.IMREAD_GRAYSCALE)
mask_left = cv.imread(r"D:\Github\PhD Code\Erythema-Progression\FFHQ-UV\topo_assets\major_valid_left_mask.png", cv.IMREAD_GRAYSCALE)
mask_right = cv.imread(r"D:\Github\PhD Code\Erythema-Progression\FFHQ-UV\topo_assets\major_valid_right_mask.png", cv.IMREAD_GRAYSCALE)
mask_front = cv.imread(r"D:\Github\PhD Code\Erythema-Progression\FFHQ-UV\topo_assets\major_valid_front_mask.png", cv.IMREAD_GRAYSCALE)
mask_mouth = cv.imread(r"D:\Github\PhD Code\Erythema-Progression\FFHQ-UV\topo_assets\mouth_constract_mask.png", cv.IMREAD_GRAYSCALE)
mask_nosal = cv.imread(r"D:\Github\PhD Code\Erythema-Progression\FFHQ-UV\topo_assets\nosal_base_mask.png", cv.IMREAD_GRAYSCALE)
mask_nostril = cv.imread(r"D:\Github\PhD Code\Erythema-Progression\FFHQ-UV\topo_assets\nostril_mask.png", cv.IMREAD_GRAYSCALE)



# Safety check for image loading
if img is None:
    raise FileNotFoundError("Base image could not be loaded. Check path.")

target_size = (img.shape[1], img.shape[0])

# 2. Resize masks
mask = cv.resize(mask, target_size, interpolation=cv.INTER_NEAREST)
mask_cen = cv.resize(mask_cen, target_size, interpolation=cv.INTER_NEAREST)
mask_left = cv.resize(mask_left, target_size, interpolation=cv.INTER_NEAREST)
mask_right = cv.resize(mask_right, target_size, interpolation=cv.INTER_NEAREST)
mask_front = cv.resize(mask_front, target_size, interpolation=cv.INTER_NEAREST)
mask_mouth = cv.resize(mask_mouth, target_size, interpolation=cv.INTER_NEAREST)
mask_nosal = cv.resize(mask_nosal, target_size, interpolation=cv.INTER_NEAREST)
mask_nostril = cv.resize(mask_nostril, target_size, interpolation=cv.INTER_NEAREST)



# 3. Create mask using cv.subtract (saturates at 0, preventing byte overflow)
# mask_combined = mask - (mask_nostril +  mask_mouth)
# mask_combinedG = mask - (mask_nostril +  mask_mouth)
# mask_combinedGs = mask - (mask_nostril +  mask_mouth)


# mask_mouthG = cv.GaussianBlur(mask_combinedG, (21,21), 8)
# mask_mouthGs = cv.GaussianBlur(mask_combinedGs, (31,31),8)

# mask_combined = mask_combined.astype(np.float32) / 255.0
# mask_combinedG = mask_mouthG.astype(np.float32) / 255.0      
# mask_combinedGs = mask_mouthGs.astype(np.float32) / 255.0

mask_combined = cv.subtract(mask, cv.add(mask_nostril, mask_mouth))
feathered_mask_combinedG = cv.GaussianBlur(mask_combined, (31, 31), 8)
feathered_mask_float = feathered_mask_combinedG.astype(np.float32) / 255.0


# single_channel_mask = cv.subtract(mask, mask_combined)
# single_channel_maskG = cv.subtract(mask, mask_combinedG)
# single_channel_maskGs = cv.subtract(mask, mask_combinedGs)



# 4. Perform bitwise operations using the 2D single-channel mask
# masked_img = cv.bitwise_and(img, img, mask=mask_combined)
# masked_imgG = cv.bitwise_and(img, img, mask=mask_combinedG)
# masked_imgGs = cv.bitwise_and(img, img, mask=mask_combinedGs)






# 5. Display results
cv.imshow('Masked Image  ', feathered_mask_float)
# cv.imshow('Masked Image with gaussian sigma 21*21*8', mask_combinedG)
# cv.imshow('Masked Image with gaussian sigma 31*31*8', mask_combinedGs)

#cv.imshow('mouth', mask_mouth)
# cv.imshow('bitXor', bitXor)
# cv.imshow('bitNot', bitNot)

cv.waitKey(0)
cv.destroyAllWindows()