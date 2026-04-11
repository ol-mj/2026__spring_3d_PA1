import numpy as np
import cv2
from core_geometry import computeH, computeH_norm, warpPerspective 
from feature_frontend import matchPicsORB

print("=== Task 1-3: Normalized Homography Comparison ===")

# TODO: Load your own captured images here
img_template = None 
img_scene = None

if img_template is None or img_scene is None:
    print("[Error] Images not loaded. Please fill in the paths and variable names.")
    exit()

# 1. Matching
matches, locs1, locs2 = matchPicsORB(img_template, img_scene)

# TODO 1: Extract the actual coordinates of matched points
# p1 = ...
# p2 = ...

# 2. [Comparison] Compute both Non-normalized and Normalized Homography
print("Computing Non-normalized Homography...")
# TODO 2-1: Call computeH
# H_basic = ...

print("Computing Normalized Homography...")
# TODO 2-2: Call computeH_norm
# H_norm = ...

# 3. Warping for both results
out_size = (img_scene.shape[1], img_scene.shape[0])

# TODO 3: Warp the template image using both H_basic and H_norm
# img_warped_basic, mask_basic = ...
# img_warped_norm, mask_norm = ...

# 4. Blending with the original scene
# TODO 4: Overlay the warped images onto the original scene using the masks
# res_basic = img_scene.copy()
# ... 
# res_norm = img_scene.copy()
# ...

# 5. Visualization of Results
# TODO 5: Combine the two results side-by-side (e.g., using np.hstack)
# comparison_res = ...

# NOTE: Placeholder for comparison_res so the script doesn't crash before you implement it.
# Remove this line once you've implemented TODO 5.
comparison_res = np.zeros((img_scene.shape[0], img_scene.shape[1] * 2, 3), dtype=np.uint8)

print("\nSaving comparison: [Left] Basic DLT | [Right] Normalized DLT")
import os
if not os.path.exists('../result'):
    os.makedirs('../result')
cv2.imwrite('../result/T1-3_comparison.png', comparison_res)
print("Saved to ../result/T1-3_comparison.png")
