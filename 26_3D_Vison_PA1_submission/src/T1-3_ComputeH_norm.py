import numpy as np
import cv2
from core_geometry import computeH, computeH_norm, warpPerspective 
from feature_frontend import matchPicsORB
import os

print("=== Task 1-3: Normalized Homography Comparison ===")

img_template = cv2.imread('../data/cv_cover.jpg')
img_scene = cv2.imread('../data/cv_desk.png')

if img_template is None or img_scene is None:
    print("[Error] Images not loaded. Please fill in the paths and variable names.")
    raise SystemExit(1)

matches, locs1, locs2 = matchPicsORB(img_template, img_scene)
if len(matches) < 4:
    print("[Error] Not enough matches to estimate a homography.")
    raise SystemExit(1)

p1 = locs1[matches[:, 0]]
p2 = locs2[matches[:, 1]]

print("Computing Non-normalized Homography...")
H_basic = computeH(p1, p2)

print("Computing Normalized Homography...")
H_norm = computeH_norm(p1, p2)

out_size = (img_scene.shape[1], img_scene.shape[0])
img_warped_basic, mask_basic = warpPerspective(img_template, H_basic, out_size)
img_warped_norm, mask_norm = warpPerspective(img_template, H_norm, out_size)

res_basic = img_scene.copy()
res_basic[mask_basic > 0] = img_warped_basic[mask_basic > 0]

res_norm = img_scene.copy()
res_norm[mask_norm > 0] = img_warped_norm[mask_norm > 0]

cv2.putText(res_basic, "Basic DLT", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
cv2.putText(res_norm, "Normalized DLT", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
comparison_res = np.hstack([res_basic, res_norm])

print("\nSaving comparison: [Left] Basic DLT | [Right] Normalized DLT")
if not os.path.exists('../result'):
    os.makedirs('../result')
cv2.imwrite('../result/T1-3_comparison.png', comparison_res)
print("Saved to ../result/T1-3_comparison.png")
