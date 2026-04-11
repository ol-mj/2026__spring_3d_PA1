import numpy as np
import cv2
from core_geometry import computeH, warpPerspective 
from feature_frontend import matchPicsORB

print("=== Task 1-2: Homography Estimation & Warping ===")

# Load the provided template and scene images
img_template = cv2.imread('../data/cv_cover.jpg')
img_scene = cv2.imread('../data/cv_desk.png')

if img_template is None or img_scene is None:
    print("[Error] Image not found. Please check paths.")
    exit()

# 1. Feature Matching (template -> scene)
matches, locs1, locs2 = matchPicsORB(img_template, img_scene)
if len(matches) < 4:
    print("[Error] Not enough matches to estimate a homography.")
    raise SystemExit(1)

p1 = locs1[matches[:min(12, len(matches)), 0]]
p2 = locs2[matches[:min(12, len(matches)), 1]]

H_best = computeH(p1, p2)

out_size = (img_scene.shape[1], img_scene.shape[0]) 
img_warped, mask = warpPerspective(img_template, H_best, out_size)

result = img_scene.copy()
result[mask > 0] = img_warped[mask > 0]

masked_scene = img_scene.copy()
masked_scene[mask > 0] = 0

import os
if not os.path.exists('../result'):
    os.makedirs('../result')
cv2.imwrite('../result/T1-2_window1.png', img_warped)
cv2.imwrite('../result/T1-2_window2.png', masked_scene)
cv2.imwrite('../result/T1-2_window3.png', result)

cv2.imshow("1. Minjae_Warped Template", img_warped)
cv2.imshow("2. Minjae_Masked Scene", masked_scene)
cv2.imshow("3. Minjae_Final Composite Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
