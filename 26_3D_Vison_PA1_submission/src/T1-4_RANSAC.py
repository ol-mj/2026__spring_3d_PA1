import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core_geometry import computeH_ransac, warpPerspective 
from feature_frontend import matchPicsORB

print("=== Task 1-4: RANSAC for Robust Homography & Analysis ===")

# Use the provided template and scene images for consistent grading
img_template = cv2.imread('../data/cv_cover.jpg')
img_scene = cv2.imread('../data/cv_desk.png')

if img_template is None or img_scene is None:
    print("[Error] Could not load images. Please check the paths.")
    exit()

# 1. Matching
matches, locs1, locs2 = matchPicsORB(img_template, img_scene)
if len(matches) < 4:
    print("[Error] Not enough matches to estimate a homography.")
    raise SystemExit(1)

p1 = locs1[matches[:, 0]]
p2 = locs2[matches[:, 1]]

print("\n[Analysis] Evaluating inlier count vs. iterations...")
iters_to_test = [1, 30, 60, 90, 120, 150, 180, 210, 240]
avg_inliers = []

for k in iters_to_test:
    counts = []
    for _ in range(10):
        _, inliers = computeH_ransac(p1, p2, max_iter=k, threshold=3.0)
        counts.append(int(np.sum(inliers)))
    avg_inliers.append(float(np.mean(counts)))

plt.figure(figsize=(7, 4.5))
plt.plot(iters_to_test, avg_inliers, marker='o', linewidth=2, color='darkorange')
plt.title("Average Inlier Count vs. RANSAC Iterations")
plt.xlabel("Iterations")
plt.ylabel("Average # Inliers over 10 runs")
plt.grid(True)
import os
if not os.path.exists('../result'):
    os.makedirs('../result')
plt.tight_layout()
plt.savefig('../result/T1-4_RANSAC_plot.png', dpi=180)
plt.close()

print("\n[Final] Computing refined Homography with RANSAC...")
H_best, inliers = computeH_ransac(p1, p2, max_iter=240, threshold=3.0)
print(f"Final inlier count: {int(np.sum(inliers))} / {len(inliers)}")

out_size = (img_scene.shape[1], img_scene.shape[0])
img_warped, mask = warpPerspective(img_template, H_best, out_size)
result = img_scene.copy()
result[mask > 0] = img_warped[mask > 0]
cv2.imwrite('../result/T1-4_RANSAC_result.png', result)

cv2.imshow("RANSAC Final Result - Minjae", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
