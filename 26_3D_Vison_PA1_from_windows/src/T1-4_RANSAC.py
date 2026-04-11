import numpy as np
import cv2
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

# TODO 1: Extract the matched point coordinates
# p1 = ...
# p2 = ...

# 2. [Analysis Section] Statistical Performance of RANSAC
print("\n[Analysis] Evaluating inlier count vs. iterations...")
iters_to_test = [1, 30, 60, 90, 120, 150, 180, 210, 240]
avg_inliers = []

# TODO 2: Compute the average number of inliers over 10 trials for each k in iters_to_test
# For each 'k', run computeH_ransac 10 times, calculate the mean of inlier counts, and append to avg_inliers
# for k in iters_to_test:
#     ...
#     avg_inliers.append(...)

# Plotting the results
# TODO 3: Plot 'iters_to_test' vs 'avg_inliers'. Customize the color of the plot.
# plt.plot(...)
# plt.title(...)
# plt.xlabel(...)
# plt.ylabel(...)
# plt.grid(True)
# plt.show()

# 3. Final Robust Estimation
print("\n[Final] Computing refined Homography with RANSAC...")

# TODO 4: Compute the final H using max_iter=240
# H_best, inliers = ...

# 4. Warping
out_size = (img_scene.shape[1], img_scene.shape[0])

# TODO 5: Warp the template image using H_best and overlay it onto the scene image using the mask
# img_warped, mask = ...
# result = img_scene.copy()
# ...

# 5. Visualization (TODO: Replace 'YourName' with your actual name)
# NOTE: Placeholder to prevent crash. Re-enable after TODO 5 is complete.
result = np.zeros_like(img_scene)

cv2.imshow("RANSAC Final Result - YourName", result)
cv2.waitKey(0)
cv2.destroyAllWindows()