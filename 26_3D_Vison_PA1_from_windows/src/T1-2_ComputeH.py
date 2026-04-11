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

# TODO 1: Extract the matched point coordinates
# p1 = ...
# p2 = ...

# 2. Compute Homography
# TODO 2: Compute the homography H using your computeH implementation
# H_best = ...

# 3. Warping
out_size = (img_scene.shape[1], img_scene.shape[0]) 
# TODO 3: Warp the template image using your warpPerspective implementation
# img_warped, mask = ...

# 4. Composite Implementation
# TODO 4: Overlay the warped template onto the scene image using the mask
# result = img_scene.copy()
# ...

# 5. Visualization (TODO: Replace 'YourName' with your actual name)
# NOTE: Placeholder to prevent crash. Re-enable after implementation.
result = np.zeros_like(img_scene)

cv2.imshow("1. YourName_Warped Template", result) # Or show img_warped if implemented
cv2.imshow("2. YourName_Masked Scene", result)
cv2.imshow("3. YourName_Final Composite Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()