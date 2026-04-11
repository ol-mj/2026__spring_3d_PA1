import numpy as np
import cv2
import os
from feature_frontend import matchPicsORB
from core_geometry import computeH_ransac, warpPerspective

def create_panorama(img_left, img_right):
    """
    TODO: Stitch two images together to create a panorama.
    1. Feature Matching & Point Alignment
    2. Homography Estimation (Right image to Left image)
    3. Coordinate Space Calculation (Bounding Box)
    4. Warp & Blending
    """
    imgL_h, imgL_w = img_left.shape[:2]
    imgR_h, imgR_w = img_right.shape[:2]

    # 1. Feature Matching using ORB
    # TODO: Get matches and locations from matchPicsORB
    # matches, locsL, locsR = ...
    
    # 2. Align matched points
    # TODO: Using 'matches' index array, align coordinates of locsL and locsR
    # matched_L = ...
    # matched_R = ...

    # 3. Estimate Homography (Right to Left)
    # TODO: Compute H using computeH_ransac
    # H_R2L, _ = ...
    
    # 4. Determine Panorama Bounding Box
    # TODO: Transform Right image corners into Left image space
    # corners_R = np.float32([[0, 0], [0, imgR_h-1], [imgR_w-1, imgR_h-1], [imgR_w-1, 0]]).reshape(-1, 1, 2)
    # corners_R_transformed = ... (use cv2.perspectiveTransform or manual projection)
    
    # TODO: Calculate x_min, y_min, x_max, y_max to encompass both images
    # ...
    
    # 5. Coordinate Adjustment (Translation)
    # TODO: Construct a translation matrix H_trans to move (x_min, y_min) to (0, 0)
    # tx, ty = ...
    # H_trans = ...
    
    # TODO: Combine H_trans and H_R2L
    # H_final = ...
    
    # 6. Warping and Blending
    # TODO: Define output canvas size
    # canvas_w, canvas_h = ...
    
    # Warp the right image
    # Note: You can use cv2.warpPerspective for the final result
    # panorama = cv2.warpPerspective(img_right, H_final, (canvas_w, canvas_h))
    
    # TODO: Copy the left image into the canvas at the translated position (tx, ty)
    # panorama[...] = img_left
    
    return None # Return the final panorama

if __name__ == "__main__":
    # Load images
    img_left = None
    img_right = None
    
    if img_left is None or img_right is None:
        print("Error: Could not load images.")
    else:
        # Resize for convenience
        img_left = cv2.resize(img_left, None, fx=0.5, fy=0.5)
        img_right = cv2.resize(img_right, None, fx=0.5, fy=0.5)
        
        result = create_panorama(img_left, img_right)
        
        if result is not None:
            cv2.imshow("Panorama Result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("T3_panorama_result.png", result)
