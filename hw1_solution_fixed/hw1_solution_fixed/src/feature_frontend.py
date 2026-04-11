
import numpy as np
import cv2


def _to_gray(img):
    if img is None:
        raise ValueError("Input image is None.")
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def matchPicsORB(I1, I2, nfeatures=2000, keep_top=300, ratio=0.75):
    """
    Match feature points between two images using ORB.

    Returns:
        matches: (P, 2) array [idx_in_locs1, idx_in_locs2]
        locs1: (N1, 2) keypoints in image 1
        locs2: (N2, 2) keypoints in image 2
    """
    gray1 = _to_gray(I1)
    gray2 = _to_gray(I2)

    orb = cv2.ORB_create(nfeatures=nfeatures)
    kps1, desc1 = orb.detectAndCompute(gray1, None)
    kps2, desc2 = orb.detectAndCompute(gray2, None)

    locs1 = np.array([kp.pt for kp in kps1], dtype=np.float32) if kps1 else np.zeros((0, 2), dtype=np.float32)
    locs2 = np.array([kp.pt for kp in kps2], dtype=np.float32) if kps2 else np.zeros((0, 2), dtype=np.float32)

    if desc1 is None or desc2 is None or len(locs1) == 0 or len(locs2) == 0:
        return np.zeros((0, 2), dtype=np.int32), locs1, locs2

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 8:
        bf_cross = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        good = bf_cross.match(desc1, desc2)

    good = sorted(good, key=lambda m: m.distance)
    if keep_top is not None:
        good = good[:keep_top]

    matches = np.array([[m.queryIdx, m.trainIdx] for m in good], dtype=np.int32)
    return matches, locs1, locs2
