import numpy as np
import cv2

def matchPicsORB(I1, I2):
	"""
	Match feature points between two images using ORB.
	
	Args:
		I1, I2: Input images to match
	Returns:
		matches: (P, 2) matrix of matched feature indices (queryIdx, trainIdx)
		locs1: (N1, 2) matrix of feature locations in image 1 (x, y)
		locs2: (N2, 2) matrix of feature locations in image 2 (x, y)
	"""
	if I1 is None or I2 is None:
		raise ValueError("Input images must not be None.")

	if I1.ndim == 3:
		I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
	if I2.ndim == 3:
		I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

	orb = cv2.ORB_create(nfeatures=2000)
	kps1, desc1 = orb.detectAndCompute(I1, None)
	kps2, desc2 = orb.detectAndCompute(I2, None)

	locs1 = np.array([kp.pt for kp in kps1], dtype=np.float32) if kps1 else np.zeros((0, 2), dtype=np.float32)
	locs2 = np.array([kp.pt for kp in kps2], dtype=np.float32) if kps2 else np.zeros((0, 2), dtype=np.float32)

	if desc1 is None or desc2 is None or len(locs1) == 0 or len(locs2) == 0:
		return np.zeros((0, 2), dtype=np.int32), locs1, locs2

	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
	knn_matches = bf.knnMatch(desc1, desc2, k=2)

	good = []
	for pair in knn_matches:
		if len(pair) < 2:
			continue
		m, n = pair
		if m.distance < 0.75 * n.distance:
			good.append(m)

	if len(good) < 8:
		bf_cross = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		good = bf_cross.match(desc1, desc2)

	good = sorted(good, key=lambda m: m.distance)[:300]
	matches = np.array([[m.queryIdx, m.trainIdx] for m in good], dtype=np.int32)
	
	return matches, locs1, locs2
