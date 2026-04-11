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
	# TODO: 1) Convert images to grayscale
	
	# TODO: 2) Initialize ORB detector using cv2.ORB_create
	
	# TODO: 3) Detect keypoints and compute descriptors using detectAndCompute
	
	# TODO: 4) Perform matching using cv2.BFMatcher and cv2.NORM_HAMMING
	
	# TODO: 5) Sort matches by distance and select top K (e.g., 100)
	
	# 6) Return match indices (matches) and all keypoint coordinates (locs1, locs2)
	# Hint: Use keypoint.pt to get the (x, y) coordinates.
	
	matches = np.array([])
	locs1 = np.array([])
	locs2 = np.array([])
	
	return matches, locs1, locs2