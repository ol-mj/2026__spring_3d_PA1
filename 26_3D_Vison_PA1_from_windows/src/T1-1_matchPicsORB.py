import numpy as np
import cv2
from core_geometry import computeH, warpPerspective 
from feature_frontend import matchPicsORB
from helper import plotMatches

print("=== Task 1-1: ORB Feature Matching ===")
# Load provided template and scene images
img_template = cv2.imread('../data/cv_cover.jpg')
img_scene = cv2.imread('../data/cv_desk.png')

if img_template is None or img_scene is None:
    print("[Error] Could not load images. Please check the paths.")
    exit()

# 1. Perform ORB Matching (TODO: Implement this in feature_frontend.py)
matches, locs1, locs2 = matchPicsORB(img_template, img_scene)

print(f"\n[Result] Number of matched points: {len(matches)}")

# 2. Visualize using the provided helper function
# TODO: Use a color other than green (e.g., 'cyan', 'orange', 'magenta') for your report
print("Rendering matches...")
plotMatches(img_template, img_scene, matches, locs1, locs2, color='cyan')