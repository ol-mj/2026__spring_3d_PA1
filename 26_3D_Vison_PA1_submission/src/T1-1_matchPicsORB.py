import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_frontend import matchPicsORB

print("=== Task 1-1: ORB Feature Matching ===")
img_template = cv2.imread('../data/cv_cover.jpg')
img_scene = cv2.imread('../data/cv_desk.png')

if img_template is None or img_scene is None:
    print("[Error] Could not load images. Please check the paths.")
    raise SystemExit(1)

matches, locs1, locs2 = matchPicsORB(img_template, img_scene)
print(f"\n[Result] Number of matched points: {len(matches)}")

if len(matches) == 0:
    print("[Error] No matched points were found.")
    raise SystemExit(1)

h1, w1 = img_template.shape[:2]
h2, w2 = img_scene.shape[:2]
vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
vis[:h1, :w1] = cv2.cvtColor(img_template, cv2.COLOR_BGR2RGB)
vis[:h2, w1:w1 + w2] = cv2.cvtColor(img_scene, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(figsize=(14, 6))
ax.imshow(vis)
for idx1, idx2 in matches:
    x1, y1 = locs1[idx1]
    x2, y2 = locs2[idx2]
    ax.plot([x1, x2 + w1], [y1, y2], color='cyan', linewidth=0.8, alpha=0.85)
    ax.scatter([x1, x2 + w1], [y1, y2], color='magenta', s=6)
ax.axis('off')
plt.tight_layout()

import os
if not os.path.exists('../result'):
    os.makedirs('../result')
fig.savefig('../result/T1-1_matches.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print("Saved to ../result/T1-1_matches.png")
