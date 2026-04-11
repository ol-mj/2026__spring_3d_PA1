import numpy as np
import cv2
import scipy.io as sio
from matplotlib import pyplot as plt

def plotMatches(im1, im2, matches, locs1, locs2, color='lime'):
    """
    Visualize matched points between two images using Matplotlib.
    
    Args:
        im1, im2: Input images
        matches: (P, 2) array of match indices [idx1, idx2]
        locs1, locs2: (N, 2) arrays of keypoint coordinates (x, y)
    """
    if len(matches) == 0:
        print("[Warning] No matched points to display.")
        return

    # 1. Prepare canvas (stack images horizontally)
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)

    def to_rgb(img):
        return np.stack([img]*3, axis=-1) if len(img.shape) == 2 else img

    vis[:h1, :w1, :] = to_rgb(im1)
    vis[:h2, w1:w1+w2, :] = to_rgb(im2)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.imshow(vis)

    # 2. Render matching points
    for i in range(matches.shape[0]):
        idx1 = int(matches[i, 0])
        idx2 = int(matches[i, 1])

        # Assign coordinates (Assume x, y order from OpenCV standard)
        x1, y1 = locs1[idx1]
        x2, y2 = locs2[idx2]

        # Add horizontal offset for the second image
        ax.plot([x1, x2 + w1], [y1, y2], color=color, linewidth=0.8, alpha=0.7)
        ax.scatter([x1, x2 + w1], [y1, y2], color='red', s=5)

    ax.axis('off')
    plt.tight_layout()
    plt.show()