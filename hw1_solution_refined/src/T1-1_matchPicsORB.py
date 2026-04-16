
from pathlib import Path
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_frontend import matchPicsORB


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def draw_matches(im1, im2, matches, locs1, locs2):
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)

    def to_rgb(img):
        if img.ndim == 2:
            return np.stack([img] * 3, axis=-1)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    vis[:h1, :w1] = to_rgb(im1)
    vis[:h2, w1:w1 + w2] = to_rgb(im2)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(vis)
    for idx1, idx2 in matches:
        x1, y1 = locs1[idx1]
        x2, y2 = locs2[idx2]
        ax.plot([x1, x2 + w1], [y1, y2], color="cyan", linewidth=0.8, alpha=0.85)
        ax.scatter([x1, x2 + w1], [y1, y2], color="magenta", s=6)
    ax.set_title(f"ORB Matches: {len(matches)} correspondences")
    ax.axis("off")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", type=str, default=str(DATA_DIR / "test_cv.jpg"))
    parser.add_argument("--img2", type=str, default=str(DATA_DIR / "test_book2.jpg"))
    args = parser.parse_args()

    img_template = cv2.imread(args.img1)
    img_scene = cv2.imread(args.img2)

    if img_template is None or img_scene is None:
        raise FileNotFoundError("Could not load cv_cover.jpg or cv_desk.png from ../data")

    matches, locs1, locs2 = matchPicsORB(img_template, img_scene)
    print(f"[Result] Number of matches: {len(matches)}")

    fig = draw_matches(img_template, img_scene, matches, locs1, locs2)
    out_path = RESULT_DIR / "T1-1_matches_test_on_copy.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")