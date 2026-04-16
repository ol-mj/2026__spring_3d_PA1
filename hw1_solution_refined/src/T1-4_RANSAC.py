
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core_geometry import computeH_ransac, warpPerspective
from feature_frontend import matchPicsORB


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
USER_NAME = "Minjae"


if __name__ == "__main__":
    print("=== Task 1-4: RANSAC for Robust Homography & Analysis ===")

    img_template = cv2.imread(str(DATA_DIR / "test_on_copy1.jpg"))
    img_scene = cv2.imread(str(DATA_DIR / "test_on_copy2.jpg"))

    if img_template is None or img_scene is None:
        raise FileNotFoundError("Could not load cv_cover.jpg or cv_desk.png from ../data")

    matches, locs1, locs2 = matchPicsORB(img_template, img_scene)
    if len(matches) < 4:
        raise RuntimeError("Not enough matches to estimate a homography.")

    p1 = locs1[matches[:, 0]]
    p2 = locs2[matches[:, 1]]

    iters_to_test = [1, 30, 60, 90, 120, 150, 180, 210, 240]
    avg_inliers = []

    for k in iters_to_test:
        counts = []
        for trial in range(10):
            rng = np.random.default_rng(1234 + 100 * k + trial)
            _, inliers = computeH_ransac(p1, p2, max_iter=k, threshold=3.0, rng=rng)
            counts.append(int(np.sum(inliers)))
        avg_inliers.append(float(np.mean(counts)))

    plt.figure(figsize=(7, 4.5))
    plt.plot(iters_to_test, avg_inliers, marker="o", linewidth=2)
    plt.title("Average Inlier Count vs. RANSAC Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Average # Inliers over 10 runs")
    plt.grid(True)
    plot_path = RESULT_DIR / "T1-4_RANSAC_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()
    print(f"Saved: {plot_path}")

    H_best, inliers = computeH_ransac(p1, p2, max_iter=800, threshold=3.0, rng=np.random.default_rng(2026))
    print(f"Final inlier count: {int(np.sum(inliers))} / {len(inliers)}")

    out_size = (img_scene.shape[1], img_scene.shape[0])
    img_warped, mask = warpPerspective(img_template, H_best, out_size)

    result = img_scene.copy()
    result[mask > 0] = img_warped[mask > 0]
    cv2.putText(result, f"RANSAC Result - {USER_NAME}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    out_img = RESULT_DIR / "T1-4_RANSAC_result.png"
    cv2.imwrite(str(out_img), result)
    print(f"Saved: {out_img}")
