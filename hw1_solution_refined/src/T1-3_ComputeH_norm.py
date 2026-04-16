
from pathlib import Path
import argparse
import cv2
import numpy as np

from core_geometry import computeH, computeH_norm, warpPerspective
from feature_frontend import matchPicsORB


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", type=str, default=str(DATA_DIR / "test_on_copy1.jpg"))
    parser.add_argument("--img2", type=str, default=str(DATA_DIR / "test_on_copy2.jpg"))
    parser.add_argument("--num_matches", type=int, default=12)
    args = parser.parse_args()

    print("=== Task 1-3: Normalized Homography Comparison ===")

    img_template = cv2.imread(args.img1)
    img_scene = cv2.imread(args.img2)

    if img_template is None or img_scene is None:
        raise FileNotFoundError("Could not load the comparison images.")

    matches, locs1, locs2 = matchPicsORB(img_template, img_scene)
    if len(matches) < 4:
        raise RuntimeError("Not enough matches to estimate a homography.")

    used = matches[: max(4, min(args.num_matches, len(matches)))]
    p1 = locs1[used[:, 0]]
    p2 = locs2[used[:, 1]]

    H_basic = computeH(p1, p2)
    H_norm = computeH_norm(p1, p2)

    out_size = (img_scene.shape[1], img_scene.shape[0])
    img_warped_basic, mask_basic = warpPerspective(img_template, H_basic, out_size)
    img_warped_norm, mask_norm = warpPerspective(img_template, H_norm, out_size)

    res_basic = img_scene.copy()
    res_basic[mask_basic > 0] = img_warped_basic[mask_basic > 0]

    res_norm = img_scene.copy()
    res_norm[mask_norm > 0] = img_warped_norm[mask_norm > 0]

    cv2.putText(res_basic, "Basic DLT", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
    cv2.putText(res_norm, "Normalized DLT", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)

    comparison_res = np.hstack([res_basic, res_norm])
    out_path = RESULT_DIR / "T1-3_comparison.png"
    cv2.imwrite(str(out_path), comparison_res)
    print(f"Compared basic vs normalized DLT using the same top {len(used)} ORB matches.")
    print(f"Saved: {out_path}")
