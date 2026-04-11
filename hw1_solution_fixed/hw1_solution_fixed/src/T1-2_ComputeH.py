
from pathlib import Path
import cv2
import numpy as np

from core_geometry import computeH, warpPerspective
from feature_frontend import matchPicsORB


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
USER_NAME = "Minjae"


if __name__ == "__main__":
    print("=== Task 1-2: Homography Estimation & Warping ===")

    img_template = cv2.imread(str(DATA_DIR / "cv_cover.jpg"))
    img_scene = cv2.imread(str(DATA_DIR / "cv_desk.png"))

    if img_template is None or img_scene is None:
        raise FileNotFoundError("Could not load cv_cover.jpg or cv_desk.png from ../data")

    matches, locs1, locs2 = matchPicsORB(img_template, img_scene)
    if len(matches) < 4:
        raise RuntimeError("Not enough matches to estimate a homography.")

    used = matches[: min(40, len(matches))]
    p1 = locs1[used[:, 0]]
    p2 = locs2[used[:, 1]]

    H_best = computeH(p1, p2)
    print("Estimated H (basic DLT):")
    print(H_best)

    out_size = (img_scene.shape[1], img_scene.shape[0])
    img_warped, mask = warpPerspective(img_template, H_best, out_size)

    result = img_scene.copy()
    result[mask > 0] = img_warped[mask > 0]

    masked_scene = img_scene.copy()
    masked_scene[mask > 0] = 0

    out1 = RESULT_DIR / "T1-2_window1.png"
    out2 = RESULT_DIR / "T1-2_window2.png"
    out3 = RESULT_DIR / "T1-2_window3.png"

    cv2.imwrite(str(out1), img_warped)
    cv2.imwrite(str(out2), masked_scene)
    cv2.imwrite(str(out3), result)

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")
    print(f"Saved: {out3}")
    print("Use these titles in GUI mode:")
    print(f"1. {USER_NAME}_Warped Template")
    print(f"2. {USER_NAME}_Masked Scene")
    print(f"3. {USER_NAME}_Final Composite Result")
