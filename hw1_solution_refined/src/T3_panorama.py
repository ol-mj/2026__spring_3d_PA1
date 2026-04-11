
from pathlib import Path
import argparse
import cv2
import numpy as np

from feature_frontend import matchPicsORB
from core_geometry import computeH_ransac, warpPerspective


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def create_panorama(img_left, img_right):
    imgL_h, imgL_w = img_left.shape[:2]
    imgR_h, imgR_w = img_right.shape[:2]

    matches, locsL, locsR = matchPicsORB(img_left, img_right)
    if len(matches) < 4:
        raise RuntimeError("Not enough matches to create a panorama.")

    matched_L = locsL[matches[:, 0]]
    matched_R = locsR[matches[:, 1]]

    H_R2L, _ = computeH_ransac(matched_R, matched_L, max_iter=300, threshold=3.0)

    corners_R = np.float32([[0, 0], [0, imgR_h - 1], [imgR_w - 1, imgR_h - 1], [imgR_w - 1, 0]]).reshape(-1, 1, 2)
    corners_R_transformed = cv2.perspectiveTransform(corners_R, H_R2L)

    corners_L = np.float32([[0, 0], [0, imgL_h - 1], [imgL_w - 1, imgL_h - 1], [imgL_w - 1, 0]]).reshape(-1, 1, 2)
    all_corners = np.vstack([corners_L, corners_R_transformed])

    x_coords = all_corners[:, 0, 0]
    y_coords = all_corners[:, 0, 1]
    x_min, y_min = np.floor([x_coords.min(), y_coords.min()]).astype(int)
    x_max, y_max = np.ceil([x_coords.max(), y_coords.max()]).astype(int)

    tx = -x_min if x_min < 0 else 0
    ty = -y_min if y_min < 0 else 0
    H_trans = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    H_final = H_trans @ H_R2L

    canvas_w = int(x_max - x_min + 1)
    canvas_h = int(y_max - y_min + 1)

    panorama, _ = warpPerspective(img_right, H_final, (canvas_w, canvas_h))
    _, right_mask = warpPerspective(
        np.ones((imgR_h, imgR_w), dtype=np.uint8) * 255,
        H_final,
        (canvas_w, canvas_h),
    )

    left_canvas = np.zeros_like(panorama)
    left_canvas[ty:ty + imgL_h, tx:tx + imgL_w] = img_left
    left_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    left_mask[ty:ty + imgL_h, tx:tx + imgL_w] = 255

    overlap = (left_mask > 0) & (right_mask > 0)
    only_left = (left_mask > 0) & ~overlap
    only_right = (right_mask > 0) & ~overlap

    blended = np.zeros_like(panorama)
    blended[only_left] = left_canvas[only_left]
    blended[only_right] = panorama[only_right]
    blended[overlap] = ((left_canvas[overlap].astype(np.float32) + panorama[overlap].astype(np.float32)) * 0.5).astype(np.uint8)

    nonzero = np.any(blended > 0, axis=2)
    ys, xs = np.where(nonzero)
    if len(xs) > 0 and len(ys) > 0:
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        blended = blended[y0:y1, x0:x1]
    return blended


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=str, default=str(DATA_DIR / "example.jpg"))
    parser.add_argument("--right", type=str, default=str(DATA_DIR / "example4.jpg"))
    args = parser.parse_args()

    img_left = cv2.imread(args.left)
    img_right = cv2.imread(args.right)

    if img_left is None or img_right is None:
        raise FileNotFoundError("Could not load panorama input images.")

    result = create_panorama(img_left, img_right)
    out_path = RESULT_DIR / "T3_panorama_result.png"
    cv2.imwrite(str(out_path), result)
    print(f"Saved: {out_path}")
