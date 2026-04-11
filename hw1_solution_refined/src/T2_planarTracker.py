
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core_geometry import computeH_ransac
from feature_frontend import matchPicsORB


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CORNERS = np.array([
    [300.0, 812.0],
    [747.0, 812.0],
    [744.0, 1017.0],
    [297.0, 1019.0],
], dtype=np.float32)


class PlanarTracker:
    def __init__(self, initial_corners):
        self.initial_corners = np.asarray(initial_corners, dtype=np.float64)
        self.corners_homo = np.hstack([self.initial_corners, np.ones((4, 1), dtype=np.float64)])
        self.H_accum = np.eye(3, dtype=np.float64)
        self.errors = []

    def update_accum(self, H_step):
        if H_step is not None:
            self.H_accum = H_step @ self.H_accum
            if abs(self.H_accum[2, 2]) > 1e-12:
                self.H_accum /= self.H_accum[2, 2]
            return True
        return False

    def get_projected_corners(self, H):
        if H is None:
            return None
        projected = (np.asarray(H, dtype=np.float64) @ self.corners_homo.T).T
        w = projected[:, 2:3]
        valid = np.abs(w[:, 0]) > 1e-12
        if not np.all(valid):
            return None
        return projected[:, :2] / w

    def compute_drift_error(self, H_direct):
        corners_accum = self.get_projected_corners(self.H_accum)
        corners_direct = self.get_projected_corners(H_direct)
        if corners_accum is None or corners_direct is None:
            self.errors.append(np.nan)
            return None

        corner_errors = np.linalg.norm(corners_accum - corners_direct, axis=1)
        error = float(np.mean(corner_errors))
        self.errors.append(float(error))
        return error


def draw_box(img, corners, color, label):
    if corners is None:
        return img
    canvas = img.copy()
    pts = np.round(corners).astype(np.int32)
    cv2.polylines(canvas, [pts], True, color, 3)
    x, y = pts[0]
    cv2.putText(canvas, label, (int(x), int(max(30, y - 10))), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return canvas


def run_planar_tracker(video_path, initial_corners=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, ref_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read the first video frame.")

    if initial_corners is None:
        initial_corners = DEFAULT_CORNERS

    tracker = PlanarTracker(initial_corners)
    prev_frame = ref_frame.copy()
    frame_idx = 0

    target_indices = sorted(set([1, max(1, total_frames // 2), max(1, total_frames - 1)])) if total_frames > 1 else [1]
    snapshots = {}

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        H_step = None
        H_direct = None

        matches_step, locs_prev, locs_curr = matchPicsORB(prev_frame, curr_frame)
        if len(matches_step) >= 4:
            matched_prev = locs_prev[matches_step[:, 0]]
            matched_curr = locs_curr[matches_step[:, 1]]
            H_step, _ = computeH_ransac(matched_prev, matched_curr, max_iter=120, threshold=3.0)
            tracker.update_accum(H_step)

        matches_ref, locs_ref, locs_curr_ref = matchPicsORB(ref_frame, curr_frame)
        if len(matches_ref) >= 4:
            matched_ref = locs_ref[matches_ref[:, 0]]
            matched_curr_ref = locs_curr_ref[matches_ref[:, 1]]
            H_direct, _ = computeH_ransac(matched_ref, matched_curr_ref, max_iter=180, threshold=3.0)

        tracker.compute_drift_error(H_direct)

        display_frame = curr_frame.copy()
        corners_accum = tracker.get_projected_corners(tracker.H_accum)
        corners_direct = tracker.get_projected_corners(H_direct)

        display_frame = draw_box(display_frame, corners_accum, (255, 255, 0), "Accumulated")
        display_frame = draw_box(display_frame, corners_direct, (255, 0, 255), "Direct")
        cv2.putText(display_frame, f"Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if frame_idx in target_indices:
            snapshots[frame_idx] = display_frame.copy()

        prev_frame = curr_frame.copy()

    cap.release()

    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(1, len(tracker.errors) + 1), tracker.errors, linewidth=2)
    plt.title("Drift Error vs Frame Number")
    plt.xlabel("Frame Number")
    plt.ylabel("Mean Corner Error (pixels)")
    plt.grid(True)
    plot_path = RESULT_DIR / "T2_drift_error_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()

    ordered_frames = [snapshots[k] for k in sorted(snapshots.keys()) if k in snapshots]
    panel_path = None
    if ordered_frames:
        target_h = 360
        resized = []
        for img in ordered_frames[:3]:
            h, w = img.shape[:2]
            new_w = int(round(w * target_h / h))
            resized.append(cv2.resize(img, (new_w, target_h)))
        panel = np.hstack(resized)
        panel_path = RESULT_DIR / "T2_tracking_frames.png"
        cv2.imwrite(str(panel_path), panel)

    return plot_path, panel_path


if __name__ == "__main__":
    video_path = DATA_DIR / "planar_video.mp4"
    plot_path, panel_path = run_planar_tracker(video_path)
    print(f"Saved: {plot_path}")
    if panel_path is not None:
        print(f"Saved: {panel_path}")
