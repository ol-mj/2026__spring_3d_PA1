import numpy as np
import cv2
import matplotlib.pyplot as plt
from core_geometry import computeH_ransac
from feature_frontend import matchPicsORB

class PlanarTracker:
    def __init__(self, initial_corners):
        self.initial_corners = initial_corners
        self.corners_homo = np.hstack((initial_corners, np.ones((4, 1))))
        self.H_accum = np.eye(3)
        self.errors = []

    def update_accum(self, H_step):
        """
        TODO: Update self.H_accum using the incremental homography H_step.
        Formula: H_accum_new = H_step @ H_accum_old
        """
        if H_step is not None:
            # self.H_accum = ...
            return True
        return False

    def get_projected_corners(self, H):
        """
        TODO: Project initial_corners using homography H.
        1. Perform matrix multiplication: projected = H @ corners_homo
        2. Normalize by dividing by the 3rd component (w).
        """
        if H is None: return None
        # projected = ...
        # return normalized_corners (N, 2)
        return None

    def compute_drift_error(self, H_direct):
        """
        TODO: Compute the Frobenius norm error between normalized H_accum and H_direct.
        1. Normalize both H matrices such that H[2, 2] = 1.
        2. Compute linalg norm of (H1 - H2).
        """
        if H_direct is None: return None
        # error = ...
        # self.errors.append(error)
        # return error
        return 0.0

def select_initial_corners(first_frame):
    """
    Mouse interface to select 4 corners in the first frame.
    Supports resizing for high-resolution images.
    """
    corners = []
    h, w = first_frame.shape[:2]
    
    # Calculate scale factor
    scale = 1.0
    if w > 1280 or h > 720:
        scale = min(1280/w, 720/h)
    
    display_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            # Scale coordinates back to original size
            x_orig, y_orig = x / scale, y / scale
            corners.append((x_orig, y_orig))
            cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select 4 Corners", display_frame)

    cv2.imshow("Select 4 Corners", display_frame)
    cv2.setMouseCallback("Select 4 Corners", mouse_callback)
    
    print("Click 4 corners of the object (Clockwise or Counter-clockwise).")
    print("Press any key after selecting 4 points.")
    cv2.waitKey(0)
    cv2.destroyWindow("Select 4 Corners")
    
    if len(corners) != 4:
        return None
    return np.array(corners, dtype=np.float32)

def run_planar_tracker(video_path, initial_corners=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    ret, ref_frame = cap.read()
    if not ret: return

    if initial_corners is None:
        initial_corners = select_initial_corners(ref_frame)
        if initial_corners is None: return

    tracker = PlanarTracker(initial_corners)
    prev_frame = ref_frame.copy()
    frame_idx = 0

    while True:
        ret, curr_frame = cap.read()
        if not ret: break

        frame_idx += 1
        
        # ---------------------------------------------------------
        # TODO: Step 1. Accumulated Tracking (Frame-to-Frame)
        # 1. Match features between prev_frame and curr_frame
        # 2. Align points using the matches array
        #    Hint: matched_prev = locs_prev[matches_step[:, 0]]
        # 3. Compute H_step using matched points
        # 4. Update tracker.H_accum
        # ---------------------------------------------------------
        # matches_step, locs_prev, locs_curr = ...
        
        # ---------------------------------------------------------
        # TODO: Step 2. Direct Tracking (Frame 0 to Current)
        # 1. Match features between ref_frame and curr_frame
        # 2. Align points using the matches array
        # 3. Compute H_direct using matched points
        # ---------------------------------------------------------
        # matches_ref, locs_ref, locs_curr_ref = ...
        
        # ---------------------------------------------------------
        # TODO: Step 3. Compute Error for Analysis
        # ---------------------------------------------------------
        # error = tracker.compute_drift_error(H_direct)
        
        # ---------------------------------------------------------
        # Step 4. Visualization (Provided)
        # ---------------------------------------------------------
        display_frame = curr_frame.copy()
        
        # --- Visualization (Separate Windows) ---
        # Window names
        win_video = "Planar Tracking: Video"
        win_plot = "Drift Error Plot"
        
        # 1. Prepare Video Window
        # Resize for display if the resolution is too high
        h_orig, w_orig = display_frame.shape[:2]
        scale = 1.0
        if w_orig > 1280 or h_orig > 720:
            scale = min(1280/w_orig, 720/h_orig)
            display_frame = cv2.resize(display_frame, None, fx=scale, fy=scale)
        
        # Draw Accumulated BB (Red) - TODO: Change color to something other than red (0, 0, 255)
        corners_accum = tracker.get_projected_corners(tracker.H_accum)
        if corners_accum is not None:
            # Note: We multiply by 'scale' to match the resized display_frame
            cv2.polylines(display_frame, [(corners_accum * scale).astype(np.int32)], True, (0, 0, 255), 2)

        # Draw Direct BB (Green) - TODO: Change color to show your own work
        # corners_direct = tracker.get_projected_corners(H_direct)
        # if corners_direct is not None:
        #     cv2.polylines(display_frame, [(corners_direct * scale).astype(np.int32)], True, (0, 255, 0), 2)

        # UI Info
        cv2.putText(display_frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "RED: Accumulated (TODO)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(display_frame, "Customized: Direct (TODO)", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 2. Render Error Plot using Matplotlib
        fig = plt.figure(figsize=(5, 4), dpi=100)
        plt.plot(tracker.errors, color='blue', label='Drift Error')
        plt.title("Numerical Error Analysis")
        plt.xlabel("Frame")
        plt.ylabel("Frobenius Norm")
        plt.grid(True)

        fig.canvas.draw()
        plot_img = np.asarray(fig.canvas.buffer_rgba())
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        plt.close(fig) 

        # 3. Handle Window Positioning
        if frame_idx == 1:
            cv2.namedWindow(win_video)
            cv2.namedWindow(win_plot)
            try:
                import tkinter as tk
                root = tk.Tk()
                sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
                root.destroy()
                
                v_w, v_h = display_frame.shape[1], display_frame.shape[0]
                p_w, p_h = plot_img.shape[1], plot_img.shape[0]
                gap = 20
                total_w = v_w + p_w + gap
                
                start_x = (sw - total_w) // 2
                start_y = (sh - max(v_h, p_h)) // 2
                
                cv2.moveWindow(win_video, start_x, start_y)
                cv2.moveWindow(win_plot, start_x + v_w + gap, start_y)
            except:
                pass

        cv2.imshow(win_video, display_frame)
        cv2.imshow(win_plot, plot_img)

        if cv2.waitKey(10) & 0xFF == 27:
            break

        prev_frame = curr_frame.copy()

    cap.release()
    cv2.destroyAllWindows()
    if len(tracker.errors) > 0:
        plt.show()

if __name__ == '__main__':
    video_path = '../data/planar_video.mp4' 
    run_planar_tracker(video_path)
