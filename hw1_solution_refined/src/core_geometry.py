
import numpy as np
import cv2


def _to_homogeneous(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Input points must have shape (N, 2).")
    return np.hstack([x, np.ones((x.shape[0], 1), dtype=np.float64)])


def _normalize_h(H):
    H = np.asarray(H, dtype=np.float64)
    if abs(H[2, 2]) > 1e-12:
        return H / H[2, 2]
    n = np.linalg.norm(H)
    return H / n if n > 0 else H


def _normalize_points(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        raise ValueError("Cannot normalize an empty point set.")
    centroid = np.mean(x, axis=0)
    shifted = x - centroid
    dist = np.sqrt(np.sum(shifted ** 2, axis=1))
    mean_dist = np.mean(dist)
    scale = 1.0 if mean_dist < 1e-12 else np.sqrt(2.0) / mean_dist
    T = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    x_h = _to_homogeneous(x)
    x_n = (T @ x_h.T).T
    x_n = x_n[:, :2] / x_n[:, 2:3]
    return x_n, T


def project_points(H, x):
    x_h = _to_homogeneous(x)
    y_h = (np.asarray(H, dtype=np.float64) @ x_h.T).T
    valid = np.abs(y_h[:, 2]) > 1e-12
    y = np.full((x.shape[0], 2), np.nan, dtype=np.float64)
    y[valid] = y_h[valid, :2] / y_h[valid, 2:3]
    return y


def computeH(x1, x2):
    """
    Compute homography H such that x2 ~ H x1 using Direct Linear Transform.
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    if x1.shape != x2.shape or x1.ndim != 2 or x1.shape[1] != 2:
        raise ValueError("x1 and x2 must both have shape (N, 2).")
    if x1.shape[0] < 4:
        raise ValueError("At least 4 correspondences are required.")

    A = []
    for (x, y), (u, v) in zip(x1, x2):
        A.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u])
        A.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v])
    A = np.asarray(A, dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return _normalize_h(H)


def warpPerspective(img, H, size_warped):
    """
    Manual inverse warping.
    Uses nearest-neighbor interpolation for stability and simplicity.
    """
    width, height = size_warped
    if width <= 0 or height <= 0:
        raise ValueError("size_warped must be positive.")

    src = np.asarray(img)
    grayscale = (src.ndim == 2)
    if grayscale:
        src = src[..., None]

    src_h, src_w, channels = src.shape
    warped = np.zeros((height, width, channels), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    H_inv = np.linalg.inv(np.asarray(H, dtype=np.float64))

    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    dst = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], axis=0)
    src_homo = H_inv @ dst
    valid_w = np.abs(src_homo[2]) > 1e-12

    src_x = np.full(xs.size, np.nan, dtype=np.float64)
    src_y = np.full(xs.size, np.nan, dtype=np.float64)
    src_x[valid_w] = src_homo[0, valid_w] / src_homo[2, valid_w]
    src_y[valid_w] = src_homo[1, valid_w] / src_homo[2, valid_w]

    src_x_nn = np.rint(src_x).astype(np.int32)
    src_y_nn = np.rint(src_y).astype(np.int32)

    valid = (
        valid_w
        & (src_x_nn >= 0) & (src_x_nn < src_w)
        & (src_y_nn >= 0) & (src_y_nn < src_h)
    )

    warped.reshape(-1, channels)[valid] = src[src_y_nn[valid], src_x_nn[valid]]
    mask.reshape(-1)[valid] = 255

    if grayscale:
        warped = warped[..., 0]
    return warped, mask


def computeH_norm(x1, x2):
    """
    Compute homography using normalized DLT.
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    if x1.shape != x2.shape or x1.ndim != 2 or x1.shape[1] != 2:
        raise ValueError("x1 and x2 must both have shape (N, 2).")
    if x1.shape[0] < 4:
        raise ValueError("At least 4 correspondences are required.")

    x1_n, T1 = _normalize_points(x1)
    x2_n, T2 = _normalize_points(x2)

    H_n = computeH(x1_n, x2_n)
    H = np.linalg.inv(T2) @ H_n @ T1
    return _normalize_h(H)


def computeH_ransac(locs1, locs2, max_iter=200, threshold=3.0, rng=None):
    """
    Robust homography estimation with RANSAC.
    """
    locs1 = np.asarray(locs1, dtype=np.float64)
    locs2 = np.asarray(locs2, dtype=np.float64)

    if locs1.shape != locs2.shape or locs1.ndim != 2 or locs1.shape[1] != 2:
        raise ValueError("locs1 and locs2 must both have shape (N, 2).")

    n = locs1.shape[0]
    if n < 4:
        return np.eye(3, dtype=np.float64), np.zeros(n, dtype=bool)

    if rng is None:
        rng = np.random.default_rng()

    best_H = np.eye(3, dtype=np.float64)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    best_mean_err = np.inf
    all_idx = np.arange(n)

    for _ in range(max_iter):
        sample_idx = rng.choice(all_idx, size=4, replace=False)
        try:
            H_candidate = computeH_norm(locs1[sample_idx], locs2[sample_idx])
        except Exception:
            continue

        proj = project_points(H_candidate, locs1)
        valid = ~np.isnan(proj).any(axis=1)
        errors = np.full(n, np.inf, dtype=np.float64)
        errors[valid] = np.linalg.norm(proj[valid] - locs2[valid], axis=1)
        inliers = errors < threshold
        count = int(np.sum(inliers))
        mean_err = float(np.mean(errors[inliers])) if count > 0 else np.inf

        if count > best_count or (count == best_count and mean_err < best_mean_err):
            best_count = count
            best_mean_err = mean_err
            best_inliers = inliers
            best_H = H_candidate

    if np.sum(best_inliers) >= 4:
        try:
            best_H = computeH_norm(locs1[best_inliers], locs2[best_inliers])
        except Exception:
            pass

    return _normalize_h(best_H), best_inliers
