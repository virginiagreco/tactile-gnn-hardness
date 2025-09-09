import math
import cv2
import numpy as np

# ==========================
# Functions
# ==========================
def build_blob_detector(params):
    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea = True
    p.minArea = float(params.get("minArea", 20))
    p.maxArea = float(params.get("maxArea", 500))
    p.filterByCircularity = True
    p.minCircularity = float(params.get("minCircularity", 0.6))
    p.filterByInertia = True
    p.minInertiaRatio = float(params.get("minInertiaRatio", 0.2))
    p.minDistBetweenBlobs = float(params.get("minDistBetweenBlobs", 6.0))
    return cv2.SimpleBlobDetector_create(p)

def detect_markers(gray, blob_cfg):
    if blob_cfg.get("blur_ksize", 0) > 0:
        k = int(blob_cfg["blur_ksize"])
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    detector = build_blob_detector(blob_cfg)
    kps = detector.detect(gray)
    pts = np.array([kp.pt for kp in kps], dtype=np.float32)
    return pts

def _principal_angle(rad):
    # map angle to [0, pi)
    a = rad % math.pi
    if a < 0:
        a += math.pi
    return a

def estimate_lattice_axes(points, knn=6, angle_tol_deg=15.0):
    if len(points) < 10:
        raise ValueError("Too few points to estimate lattice axes.")
    P = points.astype(np.float32)
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argsort(d2, axis=1)[:, :min(knn, len(P) - 1)]
    vecs = []
    for i in range(len(P)):
        for j in nn_idx[i]:
            v = P[j] - P[i]
            n = np.linalg.norm(v)
            if n > 1e-6:
                vecs.append(v / n)
    vecs = np.array(vecs, dtype=np.float32)
    if len(vecs) == 0:
        raise ValueError("No neighbor vectors found.")

    ang = np.arctan2(vecs[:, 1], vecs[:, 0])
    ang = np.array([_principal_angle(a) for a in ang], dtype=np.float32)

    bins = 90
    hist, edges = np.histogram(ang, bins=bins, range=(0, math.pi))
    k0 = np.argmax(hist)
    theta1 = 0.5 * (edges[k0] + edges[k0 + 1])
    theta2 = (theta1 + math.pi / 3.0) % math.pi

    u = np.array([math.cos(theta1), math.sin(theta1)], dtype=np.float32)
    v = np.array([math.cos(theta2), math.sin(theta2)], dtype=np.float32)

    ang_tol = math.radians(angle_tol_deg)

    def spacing_along(dir_vec):
        proj = np.dot(vecs, dir_vec)
        vangs = np.arctan2(vecs[:, 1], vecs[:, 0])
        vangs = np.array([_principal_angle(a) for a in vangs])
        mask = np.abs((vangs - _principal_angle(math.atan2(dir_vec[1], dir_vec[0])))) < ang_tol
        steps = np.abs(proj[mask])
        steps = steps[(steps > 1e-3) & (steps < np.percentile(steps, 95))]
        if len(steps) == 0:
            return 1.0
        return float(np.median(steps))

    s_u = spacing_along(u)
    s_v = spacing_along(v)

    a = u * s_u
    b = v * s_v
    return a.astype(np.float32), b.astype(np.float32)

def round_to_lattice(points, origin, A):
    a = A[:, 0]
    b = A[:, 1]
    M = np.column_stack([a, b])  # 2x2
    Minv = np.linalg.inv(M)
    U = (points - origin) @ Minv.T  # (N,2) fractional
    I = np.rint(U).astype(np.int32)
    return I, U

def refine_affine_from_coords(points, I):
    N = len(points)
    X = np.hstack([I.astype(np.float32), np.ones((N, 1), dtype=np.float32)])  # (N,3)
    Y = points.astype(np.float32)  # (N,2)
    W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # (3,2)
    A = W[:2, :].T  # (2,2)
    origin = W[2, :]  # (2,)
    return origin, A

def fit_hex_lattice(points, hex_cfg):
    a, b = estimate_lattice_axes(points, knn=hex_cfg["knn"], angle_tol_deg=hex_cfg["angle_tolerance_deg"])
    origin = np.median(points, axis=0).astype(np.float32)
    A = np.column_stack([a, b]).astype(np.float32)  # 2x2

    I, U = round_to_lattice(points, origin, A)
    for _ in range(int(hex_cfg.get("refine_iters", 3))):
        origin, A = refine_affine_from_coords(points, I)
        I, U = round_to_lattice(points, origin, A)

    order = np.lexsort((I[:, 0], I[:, 1]))  # sort by j then i
    coords_sorted = I[order]
    return origin.astype(np.float32), A.astype(np.float32), coords_sorted, order

def expected_positions_from_coords(origin, A, coords_int):
    return origin[None, :] + coords_int.astype(np.float32) @ A.T
