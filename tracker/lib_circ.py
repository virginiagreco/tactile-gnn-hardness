import cv2
import numpy as np
import re
from scipy.spatial import cKDTree

# === Natural sort helper ===
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

# === Marker detection function ===
def detect_markers(image, debug=False):
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 500
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByInertia = True
    params.minInertiaRatio = 0.2
    params.minDistBetweenBlobs = 6.0

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(blur)
    markers = np.array([kp.pt for kp in keypoints])

    if debug:
        img_with_kp = cv2.drawKeypoints(image, keypoints, None, (0,0,255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Detected markers", img_with_kp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return markers

# === Marker ID assignment ===
def assign_marker_ids(markers, n_rings=7, debug=False):
    markers = np.asarray(markers, dtype=float)
    center = np.median(markers, axis=0)
    dx, dy = markers[:,0]-center[0], markers[:,1]-center[1]
    radii  = np.hypot(dx, dy)
    angles = np.mod(np.arctan2(dy, dx), 2*np.pi)

    # ID 0 = the very center
    idx0 = int(np.argmin(radii))
    marker_ids       = {0: markers[idx0]}
    id_to_ring_angle = {0: (0,0)}
    ring_angle_to_id = {(0,0): 0}
    next_id = 1

    # remove center from the pool
    mask       = np.ones(len(markers), bool)
    mask[idx0] = False
    rem_pts    = markers[mask]
    rem_radii  = radii[mask]
    rem_angles = angles[mask]

    # build n_rings bands from r_min to r_max
    r_min, r_max = rem_radii.min(), rem_radii.max()
    thr = np.linspace(r_min, r_max, n_rings)

    for ring in range(1, n_rings):
        lo = thr[ring-1]
        hi = thr[ring]
        # make the top‐edge inclusive for the last ring
        if ring == n_rings-1:
            in_bin = np.where((rem_radii >= lo) & (rem_radii <= hi))[0]
        else:
            in_bin = np.where((rem_radii >= lo) & (rem_radii <  hi))[0]

        if in_bin.size == 0:
            continue

        # sort them by angle
        order = in_bin[np.argsort(rem_angles[in_bin])]
        for a_idx, idx in enumerate(order):
            marker_ids[next_id]           = rem_pts[idx]
            id_to_ring_angle[next_id]     = (ring, a_idx)
            ring_angle_to_id[(ring,a_idx)] = next_id
            next_id += 1

    if debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        for mid,(x,y) in marker_ids.items():
            plt.text(x,y,str(mid),fontsize=8)
            plt.plot(x,y,'ro',markersize=4)
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.xlabel("X (pixels)")        
        plt.ylabel("Y (pixels)")
        plt.show()

    return marker_ids, id_to_ring_angle, ring_angle_to_id, center

def match_to_expected_layout(expected_layout, detected_points, max_radius=12):
    exp_ids, exp_pts = zip(*expected_layout.items())
    exp_pts = np.array(exp_pts)
    tree = cKDTree(detected_points)
    matched = {}
    occupied = set()

    for mid, pos in expected_layout.items():
        dist, idx = tree.query(pos, distance_upper_bound=max_radius)
        if idx < len(detected_points) and idx not in occupied:
            matched[mid] = detected_points[idx]
            occupied.add(idx)
    return matched