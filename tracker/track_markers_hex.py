import glob
import json
import csv
import time
import cv2
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment

from lib_hex import detect_markers, fit_hex_lattice, expected_positions_from_coords

# ==========================
# Config (defaults)
# ==========================
CONFIG = {
    # I/O roots
    "videos_root": "stiffness-videos",     # contains stiffness-1 ... stiffness-5
    "csv_root": "csv_results",             # where per-video CSV folders will go
    "tracked_video_root": "tracked_outputs",

    # Visualisation
    "scale": 2,          
    "fps_out": None,     
    "show": False,       

    # Matching
    "match_threshold": 100.0,  # px threshold for a valid match (after warp prediction)

    # Detection params
    "blob": {
        "minArea": 45,
        "maxArea": 600,
        "minCircularity": 0.6,
        "minInertiaRatio": 0.2,
        "minDistBetweenBlobs": 6.0,
        "blur_ksize": 5
    },

    # Lattice fitting
    "hex": {
        "knn": 6,                 # neighbors for direction estimation
        "angle_tolerance_deg": 15.0,
        "refine_iters": 3
    }
}

# ==========================
# Core tracking for ONE video
# ==========================
def track_one_video(video_path: Path, out_csv_dir: Path, out_video_path: Path, config: dict):
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_video_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Read first frame for sizing
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Empty video: {video_path}")
    if frame.ndim == 3:
        gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray0 = frame.copy()

    # --- Detect & bootstrap lattice on first frame ---
    pts0 = detect_markers(gray0, config["blob"])
    if len(pts0) < 10:
        raise RuntimeError(f"Too few detections in first frame ({len(pts0)}) for {video_path}")
    origin, A, coords_sorted, order = fit_hex_lattice(pts0, config["hex"])
    P0 = expected_positions_from_coords(origin, A, coords_sorted)  # (N,2)

    # Output initial positions
    init_csv = out_csv_dir / "initial_positions_hex.csv"
    with init_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y"])
        for i, (x, y) in enumerate(P0):
            w.writerow([i, f"{x:.3f}", f"{y:.3f}"])

    # Prepare video writer
    h0, w0 = gray0.shape[:2]
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps_out = config["fps_out"] if config["fps_out"] else in_fps
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vw = cv2.VideoWriter(str(out_video_path), fourcc, fps_out, (int(w0*config["scale"]), int(h0*config["scale"])))

    # Tracking state
    prev_positions = P0.copy().astype(np.float32)
    t = 0
    frame_times = []

    # process first frame again 
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        start = time.time()
        det = detect_markers(gray, config["blob"])  # (M,2)
        M = len(det)
        if M == 0:
            cur_positions = prev_positions.copy()
            matched_idx, missing_idx = [], list(range(len(prev_positions)))
        else:
            # Predict warp from prev 
            try:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=1).fit(det)
                _, indices = nn.kneighbors(prev_positions)
                pairs_src = prev_positions
                pairs_dst = det[indices[:, 0]]
                M_affine, _ = cv2.estimateAffinePartial2D(
                    pairs_src.reshape(-1, 1, 2).astype(np.float32),
                    pairs_dst.reshape(-1, 1, 2).astype(np.float32),
                    method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99
                )
            except Exception:
                M_affine = None

            P_pred = prev_positions.copy() if M_affine is None else \
                     cv2.transform(prev_positions.reshape(-1, 1, 2), M_affine).reshape(-1, 2)

            # Hungarian assignment between predicted and detections
            N = len(prev_positions)
            if M == 0:
                cost = np.full((N, 1), np.inf, dtype=np.float32)
            else:
                diff = P_pred[:, None, :] - det[None, :, :]
                cost = np.linalg.norm(diff, axis=2)
            row_ind, col_ind = linear_sum_assignment(cost)

            cur_positions = P_pred.copy()
            matched_idx, missing_idx = [], []
            for r, c in zip(row_ind, col_ind):
                if c < M and cost[r, c] < config["match_threshold"]:
                    cur_positions[r] = det[c]
                    matched_idx.append(r)
                else:
                    missing_idx.append(r)
            all_r = set(range(N))
            matched_set = set(matched_idx)
            for r in sorted(all_r - matched_set):
                if r not in missing_idx:
                    missing_idx.append(r)

        # Write per-frame CSV
        csv_path = out_csv_dir / f"frame_{t:05d}.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "x", "y"])
            for i, (x, y) in enumerate(cur_positions):
                w.writerow([i, f"{x:.3f}", f"{y:.3f}"])

        # Visualisation
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        vis = cv2.resize(vis, (int(w0*config["scale"]), int(h0*config["scale"])), interpolation=cv2.INTER_NEAREST)

        # draw detections (red), matched (green), missing (yellow)
        for (x, y) in det:
            cv2.circle(vis, (int(x*config["scale"]), int(y*config["scale"])), 3, (0, 0, 255), -1)
        for i in matched_idx:
            x, y = cur_positions[i]
            cv2.circle(vis, (int(x*config["scale"]), int(y*config["scale"])), 6, (0, 255, 0), 2)
        for i in missing_idx:
            x, y = cur_positions[i]
            cv2.circle(vis, (int(x*config["scale"]), int(y*config["scale"])), 6, (0, 255, 255), 2)

        vw.write(vis)
        if config["show"]:
            cv2.imshow("HEX Tracking", vis)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

        prev_positions = cur_positions
        t += 1
        frame_times.append(time.time() - start)

    cap.release()
    vw.release()
    cv2.destroyAllWindows()

    if frame_times:
        times = np.array(frame_times)
        print(f"[DONE] {video_path.name}: {t} frames | avg {times.mean():.3f}s/frame "
              f"(std {times.std():.3f}, min {times.min():.3f}, max {times.max():.3f})")

    # Write a small metadata file
    meta = dict(CONFIG)
    meta.update({
        "video_in": str(video_path),
        "out_video": str(out_video_path),
        "out_results_dir": str(out_csv_dir),
        "frames_processed": t,
        "input_fps": float(in_fps),
        "output_fps": float(fps_out),
    })
    with (out_csv_dir / "config_used.json").open("w") as f:
        json.dump(meta, f, indent=2)

# ==========================
# Batch over dataset
# ==========================
def discover_videos(root: Path):
    """
    Yields tuples: (class_id, video_path)
    Expects subfolders named stiffness-1 ... stiffness-5 with files 'robot_video_*.*'
    """
    for cls in range(1, 6):
        sub = root / f"stiffness-{cls}"
        if not sub.is_dir():
            print(f"[WARN] missing folder: {sub}")
            continue
        # all common video extensions
        vids = []
        for ext in ("*.mp4", "*.mkv", "*.avi", "*.mov", "*.MP4", "*.MKV", "*.AVI", "*.MOV"):
            vids.extend(glob.glob(str(sub / f"robot_video_*{ext[1:]}")))
        # If pattern above misses, fall back to any video in the folder
        if not vids:
            for ext in ("*.mp4", "*.mkv", "*.avi", "*.mov", "*.MP4", "*.MKV", "*.AVI", "*.MOV"):
                vids.extend(glob.glob(str(sub / ext)))
        vids = sorted(set(map(Path, vids)))
        for vp in vids:
            yield cls, vp

def process_all_videos(config=CONFIG):
    root = Path(config["videos_root"])
    csv_root = Path(config["csv_root"])
    vid_out_root = Path(config["tracked_video_root"])

    total = 0
    for cls, vpath in discover_videos(root):
        stem = vpath.stem  # e.g., robot_video_001
        out_csv_dir = csv_root / f"tracking_results_hex_{cls}" / stem
        out_vid_dir = vid_out_root / f"stiffness-{cls}"
        out_video_path = out_vid_dir / f"{stem}_tracked.avi"

        print(f"\n[RUN] class={cls}  video={vpath.name}")
        print(f"     -> CSV: {out_csv_dir}")
        print(f"     -> AVI: {out_video_path}")
        try:
            track_one_video(vpath, out_csv_dir, out_video_path, config)
            total += 1
        except Exception as e:
            print(f"[ERROR] {vpath}: {e}")

    print(f"\n[SUMMARY] processed {total} video(s). Outputs in '{csv_root}' and '{vid_out_root}'")

# ==========================
# Entry
# ==========================
if __name__ == "__main__":
    process_all_videos(CONFIG)
