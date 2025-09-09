import cv2
import os
import csv
import time
import numpy as np
from glob import glob
from scipy.optimize import linear_sum_assignment

from lib_circ import detect_markers, assign_marker_ids, natural_sort_key

# === PARAMETERS ===
match_threshold = 2000   # px threshold for valid match
scale           = 2      # up‐scale factor for display
fps             = 10     # output video framerate
wait_for_key    = True

# === PATHS ===
init_img_path    = r"frames_bw/init_0.png"
frames_folder    = r"frames_bw/"
out_video        = r"tracked_output.avi"
out_results_dir  = "tracking_results"     # new folder for per-frame CSVs

# === 1) INITIALIZATION ===
init = cv2.imread(init_img_path, cv2.IMREAD_GRAYSCALE)
pts0 = np.array(detect_markers(init), dtype=float)
marker_ids, _, ring_angle_to_id, _ = assign_marker_ids(pts0, n_rings=7, debug=False)

all_ids            = sorted(marker_ids.keys())
N                  = len(all_ids)
initial_positions  = np.array([marker_ids[mid] for mid in all_ids], float)

out_csv = "initial_positions.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id","x","y"])
    for mid, (x,y) in zip(all_ids, initial_positions):
        writer.writerow([mid, f"{x:.3f}", f"{y:.3f}"])

# template for warp‐based prediction
P_init = initial_positions.astype(np.float32)  # (N,2)

# make sure results folder exists
os.makedirs(out_results_dir, exist_ok=True)

# === 2) PREPARE VIDEO WRITER ===
first_frame = cv2.imread(glob(os.path.join(frames_folder,"frame_*_0.png"))[0],
                         cv2.IMREAD_GRAYSCALE)
h0, w0 = first_frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(out_video, fourcc, fps, (w0*scale, h0*scale))

# === 3) MAIN LOOP ===
frame_paths = sorted(glob(os.path.join(frames_folder,"frame_*_0.png")),
                     key=natural_sort_key)
T = len(frame_paths)
frame_times = []
positions      = np.zeros((T, N, 2), float)
matched_counts = np.zeros(T, int)
missing_counts = np.zeros(T, int)
warp_errors    = np.zeros(T, float)

clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

for t, path in enumerate(frame_paths):
    start = time.time()

    frame_name = os.path.basename(path).rsplit('.',1)[0]
    frame      = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # # 3.0) PREPROCESS (CLAHE + morphology + Gaussian blur)
    # blur = cv2.GaussianBlur(frame, (5,5), 0)
    # eq   = clahe.apply(blur)
    # eq   = cv2.morphologyEx(eq, cv2.MORPH_CLOSE, kernel, iterations=1)
    # eq   = cv2.morphologyEx(eq, cv2.MORPH_OPEN,  kernel, iterations=1)

    #cv2.imshow("Processed", eq)

    # 3.1) detect blobs
    det_list = detect_markers(frame)
    D = np.array(det_list, float)
    M = len(det_list)
    if M == 0 and t>0:
        # carry forward last‐known
        positions[t]      = positions[t-1]
        matched_counts[t] = matched_counts[t-1]
        missing_counts[t] = N
    else:
        # 3.2) compute warp prediction
        if t == 0:
            P_warp = P_init.copy()
            warp_errors[t] = 0.0
        else:
            src = P_init.reshape(-1,1,2).astype(np.float32)
            dst = positions[t-1].reshape(-1,1,2).astype(np.float32)
            M_affine, inliers = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=3
            )
            if M_affine is None:
                P_warp = P_init.copy()
            else:
                P_warp = cv2.transform(P_init.reshape(-1,1,2), M_affine).reshape(-1,2)
                inl = inliers.flatten().astype(bool)
                diffs = dst[inl,0,:] - P_warp[inl]
                warp_errors[t] = np.linalg.norm(diffs,axis=1).mean()

        # 3.3) Hungarian match between P_warp and D
        cost    = np.linalg.norm(P_warp[:,None,:] - D[None,:,:], axis=2)
        row, col = linear_sum_assignment(cost)
        new_id_to_pos = {
            r: tuple(D[c]) for r,c in zip(row,col) if cost[r,c] < match_threshold
        }

        # 3.4) record matched & missing
        matched = sorted(new_id_to_pos.keys())
        missing = [i for i in range(N) if i not in matched]
        matched_counts[t] = len(matched)
        missing_counts[t] = len(missing)

        # 3.5) update positions
        for i in matched:
            positions[t,i] = new_id_to_pos[i]
        for i in missing:
            positions[t,i] = P_warp[i]
    
    elapsed = time.time() - start
    frame_times.append(elapsed)

    # 3.6) write per-frame CSV
    csv_path = os.path.join(out_results_dir, f"{frame_name}.csv")
    with open(csv_path, 'w', newline='') as f:
        from csv import writer
        w = writer(f)
        w.writerow(['id','x','y'])
        for i in range(N):
            x,y = positions[t,i]
            w.writerow([i, f"{x:.3f}", f"{y:.3f}"])

    # 3.7) create visualization frame
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    vis = cv2.resize(vis, (w0*scale, h0*scale), interpolation=cv2.INTER_NEAREST)
    for x,y in det_list:
        cv2.circle(vis,(int(x*scale),int(y*scale)),3,(0,0,255),-1)
    for i in matched:
        x,y = positions[t,i]
        cv2.circle(vis,(int(x*scale),int(y*scale)),6,(0,255,0),2)
    for i in missing:
        x,y = positions[t,i]
        cv2.circle(vis,(int(x*scale),int(y*scale)),6,(255,255,0),2)

    video_writer.write(vis)
    cv2.imshow("Tracking", vis)
    key = cv2.waitKey(1 if wait_for_key else 1) & 0xFF
    if key == ord('q'):
        break

# === 4) CLEANUP ===
video_writer.release()
cv2.destroyAllWindows()

times = np.array(frame_times)
print(f"Average processing time per frame: {times.mean():.3f} s")
print(f"  (std: {times.std():.3f} s, min: {times.min():.3f} s, max: {times.max():.3f} s)")

print(f"Per-frame CSVs written to '{out_results_dir}/'")
print(f"Annotated video saved as '{out_video}'")