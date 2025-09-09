#!/usr/bin/env python3
"""
Build a graph dataset from marker displacements using **kNN** for graph construction
(instead of neighbor_map).

Adds:
- Deformed (peak) graph plot in addition to rest-anchored quiver.
- PyTorch Geometric-friendly `.pt` files per video (dict of tensors) for easy loading.
- Same metrics & outputs as your previous script so you can compare directly.

Graph construction:
- Use kNN on the **rest positions** (first frame) to define edges.
- Undirected edges; duplicates removed.
- Optionally require **mutual** kNN (safer, sparser).

Metrics kept (per video):
- mean_disp_mag
- median_edge_strain_abs
- iqr_edge_strain_abs

Outputs (under --outdir, default: graph_outputs_knn/):
- graphs_results_by_video.csv, graphs_results_by_cube.csv
- graphs/graph_cube{c}_{video}.npz  (numpy package)
- graphs/graph_cube{c}_{video}.pt   (PyG-friendly tensors)
- graphs/nodes_*.csv, graphs/edges_*.csv
- graphs/viz_* / quiver_peak.png  (rest-anchored quiver)
- graphs/viz_* / graph_peak.png   (nodes+edges at peak positions)
"""

import argparse
import glob
import os
import re
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", os.path.basename(s))]


def read_positions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    if 'id' not in df.columns and 'ids' in df.columns:
        df['id'] = df['ids']
    missing = {'id', 'x', 'y'} - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} missing columns: {missing}")
    df['id'] = df['id'].astype(str)
    return df[['id', 'x', 'y']].copy()


def pairwise_distances(arr: np.ndarray) -> np.ndarray:
    diff = arr[:, None, :] - arr[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    return d


def expansion_signed(df0: pd.DataFrame, df1: pd.DataFrame, min_pair_percentile: float = 5.0) -> Optional[float]:
    merged = pd.merge(df0, df1, on='id', suffixes=('_0', '_1'))
    if len(merged) < 3:
        return None
    p0 = merged[['x_0', 'y_0']].to_numpy()
    p1 = merged[['x_1', 'y_1']].to_numpy()

    d0 = pairwise_distances(p0)
    d1 = pairwise_distances(p1)
    iu = np.triu_indices_from(d0, k=1)
    d0u = d0[iu]
    d1u = d1[iu]

    if d0u.size == 0:
        return None
    thresh = np.percentile(d0u, min_pair_percentile)
    eps = 1e-12
    mask = d0u > max(thresh, eps)
    if not np.any(mask):
        return None
    ratios = d1u[mask] / (d0u[mask] + eps)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return None
    return float(np.median(ratios) - 1.0)


def scan_video_frames(video_dir: str) -> List[str]:
    csvs = glob.glob(os.path.join(video_dir, "*.csv"))
    # keep only id,x,y marker CSVs; exclude neighbor maps if present
    base_excl = {"neighbor_map.csv", "neighbour_map.csv"}
    csvs = [c for c in csvs if os.path.basename(c).lower() not in base_excl]
    csvs = sorted(csvs, key=natural_key)
    return csvs


def get_cube_id_from_path(path: str) -> Optional[int]:
    # Be robust to both naming patterns used in your project
    for pat in [r"tracking_results_hex_(\d+)", r"tracking_results_hex_(\d+)"]:
        m = re.search(pat, path)
        if m:
            return int(m.group(1))
    return None


def load_labels(labels_path: str) -> pd.DataFrame:
    lab = pd.read_csv(labels_path)
    cols = [c.strip().lower() for c in lab.columns]
    lab.columns = cols
    if 'cube' not in lab.columns or 'shorea' not in lab.columns:
        raise ValueError("hardness_labels.csv must have columns: cube, shoreA")
    lab['cube_id'] = lab['cube'].astype(str).str.extract(r'(\d+)').astype(int)
    lab = lab[['cube_id', 'shorea']].drop_duplicates('cube_id')
    return lab


def compute_displacements_at_peak(ref: pd.DataFrame, peak: pd.DataFrame) -> pd.DataFrame:
    m = pd.merge(ref, peak, on='id', suffixes=('_0', '_1'))
    if m.empty:
        return m.assign(dx=np.nan, dy=np.nan, disp_mag=np.nan)
    dx = m['x_1'] - m['x_0']
    dy = m['y_1'] - m['y_0']
    disp_mag = np.sqrt(dx**2 + dy**2)
    out = m[['id', 'x_0', 'y_0', 'x_1', 'y_1']].copy()
    out['dx'] = dx
    out['dy'] = dy
    out['disp_mag'] = disp_mag
    return out


def edge_attrs(df: pd.DataFrame, edges: List[Tuple[int,int]]):
    p0 = df[['x_0', 'y_0']].to_numpy()
    p1 = df[['x_1', 'y_1']].to_numpy()
    E = len(edges)
    rest_len = np.zeros(E, dtype=float)
    peak_len = np.zeros(E, dtype=float)
    strain = np.zeros(E, dtype=float)
    eps = 1e-12
    for k, (i, j) in enumerate(edges):
        d0 = np.linalg.norm(p0[i] - p0[j])
        d1 = np.linalg.norm(p1[i] - p1[j])
        rest_len[k] = d0
        peak_len[k] = d1
        strain[k] = d1 / (d0 + eps) - 1.0 if d0 > eps else np.nan
    return rest_len, peak_len, strain, np.abs(strain)


def build_edges_knn(xy: np.ndarray, k: int = 6, mutual: bool = False) -> List[Tuple[int,int]]:
    """
    Build undirected edges from kNN on coordinates xy (N,2). Excludes self. Removes duplicates.
    If mutual=True, keep only pairs where i in kNN(j) and j in kNN(i).
    """
    N = xy.shape[0]
    if N <= 1 or k <= 0:
        return []
    # Full pairwise distances (brute force, fine for typical N~hundreds)
    D = pairwise_distances(xy)
    # argsort to get neighbor order per node
    order = np.argsort(D, axis=1)
    # skip self at position 0
    knn = order[:, 1:k+1]
    edges = set()
    if mutual:
        # Build reverse lookup sets for mutual check
        neigh_sets = [set(knn[i]) for i in range(N)]
        for i in range(N):
            for j in knn[i]:
                if i in neigh_sets[j]:
                    a, b = (i, j) if i < j else (j, i)
                    edges.add((a, b))
    else:
        for i in range(N):
            for j in knn[i]:
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                edges.add((a, b))
    return sorted(edges)


def plot_quiver_and_peak(video_outdir: str, df: pd.DataFrame, edges: List[Tuple[int,int]], title_prefix: str):
    os.makedirs(video_outdir, exist_ok=True)
    x0 = df['x_0'].to_numpy(); y0 = df['y_0'].to_numpy()
    x1 = df['x_1'].to_numpy(); y1 = df['y_1'].to_numpy()
    dx = df['dx'].to_numpy(); dy = df['dy'].to_numpy()

    # Rest-anchored quiver
    plt.figure()
    for i, j in edges:
        plt.plot([x0[i], x0[j]], [y0[i], y0[j]])
    plt.quiver(x0, y0, dx, dy, angles='xy', scale_units='xy')
    plt.gca().invert_yaxis()
    plt.title(f"{title_prefix} - rest quiver to peak")
    plt.xlabel("x (px)"); plt.ylabel("y (px)"); plt.axis('equal'); plt.tight_layout()
    qpath = os.path.join(video_outdir, "quiver_peak.png")
    plt.savefig(qpath, dpi=200); plt.close()

    # Deformed graph at peak
    plt.figure()
    for i, j in edges:
        plt.plot([x1[i], x1[j]], [y1[i], y1[j]])
    plt.scatter(x1, y1, s=10)
    plt.gca().invert_yaxis()
    plt.title(f"{title_prefix} - graph at peak")
    plt.xlabel("x (px)"); plt.ylabel("y (px)"); plt.axis('equal'); plt.tight_layout()
    gpath = os.path.join(video_outdir, "graph_peak.png")
    plt.savefig(gpath, dpi=200); plt.close()

    return qpath, gpath


def save_pyg_tensors(path_pt: str, node_ids, x, edge_index, shorea, cube_id, pos0, pos1, edge_attr_dict):
    try:
        import torch
        data = {
            'node_ids': node_ids,
            'x': torch.tensor(x, dtype=torch.float32),
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'y': torch.tensor([shorea], dtype=torch.float32),
            'cube_id': torch.tensor([cube_id], dtype=torch.int64),
            'pos0': torch.tensor(pos0, dtype=torch.float32),
            'pos1': torch.tensor(pos1, dtype=torch.float32),
            'edge_rest_len': torch.tensor(edge_attr_dict['rest_len'], dtype=torch.float32),
            'edge_peak_len': torch.tensor(edge_attr_dict['peak_len'], dtype=torch.float32),
            'edge_strain': torch.tensor(edge_attr_dict['strain'], dtype=torch.float32),
            'edge_strain_abs': torch.tensor(edge_attr_dict['strain_abs'], dtype=torch.float32),
        }
        torch.save(data, path_pt)
    except Exception as e:
        # Fallback: save as numpy if torch isn't available
        np.savez_compressed(path_pt.replace('.pt', '.npz'),
                            node_ids=np.array(node_ids),
                            x=x, edge_index=edge_index, y=np.array([shorea]), cube_id=np.array([cube_id]),
                            pos0=pos0, pos1=pos1,
                            edge_rest_len=edge_attr_dict['rest_len'],
                            edge_peak_len=edge_attr_dict['peak_len'],
                            edge_strain=edge_attr_dict['strain'],
                            edge_strain_abs=edge_attr_dict['strain_abs'])


def analyze(root: str, labels_path: str, outdir: str, min_pair_percentile: float, k: int, mutual: bool):
    # your real folders are tracking_results_hex_* / robot_video_*
    pattern = os.path.join(root, "tracking_results_hex_*", "robot_video_*")
    video_dirs = sorted(glob.glob(pattern), key=natural_key)
    labels = load_labels(labels_path)

    per_video_rows = []
    graphs_dir = os.path.join(outdir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    for vdir in video_dirs:
        csvs = scan_video_frames(vdir)
        if len(csvs) < 2:
            continue
        try:
            ref = read_positions(csvs[0])
        except Exception as e:
            print(f"[WARN] Skipping {vdir}: failed to read reference: {e}")
            continue

        # pick peak via maximal expansion magnitude
        peak_csv = None; peak_df = None; best_mag = -np.inf; signed_at_peak = None
        for c in csvs[1:]:
            try:
                cur = read_positions(c)
            except Exception:
                continue
            s = expansion_signed(ref, cur, min_pair_percentile=min_pair_percentile)
            if s is None: 
                continue
            m = abs(s)
            if m > best_mag:
                best_mag = m; signed_at_peak = s; peak_csv = c; peak_df = cur

        if peak_df is None:
            continue

        disp = compute_displacements_at_peak(ref, peak_df)
        if disp.empty:
            continue

        # kNN edges built on rest positions (x0,y0)
        node_ids = disp['id'].astype(str).tolist()
        xy0 = disp[['x_0', 'y_0']].to_numpy()
        edges = build_edges_knn(xy0, k=k, mutual=mutual)

        rest_len, peak_len, strain, strain_abs = edge_attrs(disp, edges)

        mean_disp = float(disp['disp_mag'].mean())
        med_edge_strain_abs = float(np.median(strain_abs)) if strain_abs.size else np.nan
        iqr_edge_strain_abs = float(np.percentile(strain_abs,75) - np.percentile(strain_abs,25)) if strain_abs.size else np.nan

        cube_id = get_cube_id_from_path(vdir)
        video_name = os.path.basename(vdir)
        shorea = float(labels.loc[labels['cube_id'] == cube_id, 'shorea'].iloc[0]) if (labels['cube_id'] == cube_id).any() else np.nan

        x = np.stack([disp['dx'].to_numpy(), disp['dy'].to_numpy(), disp['disp_mag'].to_numpy(),
                      disp['x_0'].to_numpy(), disp['y_0'].to_numpy()], axis=1)
        edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2,0), dtype=np.int64)
        pos0 = disp[['x_0','y_0']].to_numpy()
        pos1 = disp[['x_1','y_1']].to_numpy()

        # Save numpy package
        npz_path = os.path.join(graphs_dir, f"graph_cube{cube_id}_{video_name}.npz")
        np.savez_compressed(npz_path,
                            node_ids=np.array(node_ids),
                            x=x, edge_index=edge_index, cube_id=np.array([cube_id]), shorea=np.array([shorea]),
                            pos0=pos0, pos1=pos1,
                            edge_rest_len=rest_len, edge_peak_len=peak_len,
                            edge_strain=strain, edge_strain_abs=strain_abs)

        # Save PyG-friendly .pt (or .npz fallback if torch missing)
        pt_path = os.path.join(graphs_dir, f"graph_cube{cube_id}_{video_name}.pt")
        save_pyg_tensors(pt_path, np.array(node_ids), x, edge_index, shorea, cube_id, pos0, pos1,
                         {'rest_len':rest_len, 'peak_len':peak_len, 'strain':strain, 'strain_abs':strain_abs})

        # CSVs
        nodes_csv = os.path.join(graphs_dir, f"nodes_cube{cube_id}_{video_name}.csv")
        edges_csv = os.path.join(graphs_dir, f"edges_cube{cube_id}_{video_name}.csv")
        pd.DataFrame(disp[['id','dx','dy','disp_mag','x_0','y_0','x_1','y_1']]).to_csv(nodes_csv, index=False)
        pd.DataFrame([(node_ids[i], node_ids[j], rest_len[k], peak_len[k], strain[k], strain_abs[k])
                      for k,(i,j) in enumerate(edges)],
                     columns=['src','dst','rest_len','peak_len','strain','strain_abs']).to_csv(edges_csv, index=False)

        # Visuals
        viz_dir = os.path.join(graphs_dir, f"viz_cube{cube_id}_{video_name}")
        qpath, gpath = plot_quiver_and_peak(viz_dir, disp, edges,
                                            f"Cube {cube_id} ({shorea} ShoreA) - {video_name} peak")

        per_video_rows.append({
            'cube_id': cube_id, 'shorea': shorea, 'video': video_name,
            'n_nodes': len(node_ids), 'n_edges': len(edges),
            'expansion_signed': signed_at_peak, 'expansion_magnitude': abs(signed_at_peak) if signed_at_peak is not None else np.nan,
            'mean_disp_mag': mean_disp, 'median_edge_strain_abs': med_edge_strain_abs, 'iqr_edge_strain_abs': iqr_edge_strain_abs,
            'graph_npz': os.path.relpath(npz_path, outdir), 'graph_pt': os.path.relpath(pt_path, outdir),
            'nodes_csv': os.path.relpath(nodes_csv, outdir), 'edges_csv': os.path.relpath(edges_csv, outdir),
            'quiver_png': os.path.relpath(qpath, outdir), 'graph_peak_png': os.path.relpath(gpath, outdir),
        })

    if not per_video_rows:
        raise RuntimeError("No graphs built. Check data paths.")

    dfv = pd.DataFrame(per_video_rows).sort_values(['cube_id','video']).reset_index(drop=True)
    aggc = (dfv.groupby(['cube_id','shorea'])
              .agg(n_videos=('video','count'),
                   mean_disp_mag=('mean_disp_mag','mean'), std_disp_mag=('mean_disp_mag','std'),
                   mean_med_edge_strain_abs=('median_edge_strain_abs','mean'),
                   std_med_edge_strain_abs=('median_edge_strain_abs','std'))
              .reset_index().sort_values('cube_id'))

    os.makedirs(outdir, exist_ok=True)
    by_video_csv = os.path.join(outdir, "graphs_results_by_video.csv")
    by_cube_csv = os.path.join(outdir, "graphs_results_by_cube.csv")
    dfv.to_csv(by_video_csv, index=False); aggc.to_csv(by_cube_csv, index=False)
    return dfv, aggc, [by_video_csv, by_cube_csv]


def plot_metric(df_video: pd.DataFrame, df_cube: pd.DataFrame, outdir: str, value_col: str, base_name: str, y_label: str, title_suffix: str):
    os.makedirs(outdir, exist_ok=True)
    if value_col == 'mean_disp_mag':
        mean_col, std_col = 'mean_disp_mag', 'std_disp_mag'
    elif value_col == 'median_edge_strain_abs':
        mean_col, std_col = 'mean_med_edge_strain_abs', 'std_med_edge_strain_abs'
    else:
        raise ValueError("Unknown value_col for plotting.")

    labels = [f"Cube {row.cube_id}\n(ShoreA {row.shorea})" for _, row in df_cube.iterrows()]
    means = df_cube[mean_col].to_numpy()
    stds = df_cube[std_col].fillna(0.0).to_numpy()

    x = np.arange(len(labels))

    plt.figure()
    plt.bar(x, means, yerr=stds, capsize=6)
    plt.xticks(x, labels); plt.ylabel(y_label)
    plt.title(f"{title_suffix} by Cube (with Shore A)")
    plt.tight_layout()
    bar_path = os.path.join(outdir, f"cube_{base_name}_bars.png")
    plt.savefig(bar_path, dpi=200); plt.close()

    plt.figure()
    plt.scatter(df_video['shorea'], df_video[value_col])
    valid = df_video[['shorea', value_col]].dropna()
    if len(valid) >= 2:
        m, b = np.polyfit(valid['shorea'].to_numpy(), valid[value_col].to_numpy(), deg=1)
        xs = np.linspace(valid['shorea'].min(), valid['shorea'].max(), 100)
        ys = m * xs + b
        plt.plot(xs, ys)
        r = np.corrcoef(valid['shorea'], valid[value_col])[0, 1]
        plt.title(f"Per-video {title_suffix} vs. Shore A (r = {r:.3f})")
    else:
        plt.title(f"Per-video {title_suffix} vs. Shore A")
    plt.xlabel("Shore A hardness"); plt.ylabel(y_label); plt.tight_layout()
    scatter_path = os.path.join(outdir, f"{base_name}_vs_shorea.png")
    plt.savefig(scatter_path, dpi=200); plt.close()
    return bar_path, scatter_path


def main():
    parser = argparse.ArgumentParser(description="Build graph dataset with kNN; save PyG tensors; make visuals & metrics.")
    parser.add_argument("--root", type=str, default="csv_results_all", help="Root containing tracking_results_hex_* folders.")
    parser.add_argument("--labels", type=str, default="hardness_labels.csv", help="Path to hardness_labels.csv (cube, shoreA)." )
    parser.add_argument("--outdir", type=str, default="graph_outputs_knn8", help="Directory to write graphs and summaries.")
    parser.add_argument("--min-pair-percentile", type=float, default=5.0, help="Robustness threshold for expansion ratios.")
    parser.add_argument("--knn", type=int, default=8, help="Number of neighbors per node for kNN graph on rest positions.")
    parser.add_argument("--mutual", action="store_true", help="Keep only mutual kNN edges (safer, sparser)." )
    args = parser.parse_args()

    df_video, df_cube, csv_paths = analyze(args.root, args.labels, args.outdir, args.min_pair_percentile, args.knn, args.mutual)

    plots = []
    plots += list(plot_metric(df_video, df_cube, args.outdir, 'mean_disp_mag', 'mean_disp', 'Mean displacement magnitude (px)', 'Mean Displacement Magnitude'))
    plots += list(plot_metric(df_video, df_cube, args.outdir, 'median_edge_strain_abs', 'edge_strain', 'Median |edge strain|', 'Median Edge Strain |ε|'))

    print("\n=== Graph per-cube summary ===")
    print(df_cube.to_string(index=False))
    print("\nWrote:")
    for p in csv_paths:
        print(" -", p)
    for p in plots:
        print("Saved figure:", p)


if __name__ == "__main__":
    main()