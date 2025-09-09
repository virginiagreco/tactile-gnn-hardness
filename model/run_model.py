"""
Train/validate/test a GNN (GINE/GCN/GAT) to regress Shore A from .pt graph files.
"""
import argparse
import os
import glob
import json 
import time
import random
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from visuals import plot_pred_vs_true, plot_time_per_epoch, plot_efficiency_curve, plot_learning_curves

from GINEConv import GINENet, GCNNet, GATNet  

# ==========================
# Functions
# ==========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def metrics(y_true_t, y_pred_t):
    y_true = y_true_t.detach().cpu().numpy()
    y_pred = y_pred_t.detach().cpu().numpy()
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - y_true.mean())**2) + 1e-12)
    r2   = float(1.0 - ss_res/ss_tot)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# ==========================
# Dataset
# ==========================
class GraphFolderDataset(Dataset):
    def __init__(self, folder: str, file_glob: str = "graph_*.pt",
                 use_pos: bool = False, use_edge_attr: bool = False, scalers: dict = None):
        super().__init__(root=folder)
        self.folder = folder
        self.files = sorted(glob.glob(os.path.join(folder, file_glob)))
        if not self.files:
            raise FileNotFoundError(f"No .pt graph files found under {folder}")
        self.use_pos = use_pos
        self.use_edge_attr = use_edge_attr
        self.scalers = scalers or {}

    def len(self): return len(self.files)

    def get(self, idx: int) -> Data:
        d = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        x = d["x"]
        if self.use_pos and "pos0" in d:
            x = torch.cat([x, d["pos0"]], dim=1)

        # standardize node features
        if "x_mean" in self.scalers and "x_std" in self.scalers:
            xm = torch.tensor(self.scalers["x_mean"], dtype=x.dtype)
            xs = torch.tensor(self.scalers["x_std"], dtype=x.dtype).clamp_min(1e-8)
            x = (x - xm) / xs

        edge_index = d["edge_index"]
        edge_attr = None
        if self.use_edge_attr:
            ea_list = []
            for key in ["edge_rest_len", "edge_peak_len", "edge_strain", "edge_strain_abs"]:
                if key in d: 
                    ea_list.append(d[key].view(-1, 1))
            if ea_list:
                edge_attr = torch.cat(ea_list, dim=1)
                if "ea_mean" in self.scalers and "ea_std" in self.scalers:
                    eam = torch.tensor(self.scalers["ea_mean"], dtype=edge_attr.dtype)
                    eas = torch.tensor(self.scalers["ea_std"], dtype=edge_attr.dtype).clamp_min(1e-8)
                    edge_attr = (edge_attr - eam) / eas

        y = d["y"].view(-1)
        cube_id = int(d.get("cube_id", torch.tensor([-1]))[0])

        data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
        data.cube_id = cube_id
        data.path = self.files[idx]
        return data

def compute_scalers(files: List[str], use_pos: bool, use_edge_attr: bool) -> dict:
    xs, eas = [], []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        x = d["x"]
        if use_pos and "pos0" in d: 
            x = torch.cat([x, d["pos0"]], dim=1)
        xs.append(x)
        if use_edge_attr:
            ea = []
            for key in ["edge_rest_len", "edge_peak_len", "edge_strain", "edge_strain_abs"]:
                if key in d: 
                    ea.append(d[key].view(-1, 1))
            if ea: 
                eas.append(torch.cat(ea, dim=1))
    x_all = torch.cat(xs, dim=0)
    scalers = {"x_mean": x_all.mean(dim=0).numpy(), "x_std": x_all.std(dim=0).numpy()}
    if eas:
        ea_all = torch.cat(eas, dim=0)
        scalers["ea_mean"] = ea_all.mean(dim=0).numpy()
        scalers["ea_std"]  = ea_all.std(dim=0).numpy()
    return scalers

def stratified_split_by_cube(files: List[str], val_frac=0.2, test_frac=0.2, seed=42) -> Tuple[List[str], List[str], List[str]]:
    rng = np.random.RandomState(seed)
    meta = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        cid = int(d.get("cube_id", torch.tensor([-1]))[0])
        meta.append((f, cid))
    files_by_cube = {}
    for f, cid in meta:
        files_by_cube.setdefault(cid, []).append(f)

    train, val, test = [], [], []
    for cid, lst in files_by_cube.items():
        lst = sorted(lst)
        rng.shuffle(lst)
        n = len(lst)
        n_test = max(1, int(round(test_frac * n)))
        n_val  = max(1, int(round(val_frac  * n)))
        n_train = max(1, n - n_val - n_test)
        train += lst[:n_train]
        val   += lst[n_train:n_train+n_val]
        test  += lst[n_train+n_val:]
    return train, val, test

# ==========================
# Train/Eval
# ==========================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.batch, getattr(batch, "edge_attr", None))
        loss = nn.functional.mse_loss(pred, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.num_graphs
    return total / max(1, len(loader.dataset))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.batch, getattr(batch, "edge_attr", None))
        ys.append(batch.y.view(-1).cpu())
        ps.append(pred.cpu())
    y = torch.cat(ys)
    p = torch.cat(ps)
    return metrics(y, p), y.numpy(), p.numpy()

# ==========================
# Main
# ==========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="graph_outputs/graphs")
    ap.add_argument("--outdir", type=str, default="runs/gnn_raw")
    ap.add_argument("--model", type=str, choices=["gine", "gcn", "gat"], default="gine")
    ap.add_argument("--use-edge-attr", action="store_true")
    ap.add_argument("--use-pos", action="store_true")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--standardize-on-train", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data_dir, "graph_*.pt")))
    if not files: 
        raise SystemExit(f"No .pt graphs found in {args.data_dir}")

    train_files, val_files, test_files = stratified_split_by_cube(files, args.val_frac, args.test_frac, args.seed)

    # scalers
    source_for_scalers = train_files if args.standardize_on_train else files
    scalers = compute_scalers(source_for_scalers, args.use_pos, args.use_edge_attr)

    ds_train = GraphFolderDataset(args.data_dir, use_pos=args.use_pos, use_edge_attr=args.use_edge_attr, scalers=scalers)
    ds_train.files = train_files
    ds_val   = GraphFolderDataset(args.data_dir, use_pos=args.use_pos, use_edge_attr=args.use_edge_attr, scalers=scalers)
    ds_val.files   = val_files
    ds_test  = GraphFolderDataset(args.data_dir, use_pos=args.use_pos, use_edge_attr=args.use_edge_attr, scalers=scalers)
    ds_test.files  = test_files

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False)

    in_dim = ds_train.get(0).x.size(-1)
    edge_dim = ds_train.get(0).edge_attr.size(-1) if (args.use_edge_attr and ds_train.get(0).edge_attr is not None) else 0

    if args.model == "gine":
        model = GINENet(in_dim, args.hidden, args.layers, use_edge_attr=args.use_edge_attr, edge_dim=edge_dim, dropout=args.dropout)
    elif args.model == "gcn":
        model = GCNNet(in_dim, args.hidden, args.layers, dropout=args.dropout)
    else:
        model = GATNet(in_dim, args.hidden, args.layers, heads=4, dropout=args.dropout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val = float("inf")
    best_state = None
    # logs
    epoch_times, cum_times = [], []
    train_mse_hist, val_mae_hist, val_rmse_hist, val_r2_hist = [], [], [], []
    t_cum = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["RMSE"])
        dt = time.perf_counter() - t0

        t_cum += dt
        epoch_times.append(dt)
        cum_times.append(t_cum)
        train_mse_hist.append(float(tr_loss))
        val_mae_hist.append(float(val_metrics["MAE"]))
        val_rmse_hist.append(float(val_metrics["RMSE"]))
        val_r2_hist.append(float(val_metrics["R2"]))

        print(f"Epoch {epoch:03d} | {dt:.2f}s | Train MSE: {tr_loss:.4f} | "
              f"Val MAE: {val_metrics['MAE']:.3f} RMSE: {val_metrics['RMSE']:.3f} R2: {val_metrics['R2']:.3f}")

        if val_metrics["RMSE"] < best_val:
            best_val = val_metrics["RMSE"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "scalers": scalers, "args": vars(args)},
                       os.path.join(args.outdir, "best_model.pt"))

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, y_true, y_pred = evaluate(model, test_loader, device)
    print("Test:", test_metrics)

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"val_best_RMSE": best_val, "test": test_metrics,
                   "n_train": len(ds_train), "n_val": len(ds_val), "n_test": len(ds_test),
                   "model": args.model, "use_edge_attr": args.use_edge_attr, "use_pos": args.use_pos}, f, indent=2)

    # Save predictions + training log
    dfp = pd.DataFrame({"path": [ds_test.files[i] for i in range(len(ds_test))],
                        "y_true": y_true, "y_pred": y_pred, "abs_err": np.abs(y_true - y_pred)})
    dfp.to_csv(os.path.join(args.outdir, "predictions.csv"), index=False)

    log_df = pd.DataFrame({
        "epoch": np.arange(1, args.epochs + 1, dtype=int),
        "sec_per_epoch": epoch_times,
        "cum_seconds": cum_times,
        "train_mse": train_mse_hist,
        "val_mae": val_mae_hist,
        "val_rmse": val_rmse_hist,
        "val_r2": val_r2_hist,
    })
    log_df.to_csv(os.path.join(args.outdir, "training_log.csv"), index=False)

    # Final plots
    plot_pred_vs_true(dfp, test_metrics, os.path.join(args.outdir, "pred_vs_true.png"))
    plot_time_per_epoch(log_df, os.path.join(args.outdir, "time_per_epoch.png"))
    plot_efficiency_curve(log_df, os.path.join(args.outdir, "efficiency_rmse_vs_time.png"))
    plot_learning_curves(log_df, os.path.join(args.outdir, "learning_curves.png"))

if __name__ == "__main__":
    main()
