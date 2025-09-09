# viz.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_pred_vs_true(df_preds: pd.DataFrame, test_metrics: dict, outpath: str):
    plt.figure()
    lims = [min(df_preds["y_true"].min(), df_preds["y_pred"].min()),
            max(df_preds["y_true"].max(), df_preds["y_pred"].max())]
    plt.scatter(df_preds["y_true"], df_preds["y_pred"])
    plt.plot(lims, lims)
    plt.xlabel("True Shore A")
    plt.ylabel("Predicted Shore A")
    plt.title(f"Pred vs True (Test) | MAE={test_metrics['MAE']:.2f}, RMSE={test_metrics['RMSE']:.2f}, R2={test_metrics['R2']:.2f}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_time_per_epoch(log_df: pd.DataFrame, outpath: str):
    plt.figure()
    plt.plot(log_df["epoch"], log_df["sec_per_epoch"], marker="o", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Seconds per epoch")
    plt.title("Training speed per epoch")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_efficiency_curve(log_df: pd.DataFrame, outpath: str):
    plt.figure()
    plt.plot(log_df["cum_seconds"], log_df["val_rmse"], marker="o", linewidth=1)
    plt.gca().invert_yaxis()  # lower RMSE = better
    plt.xlabel("Cumulative training time (s)")
    plt.ylabel("Validation RMSE (↓ better)")
    plt.title("Training efficiency: RMSE vs wall-clock time")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_learning_curves(log_df: pd.DataFrame, outpath: str):
    plt.figure()
    plt.plot(log_df["epoch"], log_df["train_mse"], label="Train MSE")
    plt.plot(log_df["epoch"], log_df["val_rmse"], label="Val RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Error")
    plt.title("Learning curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
