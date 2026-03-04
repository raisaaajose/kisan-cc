import numpy as np
import torch
from sklearn.metrics import r2_score
from scipy.spatial import KDTree


def build_spatial_edges(locs, k=12):
    n = len(locs)
    if k <= 0 or locs.ndim == 1:
        return torch.stack([torch.arange(n), torch.arange(n)], dim=0), torch.ones(
            (n, 1)
        )
    tree = KDTree(locs)
    dist, col = tree.query(locs, k=k + 1)
    row = np.repeat(np.arange(n), k)
    dist_flat = dist[:, 1:].flatten()
    sigma = np.median(dist_flat) + 1e-6
    edge_weight = np.exp(-(dist_flat**2) / (2 * sigma**2))
    return torch.tensor([row, col[:, 1:].flatten()], dtype=torch.long), torch.tensor(
        edge_weight, dtype=torch.float
    ).view(-1, 1)


def robust_r2(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_pred)
    return r2_score(y_true[mask], y_pred[mask]) if np.sum(mask) > 10 else -1.0


def load_data(path):
    h = np.nan_to_num(
        np.concatenate(
            [
                np.load(f"{path}/train_hists.npz", allow_pickle=True)["data"],
                np.load(f"{path}/dev_hists.npz", allow_pickle=True)["data"],
            ],
            axis=0,
        )
    ).astype(np.float32)
    n = np.nan_to_num(
        np.concatenate(
            [
                np.load(f"{path}/train_ndvi.npz", allow_pickle=True)["data"],
                np.load(f"{path}/dev_ndvi.npz", allow_pickle=True)["data"],
            ],
            axis=0,
        )
    ).astype(np.float32)
    y = np.concatenate(
        [
            np.load(f"{path}/train_yields.npz", allow_pickle=True)["data"],
            np.load(f"{path}/dev_yields.npz", allow_pickle=True)["data"],
        ],
        axis=0,
    ).astype(np.float32)
    yr = np.concatenate(
        [
            np.load(f"{path}/train_years.npz", allow_pickle=True)["data"],
            np.load(f"{path}/dev_years.npz", allow_pickle=True)["data"],
        ],
        axis=0,
    )
    l = np.concatenate(
        [
            np.load(f"{path}/train_locs.npz", allow_pickle=True)["data"],
            np.load(f"{path}/dev_locs.npz", allow_pickle=True)["data"],
        ],
        axis=0,
    )
    cleaned_l = np.array(
        [[float(e[0]), float(e[1])] if e is not None else [0.0, 0.0] for e in l]
    )
    return h, n, y, yr, cleaned_l


import yaml
import os


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


