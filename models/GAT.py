import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from sklearn.metrics import r2_score
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler
import os

SPATIAL_CONTEXT = 12
TASK = "DOMESTIC_BRA"
DATA_PATH = "/data/brazil" if "BRA" in TASK else "/kaggle/input/argentina"

MODEL_PATH = "models_saved/best_gat.pth"
SPATIAL_REFINER_DROPOUT = 0.3
REGRESSOR_DROPOUT = 0.3
ENCODER_FEATURES = [32, 64]
ENCODER_OUTPUT = 256
GAT_HEADS = 4
GAT_OUTPUT_FEATURES_PER_HEAD = 64
EPOCH = 201


class CustomGAT(nn.Module):
    def __init__(
        self,
        task="DOMESTIC",
        use_ndvi=True,
        hidden_dim=256,
        num_years=5,
        spatial_refiner_dropout=0.3,
        regressor_dropout=0.3,
        encoder_features=[32, 64],
        encoder_output=256,
        gat_heads=4,
        gat_output_features_per_head=64,
    ):
        super(CustomGAT, self).__init__()

        self.task = task
        self.hist_encoder = nn.Sequential(
            nn.Conv2d(9, encoder_features[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                encoder_features[0], encoder_features[1], kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(encoder_features[1] * 16, encoder_output),
            nn.ReLU(),
        )
        self.input_dim = encoder_output + (32 if use_ndvi else 0) + num_years
        self.feat_norm = nn.LayerNorm(self.input_dim)

        self.gat_layer = GATv2Conv(
            self.input_dim,
            gat_output_features_per_head,
            heads=gat_heads,
            concat=True,
            edge_dim=1,
        )
        self.spatial_refiner = nn.Sequential(
            nn.Linear(gat_output_features_per_head * gat_heads, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(spatial_refiner_dropout),
        )

        self.spatial_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()
        )

        self.graph_alpha = nn.Parameter(torch.tensor([0.2]))
        self.project_local = nn.Linear(self.input_dim, hidden_dim)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(regressor_dropout),
            nn.Linear(256, 1),
        )

    def forward(self, data, warmup=False):
        h_cnn = self.hist_encoder(data.x_hist)
        h_local = self.feat_norm(torch.cat([h_cnn, data.x_ndvi, data.x_year], dim=-1))

        h_regional = F.elu(
            self.gat_layer(h_local, data.edge_index, edge_attr=data.edge_attr)
        )
        h_regional = self.spatial_refiner(h_regional)

        h_local_proj = self.project_local(h_local)

        gate_weight = self.spatial_gate(h_regional)
        h_regional_gated = gate_weight * h_regional

        h_delta = h_local_proj - h_regional_gated
        if "TRANSFER" in self.task:
            alpha = self.graph_alpha if not warmup else 0.05
        else:
            alpha = self.graph_alpha
        h_combined = h_local_proj + (alpha * h_delta)

        return self.regressor(h_combined)


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


if __name__ == "__main__":
    config = load_config("../config/params.yml")

    TASK = config["TASK"]
    DATA_PATH = config["DATA_PATH"]
    MODEL_PATH = config["MODEL_PATH"]
    SPATIAL_CONTEXT = config["SPATIAL_CONTEXT"]
    ENCODER_FEATURES = config["ENCODER_FEATURES"]
    ENCODER_OUTPUT = config["ENCODER_OUTPUT"]
    GAT_HEADS = config["GAT_HEADS"]
    GAT_FEATURES = config["GAT_OUTPUT_FEATURES_PER_HEAD"]
    HIDDEN_DIM = config["HIDDEN_DIM"]
    S_DROPOUT = config["SPATIAL_REFINER_DROPOUT"]
    R_DROPOUT = config["REGRESSOR_DROPOUT"]
    EPOCHS = config["EPOCHS"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b_h, b_n, b_y, b_yr, b_l = load_data(DATA_PATH)
    target_years = ["2012", "2013", "2014", "2015", "2016"]
    yr_map = {y: i for i, y in enumerate(target_years)}

    loyo_results = {}

    for test_year in target_years:
        print(f"\n{'='*10} TEST YEAR: {test_year} {'='*10}")
        train_mask = (b_yr != test_year) & np.isin(b_yr, target_years)
        test_mask = b_yr == test_year

        h_scaler = StandardScaler().fit(b_h[train_mask].reshape(-1, 9))
        n_scaler = StandardScaler().fit(b_n[train_mask])
        y_scaler = StandardScaler().fit(
            np.log1p(np.maximum(b_y[train_mask], 0)).reshape(-1, 1)
        )

        def prep_fold_data(m):
            x_h = h_scaler.transform(b_h[m].reshape(-1, 9)).reshape(b_h[m].shape)
            x_n = n_scaler.transform(b_n[m])
            y_norm = y_scaler.transform(np.log1p(np.maximum(b_y[m], 0)).reshape(-1, 1))
            yr_oh = F.one_hot(
                torch.tensor([yr_map[v] for v in b_yr[m]]), num_classes=5
            ).float()
            edges, attr = build_spatial_edges(b_l[m], k=SPATIAL_CONTEXT)
            return Data(
                x_hist=torch.tensor(x_h).permute(0, 3, 1, 2).float(),
                x_ndvi=torch.tensor(x_n).float(),
                x_year=yr_oh,
                edge_index=edges,
                edge_attr=attr,
                y=torch.tensor(y_norm).float(),
            ).to(device)

        train_data = prep_fold_data(train_mask)
        test_data = prep_fold_data(test_mask)

        model = CustomGAT(
            task=TASK,
            hidden_dim=HIDDEN_DIM,
            spatial_refiner_dropout=S_DROPOUT,
            regressor_dropout=R_DROPOUT,
            encoder_features=ENCODER_FEATURES,
            encoder_output=ENCODER_OUTPUT,
            gat_heads=GAT_HEADS,
            gat_output_features_per_head=GAT_FEATURES,
        ).to(device)
        if os.path.exists(MODEL_PATH):
            sd = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(
                {k: v for k, v in sd.items() if "hist_encoder" in k}, strict=False
            )

        for param in model.hist_encoder.parameters():
            param.requires_grad = False
        for param in model.feat_norm.parameters():
            param.requires_grad = False

        optimizer = optim.Adam(
            [
                {"params": model.hist_encoder.parameters(), "lr": 0},
                {"params": model.gat_layer.parameters(), "lr": 5e-4},
                {"params": model.spatial_refiner.parameters(), "lr": 5e-4},
                {"params": model.spatial_gate.parameters(), "lr": 1e-3},
                {"params": model.regressor.parameters(), "lr": 2e-4},
                {"params": [model.graph_alpha], "lr": 1e-2},
            ]
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        best_r2_fold = -np.inf

        for epoch in range(1, EPOCH):
            model.train()

            if epoch == 40:
                for param in model.hist_encoder.parameters():
                    param.requires_grad = True
                for param in model.feat_norm.parameters():
                    param.requires_grad = True
                optimizer.param_groups[0]["lr"] = 5e-7

            if epoch == 80:
                optimizer.param_groups[0]["lr"] = 5e-6
            if epoch == 120:
                optimizer.param_groups[0]["lr"] = 1e-5

            optimizer.zero_grad()
            out = model(train_data, warmup=(epoch < 30))
            loss = F.mse_loss(out, train_data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if epoch % 20 == 0:
                model.eval()
                with torch.no_grad():
                    pred = model(test_data)
                    curr_r2 = robust_r2(test_data.y.cpu().numpy(), pred.cpu().numpy())

                    if curr_r2 > best_r2_fold:
                        best_r2_fold = curr_r2

                        if "DOMESTIC" in TASK:
                            
                            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                            torch.save(model.state_dict(), MODEL_PATH)
                            print(f"Saved New Best Model (R2: {curr_r2:.4f})")

                    scheduler.step(curr_r2)
                
                print(
                    f"Epoch {epoch:03d} | R2: {curr_r2:.4f} | Alpha: {model.graph_alpha.item():.4f}"
                )

        loyo_results[test_year] = best_r2_fold

    print("\n" + "=" * 30)
    print("--- FINAL RESULTS ---")
    for yr, r2 in loyo_results.items():
        print(f"Year {yr}: R2 = {r2:.4f}")
    print(f"Mean R2: {np.mean(list(loyo_results.values())):.4f}")
