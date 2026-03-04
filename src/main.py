import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import os
from utils import load_config,load_data,build_spatial_edges,robust_r2
from gat import CustomGAT

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
    EPOCH = config["EPOCHS"]

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
