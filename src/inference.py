import torch
import numpy as np
import pandas as pd
import yaml
import os

from models.stgnn import STGNN
from utils import normalize_laplacian


def run_inference():
    config = yaml.safe_load(open("config/params.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- Starting Inference ---")

    processed_dir = config["data"]["processed_graph"]

    adj_path = os.path.join(processed_dir, "adj_matrix.npz")
    A_hat = normalize_laplacian(adj_path).to(device)

    test_mask = (
        torch.from_numpy(np.load(os.path.join(processed_dir, "test_mask.npy")))
        .bool()
        .to(device)
    )

    graph_field_ids = np.load(
        os.path.join(processed_dir, "graph_field_id.npy"),
        allow_pickle=True,
    )
    num_nodes = len(graph_field_ids)
    tcn_hidden = config["model"]["tcn"]["hidden"]
    gcn_hidden = config["model"]["gcn"]["hidden"]

    print(f"Reconstructing graph with {num_nodes} nodes...")

    train_img_dir = config["data"]["train_img"]
    test_img_dir = config["data"].get("test_img", train_img_dir)

    in_channels = 30
    num_time = 12
    X_all = torch.zeros(1, in_channels, num_nodes, num_time)

    missing_files = 0
    for i, field_id in enumerate(graph_field_ids):
        try:
            img_path = os.path.join(train_img_dir, f"{field_id}.npy")
            if not os.path.exists(img_path):
                img_path = os.path.join(test_img_dir, f"{field_id}.npy")

            img = np.load(img_path)
            feat = np.mean(img, axis=(1, 2)).reshape(12, 30).T
            X_all[0, :, i, :] = torch.FloatTensor(feat)

        except FileNotFoundError:
            missing_files += 1

    if missing_files > 0:
        print(f"Warning: {missing_files} images missing during inference.")

    X_all = X_all.to(device)

    print("Loading model weights...")
    model = STGNN(
        num_nodes=num_nodes,
        in_channels=in_channels,
        tcn_hidden=tcn_hidden,
        gcn_hidden=gcn_hidden,
        dropout=0.3,
    ).to(device)

    model_path = os.path.join(config["models_saved"], "stgnn_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Running forward pass...")
    with torch.no_grad():
        predictions = model(X_all, A_hat)

        preds_flat = predictions.view(-1)

        test_preds = preds_flat[test_mask].cpu().numpy()

        test_ids = graph_field_ids[test_mask.cpu().numpy()]

    print("Saving submission...")
    submission_df = pd.DataFrame({"Field_ID": test_ids, "Yield": test_preds})

    # sample = pd.read_csv(config["data"]["sample_submission"])
    # submission_df = sample[['Field_ID']].merge(submission_df, on='Field_ID', how='left')

    save_path = "data/submission.csv"
    submission_df.to_csv(save_path, index=False)
    print(f"Done! Saved {len(submission_df)} predictions to {save_path}")


if __name__ == "__main__":
    run_inference()
