import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import pandas as pd
import os

from models.stgnn import STGNN
from utils import normalize_laplacian

config = yaml.safe_load(open("config/params.yaml"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Preparing graph..")
adj_path = os.path.join(config["data"]["processed_graph"], "adj_matrix.npz")
train_mask = torch.tensor(
    np.load(os.path.join(config["data"]["processed_graph"], "train_mask.npy"))
).bool().to(device)
test_mask = torch.tensor(
    np.load(os.path.join(config["data"]["processed_graph"], "test_mask.npy"))
).bool().to(device)
tcn_hidden=config["model"]["tcn"]["hidden"]
gcn_hidden=config["model"]["gcn"]["hidden"]
graph_field_ids = np.load(
    os.path.join(config["data"]["processed_graph"], "graph_field_id.npy"),
    allow_pickle=True,
)
num_nodes = len(graph_field_ids)

print(f"Graph Total Nodes: {num_nodes}")

A_hat = normalize_laplacian(adj_path).to(device)

df_train = pd.read_csv(config["data"]["train"])
yield_map = dict(zip(df_train["Field_ID"], df_train["Yield"]))

print("Loading all image data...")
train_img_dir = config["data"]["train_img"]
test_img_dir = config["data"]["test_img"]

in_channels = 30
num_time = 12

X_all = torch.zeros(1, in_channels, num_nodes, num_time)
y_all = torch.zeros(1, num_nodes)

print(f"Loading {num_nodes} fields into memory...")
missing_files=0
for i, field_id in enumerate(graph_field_ids):
    try:
        img_path = os.path.join(train_img_dir, f"{field_id}.npy")
        if not os.path.exists(img_path):
             img_path = os.path.join(test_img_dir, f"{field_id}.npy")
        img = np.load(img_path)

        feat = np.mean(img, axis=(1, 2)).reshape(12, 30).T

        X_all[0, :, i, :] = torch.FloatTensor(feat)

    except FileNotFoundError:
        missing_files+=1
    if field_id in yield_map:
        y_all[0, i] = yield_map[field_id]
    else:
        y_all[0, i] = 0.0

if missing_files > 0:
    print(f"Warning: {missing_files} images were missing and set to zero.")

X_all = X_all.to(device)
y_all = y_all.to(device)

print("Initializing Model...")
model = STGNN(
    num_nodes=num_nodes,
    in_channels=in_channels,
    tcn_hidden=tcn_hidden,
    gcn_hidden=gcn_hidden,
    dropout=0.3,
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["model"]["learn"])

print("Start training..")
epochs = config["model"]["epoch"]

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_all, A_hat)
    preds_flat = predictions.view(-1)
    y_flat = y_all.view(-1)
    
    loss = criterion(preds_flat[train_mask], y_flat[train_mask])

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        rmse = torch.sqrt(loss).item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | RMSE: {rmse:.4f}")

torch.save(model.state_dict(), os.path.join(config["models_saved"], "stgnn_model.pth"))
print("Training complete..")
