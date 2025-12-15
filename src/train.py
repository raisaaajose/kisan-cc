import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import pandas as pd
import os

from src.models.stgnn import STGNN
from src.utils import normalize_laplacian

config = yaml.safe_load(open("config/params.yaml"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Preparing graph..")
adj_path = os.path.join(config["config"]["data"]["processed_graph"], "adj_matrix.npz")
A_hat = normalize_laplacian(adj_path).to(device)

print("Loading all image data...")
csv_path = os.path.join(config["config"]["train"], "Train.csv")
img_dir = "image_arrays_train"

df = pd.read_csv(csv_path)
df = df[df['Quality'] != 1].reset_index(drop=True)

df = df.sort_values(by="Field_ID")

num_nodes = len(df)
in_channels = 30
num_time = 12 

X_all = torch.zeros(1, in_channels, num_nodes, num_time)
y_all = torch.zeros(1, num_nodes)

print(f"Loading {num_nodes} fields into memory...")

for i, row in df.iterrows():
    field_id = row['Field_ID']
    try:
        img_path = os.path.join(img_dir, f"{field_id}.npy")
        img = np.load(img_path)
        
        feat = np.mean(img, axis=(1, 2)).reshape(12, 30).T
        
        X_all[0, :, i, :] = torch.FloatTensor(feat)
        y_all[0, i] = row['Yield']
        
    except FileNotFoundError:
        print(f"Warning: File for {field_id} not found.")

X_all = X_all.to(device)
y_all = y_all.to(device)

print("Initializing Model...")
model = STGNN(
    num_nodes=num_nodes,
    in_channels=in_channels,
    tcn_hidden=[64, 128],
    gcn_hidden=128,
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

    loss = criterion(predictions.view(-1), y_all.view(-1))

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        rmse = torch.sqrt(loss).item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | RMSE: {rmse:.4f}")

torch.save(model.state_dict(), os.path.join(config["models_saved"], "stgnn_model.pth"))
print("Training complete..")