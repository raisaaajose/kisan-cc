import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yaml
import pandas as pd
import os

from src.model.stgnn import STGNN
from src.utils import normalize_laplacian

config = yaml.safe_load(open("config/params.yaml"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(" Loading data..")
train_data = np.load(config["data"]["train"])

X_train = pd.DataFrame(train_data)[:][:-1]
y_train = pd.DataFrame(train_data)[:][-1]

tensor_x = torch.Tensor(X_train)
tensor_y = torch.Tensor(y_train)

train_dataset = TensorDataset(tensor_x, tensor_y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("Preparing graph..")
A_hat = normalize_laplacian(
    os.path.join(config["config"]["data"]["processed_graph"], "/adj_matrix.npz")
).to(device)

num_nodes = X_train.shape[1]
in_channels = X_train.shape[2]

model = STGNN(
    num_nodes=num_nodes,
    in_channels=in_channels,
    tcn_hidden=[32, 64],
    gcn_hidden=64,
    dropout=0.3,
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.paramters(), lr=config["model"]["learn"])

print("Start training..")
epochs = config["model"]["epoch"]

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        predictions = model(batch_x, A_hat)

        loss = criterion(predictions, batch_y)

        loss.bakcward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(
        f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | RSME {np.sqrt(avg_loss):.4f}"
    )

torch.save(model.stat_dict(), os.path.join(config["models_saved"], "stgnn_model.pth"))
print("Training complete..")
