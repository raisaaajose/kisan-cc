import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv


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


