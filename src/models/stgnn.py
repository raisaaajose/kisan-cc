import torch
import torch.nn as nn
from src.models.tcn import CausalTCN
from src.models.gcn import StaticSpectralGCN


class STGNN(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_channels,
        tcn_hidden,
        gcn_hidden,
        out_channels=1,
        dropout=0.2,
    ):
        super(STGNN, self).__init__()

        self.tcn_lower = CausalTCN(
            num_inputs=in_channels,
            num_channels=tcn_hidden,
            kernel_size=3,
            dropout=dropout,
        )

        gcn_in_dim = tcn_hidden[-1]
        self.gcn = StaticSpectralGCN(
            in_features=gcn_in_dim,
            out_features=gcn_hidden,
            num_nodes=num_nodes,
            dropout=dropout,
        )

        self.tcn_upper = CausalTCN(
            num_inputs=gcn_hidden,
            num_channels=tcn_hidden,
            kernel_size=3,
            dropout=dropout,
        )

        final_dim = tcn_hidden[-1]
        self.regressor = nn.Linear(final_dim, out_channels)

    def forward(self, x, A_hat):

        B, C, N, T = x.shape

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B * N, C, T)

        x = self.tcn_lower(x)

        _, Hidden, _ = x.shape
        x = x.view(B, N, Hidden, T)
        x = x.permute(0, 2, 1, 3)

        x = self.gcn(x, A_hat)
        B, Hidden, N, T = x.shape

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B * N, Hidden, T)
        x = self.tcn_upper(x)
        x = x[:, :, -1]

        out = self.regressor(x)

        out = out.view(B, N)
        return out
