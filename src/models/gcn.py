import torch
import torch.nn as nn


class StaticSpectralGCN(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, dropout):
        super(StaticSpectralGCN, self).__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, A_hat):
        B, C, N, T = x.shape
        x=x.permute(0,3,2,1)
        out=torch.matmul(A_hat,x)
        out=self.linear(out)
        out=self.act(out)
        out=self.dropout(out)

        out=out.permute(0,3,2,1)
        return out