import torch
import numpy as np
from scipy import sparse


def normalize_laplacian(adj_matrix_path):

    adj = sparse.load_npz(adj_matrix_path)

    row_sum = np.array(adj.sum(1))

    r_inv = np.power(row_sum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sparse.diags(r_inv)

    mx = r_mat_inv.dot(adj).dot(r_mat_inv)

    mx = mx.tocoo()
    indices = torch.LongTensor([mx.row, mx.col])
    values = torch.FloatTensor(mx.data)
    shape = torch.size(mx.shape)

    laplacian = torch.sparse.FloatTensor(indices, values, shape).to_dense()

    return laplacian
