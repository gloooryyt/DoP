import torch
from torch import nn
from positional_encodings.torch_encodings import PositionalEncoding1D
from scipy.signal import welch
import numpy as np
import scipy.sparse as sp

class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, embed_dim, adj_mx):
        super().__init__()
        self.embed_dim = embed_dim
        weekday_size = 7
        frequency_band_num = 4
        self.steps_per_day = 288
        minute_size = self.steps_per_day
        self.dow_pe = nn.Embedding(weekday_size, embed_dim)
        self.tod_pe = nn.Embedding(minute_size, embed_dim)
        self.psd_pe = nn.Embedding(frequency_band_num, embed_dim)
        self.lape_dim = 8
        self.lape_to_pe = nn.Linear(self.lape_dim, self.embed_dim)
        self.lape_vec = self._cal_lape(adj_mx)



    def get_temporal_pe(self, input_data):
        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        pe1d = PositionalEncoding1D(num_feat).to(input_data.device)
        tp_pe = pe1d(input_data.view(batch_size * num_nodes, num_patches, num_feat))
        return tp_pe.view(batch_size, num_nodes, num_patches, num_feat)



    def get_dow_pe(self, input_data):
        batch_size, num_nodes, _, num_points = input_data.shape
        num_patches = num_points // 12
        time_point = input_data[:, :, :, ::12].flatten().long()
        return self.dow_pe(time_point).view(batch_size, num_nodes, num_patches, -1)


    def get_tod_pe(self, input_data):
        batch_size, num_nodes, _, num_points = input_data.shape
        num_patches = num_points // 12
        time_point = input_data[:, :, :, ::12].flatten()
        time_point = (time_point * self.steps_per_day).round().long()
        return self.tod_pe(time_point).view(batch_size, num_nodes, num_patches, -1)


    def get_psd_pe(self, input_data):
        fs = 1 / 300
        batch_size, num_nodes, _, num_points = input_data.shape
        num_patches = num_points // 12
        patch_data = input_data.squeeze(-2).view(batch_size, num_nodes, num_patches, -1).cpu()
        f, Pxx = welch(patch_data, fs, nperseg=6)
        Pxx_sum = torch.tensor(Pxx).sum(dim=-1, keepdim=True)
        Pxx = Pxx / Pxx_sum
        Pxx = torch.nan_to_num(Pxx, nan=0.0)
        psd_emb = self.psd_pe(torch.LongTensor([0, 1, 2, 3]).to(input_data.device))
        return torch.matmul(Pxx.to(input_data.device), psd_emb)



    def get_spectral_pe(self):
        return self.lape_to_pe(self.lape_vec.to(self.lape_to_pe.weight.device))



    def _cal_lape(self, adj_mx):
        L, isolated_point_num = self._calculate_normalized_laplacian(adj_mx)
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort()
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
        laplacian_pe = torch.from_numpy(EigVec[:, isolated_point_num + 1: self.lape_dim + isolated_point_num + 1]).float()
        laplacian_pe.require_grad = False
        return laplacian_pe


    def _calculate_normalized_laplacian(self, adj):
        adj = sp.coo_matrix(adj)
        d = np.array(adj.sum(1))
        isolated_point_num = np.sum(np.where(d, 0, 1))
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        return normalized_laplacian, isolated_point_num


