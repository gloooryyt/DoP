import numpy as np
import torch
from torch import nn

from .patch import PatchEmbedding
from .positional_encoding import PositionalEncoding
from .transformer_decoder_layers import TransformerDecoderLayers

import pickle

def load_pkl(pickle_file: str) -> object:
    """Load pickle data.

    Args:
        pickle_file (str): file path

    Returns:
        object: loaded objected
    """

    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


def unshuffle(shuffled_tokens):
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index


class DoPPretrain(nn.Module):
    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout,
                 temporal_decoder_1_depth, temporal_decoder_2_depth, adj_mx, used_length=288*7, mode="pre-train"):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.selected_feature = 0

        # # patchify & embedding
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        # # positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, adj_mx)
        self.adj_mx = adj_mx
        self.temporal_decoder_1 = TransformerDecoderLayers(embed_dim, temporal_decoder_1_depth, mlp_ratio, num_heads, dropout)
        self.temporal_decoder_1_norm = nn.LayerNorm(embed_dim)
        self.temporal_decoder_2 = TransformerDecoderLayers(embed_dim, temporal_decoder_2_depth, mlp_ratio, num_heads, dropout)
        self.temporal_decoder_2_norm = nn.LayerNorm(embed_dim)
        self.predictor = nn.Linear(embed_dim, patch_size)

        self.used_length = used_length



    def dop_model(self, long_term_history, pre_train):
        # patchify and embed input
        long_term_ts = long_term_history[:, :, [0], :]
        long_term_tod = long_term_history[:, :, [1], :]
        long_term_dow = long_term_history[:, :, [2], :]

        patches = self.patch_embedding(long_term_ts)  # B, N, d, P
        patches = patches.transpose(-1, -2)  # B, N, P, d
        batch_size, num_nodes, num_time, num_dim = patches.shape

        # positional embedding
        tp_pe = self.positional_encoding.get_temporal_pe(patches)
        tod_pe = self.positional_encoding.get_tod_pe(long_term_tod)
        dow_pe = self.positional_encoding.get_dow_pe(long_term_dow)
        psd_pe = self.positional_encoding.get_psd_pe(long_term_history[:, :, [0], :])
        temporal_emb = tp_pe + tod_pe + dow_pe + psd_pe

        patches = patches + temporal_emb

        patches = self.temporal_decoder_1(patches)
        patches = self.temporal_decoder_1_norm(patches)

        spectral_pe = self.positional_encoding.get_spectral_pe()
        spatial_emb = spectral_pe.unsqueeze(0).unsqueeze(2)
        patches = patches + spatial_emb

        patches = self.temporal_decoder_2(patches)
        patches = self.temporal_decoder_2_norm(patches)

        if pre_train:
            prediction = self.predictor(patches)
            return prediction

        else:
            return patches



    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None,
                epoch: int = None, **kwargs) -> torch.Tensor:
        # reshape
        history_data = history_data.permute(0, 2, 3, 1)  # B, N, 1, L * P

        if self.used_length != 288*7:
            history_len = self.used_length
            time_point_num = history_data.shape[-1]
            history_data = history_data[:, :, :, time_point_num - history_len:]

        # feed forward
        if self.mode == "pre-train":
            # encoding
            prediction = self.dop_model(history_data, True)
            label_full = history_data.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, 1:, :,
                         self.selected_feature, :].transpose(1, 2)
            B, N, L, D = label_full.shape
            return prediction[:, :, :-1, :].view(B, N, -1).transpose(1, 2), label_full.view(B, N, -1).transpose(1, 2)

        else:
            patches = self.dop_model(history_data, False)
            return patches


