import math
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerDecoderLayers(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        # Here we use Transformer Encoder in torch.nn with a mask_matrix to implement Transformer decoder.
        decoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout)
        self.transformer_decoder = TransformerEncoder(decoder_layers, nlayers)

    def forward(self, src):
        B, N, L, D = src.shape
        src = src * math.sqrt(self.d_model)
        src=src.contiguous()
        src = src.view(B*N, L, D)
        src = src.transpose(0, 1)
        mask_matrix = torch.triu(torch.ones(L, L) * torch.tensor(float('-inf')), diagonal=1).to(src.device)
        output = self.transformer_decoder(src, mask=mask_matrix)
        output = output.transpose(0, 1).view(B, N, L, D)
        return output

