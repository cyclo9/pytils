from .positional import encode_position
from .mlp import MLP
from ..attention import AttentionStack

import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, N, d_model, n_heads, hn_size):
        super().__init__()
        self.attn_layers = nn.ModuleList(
            [AttentionStack(d_model, n_heads) for _ in range(N)]
        )
        self.mlp_layers = nn.ModuleList([MLP(d_model, hn_size) for _ in range(N)])

    def forward(self, x):
        x = encode_position(x)
        for attn, mlp in zip(self.attn_layers, self.mlp_layers):
            x = attn(x, x, x)
            x = mlp(x)
        return x
