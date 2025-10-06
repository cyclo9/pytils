import torch.nn as nn
from .block import AttentionBlock


class AttentionStack(nn.Module):
    def __init__(self, d_model, n_heads=1, n_layers=1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [AttentionBlock(d_model, n_heads) for _ in range(n_layers - 1)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, queries, keys, values):
        x = queries
        for block in self.blocks:
            x = block(x, keys, values) + x
        return self.norm(x)
