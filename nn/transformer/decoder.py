from .positional import encode_position
from .mlp import MLP
from ..attention import AttentionStack

import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    def __init__(self, N, d_model, n_heads, hn_size, use_cross_attn=True):
        super().__init__()
        self.self_attn = nn.ModuleList(
            [AttentionStack(d_model, n_heads) for _ in range(N)]
        )
        self.cross_attn = nn.ModuleList(
            [AttentionStack(d_model, n_heads) for _ in range(N)]
        )

        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = nn.ModuleList(
                [AttentionStack(d_model, n_heads) for _ in range(N)]
            )
        else:
            self.cross_attn = None

        self.mlp = nn.ModuleList([MLP(d_model, hn_size) for _ in range(N)])

    def _casual_mask(self, seq_len, device):
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, x, enc_out=None):
        x = encode_position(x)
        seq_len = x.size(1)
        mask = self._casual_mask(seq_len, x.device)

        for i, (sa, mlp) in enumerate(zip(self.self_attn, self.mlp)):
            x = sa(x, x, x, attn_mask=mask)

            if self.use_cross_attn:
                x = self.cross_attn[i](x, enc_out, enc_out)
            x = mlp(x)
        return x
