import torch
import torch.nn as nn

from ...nn.lstm import LSTM
from ...nn.attention import AttentionStack


class InterSeqMem(nn.Module):
    def __init__(
        self,
        maxlen: int,
        d_model: int,
        n_units: int,
        n_layers: int,
    ):
        super().__init__()
        self.maxlen = maxlen
        self.memory = torch.zeros(0)
        self.encoder = LSTM(d_model, d_model, n_units, n_layers)
        self.attn = AttentionStack(d_model)

    def forward(self, x):
        if len(self.memory) == 0:
            out = x
        else:
            out = self.attn(x, self.memory, self.memory)

        new_mem, _ = self.encoder(x)
        self._update_memory(new_mem)
        return out

    def _update_memory(self, new_mem):
        if len(self.memory):
            self.memory = torch.cat([self.memory, new_mem], dim=1)
        else:
            self.memory = new_mem
        self.memory = self.memory[:, -self.maxlen :, :]
