import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
        self,
        in_size: int,
        n_units: int,
        n_layers: int,
        dropout: float = 0.0,
        states=True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            in_size, n_units, n_layers, batch_first=True, dropout=dropout
        )
        self.states = states

    def forward(self, x, hidden=None):
        out, (hn, cn) = self.lstm(x, hidden)
        return (out, (hn, cn)) if self.states else out
