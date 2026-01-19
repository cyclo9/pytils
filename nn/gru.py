from common_imports import nn

class GRU(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        n_layers: int,
        n_units: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gru = nn.GRU(in_size, n_units, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(n_units, out_size)

    def forward(self, x, hn=None):
        out, hn = self.gru(x, hn)
        out = self.fc(out[:, -1, :])
        return out, hn
