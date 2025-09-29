from common_imports import nn

class LSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        out_size: int,
        n_layers: int,
        n_units: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, n_units, n_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(n_units, out_size)

    def forward(self, x, hidden=None):
        out, (hn, cn) = self.lstm(x, hidden)
        out = self.fc(out[:, -1:, :])
        return out, (hn, cn)
