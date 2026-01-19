from common_imports import nn

class FeedForward(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        n_layers: int,
        n_units: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        for _ in range(n_layers + 1):
            layers.append(nn.Linear(in_size, n_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = n_units
        layers.append(nn.Linear(n_units, out_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
