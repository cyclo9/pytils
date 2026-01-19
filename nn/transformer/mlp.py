import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.model(x))
