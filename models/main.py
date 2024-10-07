import torch, math
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_layers: int,
        n_units: int,
        dropout: float = 0.5,
    ):
        super().__init__()

        layers = []
        for _ in range(n_layers + 1):
            layers.append(nn.Linear(in_features, n_units))
            layers.append(nn.ReLU())
            in_features = n_units

        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(n_units, out_features))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class LSTM(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_layers: int,
        n_units: int,
        dropout: float,
    ):
        super().__init__()
        self.lstm = nn.LSTM(in_features, n_units, n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_units, out_features)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        return self.fc(out[:, -1, :])


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        out_features: int,
        n_layers: int,
        dropout: float,
        max_len: int = 5000,
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=8,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(d_model, out_features)

    def forward(self, src, tgt):
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # src = src.permute(1, 0, 2)  # (seq_len, batch, features)
        # tgt = tgt.permute(1, 0, 2)  # (seq_len, batch, features)

        output = self.transformer(src, tgt)
        output = output[-1, :, :]  # Take the last output
        output = self.fc(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
