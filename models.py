import torch, math
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        n_layers: int,
        n_units: int,
    ):
        super().__init__()

        layers = []
        for _ in range(n_layers + 1):
            layers.append(nn.Linear(in_size, n_units))
            layers.append(nn.ReLU())
            in_size = n_units
        layers.append(nn.Linear(n_units, out_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class LSTM(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        n_layers: int,
        n_units: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            in_size, n_units, n_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(n_units, out_size)

    def forward(self, x, hidden=None):
        out, (hn, cn) = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, (hn, cn)


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


class CNN1D(nn.Module):
    def __init__(
        self,
        channels: int,
        out_size: int,
        n_layers: int,
        out_channels: int,
        seq_len: int,
        min_seq_len: int,
        kernel_size: int = 2,
        dropout: float = 0.0,
        padding: int = -1,
    ):
        """`kernel_size`, `padding` all have to be <=`seq_len`"""

        super().__init__()

        layers = []
        calc_len = lambda x, s: ((x - kernel_size + (2 * padding)) // s) + 1
        post_pool_len = lambda x: calc_len(x, kernel_size)

        for _ in range(n_layers):
            padding = (kernel_size - 1) // 2 if padding == -1 else padding

            layers.append(nn.Conv1d(channels, out_channels, kernel_size, 1, padding))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            seq_len = calc_len(seq_len, 1)

            if post_pool_len(seq_len) >= min_seq_len:
                layers.append(nn.MaxPool1d(kernel_size))
                seq_len //= kernel_size

            channels = out_channels

        layers.append(nn.Flatten())
        self.model = nn.Sequential(*layers)
        self.fc = nn.Linear(seq_len * out_channels, out_size)
        self.out_size = out_size

    def forward(self, x):
        x = self.model(x)
        return self.fc(x)


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


class Transformer(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        n_layers: int,
        d_model: int,  # must be even; divisible by nhead; like hidden size
        nhead: int = 8,
        dropout: float = 0.0,
    ):
        super(Transformer, self).__init__()

        self.encoder = nn.Linear(in_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.fc = nn.Linear(d_model, out_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)[:, -1, :]
        return x
