from common_imports import nn

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


