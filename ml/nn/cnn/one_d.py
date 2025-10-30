import torch.nn as nn


class CNN1d(nn.Module):
    def __init__(
        self,
        in_size,
        hidden_sizes,
        conv_kernel=3,  # need to be odd
        pool_kernel=2,  # needs to be less than seq len
    ):
        super().__init__()
        padding = (conv_kernel - 1) // 2
        layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            in_ch = in_size if i == 0 else hidden_sizes[i - 1]
            layers.append(
                nn.Conv1d(in_ch, hidden_size, kernel_size=conv_kernel, padding=padding)
            )
            layers.append(nn.ReLU())

            if (i + 1) % 2 == 0 or i == len(hidden_sizes) - 1:
                layers.append(nn.MaxPool1d(pool_kernel))
                self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
