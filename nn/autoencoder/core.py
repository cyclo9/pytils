from typing import Callable
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, dims: list, Model: Callable):
        super().__init__()

        encoders = [Model(inp, out) for inp, out in zip(dims, dims[1:])]
        decoders = [Model(out, inp) for inp, out in zip(dims, dims[1:])]

        self.encoders = nn.Sequential(*encoders)
        decoders = list(reversed(decoders))
        self.decoders = nn.Sequential(*decoders)

    def forward(self, x):
        z = self.encoders(x)
        return self.decoders(z)

    def encode(self, x):
        return self.encoders(x)

    def decode(self, x):
        return self.decoders(x)

    def freeze_decoder(self):
        for param in self.decoders.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self):
        for param in self.decoders.parameters():
            param.requires_grad = True
