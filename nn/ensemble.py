import torch
import torch.nn as nn
import torch.optim as optim
from enum import Enum, auto

import random


class Mode(Enum):
    TRAIN = auto()
    EVAL = auto()


class Ensemble(nn.Module):
    def __init__(self, Model, params, device, K, p=0.5, lr=1e-3):
        super().__init__()
        self.mode = Mode.TRAIN
        self.ensemble = [Model(*params).to(device) for _ in range(K)]
        self.optims = [optim.Adam(model.parameters(), lr=lr) for model in self.ensemble]
        self.p = p

    def forward(self, x):
        if self.mode is Mode.TRAIN:
            model = random.choice(self.ensemble)
            return model(x)
        else:
            outputs = torch.stack([model(x) for model in self.ensemble], dim=0)
            return outputs.mean(dim=0)

    def update(self, x, y, loss_fn):
        for model, optim in zip(self.ensemble, self.optims):
            if random.random() < self.p:
                optim.zero_grad()
                output = model(x)
                loss = loss_fn(output, y)
                loss.backward()
                optim.step()

    def get_std(self, x):
        outputs = torch.stack([model(x) for model in self.ensemble], dim=0)
        return outputs.std(dim=0)

    def train(self, mode=True):
        self.mode = Mode.TRAIN
        super().train(mode)
        return self

    def eval(self):
        self.mode = Mode.EVAL
        super().eval()
        return self
