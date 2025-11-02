import os
from pytils.ml.rl.learning.q_learning import bandit_q_loss

import torch
import torch.optim as optim
import random


class BootstrappedThompson:
    def __init__(self, Model, model_params, device, K, p=0.5, lr=1e-3):
        self.ensemble = [Model(*model_params).to(device) for _ in range(K)]
        self.optims = [optim.Adam(model.parameters(), lr=lr) for model in self.ensemble]
        self.p = p

    def act(self, obs: torch.Tensor):
        model = random.choice(self.ensemble)
        q_values = model(obs).squeeze(0)

        if torch.all(q_values < 0):
            action = -1
        else:
            action = torch.argmax(q_values).item()
        return action

    def update(self, obs, action_idx, reward):
        for model, optim in zip(self.ensemble, self.optims):
            if random.random() < self.p:
                optim.zero_grad()
                loss = bandit_q_loss(model, obs, action_idx, reward)
                loss.backward()
                optim.step()

    def save(self, name):
        os.makedirs(name, exist_ok=True)
        for i, model in enumerate(self.ensemble):
            path = os.path.join(name, f"{i}.pt")
            torch.save(model.state_dict(), path)

    def load(self, name):
        for i, model in enumerate(self.ensemble):
            path = os.path.join(name, f"{i}.pt")
            model.load_state_dict(torch.load(path))
            model.eval()
