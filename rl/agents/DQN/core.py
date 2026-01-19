import torch
import torch.optim as optim
import random
from collections import namedtuple

Transition = namedtuple("Transition", ["obs", "action", "signal", "next_obs"])


class DQNAgent:
    def __init__(self, Model, params, device, K, p=0.5, lr=1e-3):
        self.ensemble = [Model(*params).to(device) for _ in range(K)]
        self.optims = [optim.Adam(model.parameters(), lr=lr) for model in self.ensemble]
        self.p = p
        self.buffer = []

    def act(self, obs: torch.Tensor):
        model = random.choice(self.ensemble)
        out = model(obs)
        return out.argmax()

    def push(self, obs, action, signal, next_obs):
        self.buffer.append(Transition(obs, action, signal, next_obs))

    def train(self):
        # TODO: take care of sampling & training
        pass

    # def decay(self):
    #     self.eps_greedy.decay()

    # def train(self, obs, action_idx, reward):
    #     loss = bandit_q_loss(self.model, obs, action_idx, reward)
    #     self.optim.zero_grad()
    #     loss.backward()
    #     self.optim.step()
