import torch
import random


class EpsilonGreedy:
    def __init__(self, model, n_actions, tau, epsilon, min_eps):
        self.model = model
        self.n_actions = n_actions
        self.tau = tau
        self.epsilon = epsilon
        self.min_eps = min_eps

    def decay(self):
        self.epsilon = max(self.min_eps, self.epsilon * self.tau)

    def act(self, obs: torch.Tensor):
        if random.random() < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                out = self.model(obs)
                action = torch.argmax(out).item()
        return action
