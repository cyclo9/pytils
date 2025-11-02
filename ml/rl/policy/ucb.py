import torch
import numpy as np


class UCB1:
    def __init__(self, model, n_actions, c):
        self.model = model
        self.N = np.zeros(n_actions)
        self.t = 0
        self.c = c

    def act(self, obs: torch.Tensor):
        self.t += 1
        if 0 in self.N:
            action = np.where(self.N == 0)[0][0]
        else:
            q_values = self.model(obs).squeeze(0)
            ucb = self.c * np.sqrt(np.log(self.t) / self.N)
            ucb = torch.tensor(ucb, device=q_values.device)
            action = torch.argmax(q_values + ucb)

        action = int(action.item())
        self.N[action] += 1
        return action
