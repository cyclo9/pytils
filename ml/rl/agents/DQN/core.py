import torch
import torch.optim as optim
from pytils.ml.rl.exploration.eps_greedy import EpsilonGreedy
from pytils.ml.rl.learning.d_learning import bandit_q_loss


class DQNBanditAgent:
    def __init__(self, model, n_actions, tau, epsilon, min_eps, lr=1e-3):
        self.model = model
        self.eps_greedy = EpsilonGreedy(model, n_actions, tau, epsilon, min_eps)
        self.optim = optim.Adam(model.parameters(), lr=lr)

    def act(self, obs: torch.Tensor):
        return self.eps_greedy.act(obs)

    def decay(self):
        self.eps_greedy.decay()

    def train(self, obs, action_idx, reward):
        loss = bandit_q_loss(self.model, obs, action_idx, reward)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
