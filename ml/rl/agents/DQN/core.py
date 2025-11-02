import torch
import torch.optim as optim
from pytils.ml.rl.policy import BootstrappedThompson


class DQNBanditAgent:
    def __init__(self, Model, model_params, n_actions, p, lr=1e-3):
        # def __init__(self, model, n_actions, c, lr=1e-3):
        # def __init__(self, model, n_actions, tau, epsilon, min_eps, lr=1e-3):
        # self.eps_greedy = EpsilonGreedy(model, n_actions, tau, epsilon, min_eps)
        # self.policy = UCB1(model, n_actions, c)
        self.policy = BootstrappedThompson(Model, model_params, n_actions, p)

    def act(self, obs: torch.Tensor):
        return self.policy.act(obs)

    def train(self, obs, action_idx, reward):
        self.policy.update(obs, action_idx, reward)

    # def decay(self):
    #     self.eps_greedy.decay()

    # def train(self, obs, action_idx, reward):
    #     loss = bandit_q_loss(self.model, obs, action_idx, reward)
    #     self.optim.zero_grad()
    #     loss.backward()
    #     self.optim.step()
