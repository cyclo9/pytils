import random

import torch
import torch.nn as nn
from torch.optim.adamw import AdamW

from .rlutils import apply_mask
from .buffers import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RDQN:
    def __init__(self, n_acts, policy_net, target_net):
        self._init_hyperparameters()
        self.n_actions = n_acts

        self.policy_net = policy_net.to(device)
        self.target_net = target_net.to(device)
        self.target_net.load_state_dict(policy_net.state_dict())

        self.optimizer = AdamW(policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayBuffer(
            10000, ("obs", "action", "next_obs", "reward", "hidden")
        )

    def _init_hyperparameters(
        self,
        batch_size: int = 128,
        gamma: float = 0.99,
        epsilon: float = 0.99,
        min_eps: float = 0.05,
        eps_decay: int = 100_000,
        lr: float = 1e-4,
        tau: float = 5e-3,
        lmbda: float = 1e-3,
    ):
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_eps = epsilon
        self.min_eps = min_eps
        self.eps_decay = eps_decay
        self.lr = lr
        self.tau = tau
        self.lmbda = lmbda

    def get_action(self, obs, hidden=None):
        sample = random.random()
        with torch.no_grad():
            _, hidden = self.policy_net(obs, hidden)

            if sample < self.epsilon:
                action = torch.randint(0, self.n_actions, (1, 1))
            else:
                q_values, hidden = self.policy_net(obs, hidden)
                action = q_values.argmax().view(1, 1)
        return action, hidden

    def decay_epsilon(self):
        """Call episode-wise"""
        self.epsilon = max(
            self.min_eps, self.epsilon - (self.max_eps - self.min_eps) / self.eps_decay
        )

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)

        # extract non final next obs
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_obs)),
            dtype=torch.bool,
            device=device,
        )
        non_final_next_obs = torch.cat([s for s in batch.next_obs if s is not None])

        obs = torch.cat(batch.obs)
        actions = torch.cat(batch.action)
        reward = torch.cat(batch.reward)
        hidden = torch.stack(batch.hidden, dim=1).squeeze(2)

        q_values, _ = self.policy_net(obs, hidden)
        q_values = q_values.gather(1, actions).flatten()

        # compute target q values
        next_q_values = torch.zeros(self.batch_size, device=device)
        non_final_hidden = hidden[:, non_final_mask, :]
        with torch.no_grad():
            n_q_values, _ = self.target_net(non_final_next_obs, non_final_hidden)
            n_q_values = n_q_values.max(1).values.squeeze()
            next_q_values[non_final_mask] = n_q_values
        target_q_values = (next_q_values * self.gamma) + reward

        criterion = nn.SmoothL1Loss()
        loss = criterion(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return round(loss.item(), 4)

    def update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
