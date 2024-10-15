import random

import torch
import torch.nn as nn
from torch.optim import AdamW

from buffers import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN:
    def __init__(self, n_acts, policy_net, target_net):
        self._init_hyperparameters()
        self.n_actions = n_acts

        self.policy_net = policy_net.to(device)
        self.target_net = target_net.to(device)
        self.target_net.load_state_dict(policy_net.state_dict())

        self.optimizer = AdamW(policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayBuffer(10000)

    def _init_hyperparameters(self):
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 0.99
        self.min_eps = 0.05
        self.eps_decay = 10000
        self.tau = 5e-3
        self.lr = 1e-4

    def get_action(self, obs):
        sample = random.random()
        self.epsilon = max(self.min_eps, self.epsilon - (self.epsilon / self.eps_decay))

        if sample > self.epsilon:
            with torch.no_grad():
                return self.policy_net(obs).argmax(dim=1, keepdim=True)
        else:
            action = random.randint(0, self.n_actions - 1)
            return torch.tensor([[action]], dtype=torch.long, device=device)

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

        q_values = self.policy_net(obs).gather(1, actions)

        # compute target q values
        next_q_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_q_values[non_final_mask] = (
                self.target_net(non_final_next_obs).max(1).values
            )
        target_q_values = (next_q_values * self.gamma) + reward

        criterion = nn.SmoothL1Loss()
        loss = criterion(q_values, target_q_values.unsqueeze(1))

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
