import torch, numpy as np, torch.nn as nn
from torch.distributions import Normal, Categorical
from collections import defaultdict
import torch.nn.functional as F


def apply_mask(action: torch.Tensor, mask: list[int]):
    mask_tensor = torch.tensor(mask, dtype=torch.bool)
    action = action.flatten()
    action[mask_tensor == 0] = float("-inf")
    return action


def make_mask(mask):
    mask = torch.tensor(mask, dtype=torch.float)
    return mask.masked_fill(mask == 0, float("-1e10"))


class StochasticActor:
    def __init__(self, actor_net, categorical=False):
        self.actor_net = actor_net
        self.categorical = categorical

    def __call__(self, x, mask=None, min=-float("inf"), max=float("inf")):
        if self.categorical:
            logits = self.actor_net(x)

            mask = make_mask(mask or [1] * len(logits))
            logits = logits * mask

            probs = F.softmax(logits, dim=0)
            dist = Categorical(probs)
            action = dist.sample()
        else:
            mean, std = self.actor_net(x)
            std = F.softplus(std)
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, min, max)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate(self, x, actions):
        if self.categorical:
            logits = self.actor_net(x)
            probs = F.softmax(logits, dim=0)
            dist = Categorical(probs)
        else:
            mean, std = self.actor_net(x)
            std = F.softplus(std)
            dist = Normal(mean, std)

        return dist.log_prob(actions)


class RolloutBuffer:
    def __init__(self):
        self.data = defaultdict(list)

    def add(self, entries):
        for key, value in entries.items():
            self.data[key].append(value)

    def __getitem__(self, key):
        if key in self.data:
            return torch.stack(self.data[key])
        return torch.tensor([])

    def sample(self):
        key = next(iter(self.data))
        length = len(self.data[key])

        indices = np.random.permutation(length).tolist()
        return indices

    def clear(self):
        self.data = defaultdict(list)


class GAE:
    def __init__(self, gamma, lmbda, value_net):
        self.gamma = gamma
        self.lmbda = lmbda
        self.value_net = value_net

    def __call__(self, states, next_states, rewards, dones):
        values = self.value_net(states)
        next_values = self.value_net(next_states)

        advantages = torch.zeros_like(rewards, dtype=torch.float)
        returns = torch.zeros_like(rewards, dtype=torch.float)
        gae = 0

        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t] + self.gamma * (1 - dones[t]) * next_values[t] - values[t]
            )
            gae = delta + self.gamma * self.lmbda * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        return advantages, returns


class ClipPPOLoss:

    def __init__(
        self,
        actor,
        value_net,
        epsilon: float = 0.2,
        c1: float = 0.5,  # critic coef
        c2: float = 0.01,  # entropy coef
    ):
        self.actor = actor
        self.value_net = value_net
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.criterion = nn.SmoothL1Loss()

    def __call__(self, old_probs, states, actions, advantages, entropy, returns):
        new_probs = self.actor.evaluate(states, actions)

        ratio = new_probs / (old_probs + 1e-10)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        policy_loss = torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        values = self.value_net(states)
        value_loss = self.criterion(values, returns)

        entropy_loss = -entropy.mean()

        total_loss = -policy_loss + (self.c1 * value_loss) - (self.c2 * entropy_loss)
        return total_loss
