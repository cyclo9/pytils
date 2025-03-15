import torch, numpy as np, torch.nn as nn
from torch.distributions import Normal, Categorical
from collections import defaultdict
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


def apply_mask(action: torch.Tensor, mask: list[int]):
    mask_tensor = torch.tensor(mask, dtype=torch.bool)
    action = action.flatten()
    action[mask_tensor == 0] = float("-inf")
    return action


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
