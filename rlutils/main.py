import torch
from torch.distributions import Normal, Categorical
from collections import defaultdict
import torch.nn.functional as F
from torch.serialization import get_default_mmap_options

def make_mask(mask):
    mask = torch.tensor(mask, dtype=torch.float)
    return mask.masked_fill(mask == 0, float('-1e10'))

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

class Collector:
    def __init__(self):
        self.data = defaultdict(list)

    def add(self, entries):
        for key, value in entries.items():
            self.data[key].append(value)

    def __getitem__(self, key):
        if key in self.data:
            return torch.stack(self.data[key])
        return torch.tensor([])

class GAE:
    def __init__(self, gamma, lmbda, value_net):
        self.gamma = gamma
        self.lmbda = lmbda
        self.value_net = value_net

    def __call__(self, states, next_states, rewards, dones):
        values = self.value_net(states)
        next_values = self.value_net(next_states)
    
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0 # the last advantage
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * ((1 - dones[t]) * next_values[t]) - values[t]
            gae = delta + (self.gamma * self.lmbda * gae)
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            
        return advantages, returns
