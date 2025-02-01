import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import namedtuple
from pytils.models import FeedForward

Transition = namedtuple(
    'Transition', ['reward', 'value', 'next_value', 'log_prob', 'entropy']
)


def compute_advantages(reward, value, next_value, gamma=0.99):
    return reward + gamma * next_value - value


def compute_loss(
    log_prob, value, reward, next_value, advantage, entropy, beta, gamma=0.99
):
    actor_loss = -log_prob * advantage
    returns = reward + gamma * next_value
    critic_loss = (value - returns) ** 2
    entropy = entropy.mean()
    total_loss = (actor_loss + critic_loss).mean()
    return total_loss - (beta * entropy), entropy


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, beta):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.beta = beta

        self.buffer = []

    def forward(self, state):
        x = self.shared(state)
        probs = self.actor(x)
        value = self.critic(x)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(0)
        entropy = dist.entropy().unsqueeze(0)
        return action.item(), log_prob, entropy, value

    def save(self, reward, value, next_value, log_prob, entropy):
        new_transition = Transition(
            reward, value, next_value, log_prob, entropy
        )
        self.buffer.append(new_transition)

    def learn(self):
        batch = Transition(*zip(*self.buffer))

        rewards = torch.cat(batch.reward)
        values = torch.cat(batch.value)
        next_values = torch.cat(batch.next_value)
        log_probs = torch.cat(batch.log_prob)
        entropy = torch.cat(batch.entropy)
        advantages = compute_advantages(rewards, values, next_values)

        loss, entropy = compute_loss(
            log_probs,
            values,
            next_values,
            rewards,
            advantages,
            entropy,
            self.beta,
        )
        self.beta = max(0.01, self.beta - 0.001)   # anneal beta
        print(loss.item(), entropy.item(), self.beta)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.75)
        self.optimizer.step()

        self.buffer = []
