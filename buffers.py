import random
import torch
from collections import namedtuple, deque

RolloutTransition = namedtuple(
    "RolloutTransition", ["obs", "act", "log_prob", "reward", "value", "done"]
)


class ReplayBuffer(object):
    def __init__(self, capacity, features):
        self.Transition = namedtuple("Transition", features)
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return self.Transition(*zip(*batch))

    def clear(self):
        self.memory = deque([], maxlen=self.capacity)

    def __len__(self):
        return len(self.memory)


class RolloutBuffer:
    def __init__(self):
        self.batch_obs = []
        self.batch_act = []
        self.batch_log_probs = []
        self.batch_rewards = []
        self.batch_vals = []
        self.batch_dones = []

    def last_episode_transition(self):
        return not self.batch_dones or self.batch_dones[-1][-1]

    def add(self, obs, act, log_prob, reward, value, done):
        self.batch_obs.append(obs)
        self.batch_act.append(act)
        self.batch_log_probs.append(log_prob)

        if self.last_episode_transition():
            self.batch_rewards.append([reward])
            self.batch_vals.append([value])
            self.batch_dones.append([done])
        else:
            self.batch_rewards[-1].append(reward)
            self.batch_vals[-1].append(value)
            self.batch_dones[-1].append(done)

    def sample(self):
        obs = torch.cat(self.batch_obs)
        actions = torch.tensor(self.batch_act)
        log_probs = torch.tensor(self.batch_log_probs)

        rewards = [torch.tensor(batch) for batch in self.batch_rewards]
        values = [torch.tensor(batch) for batch in self.batch_vals]
        dones = [torch.tensor(batch) for batch in self.batch_dones]

        self.clear()
        return obs, actions, log_probs, rewards, values, dones

    def clear(self):
        self.batch_obs.clear()
        self.batch_act.clear()
        self.batch_log_probs.clear()
        self.batch_rewards.clear()
        self.batch_vals.clear()
        self.batch_dones.clear()


# class Episode:
#     def __init__(self):
#         self.transitions = []
#
#     def add_transition(self, transition):
#         self.transitions.append(transition)
#
#     def get_data(self):
#         obs = torch.cat([t.obs for t in self.transitions])
#         acts = torch.tensor([t.act for t in self.transitions], dtype=torch.float)
#         log_probs = torch.tensor(
#             [t.log_prob for t in self.transitions], dtype=torch.float
#         )
#         rewards = torch.tensor([t.reward for t in self.transitions], dtype=torch.float)
#         values = torch.cat([t.value for t in self.transitions])
#         dones = [t.done for t in self.transitions]
#         return obs, acts, log_probs, rewards, values, dones
#
#
# class RolloutBuffer:
#     def __init__(self):
#         self.episodes = []
#
#     def push(self, obs, act, log_prob, reward, value, done):
#         """All arguments are expected to be tensors, except for `done`"""
#
#         transition = RolloutTransition(obs, act, log_prob, reward, value, done)
#
#         if not self.episodes or self.episodes[-1].transitions[-1].done:
#             episode = Episode()
#             episode.add_transition(transition)
#             self.episodes.append(episode)
#         else:
#             self.episodes[-1].add_transition(transition)
#
#     def sample(self):
#         obs = torch.cat([(episode.get_data()[0]) for episode in self.episodes])
#         actions = torch.cat([episode.get_data()[1] for episode in self.episodes])
#         log_probs = torch.cat([episode.get_data()[2] for episode in self.episodes])
#
#         batch_rewards = [episode.get_data()[3] for episode in self.episodes]
#         batch_values = [episode.get_data()[4] for episode in self.episodes]
#         batch_dones = [episode.get_data()[5] for episode in self.episodes]
#
#         self.clear()
#         return {
#             "obs": obs,
#             "actions": actions,
#             "log_probs": log_probs,
#             "rewards": batch_rewards,
#             "values": batch_values,
#             "dones": batch_dones,
#         }
#
#     def clear(self):
#         self.episodes = []
#
#     def __len__(self):
#         return len(self.episodes)
