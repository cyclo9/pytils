import torch

from .models import FeedForward, LSTM
from .actor import ActorCritic

from .buffers import RolloutBuffer
from .awr import compute_gae


def test_buffer():
    buffer = RolloutBuffer()

    def rand_transition():
        obs = torch.randn(1, 4)
        act = torch.randn(1)
        log_prob = torch.randn(1)
        reward = torch.randn(1)
        value = torch.randn(1)
        return obs, act, log_prob, reward, value

    for _ in range(5):
        obs, act, log_prob, reward, value = rand_transition()
        done = 1 if _ == 2 else 0
        buffer.push(obs, act, log_prob, reward, value, done)

    rewards = [torch.tensor([1, 1, 1]), torch.tensor([1, 2, 3])]
    values = [torch.tensor([1, 1, 1]), torch.tensor([1, 2, 3])]
    dones = [[0, 0, 0], [0, 0, 0]]

    compute_gae(rewards, values, dones, 1, 1)


def test_ac_feedforward():
    actor_net = FeedForward(4, 2, 2, 64)
    critic_net = FeedForward(4, 1, 2, 64)

    ac = ActorCritic(actor_net, critic_net, True)
    obs = torch.randn(1, 4)
    action, log_prob, value = ac(obs)

    assert action.shape == torch.Size([1])
    assert log_prob.shape == torch.Size([1])
    assert value.shape == torch.Size([1])


def test_ac_lstm():
    actor_net = LSTM(4, 2, 2, 64)
    critic_net = LSTM(4, 1, 2, 64)

    ac = ActorCritic(actor_net, critic_net, True)
    obs = torch.randn(1, 3, 4)
    action, log_prob, value = ac(obs)

    assert action.shape == torch.Size([1])
    assert log_prob.shape == torch.Size([1])
    assert value.shape == torch.Size([1])
