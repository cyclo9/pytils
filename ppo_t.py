import torch

import gymnasium as gym
from buffers import RolloutBuffer
from models import FeedForward
from ppo import PPO

env = gym.make("CartPole-v1", render_mode="human")

actor_net = FeedForward(4, 2, 2, 64)
critic_net = FeedForward(4, 1, 2, 64)

agent = PPO(env, 4, 2, actor_net, critic_net)
agent._init_hyperparameters(epochs=20, minibatch_size=8, c1=0.1, clip=0.5)
buffer = RolloutBuffer()

for ep in range(100_000):
    obs, mask = env.reset()

    while True:
        obs = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
        action, log_prob, value = agent.ac(obs)

        next_obs, reward, terminated, truncated, mask = env.step(action.item())
        done = int(terminated or truncated)

        buffer.add(obs, action, log_prob, reward, value, done)

        obs = next_obs

        if done:
            break

    if (ep + 1) % 8 == 0:
        agent.learn(*buffer.sample())
