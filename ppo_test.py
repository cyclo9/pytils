import gymnasium as gym
from ppo import PPO

env = gym.make("CartPole-v1", render_mode="human")

model = PPO(env)

model.learn(500_000)
