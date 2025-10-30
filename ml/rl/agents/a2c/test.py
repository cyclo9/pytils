import torch
import torch.optim as optim
import gymnasium as gym
from script import ActorCritic, compute_advantages, compute_loss

env = gym.make('CartPole-v1', render_mode='human')
model = ActorCritic(4, 2, 0.1)

for i in range(100_000):
    obs, _ = env.reset()
    done = False

    while True:
        obs = torch.tensor(obs, dtype=torch.float32)
        action, log_prob, entropy, value = model(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = terminated or truncated

        next_obs = torch.tensor(next_obs, dtype=torch.float)
        _, _, _, next_value = model(next_obs)

        model.save(reward, value, next_value, log_prob, entropy)

        obs = next_obs

        if done:
            break

    if i % 100 == 0:
        model.learn()
