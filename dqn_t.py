import gymnasium as gym
import torch
from itertools import count

from models import FeedForward
from dqn import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("Acrobot-v1", render_mode="human")

n_obs = env.observation_space.shape[0]
n_act = env.action_space.n

policy_net = FeedForward(n_obs, n_act, 2, 64)
target_net = FeedForward(n_obs, n_act, 2, 64)

agent = DQN(n_act, policy_net, target_net)
agent._init_hyperparameters(eps_decay=50000)

for _ in range(10000):
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float, device=device).unsqueeze(0)
    for _ in count():
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_obs = None
        else:
            next_obs = torch.tensor(
                next_obs, dtype=torch.float, device=device
            ).unsqueeze(0)

        agent.memory.push(obs, action, next_obs, reward)

        obs = next_obs

        agent.train()
        agent.update_target_net()

        print(f"{agent.epsilon:.2f}")

        if done:
            break
