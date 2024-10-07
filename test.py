import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from models import *
import gymnasium as gym
from rlutils import *
# from tensordict import TensorDict
# from torchrl.data import ReplayBuffer, LazyTensorStorage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_features = 2
out_features = 3
n_layers = 2
n_units = 64

actor_net = FeedForward(in_features, out_features, n_layers, n_units)
value_net = FeedForward(in_features, 1, n_layers, n_units)

actor = StochasticActor(actor_net, True)
env = gym.make("CartPole-v1", render_mode='human')

buffer = Collector()
gae = GAE(gamma=0.99, lmbda=0.95, value_net=value_net)

for i in range(200):
    state = env.reset()
    state = torch.tensor(state[0], dtype=torch.float, device=device)
    action, log_prob, entropy = actor(state)

    next_state, reward, terminated, truncated, info = env.step(action.item())
    done = terminated or truncated

    transition = {
        "state": state, 
        "next_state": torch.tensor(next_state, dtype=torch.float, device=device),
        "reward": torch.tensor([reward], dtype=torch.float, device=device),
        "done": torch.tensor([float(terminated or truncated)], dtype=torch.float, device=device)
    }

    buffer.add(transition)

    if done:
        break

states = buffer["state"]
next_states = buffer["next_state"]
rewards = buffer["reward"]
dones = buffer["done"]

advantages, returns = gae(states, next_states, rewards, dones)

print(advantages)
