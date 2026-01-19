from datetime import datetime, timezone
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import os


def chunk_1d(x, w):
    x = np.asarray(x)
    n = (x.size // w) * w
    return x[:n].reshape(-1, w)


def clt():
    os.system("cls" if os.name == "nt" else "clear")


def convert_units(q_per_b, b=None, q=None, d=8):
    if b is None and q is None:
        raise ValueError("Provide exactly one of 'b' or 'q'")

    if b is not None:
        return round(b * q_per_b, d)
    else:
        return round(q / q_per_b, d)


def shift(arr, n, fill=np.nan):
    shifted = np.full_like(arr, fill)
    if n >= 0:
        rolled = np.roll(arr, n)[n:]
        shifted[n:] = rolled
        return shifted
    else:
        rolled = np.roll(arr, n)[:n]
        shifted[:n] = rolled
        return shifted


def rolling_avg(old_mean, n, value):
    """
    `m`: the current mean
    `n`: number of elements averaged so far
    `x`: a new value to include in mean
    """
    return (old_mean * n + value) / (n + 1)


def unix_to_utc(unix_ts: int):
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime(
        "%d-%m-%Y %H:%M:%S"
    )


def rfc3339_to_hhmm(timestamp: str) -> str:
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return dt.strftime("%m-%d-%Y %H:%M:%S")


class Nil:
    def __repr__(self):
        return "nil"

    def __bool__(self):
        return False


nil = Nil()

# def make_mask(mask):
#     mask = torch.tensor(mask, dtype=torch.float)
#     return mask.masked_fill(mask == 0, float("-1e10"))

# class GAE:
#     def __init__(self, gamma, lmbda, value_net):
#         self.gamma = gamma
#         self.lmbda = lmbda
#         self.value_net = value_net
#
#     def __call__(self, states, next_states, rewards, dones):
#         values = self.value_net(states)
#         next_values = self.value_net(next_states)
#
#         advantages = torch.zeros_like(rewards, dtype=torch.float)
#         returns = torch.zeros_like(rewards, dtype=torch.float)
#         gae = 0
#
#         for t in reversed(range(len(rewards))):
#             delta = (
#                 rewards[t] + self.gamma * (1 - dones[t]) * next_values[t] - values[t]
#             )
#             gae = delta + self.gamma * self.lmbda * gae
#             advantages[t] = gae
#             returns[t] = gae + values[t]
#
#         return advantages, returns
