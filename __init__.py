from datetime import datetime, timezone
from numpy import nan, full_like


def shift(arr, shift=1, fill_value=nan):
    result = full_like(arr, fill_value, dtype=arr.dtype)
    if shift > 0:
        result[shift:] = arr[:-shift]
    elif shift < 0:
        result[:shift] = arr[-shift:]
    else:
        result[:] = arr
    return result


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


# def make_mask(mask):
#     mask = torch.tensor(mask, dtype=torch.float)
#     return mask.masked_fill(mask == 0, float("-1e10"))


class Nil:
    def __repr__(self):
        return "nil"

    def __bool__(self):
        return False


nil = Nil()


def r(num, d=0):
    return round(num) if d == 0 else round(num, d)


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
