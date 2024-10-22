import numpy as np
from . import r


def weight_fn(n, k):
    """The weighting function that returns a weight based on
    `n` and `k` where `n` represents the number of data points
    and `k` represents the rate or sensitivity to `n`"""
    return n / (n + k)


def winrate(returns):
    if len(returns) == 0:
        return 0
    else:
        return r(sum(r > 0 for r in returns) / len(returns) * 100, 2)


def profit_factor(returns: list[float]) -> float:
    gross_profits = sum(r for r in returns if r > 0)
    gross_losses = sum(r for r in returns if r < 0)
    return r(gross_profits / gross_losses, 4) if gross_losses != 0 else 0


def sharpe(returns: list[float]) -> float:
    net_returns = sum(returns)
    avg_return = net_returns / len(returns) if len(returns) != 0 else 0

    returns_std = np.std(returns)
    if returns_std == 0 or np.isnan(returns_std):
        return 0
    return r(avg_return / returns_std, 2)
