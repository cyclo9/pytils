import numpy as np
import pandas as pd


def monte_carlo_test(df, strategy_fn, params, observed_score, n=1000):
    scores = []
    for i in range(n):
        df_perm = get_permutation(df, start_index=50, seed=i)
        score = strategy_fn(df_perm.copy(), params)
        scores.append(score)

    p_value = (np.sum(np.array(scores) >= observed_score) + 1) / (n + 1)
    return p_value, scores


def get_permutation(ohlc: pd.DataFrame, start_index: int = 0, seed=None):
    np.random.seed(seed)

    # Log-transform the OHLC prices
    log_bars = np.log(ohlc[["open", "high", "low", "close"]])

    # Compute relative changes (open, high, low, close)
    r_o = (log_bars["open"] - log_bars["close"].shift()).to_numpy()
    r_h = (log_bars["high"] - log_bars["open"]).to_numpy()
    r_l = (log_bars["low"] - log_bars["open"]).to_numpy()
    r_c = (log_bars["close"] - log_bars["open"]).to_numpy()

    # Permutation starting from start_index
    perm_index = start_index + 1
    n_bars = len(ohlc)

    # Shuffle the relative changes (high/low/close and open-to-close)
    idx = np.arange(n_bars - perm_index)
    perm1 = np.random.permutation(idx)
    perm2 = np.random.permutation(idx)

    # Apply permutations
    perm_r_h = r_h[perm_index:][perm1]
    perm_r_l = r_l[perm_index:][perm1]
    perm_r_c = r_c[perm_index:][perm1]
    perm_r_o = r_o[perm_index:][perm2]

    # Reconstruct the permuted OHLC data
    perm_bars = np.zeros((n_bars, 4))
    perm_bars[:start_index] = log_bars.iloc[:start_index].to_numpy()
    perm_bars[start_index, :] = log_bars.iloc[start_index].to_numpy()

    for i in range(perm_index, n_bars):
        k = i - perm_index
        perm_bars[i, 0] = perm_bars[i - 1, 3] + perm_r_o[k]
        perm_bars[i, 1] = perm_bars[i, 0] + perm_r_h[k]
        perm_bars[i, 2] = perm_bars[i, 0] + perm_r_l[k]
        perm_bars[i, 3] = perm_bars[i, 0] + perm_r_c[k]

    # Exponentiate back to original price scale
    perm_bars = np.exp(perm_bars)

    # Return the permuted OHLC as DataFrame
    return pd.DataFrame(
        perm_bars, index=ohlc.index, columns=pd.Index(["open", "high", "low", "close"])
    )


def weight_fn(n, k):
    """The weighting function that returns a weight based on
    `n` and `k` where `n` represents the number of data points
    and `k` represents the rate or sensitivity to `n`"""
    return n / (n + k)
