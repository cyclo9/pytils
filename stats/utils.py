import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def SVA(log_returns, w):
    """Semi-deviation-based Volatilty Asymmetry

    Computes the positive and negative volatility
    """
    windows = sliding_window_view(log_returns, w)
    pos_returns = np.where(windows > 0, windows, np.nan)
    neg_returns = np.where(windows < 0, windows, np.nan)

    pos_vol = np.nanstd(pos_returns, axis=1, ddof=1)
    neg_vol = np.nanstd(neg_returns, axis=1, ddof=1)

    w_pos = np.count_nonzero(~np.isnan(pos_returns), axis=1) / w
    w_neg = np.count_nonzero(~np.isnan(neg_returns), axis=1) / w

    norm_pos_vol = pos_vol * w_pos
    norm_neg_vol = neg_vol * w_neg

    return norm_pos_vol, norm_neg_vol
