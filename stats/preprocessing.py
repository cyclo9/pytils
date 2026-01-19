import numpy as np


def standard_scale(x, axis=0, eps=1e-8):
    """
    Column-wise: `axis=0`
    Row-wise: axis=1
    """
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    return (x - mean) / (std + eps)


def min_max_scale(x, axis=0, eps=1e-8):
    x = np.asarray(x)
    min_val = np.min(x, axis=axis, keepdims=True)
    max_val = np.max(x, axis=axis, keepdims=True)
    return (x - min_val) / (max_val - min_val + eps)


def max_abs_scale(x, target=1.0, eps=1e-8):
    x = np.asarray(x)
    peak = np.max(np.abs(x))
    return x / (peak + eps) * target
