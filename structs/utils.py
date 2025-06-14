from numpy import nan, full, asarray, full_like, float64, apply_along_axis
from numpy.lib.stride_tricks import sliding_window_view


def transpose_list_to_dict(arr):
    """Transposes a list of dicts (all the same) into a single dict of lists"""
    map = {key: [d[key] for d in arr] for key in arr[0]}
    return map


def rolling_window_apply(y, func, w):
    y = asarray(y)
    arr = full_like(y, nan, dtype=float64)
    if len(y) < w:
        return arr
    windows = sliding_window_view(y, w)
    result = apply_along_axis(func, 1, windows)
    arr[w - 1 :] = result
    return arr
