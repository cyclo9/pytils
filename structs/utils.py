from numpy import nan, full, asarray
from numpy.lib.stride_tricks import sliding_window_view


def transpose_list_to_dict(arr):
    """Transposes a list of dicts (all the same) into a single dict of lists"""
    map = {key: [d[key] for d in arr] for key in arr[0]}
    return map


def slide_window_apply(y, func, w):
    y = asarray(y)
    windows = sliding_window_view(y, window_shape=w)
    arr = full(len(y), nan)
    arr[w - 1 :] = [func(win)[-1] for win in windows]
    return arr
