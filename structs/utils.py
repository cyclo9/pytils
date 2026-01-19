from numpy import (
    nan,
    pad,
    isnan,
    asarray,
    full_like,
    float64,
    apply_along_axis,
    column_stack,
)
from .. import sliding_window_view

# from numpy.lib.stride_tricks import sliding_window_view


def interweave(a, b):
    max_len = max(len(a), max(b))
    init_type = a.dtype

    a, b = a.astype(float), b.astype(float)

    a = pad(a, (0, max_len - len(a)), constant_values=nan)
    b = pad(b, (0, max_len - len(b)), constant_values=nan)

    interwoven = column_stack((a, b)).ravel()
    interwoven = interwoven[~isnan(interwoven)]
    interwoven = interwoven.astype(init_type)

    return interwoven


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
