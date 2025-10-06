import numpy as np


def monotonic_stretches(x):
    d = np.sign(np.diff(x))
    changes = np.r_[True, d[1:] != d[:-1]]

    starts = np.where(changes)[0]
    ends = np.r_[starts[1:], len(x) - 1]
    stretch_dir = d[starts]

    increase = [
        list(range(s, e + 1))
        for s, e, dir in zip(starts, ends, stretch_dir)
        if dir == 1
    ]
    decrease = [
        list(range(s, e + 1))
        for s, e, dir in zip(starts, ends, stretch_dir)
        if dir == -1
    ]

    return increase, decrease
