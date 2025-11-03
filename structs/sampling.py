import numpy as np


def sample_windows_w_offset(srcs, N, w_size, offset_range):
    srcs = np.asarray(srcs)
    src_lens = np.array([s.shape[0] for s in srcs])
    n_srcs = len(srcs)

    src_ids = np.random.randint(0, n_srcs, size=N)
    sample = srcs[src_ids]

    max_start_ids = np.array(
        [src_lens[idx] - w_size - offset_range[-1] for idx in src_ids]
    )
    start_ids = np.random.randint(0, max_start_ids)[:, None]

    rows = np.arange(N)[:, None]
    cols = np.arange(w_size)
    data = sample[rows, start_ids + cols]

    last_ids = start_ids + w_size - 1
    ends_ids = last_ids + offset_range
    offsets = sample[rows, ends_ids]

    return data, offsets
