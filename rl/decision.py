import torch


def interleave(a, b, c):
    if c.size(0) != a.size(0):
        padding = torch.full((1, c.size(1)), float("nan"), dtype=torch.float)
        c = torch.cat([c, padding], dim=0)
    out = torch.stack([a, b, c], dim=1)
    return out


def pair_sequences(interleave, S):
    N = interleave.size(0)
    n_batches = N // S
    trimmed = interleave[: n_batches * S]
    new_shape = (n_batches, S * interleave.size(1)) + interleave.shape[2:]
    return trimmed.view(new_shape)


def split_X_y(full_seq):
    X = full_seq[:, :-1, :]
    y = full_seq[:, -1:, :].squeeze(1)
    return X, y


def drop_nan(x):
    mask = ~torch.isnan(x).all(dim=2).squeeze(0)
    return x[:, mask, :]


class TrajBuilder:
    @staticmethod
    def get_train(R, s, a, S):
        assert S <= R.size(0), f"S={S} is larger than R"
        interleaved = interleave(R, s, a)
        interleaved = pair_sequences(interleaved, S)
        X, y = split_X_y(interleaved)
        return X, y

    @staticmethod
    def get_infer(R, s, a, S):
        assert S <= R.size(0), f"S={S} is larger than R"
        interleaved = interleave(R, s, a)
        interleaved = interleaved[-S:, :, :]
        interleaved = interleaved.view(1, -1, interleaved.size(2))
        return drop_nan(interleaved)
