import torch


def drop_nan(x):
    mask = ~torch.isnan(x).all(dim=2).squeeze(0)
    return x[:, mask, :]


class Interleaver:
    def __init__(self, a, b, c, S):
        self.a = a
        self.b = b
        self.c = c
        self.S = S
        if c.size(0) != a.size(0):
            padding = torch.full((1, c.size(1)), float("nan"), dtype=torch.float)
            self.c = torch.cat([c, padding], dim=0)

        assert S <= c.size(0), "S is too large"

    def stack(self):
        self.active = torch.stack([self.a, self.b, self.c], dim=1)
        return self

    def to_ids(self):
        to_ids = lambda t: torch.arange(t.size(0)).unsqueeze(1)
        self.c = self.c[~torch.isnan(self.c).any(dim=1)]
        self.active = torch.stack([to_ids(self.c)], dim=1)
        return self

    def pair(self):
        N = self.active.size(0)
        S = N if self.S == 1 else self.S
        n_batches = N // S
        trimmed = self.active[: n_batches * S]
        new_shape = (n_batches, S * self.active.size(1)) + self.active.shape[2:]
        self.active = trimmed.view(new_shape)
        return self

    def drop_tail(self):
        return self.active[:, :-1, :]

    def take_tail(self):
        return self.active[:, -1:, :].flatten()


# def interleave_drop_tail(stacked, S=1):
#     interleaved = pair_sequences(stacked, S)
#     X = drop_tail(interleaved)
#     return X
#
#
# def interleave_take_tail(ids, S=1):
#     # assert S <= c.size(0), "S is too large"
#     ids = pair_sequences(ids, S)
#     return take_tail(ids)
#
#
# def stack(a, b, c):
#     if c.size(0) != a.size(0):
#         padding = torch.full((1, c.size(1)), float("nan"), dtype=torch.float)
#         c = torch.cat([c, padding], dim=0)
#
#     to_ids = lambda t: torch.arange(t.size(0)).unsqueeze(1)
#     out = torch.stack([a, b, c], dim=1)
#     a_ids = torch.stack([to_ids(c)], dim=1)
#     return out, a_ids
#
#
# def pair_sequences(interleave, S=1):
#     N = interleave.size(0)
#     S = N if S == 1 else S
#     n_batches = N // S
#     trimmed = interleave[: n_batches * S]
#     new_shape = (n_batches, S * interleave.size(1)) + interleave.shape[2:]
#     return trimmed.view(new_shape)
#
#
# def drop_tail(full_seq):
#     return full_seq[:, :-1, :]
#
#
# def take_tail(full_seq):
#     return full_seq[:, -1:, :].flatten()
#
#
# # def interleave(a, b, c, S=1):
# #     assert S <= a.size(0), "S is too large"
# #     stacked, a_ids = stack(a, b, c)
# #     interleaved = pair_sequences(stacked, S)
# #     a_ids = pair_sequences(a_ids, S)
# #     X = drop_tail(interleaved)
# #     y = take_tail(a_ids)
# #     return X, y
