import torch


class Nil:
    def __repr__(self):
        return 'nil'

    def __bool__(self):
        return False


nil = Nil()


def r(num, d=0):
    return round(num) if d == 0 else round(num, d)


def apply_mask(action: torch.Tensor, mask: list[int]):
    mask_tensor = torch.tensor(mask, dtype=torch.bool)
    action = action.flatten()
    action[mask_tensor == 0] = float('-inf')
    return action
