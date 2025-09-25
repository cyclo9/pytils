from ..common_imports import np


def mseloss(pred, target, d=8):
    """
    pred: `ArrayLike`
    target: `ArrayLike`
    """

    pred, target = np.array(pred), np.array(target)
    sq_diff = (target - pred) ** 2
    return round(float(np.mean(sq_diff)), d)
