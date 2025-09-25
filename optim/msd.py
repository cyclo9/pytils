from ..common_imports import np


def msdloss(pred, target, d=8):
    """
    Mean Directional Squared Loss
    pred: `ArrayLike`
    target: `ArrayLike`
    """

    pred, target = np.array(pred), np.array(target)
    diff = target - pred
    sq_diff = np.sign(diff) * diff**2
    return round(float(np.mean(sq_diff)), d)
