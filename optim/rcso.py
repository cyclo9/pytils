import numpy as np


class RCSO:
    """Radial Centerend Surplus Optimizer (RCSO)

    Optimizes `n` parameters arranged radially by minimizing the squared distance
    between their center of mass and a target point. Also incorporates a surplus term
    that rewards parameters exceeding target magnitudes."""

    def __init__(self, *radii):
        self.num_params = len(radii)
        self.target_radii = np.array(radii)
        self._thetas = np.arange(len(radii)) * (2 * np.pi) / len(radii)
        self.target_center = self._center_of_mass(*radii)

    def loss(self, *radii):
        if len(radii) != self.num_params:
            raise ValueError(
                f"Number of parameters don't match. Expected {self.num_params} but received {len(radii)}"
            )

        com = self._center_of_mass(*radii)
        d = np.sum((com - self.target_center) ** 2)
        return d

        # surplus = 0
        # if all(abs(r - t) <= 1e-6 for r, t in zip(radii, self.target_radii)):
        #     surplus = self._surplus(*radii)
        # return d - surplus

    def _encode_params(self, *radii):
        unit_vectors = np.stack([np.cos(self._thetas), np.sin(self._thetas)])
        vectors = radii * unit_vectors
        return vectors

    def _center_of_mass(self, *radii):
        vectors = self._encode_params(radii)
        return np.mean(vectors, axis=1)

    def _surplus(self, *radii):
        radii = np.array(radii)
        h = np.maximum(radii - self.target_radii, 0)
        return np.mean(h**2)
