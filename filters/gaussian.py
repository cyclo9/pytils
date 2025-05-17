from numpy import copy, array


def causal_gaussian_approx(x, sigma, passes=3):
    alpha = 1 / sigma
    y = copy(x)
    for _ in range(passes):
        for t in range(1, len(y)):
            y[t] = alpha * x[t] + (1 - alpha) * y[t - 1]
    return y


# class CausalGaussianApprox:
#     def __init__(self, sigma, passes=3):
#         self.alpha = 1 / sigma
#         self.passes = passes
#         self.state = [None] * passes
#
#     def update(self, x):
#         for i in range(self.passes):
#             prev = x if self.state[i] is None else self.state[i]
#             x = self.alpha * x + (1 - self.alpha) * prev
#             self.state[i] = x
#         return x
#
#     def apply(self, x_array):
#         return array([self.update(x_t) for x_t in x_array])
