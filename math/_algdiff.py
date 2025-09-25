import numpy as np
from math import factorial


def algdiff(x, n, k, tau=None):
    """`tau`: independent variable
    `x`: dependent variable
    `n`: the assumed order of the polynomial, locally
    `k`: order of derivative to compute
    """
    tau = np.arange(len(x)) if tau is None else np.asarray(tau)
    x = np.asarray(x)

    tau = tau - tau[-1]  # shift so last tau is 0
    dt = tau[1] - tau[0]

    powers = np.vstack([tau**k for k in range(n + 1)])
    I = np.sum(x * powers, axis=1) * dt

    exponents = np.add.outer(np.arange(n + 1), np.arange(n + 1))
    M = np.sum(tau[:, None, None] ** exponents[None, :, :], axis=0) * dt

    a = np.linalg.solve(M, I)
    return a[k] * factorial(k)


"""Deprecated, but useful to understand the concept"""
# def build_I_vector(tau, x, n):
#     """`x`: list of sampled values (like y values)
#     `tau`: list of values corresponding to each sample value (like x values)
#     `n`: is the assumed order of the polynomial
#     `k`: the order of the integral to compute
#     """
#     dt = tau[1] - tau[0]
#     I = np.array([np.sum(x * tau**k) * dt for k in range(n + 1)])
#     return I
#
#
# def build_M_matrix(tau, n):
#     """Builds the Hilber matrix. `n` is the assumed order of the polynomial"""
#     dt = tau[1] - tau[0]
#     size = n + 1
#     M = np.zeros((size, size))
#     tau = np.array(tau)
#
#     for i in range(size):
#         for j in range(size):
#             order = i + j
#             M[i, j] = np.sum(tau**order) * dt
#
#     return M
#
#
# def solve_coeff(M, I):
#     """Returns an array `a` containing derivative coefficients up to degree `n`"""
#     return np.linalg.solve(M, I)
#
#
# def get_derivative(a, k):
#     return a[k] * factorial(k)
