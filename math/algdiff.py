import numpy as np
from math import factorial


def algdiff(tau, x, n, k):
    """`tau`: independent variable
    `x`: dependent variable
    `n`: the assumed order of the polynomial, locally
    `k`: order of derivative to compute
    """
    I = build_I_vector(tau, x, n)
    M = build_M_matrix(tau, n)
    a = solve_coeff(M, I)
    return get_derivative(a, k)


def build_I_vector(tau, x, n):
    """`x`: list of sampled values (like y values)
    `tau`: list of values corresponding to each sample value (like x values)
    `n`: is the assumed order of the polynomial
    `k`: the order of the integral to compute
    """
    x = np.asarray(x)
    tau = np.asarray(tau)
    tau = tau - tau[-1]  # shift so last tau is 0

    dt = tau[1] - tau[0]
    I = np.array([np.sum(x * tau**k) * dt for k in range(n + 1)])
    return I


def build_M_matrix(tau, n):
    """Builds the Hilber matrix. `n` is the assumed order of the polynomial"""
    tau = np.asarray(tau)
    tau = tau - tau[-1]  # shift so last tau is 0
    dt = tau[1] - tau[0]
    size = n + 1
    M = np.zeros((size, size))
    tau = np.array(tau)

    for i in range(size):
        for j in range(size):
            order = i + j
            M[i, j] = np.sum(tau**order) * dt

    return M


def solve_coeff(M, I):
    """Returns an array `a` containing derivative coefficients up to degree `n`"""
    return np.linalg.solve(M, I)


def get_derivative(a, k):
    return a[k] * factorial(k)
