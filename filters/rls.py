import numpy as np


class RLS:
    def __init__(self, degree, lambda_=0.99, delta=1e4):
        self.degree = degree
        self.lambda_ = lambda_  # forgetting factor (close to 1)
        self.delta = delta  # initial covariance scale
        self.n = degree + 1  # number of coefficients
        self.P = np.eye(self.n) * self.delta  # covariance matrix
        self.theta = np.zeros(self.n)  # coefficients
        self.initialized = False

    def _phi(self, x):
        # Design vector: [1, x, x^2, ..., x^degree]
        return np.array([x**i for i in range(self.n)])

    def update(self, x, y):
        phi = self._phi(x)
        if not self.initialized:
            self.theta = np.zeros(self.n)
            self.P = np.eye(self.n) * self.delta
            self.initialized = True

        P_phi = self.P @ phi
        gain = P_phi / (self.lambda_ + phi @ P_phi)
        error = y - phi @ self.theta
        self.theta = self.theta + gain * error
        self.P = (self.P - np.outer(gain, P_phi)) / self.lambda_

    def predict(self, x):
        phi = self._phi(x)
        return phi @ self.theta

    def derivative(self, x, order=1):
        # Compute derivative of fitted polynomial at x
        # order=1 for first derivative, etc.
        if order > self.degree:
            return 0.0
        deriv_coeffs = [i * self.theta[i] for i in range(order, self.n)]
        val = 0.0
        power = 1
        for i in range(len(deriv_coeffs) - 1, -1, -1):
            val += deriv_coeffs[i] * power
            power *= x
        # Multiply by factorial for order-th derivative
        for k in range(order - 1):
            val *= order - k
        return val


# rls = RLSPoly(degree=3, forgetting_factor=0.98)
#
# for i in range(len(time)):
#     rls.update(time[i], log_price[i])
#     smooth_val = rls.predict(time[i])
#     velocity = rls.derivative(time[i], order=1)
#     acceleration = rls.derivative(time[i], order=2)
#     # store or plot smooth_val, velocity, acceleration
