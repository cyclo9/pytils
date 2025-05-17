import numpy as np


class CausalSavitzkyGolay:
    def __init__(self, window, poly_order, deriv, dt=1.0):
        self.window_size = window
        self.poly_order = poly_order
        self.deriv_order = deriv
        self.dt = dt
        self.buffer = []
        self.coeffs = self._compute_coeffs()

    def _compute_coeffs(self):
        # Fit only to past and current samples: [0, -1, ..., -(window_size - 1)]
        x = np.arange(0, -self.window_size, -1)
        A = np.vander(x, self.poly_order + 1, increasing=True)
        pinv = np.linalg.pinv(A)
        return pinv[self.deriv_order] / (self.dt**self.deriv_order)

    def update(self, new_val):
        self.buffer.insert(0, new_val)
        if len(self.buffer) > self.window_size:
            self.buffer.pop()
        if len(self.buffer) < self.window_size:
            return np.nan
        return np.dot(self.coeffs, self.buffer)


"""Deprecated but kept just incase; you never know"""
# class IncrementalCausalSavGol:
#     def __init__(self, window_size, poly_order, deriv=0, dt=1.0):
#         self.window_size = window_size
#         self.poly_order = poly_order
#         self.deriv = deriv
#         self.delta = dt
#         self.buffer = deque(maxlen=window_size)
#
#         x = np.arange(window_size)
#         X = np.vander(x, N=poly_order + 1, increasing=True)
#         B = pinv(X)
#
#         self.coeffs = {}
#         for d in range(deriv + 1):
#             factor = factorial(d) / (dt**d)
#             self.coeffs[d] = factor * B[d]
#
#     def update(self, new_value):
#         self.buffer.append(new_value)
#         if len(self.buffer) < self.window_size:
#             return np.nan
#         return np.dot(self.coeffs[self.deriv], np.array(self.buffer))
#
# def causal_savgol_coeffs(window_size, poly_order):
#     x = np.arange(window_size)
#     X = np.vander(x, N=poly_order + 1, increasing=True)
#     B = pinv(X)
#     return B[0]  # coefficients for smoothing (0th derivative)
#
#
# def causal_savgol_filter(data, window_size, poly_order):
#     coeffs = causal_savgol_coeffs(window_size, poly_order)
#     smoothed = []
#     for i in range(window_size - 1, len(data)):
#         segment = data[i - window_size + 1 : i + 1]
#         smoothed.append(np.dot(coeffs, segment))
#     return np.concatenate([np.full(window_size - 1, np.nan), smoothed])
