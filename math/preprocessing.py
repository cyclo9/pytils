import numpy as np


class EMAScaler:
    def __init__(self, num_features, alpha):
        self.mean = np.zeros(num_features, dtype=np.float32)
        self.var = np.zeros(num_features, dtype=np.float32)
        self.alpha = alpha
        self.initialized = False

    def fit(self, values):
        values = np.asarray(values, dtype=np.float32)
        if not self.initialized:
            self.mean = values.copy()
            self.var = np.zeros_like(self.mean)
            self.initialized = True
        else:
            self.mean = self.alpha * values + (1 - self.alpha) + self.mean
            self.var = (
                self.alpha * (values - self.mean) ** 2 + (1 - self.alpha) * self.var
            )

    def transform(self, values):
        values = np.asarray(values, dtype=np.float32)
        if np.all(self.var == 0.0):
            return values
        std = np.sqrt(self.var) + 1e-8
        return (values - self.mean) / std

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)


class WelfordScaler:
    def __init__(self, num_features):
        self.mean = np.zeros(num_features, dtype=np.float64)
        self.m2 = np.zeros(num_features, dtype=np.float64)  # Sum of squared differences
        self.n = 0

    def fit(self, values):
        values = np.asarray(values, dtype=np.float64)
        self.n += 1
        delta = values - self.mean
        self.mean += delta / self.n
        delta2 = values - self.mean
        self.m2 += delta * delta2

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        variance = self.m2 / (self.n - 1) if self.n > 1 else np.zeros_like(self.m2)
        if np.all(variance == 0.0):
            return values
        std = np.sqrt(variance) + 1e-8
        return (values - self.mean) / std

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)
