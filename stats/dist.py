import torch
from torch.distributions import StudentT, Normal
from torch.types import _size
import scipy.special as sp


class SkewT:
    def __init__(self, mean, std, df, skew):
        self.mean = mean
        self.std = std
        self.df = df
        self.skew = skew

        assert torch.all(std > 0), "Scale must be positive"
        assert torch.all(df >= 2), "Degrees of freedom must be greater than 2."

    def sample(self, sample_shape: _size = torch.Size()):
        # based on Azzalini and Capitanio (2003)
        U = StudentT(self.df).rsample(sample_shape)
        U = self.mean + U * self.std

        means = torch.zeros(self.mean.shape)
        stds = torch.ones(self.std.shape)
        V = Normal(means, stds).rsample(sample_shape)

        delta = (self.skew / torch.sqrt(1 + self.skew**2)).unsqueeze(0)
        Z = delta * torch.abs(U) + torch.sqrt(1 - delta**2) * V
        return (self.mean + self.std * Z).squeeze(0)

    def log_prob(self, value: torch.Tensor):
        y = (value - self.mean) / self.std
        nu = self.df
        alpha = self.skew

        t_pdf = StudentT(nu).log_prob(y).exp()
        d = y.pow(2)

        cdf_arg = alpha * y * torch.sqrt((nu + 1) / nu + d)
        with torch.no_grad():
            t_cdf = sp.stdtr(nu + 1, cdf_arg.detach().cpu().numpy())
        t_cdf = t_cdf.detach().clone().to(cdf_arg.device)

        pdf = 2 * t_pdf * t_cdf
        return torch.log(pdf + 1e-12)

    def ci(self, value, n_bootstraps=1000, sample_size=10_000, level=0.95):
        samples = self.sample((sample_size,))
        diffs = samples - value.unsqueeze(1)
        diffs = diffs.unsqueeze(1).expand(-1, n_bootstraps, -1)

        indices = torch.randint(
            0, sample_size, (diffs.shape[0], n_bootstraps, sample_size)
        )
        resampled = torch.gather(diffs, -1, indices)
        stats = resampled.median(dim=-1).values

        lowers = stats.quantile((1 - level) / 2, dim=-1)
        uppers = stats.quantile(1 - (1 - level) / 2, dim=-1)
        ci = torch.vstack([lowers, uppers]).T

        l, u = ci
        r = torch.diff(ci, dim=-1).squeeze(1)
        return l, u, r
