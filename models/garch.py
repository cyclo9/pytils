from numpy import sqrt
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox


class GARCH:
    """Fits a predictive model on squared log returns (variance) and forecasts volatility (std)"""

    def __init__(self, log_returns, scale, p=1, q=1):
        model = arch_model(log_returns * scale, vol="GARCH", p=p, q=q)
        self.scale = scale
        self.res = model.fit(disp="off")
        self.std_resid = self.res.std_resid

    def is_good_fit(self, lags=20, alpha=0.05):
        lb_resid = acorr_ljungbox(self.std_resid, lags=lags, return_df=True)
        lb_sq = acorr_ljungbox(self.std_resid**2, lags=lags, return_df=True)
        return (lb_resid["lb_pvalue"] > alpha).all() and (
            lb_sq["lb_pvalue"] > alpha
        ).all()

    def forecast(self, horizon):
        forecast = self.res.forecast(horizon=horizon)
        variance = forecast.variance.values[0][-1] / (self.scale**2)
        return sqrt(variance)
