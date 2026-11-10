"""Multi-factor ETF momentum - the ORIGINAL (old_model/) idea rescued onto daily bars.

The legacy minute engine took the old daily model's windows (22..132 *trading days*,
i.e. 1-6 months) and applied them to *minutes*. This module restores the daily horizon
and fixes the scoring:

* six factors per asset, each the best value over several windows (as in old_model):
  high/low differential, regression trend ratio, percent rank, RSI, stochastic,
  plus the plain 6-month return in place of the 1-bar "SMA slope";
* all factors are scale-free (no division by price std);
* factors are combined by *averaging cross-sectional percentile ranks* (robust),
  not by multiplying raw values of wildly different scales;
* absolute-momentum filter: an asset must have positive 6-month return to be held;
* risk-off switch: when SPY is below its 200-day SMA, only defensive assets are eligible
  (no leveraged inverse ETFs);
* inverse-volatility weights over the top-K, rebalanced weekly or monthly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import DailyStrategy, inverse_vol_weights, realized_vol

# Liquid subset of the legacy universe (thin ETFs like PGJ/IWC/DLS removed).
DEFAULT_RISK_ON = [
    "SPY", "QQQ", "IWM", "IJT", "IJS", "RSP", "VNQ", "EFA", "EEM", "EWJ", "EWY", "EWT",
    "FXI", "KWEB", "ICLN", "XLE", "XLK", "XLV", "XLF", "XLI", "GLD", "DBC", "HYG", "TLT", "IEF",
]
DEFAULT_DEFENSIVE = ["IEF", "TLT", "GLD", "SHY", "BIL"]


def _rolling_linreg_ratio(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """Old-model 'linear regression ratio': forecast at x=n over intercept (x=0), minus 1,
    computed on prices normalised by the window's last price (scale-free). Vectorised
    with cumulative sums: y over x=1..n in each trailing window."""
    y = prices.to_numpy(dtype=float)
    T = len(y)
    out = np.full_like(y, np.nan)
    if T < n:
        return pd.DataFrame(out, prices.index, prices.columns)
    i = np.arange(1, T + 1, dtype=float)[:, None]
    S1 = np.vstack([np.zeros((1, y.shape[1])), np.nancumsum(y, axis=0)])
    S2 = np.vstack([np.zeros((1, y.shape[1])), np.nancumsum(y * i, axis=0)])
    t = np.arange(n, T + 1)  # window end (1-based)
    sum_y = S1[t] - S1[t - n]
    sum_iy = S2[t] - S2[t - n]
    offs = (t - n)[:, None].astype(float)
    sum_xy = sum_iy - offs * sum_y            # x = i - (t-n) in 1..n
    sx = n * (n + 1) / 2
    sx2 = n * (n + 1) * (2 * n + 1) / 6
    den = n * sx2 - sx * sx
    b = (n * sum_xy - sx * sum_y) / den
    a = (sum_y * sx2 - sx * sum_xy) / den
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (a + b * n) / a - 1
    window_has_nan = pd.DataFrame(y).isna().rolling(n).sum().to_numpy()[n - 1:] > 0
    ratio[window_has_nan] = np.nan
    out[n - 1:] = ratio
    return pd.DataFrame(out, prices.index, prices.columns)


def multi_window_factors(prices: pd.DataFrame, windows=(22, 44, 66, 88, 110, 132)) -> dict[str, pd.DataFrame]:
    rets = prices.pct_change(fill_method=None)
    fac: dict[str, list[pd.DataFrame]] = {k: [] for k in ("hl", "lr", "pr", "rsi", "stoch")}
    for p in windows:
        mx = prices.rolling(p).max()
        mn = prices.rolling(p).min()
        sd_norm = prices.rolling(p).std() / prices          # std of (window / current price)
        fac["hl"].append(((prices / mn) - (mx / prices)) / sd_norm.replace(0, np.nan))
        fac["lr"].append(_rolling_linreg_ratio(prices, p))
        fac["pr"].append(prices.rolling(p).rank(pct=True))
        g = rets.clip(lower=0).rolling(p).mean()
        l_ = (-rets.clip(upper=0)).rolling(p).mean()
        fac["rsi"].append(100 - 100 / (1 + g / l_.replace(0, np.nan)))
        fac["stoch"].append((prices - mn) / (mx - mn).replace(0, np.nan))
    out = {k: pd.concat(v).groupby(level=0).max().reindex(prices.index) for k, v in fac.items()}
    out["ret126"] = prices / prices.shift(126) - 1
    return out


@dataclass
class MultiFactorMomentum(DailyStrategy):
    name: str = "multifactor_momentum"
    rebalance: str = "W"
    risk_on: list = field(default_factory=lambda: list(DEFAULT_RISK_ON))
    defensive: list = field(default_factory=lambda: list(DEFAULT_DEFENSIVE))
    top_k: int = 4
    vol_lookback: int = 63
    regime_symbol: str = "SPY"
    regime_sma: int = 200
    abs_mom_days: int = 126

    @property
    def symbols(self) -> list[str]:
        return list(dict.fromkeys(self.risk_on + self.defensive + [self.regime_symbol]))

    @property
    def warmup_days(self) -> int:
        return max(self.regime_sma, 132 + 5, self.abs_mom_days)

    def scores(self, prices: pd.DataFrame) -> pd.DataFrame:
        f = multi_window_factors(prices)
        ranks = [v.rank(axis=1, pct=True) for v in f.values()]
        return sum(ranks) / len(ranks)

    def signal_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        score = self.scores(prices)
        absmom = prices / prices.shift(self.abs_mom_days) - 1
        reg = prices[self.regime_symbol]
        risk_on = (reg > reg.rolling(self.regime_sma).mean()).to_numpy()[:, None]
        cols = prices.columns
        in_on = np.isin(cols, self.risk_on)[None, :]
        in_def = np.isin(cols, self.defensive)[None, :]
        eligible = np.where(risk_on, in_on, in_def) & (absmom > 0).to_numpy() & score.notna().to_numpy()
        s = score.where(eligible)
        rank = s.rank(axis=1, ascending=False, method="first")
        selected = rank <= self.top_k
        w = inverse_vol_weights(selected, realized_vol(prices, self.vol_lookback))
        # if nothing qualifies, hold the safest defensive asset available (cash proxy)
        empty = w.sum(axis=1) == 0
        cash = next((c for c in ("BIL", "SHY") if c in cols), None)
        if cash is not None:
            w.loc[empty & prices[cash].notna(), cash] = 1.0
        w[score.isna().all(axis=1)] = np.nan
        return w
