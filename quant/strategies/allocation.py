"""Low-turnover, evidence-backed tactical allocation strategies.

* DualMomentum  - Antonacci, *Dual Momentum Investing* (2014), Global Equities Momentum.
* GTAA          - Faber, "A Quantitative Approach to Tactical Asset Allocation" (2007).
* TrendFollowing- time-series momentum (Moskowitz, Ooi & Pedersen 2012), long-only,
                  inverse-volatility weighted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import DailyStrategy, inverse_vol_weights, realized_vol


@dataclass
class DualMomentum(DailyStrategy):
    """Monthly: if US equities' 12m return beats T-bills, hold the stronger of US vs
    ex-US equities; otherwise hold bonds."""
    name: str = "dual_momentum"
    rebalance: str = "M"
    us: str = "SPY"
    intl: str = "VEU"
    bills: str = "BIL"
    bonds: str = "AGG"
    lookback: int = 252

    @property
    def symbols(self) -> list[str]:
        return [self.us, self.intl, self.bills, self.bonds]

    @property
    def warmup_days(self) -> int:
        return self.lookback + 1

    def signal_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        r = prices / prices.shift(self.lookback) - 1
        w = pd.DataFrame(0.0, prices.index, prices.columns)
        risk_on = r[self.us] > r[self.bills].fillna(0.0)
        us_wins = r[self.us] >= r[self.intl].fillna(-np.inf)
        w.loc[risk_on & us_wins, self.us] = 1.0
        w.loc[risk_on & ~us_wins, self.intl] = 1.0
        w.loc[~risk_on, self.bonds] = 1.0
        w[r[self.us].isna()] = np.nan
        return w


@dataclass
class GTAA(DailyStrategy):
    """Faber: equal weight each asset class, held only if month-end price > 10-month SMA
    (~210 trading days); otherwise that slice sits in cash (or ``cash_symbol``)."""
    name: str = "gtaa"
    rebalance: str = "M"
    assets: list = field(default_factory=lambda: ["SPY", "EFA", "IEF", "VNQ", "DBC"])
    sma_days: int = 210
    cash_symbol: str | None = "BIL"

    @property
    def symbols(self) -> list[str]:
        return self.assets + ([self.cash_symbol] if self.cash_symbol else [])

    @property
    def warmup_days(self) -> int:
        return self.sma_days + 1

    def signal_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        a = prices[self.assets]
        above = a > a.rolling(self.sma_days).mean()
        n = len(self.assets)
        w = pd.DataFrame(0.0, prices.index, prices.columns)
        w[self.assets] = above.astype(float) / n
        if self.cash_symbol:
            w[self.cash_symbol] = 1.0 - w[self.assets].sum(axis=1)
        w[a.rolling(self.sma_days).mean().isna().all(axis=1)] = np.nan
        return w


@dataclass
class TrendFollowing(DailyStrategy):
    """Long-only time-series momentum: hold each asset whose 12-month return is positive,
    inverse-vol weighted across the holdings; remainder in cash."""
    name: str = "trend_following"
    rebalance: str = "M"
    assets: list = field(default_factory=lambda: ["SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC", "VNQ"])
    lookback: int = 252
    vol_lookback: int = 63

    @property
    def symbols(self) -> list[str]:
        return list(self.assets)

    @property
    def warmup_days(self) -> int:
        return self.lookback + 1

    def signal_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        r = prices / prices.shift(self.lookback) - 1
        sel = r > 0
        w = inverse_vol_weights(sel, realized_vol(prices, self.vol_lookback))
        w[r.isna().all(axis=1)] = np.nan
        return w
