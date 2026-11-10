"""Strategy contract and shared helpers.

A daily strategy is a *pure, causal* mapping::

    target_weights(prices) -> DataFrame (same index/columns as prices)

Row t uses only prices up to and including row t. Rows that are entirely NaN mean
"no rebalance today". The same function is used in backtests (full history) and live
(history up to now -> take the last row), so research and production cannot diverge.
``tests/test_causality.py`` enforces causality for every registered strategy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DailyStrategy(ABC):
    name: str = "base"
    rebalance: str = "M"            # 'D' daily, 'W' last trading day of week, 'M' last of month
    params: dict = field(default_factory=dict)

    @property
    @abstractmethod
    def symbols(self) -> list[str]:
        """All symbols the strategy needs prices for."""

    @property
    def warmup_days(self) -> int:
        return 260

    @abstractmethod
    def signal_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Desired weights for every row (causal). Rebalance masking is applied by caller."""

    def target_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        prices = prices.reindex(columns=self.symbols)
        w = self.signal_weights(prices).reindex(index=prices.index, columns=prices.columns)
        w = w.where(prices.notna(), 0.0)
        mask = rebalance_mask(prices.index, self.rebalance)
        warm = np.arange(len(prices)) >= self.warmup_days
        w.loc[~(mask & warm)] = np.nan
        # rows where signal could not be computed (all NaN) remain NaN = hold
        return w


def rebalance_mask(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """True on the last trading day of each period. Uses the calendar of ``index`` itself;
    knowing that tomorrow starts a new week/month is calendar information, not price data."""
    idx = pd.DatetimeIndex(index)
    if freq == "D":
        return np.ones(len(idx), dtype=bool)
    if freq == "W":
        key = idx.isocalendar().week.to_numpy() + 100 * idx.isocalendar().year.to_numpy()
    elif freq == "M":
        key = idx.year * 100 + idx.month
    elif freq == "Q":
        key = idx.year * 10 + (idx.month - 1) // 3
    else:
        raise ValueError(f"unknown rebalance freq {freq!r}")
    key = np.asarray(key)
    nxt = np.append(key[1:], -1)
    mask = key != nxt
    mask[-1] = True  # the latest bar is always a candidate (live: "is today the last day?" decided by caller)
    return mask


def trailing_return(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    return prices / prices.shift(days) - 1


def realized_vol(prices: pd.DataFrame, days: int = 63) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).rolling(days, min_periods=max(10, days // 2)).std() * np.sqrt(252)


def inverse_vol_weights(selected: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """Row-wise inverse-vol weights over a boolean selection mask (rows sum to 1 or 0)."""
    inv = (1.0 / vol.replace(0, np.nan)).where(selected)
    s = inv.sum(axis=1)
    return inv.div(s.replace(0, np.nan), axis=0).fillna(0.0)


def rsi(prices: pd.DataFrame | pd.Series, period: int) -> pd.DataFrame | pd.Series:
    """Wilder RSI."""
    d = prices.diff()
    up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0)
