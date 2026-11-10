"""Portfolio construction: combine strategy sleeves, cap positions, target volatility."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..strategies.base import DailyStrategy


def combine_sleeves(prices: pd.DataFrame, sleeves: list[tuple[DailyStrategy, float]]) -> pd.DataFrame:
    """Weighted sum of sleeve targets. Each sleeve keeps its own rebalance schedule: its
    last target is carried forward, and the combined target is emitted on any day where
    at least one sleeve rebalances (other sleeves contribute their standing targets)."""
    cols = sorted({s for strat, _ in sleeves for s in strat.symbols})
    prices = prices.reindex(columns=cols)
    total = pd.DataFrame(0.0, prices.index, cols)
    any_reb = pd.Series(False, prices.index)
    for strat, weight in sleeves:
        t = strat.target_weights(prices[strat.symbols])
        reb = ~t.isna().all(axis=1)
        any_reb |= reb
        total = total.add(t.ffill().fillna(0.0).reindex(columns=cols, fill_value=0.0) * weight, fill_value=0.0)
    total[~any_reb] = np.nan
    return total


@dataclass
class RiskLimits:
    max_position_weight: float = 1.0    # set lower (e.g. 0.35) for multi-sleeve portfolios
    max_gross_leverage: float = 1.0
    target_vol: float | None = 0.10     # annualised; None disables vol targeting
    vol_lookback: int = 63
    min_scale: float = 0.25


def apply_risk(prices: pd.DataFrame, targets: pd.DataFrame, limits: RiskLimits) -> pd.DataFrame:
    """Caps + causal volatility targeting applied on rebalance rows only.

    Portfolio vol is estimated from trailing returns of the *proposed* weights
    (w' Sigma w with a trailing covariance), so no future data is used.
    """
    out = targets.copy()
    rets = prices.pct_change(fill_method=None)
    rows = np.where(~targets.isna().all(axis=1))[0]
    for i in rows:
        w = targets.iloc[i].fillna(0.0).clip(upper=limits.max_position_weight)
        if limits.target_vol is not None and i >= limits.vol_lookback:
            window = rets.iloc[i - limits.vol_lookback + 1:i + 1][w.index]
            cov = window.cov().fillna(0.0).to_numpy() * 252
            pv = float(np.sqrt(max(0.0, w.to_numpy() @ cov @ w.to_numpy())))
            if pv > 0:
                w = w * float(np.clip(limits.target_vol / pv, limits.min_scale, limits.max_gross_leverage / max(w.abs().sum(), 1e-9)))
        gross = w.abs().sum()
        if gross > limits.max_gross_leverage:
            w = w * limits.max_gross_leverage / gross
        out.iloc[i] = w
    return out
