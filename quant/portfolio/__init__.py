"""Portfolio construction: combine strategy sleeves, cap positions, target volatility."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..strategies.base import DailyStrategy
from .universe import ASSET_CLASS, UNCAPPED_GROUPS, asset_class


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
    # per asset-class caps, e.g. {"us_equity": 0.6, "commodities": 0.25}; see universe.ASSET_CLASS
    group_caps: dict | None = None
    groups: dict | None = None          # symbol -> group overrides
    # max one-way turnover per rebalance (sum |w_new - w_old|); None disables.
    # The first allocation from (almost) cash is exempt so a new book can be built.
    max_turnover: float | None = None


def _apply_group_caps(w: pd.Series, limits: RiskLimits) -> pd.Series:
    if not limits.group_caps:
        return w
    g = pd.Series({s: asset_class(s, limits.groups) for s in w.index})
    for grp, cap in limits.group_caps.items():
        if grp in UNCAPPED_GROUPS:
            continue
        members = g.index[g == grp]
        tot = float(w[members].clip(lower=0).sum())
        if tot > cap > 0:
            w.loc[members] = w[members] * (cap / tot)   # excess goes to cash
    return w


def _apply_turnover_budget(w: pd.Series, prev: pd.Series, limits: RiskLimits) -> pd.Series:
    if limits.max_turnover is None or prev.abs().sum() < 0.05:
        return w
    tr = float((w - prev).abs().sum())
    if tr <= limits.max_turnover or tr == 0:
        return w
    return prev + (w - prev) * (limits.max_turnover / tr)   # move part of the way


def apply_risk(prices: pd.DataFrame, targets: pd.DataFrame, limits: RiskLimits,
               current_weights: dict | pd.Series | None = None, current_at=None) -> pd.DataFrame:
    """Caps, group caps, causal volatility targeting and a turnover budget, applied on
    rebalance rows only (in that order).

    Portfolio vol is estimated from trailing returns of the *proposed* weights
    (w' Sigma w with a trailing covariance), so no future data is used.
    The turnover budget is measured against the previous risk-adjusted target; for the
    row ``current_at`` (default: the last rebalance row), ``current_weights`` (live
    holdings / book equity) replaces it when given.
    """
    out = targets.copy()
    rets = prices.pct_change(fill_method=None)
    rows = np.where(~targets.isna().all(axis=1))[0]
    prev = pd.Series(0.0, targets.columns)
    for k, i in enumerate(rows):
        w = targets.iloc[i].fillna(0.0).clip(upper=limits.max_position_weight)
        w = _apply_group_caps(w, limits)
        if limits.target_vol is not None and i >= limits.vol_lookback:
            window = rets.iloc[i - limits.vol_lookback + 1:i + 1][w.index]
            cov = window.cov().fillna(0.0).to_numpy() * 252
            pv = float(np.sqrt(max(0.0, w.to_numpy() @ cov @ w.to_numpy())))
            if pv > 0:
                w = w * float(np.clip(limits.target_vol / pv, limits.min_scale, limits.max_gross_leverage / max(w.abs().sum(), 1e-9)))
        gross = w.abs().sum()
        if gross > limits.max_gross_leverage:
            w = w * limits.max_gross_leverage / gross
        at = (targets.index[i] == pd.Timestamp(current_at)) if current_at is not None else (k == len(rows) - 1)
        if current_weights is not None and at:
            prev = pd.Series(current_weights, dtype=float).reindex(targets.columns).fillna(0.0)
        w = _apply_turnover_budget(w, prev, limits)
        out.iloc[i] = w
        prev = w
    return out

