"""Daily portfolio backtester.

Semantics (chosen to make look-ahead impossible by construction):

* ``targets`` row t holds weights decided with information up to and including the
  close of bar t. Rows that are all-NaN mean "no new decision" (hold; weights drift).
* The decision is executed at the close of bar ``t + lag`` (default lag=1 -> next close).
  lag=0 approximates a market-on-close order computed from a price snapshot taken
  minutes before the close; use it only for strategies built for MOC execution and
  always report the lag=1 result alongside.
* Between rebalances, holdings drift with prices. Unallocated weight is cash, earning
  ``cash_rate`` (annual, default 0).
* Costs: ``cost_bps`` per unit of one-way traded notional (half-spread + slippage +
  fees), charged on every rebalance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import summarize


@dataclass
class BacktestResult:
    returns: pd.Series          # daily net portfolio returns
    gross_returns: pd.Series
    weights: pd.DataFrame       # end-of-day holdings weights (after trading)
    turnover: pd.Series         # one-way turnover traded each day
    costs: pd.Series
    stats: dict

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()


def run_backtest(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    cost_bps: float = 5.0,
    lag: int = 1,
    cash_rate: float = 0.0,
    periods_per_year: int = 252,
    start=None,
) -> BacktestResult:
    if lag < 0:
        raise ValueError("lag must be >= 0")
    prices = prices.sort_index()
    targets = targets.reindex(index=prices.index, columns=prices.columns)
    rets = prices.pct_change(fill_method=None).fillna(0.0).to_numpy()
    tgt = targets.shift(lag).to_numpy()
    has_tgt = ~np.all(np.isnan(tgt), axis=1)
    tradable = prices.notna().to_numpy()

    n, m = rets.shape
    w = np.zeros(m)
    out_w = np.zeros((n, m))
    gross = np.zeros(n)
    net = np.zeros(n)
    turn = np.zeros(n)
    cost = np.zeros(n)
    daily_cash = (1 + cash_rate) ** (1 / periods_per_year) - 1
    c = cost_bps / 1e4
    for t in range(n):
        r = rets[t]
        cash_w = 1.0 - w.sum()
        g = float(w @ r) + cash_w * daily_cash
        # drift weights to end of bar t
        denom = 1.0 + g
        if denom > 0:
            w = w * (1.0 + r) / denom
        if has_tgt[t]:
            new = np.nan_to_num(tgt[t], nan=0.0)
            new = np.where(tradable[t], new, 0.0)  # cannot hold what has no price
            tr = float(np.abs(new - w).sum())
            turn[t] = tr
            cost[t] = tr * c
            w = new
        gross[t] = g
        net[t] = (1.0 + g) * (1.0 - cost[t]) - 1.0
        out_w[t] = w

    idx = prices.index
    res = BacktestResult(
        returns=pd.Series(net, idx, name="net"),
        gross_returns=pd.Series(gross, idx, name="gross"),
        weights=pd.DataFrame(out_w, idx, prices.columns),
        turnover=pd.Series(turn, idx, name="turnover"),
        costs=pd.Series(cost, idx, name="costs"),
        stats={},
    )
    # trim warm-up: start at first day with any exposure (or explicit start)
    first = start if start is not None else (res.weights.abs().sum(axis=1) > 0).idxmax()
    for name in ("returns", "gross_returns", "turnover", "costs"):
        setattr(res, name, getattr(res, name).loc[first:])
    res.weights = res.weights.loc[first:]
    res.stats = summarize(res.returns, periods_per_year=periods_per_year,
                          turnover=res.turnover, weights=res.weights)
    return res
