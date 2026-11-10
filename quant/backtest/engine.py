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
  fees), charged on every rebalance. A scalar, a per-symbol Series, or a day x symbol
  DataFrame (see ``quant.backtest.costs.cost_model_bps``).
* ``open_prices``: if given (requires lag >= 1), trades execute at the OPEN of bar t+lag
  instead of the close: the old weights earn close(t-1)->open(t), the new weights earn
  open(t)->close(t). Use it for strategies traded with opening-auction (OPG) orders.
* ``trade_band``: no-trade band in weight units, same rule as ``RebalanceConfig.drift_band``.
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


def _cost_matrix(cost_bps, prices: pd.DataFrame) -> np.ndarray:
    """Normalise scalar / per-symbol Series / day x symbol DataFrame costs (bps) to a decimal matrix."""
    n, m = prices.shape
    if isinstance(cost_bps, pd.DataFrame):
        c = cost_bps.reindex(columns=prices.columns).reindex(prices.index, method="ffill")
        c = c.fillna(c.median()).fillna(5.0)
        return c.to_numpy(dtype=float) / 1e4
    if isinstance(cost_bps, (pd.Series, dict)):
        c = pd.Series(cost_bps, dtype=float).reindex(prices.columns).fillna(5.0).to_numpy()
        return np.broadcast_to(c / 1e4, (n, m))
    return np.full((n, m), float(cost_bps) / 1e4)


def _apply_band(new: np.ndarray, w: np.ndarray, band: float) -> np.ndarray:
    """Mimic the live rebalancer's drift band: skip a symbol's trade when |delta| < band,
    except full exits (target 0) and new entries, which always trade."""
    keep = (np.abs(new - w) < band) & (new != 0) & (w != 0)
    return np.where(keep, w, new)


def run_backtest(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    cost_bps=5.0,
    lag: int = 1,
    cash_rate: float = 0.0,
    periods_per_year: int = 252,
    start=None,
    open_prices: pd.DataFrame | None = None,
    trade_band: float = 0.0,
) -> BacktestResult:
    if lag < 0:
        raise ValueError("lag must be >= 0")
    if open_prices is not None and lag < 1:
        raise ValueError("open execution needs lag >= 1 (decide at close t, trade at open t+1)")
    prices = prices.sort_index()
    targets = targets.reindex(index=prices.index, columns=prices.columns)
    rets = prices.pct_change(fill_method=None).fillna(0.0).to_numpy()
    C = _cost_matrix(cost_bps, prices)
    if open_prices is not None:
        op = open_prices.reindex(index=prices.index, columns=prices.columns)
        r_on = (op / prices.shift(1) - 1).fillna(0.0).to_numpy()     # close(t-1) -> open(t)
        r_id = (prices / op - 1).fillna(0.0).to_numpy()              # open(t) -> close(t)
        bad = ~np.isfinite(r_on) | ~np.isfinite(r_id) | op.isna().to_numpy()
        r_on = np.where(bad, rets, r_on)                             # missing open: whole day overnight
        r_id = np.where(bad, 0.0, r_id)
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

    def _drift(w, r, extra_cash=0.0):
        g = float(w @ r) + (1.0 - w.sum()) * extra_cash
        denom = 1.0 + g
        return (w * (1.0 + r) / denom if denom > 0 else w), g

    for t in range(n):
        if has_tgt[t]:
            new = np.nan_to_num(tgt[t], nan=0.0)
            new = np.where(tradable[t], new, 0.0)  # cannot hold what has no price
        if open_prices is not None and has_tgt[t]:
            w, g1 = _drift(w, r_on[t], daily_cash)          # old weights overnight
            if trade_band > 0:
                new = _apply_band(new, w, trade_band)
            tr_vec = np.abs(new - w)
            turn[t] = float(tr_vec.sum())
            cost[t] = float(tr_vec @ C[t])
            w = new
            w, g2 = _drift(w, r_id[t])                      # new weights intraday
            g = (1.0 + g1) * (1.0 + g2) - 1.0
            net[t] = (1.0 + g1) * (1.0 - cost[t]) * (1.0 + g2) - 1.0
        else:
            w, g = _drift(w, rets[t], daily_cash)            # drift to end of bar t
            if has_tgt[t]:
                if trade_band > 0:
                    new = _apply_band(new, w, trade_band)
                tr_vec = np.abs(new - w)
                turn[t] = float(tr_vec.sum())
                cost[t] = float(tr_vec @ C[t])
                w = new
            net[t] = (1.0 + g) * (1.0 - cost[t]) - 1.0
        gross[t] = g
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
