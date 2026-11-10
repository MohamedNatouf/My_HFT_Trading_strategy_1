import numpy as np
import pandas as pd

from quant.backtest import run_backtest
from quant.backtest.metrics import summarize


def test_no_lookahead_perfect_foresight_signal_earns_nothing_with_lag(rw):
    """A 'cheating' target that knows bar t's return and is decided AT t must not profit:
    with lag>=1 it is executed after the move it reacts to."""
    p = rw(["A", "B"], n=2000, seed=1)
    ret = p.pct_change()
    # target at t = long the asset that went up most ON bar t (pure reaction, no forecast)
    t = pd.DataFrame(0.0, p.index, p.columns)
    best = ret.fillna(0.0).idxmax(axis=1)
    for c in p.columns:
        t.loc[best == c, c] = 1.0
    t.iloc[0] = np.nan
    res = run_backtest(p, t, cost_bps=0.0, lag=1)
    assert abs(res.returns.mean()) < 3e-4
    # whereas crediting the same-bar return (the legacy bug) would look hugely profitable
    same_bar = (t.shift(0).fillna(0) * ret.fillna(0)).sum(axis=1).mean()
    assert same_bar > 3e-3


def test_buy_and_hold_matches_asset_return(rw):
    p = rw(["A"], n=500, seed=2)
    t = pd.DataFrame(np.nan, p.index, p.columns)
    t.iloc[0] = 1.0
    res = run_backtest(p, t, cost_bps=0.0, lag=1)
    expected = p["A"].iloc[-1] / p["A"].iloc[1] - 1
    assert abs(res.equity.iloc[-1] - 1 - expected) < 1e-9


def test_costs_charged_on_turnover(rw):
    p = rw(["A", "B"], n=300, seed=3)
    t = pd.DataFrame(np.nan, p.index, p.columns)
    t.iloc[::2] = [1.0, 0.0]
    t.iloc[1::2] = [0.0, 1.0]
    free = run_backtest(p, t, cost_bps=0.0)
    costly = run_backtest(p, t, cost_bps=10.0)
    assert costly.returns.sum() < free.returns.sum()
    assert costly.costs.sum() > 0
    assert abs(costly.turnover.iloc[5:].mean() - 2.0) < 0.2


def test_weights_drift_between_rebalances(rw):
    p = rw(["A", "B"], n=50, seed=4, vol=0.05)
    t = pd.DataFrame(np.nan, p.index, p.columns)
    t.iloc[0] = [0.5, 0.5]
    res = run_backtest(p, t, cost_bps=0.0)
    w = res.weights.iloc[-1]
    rel = (p.iloc[-1] / p.iloc[1])
    assert abs(w["A"] / w["B"] - rel["A"] / rel["B"]) < 1e-9


def test_summarize_basic():
    r = pd.Series([0.01, -0.005] * 252, index=pd.bdate_range("2020-01-01", periods=504))
    s = summarize(r)
    assert s["sharpe"] > 0 and s["max_drawdown"] < 0 and s["years"] == 2.0
