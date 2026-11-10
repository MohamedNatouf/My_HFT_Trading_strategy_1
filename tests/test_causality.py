"""Every strategy must be causal: weights on day t computed from the full history must
equal weights computed from history truncated at t."""
import numpy as np
import pandas as pd
import pytest

from quant.strategies import DAILY_STRATEGIES, build_daily
from quant.strategies.base import rebalance_mask


@pytest.mark.parametrize("name", sorted(DAILY_STRATEGIES))
def test_strategy_is_causal(name, rw):
    st = build_daily(name)
    prices = rw(st.symbols, n=700, seed=11, drift=0.0003)
    full = st.signal_weights(prices)
    for k in (400, 523, 650, 699):
        trunc = st.signal_weights(prices.iloc[: k + 1])
        a, b = full.iloc[k], trunc.iloc[k]
        assert np.allclose(a.fillna(-9).to_numpy(), b.fillna(-9).to_numpy()), f"{name} not causal at row {k}"


@pytest.mark.parametrize("name", sorted(DAILY_STRATEGIES))
def test_targets_valid(name, rw):
    st = build_daily(name)
    prices = rw(st.symbols, n=700, seed=12, drift=0.0003)
    t = st.target_weights(prices)
    rows = t.dropna(how="all")
    assert len(rows) > 0
    assert (rows.fillna(0) >= -1e-12).all().all(), "long-only strategies"
    assert (rows.fillna(0).sum(axis=1) <= 1 + 1e-9).all()
    assert t.iloc[: st.warmup_days].isna().all().all(), "no targets during warm-up"


def test_rebalance_mask_month_end():
    idx = pd.bdate_range("2024-01-01", "2024-03-31")
    m = rebalance_mask(idx, "M")
    picked = idx[m]
    assert [d.strftime("%Y-%m-%d") for d in picked] == ["2024-01-31", "2024-02-29", "2024-03-29"]


def test_multifactor_scale_invariant(rw):
    st = build_daily("multifactor_momentum")
    prices = rw(st.symbols, n=500, seed=5, drift=0.0002)
    s1 = st.scores(prices).iloc[-1]
    scaled = prices.copy()
    scaled[st.symbols[0]] *= 37.0
    s2 = st.scores(scaled).iloc[-1]
    assert np.allclose(s1.to_numpy(), s2.to_numpy(), equal_nan=True)


def test_multifactor_prefers_uptrends():
    from quant.strategies.momentum import MultiFactorMomentum

    idx = pd.bdate_range("2018-01-01", periods=400)
    rng = np.random.default_rng(3)
    syms = ["SPY", "AAA", "BBB", "CCC", "BIL"]
    drifts = {"SPY": 0.0005, "AAA": 0.002, "BBB": -0.001, "CCC": 0.0, "BIL": 0.0001}
    prices = pd.DataFrame({s: 100 * np.exp(np.cumsum(d + 0.005 * rng.standard_normal(400))) for s, d in drifts.items()}, idx)
    st = MultiFactorMomentum(risk_on=["SPY", "AAA", "BBB", "CCC"], defensive=["BIL"], top_k=1)
    w = st.signal_weights(prices).iloc[-1]
    assert w.idxmax() == "AAA"
