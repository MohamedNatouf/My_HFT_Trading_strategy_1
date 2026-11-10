"""Regression tests for the bug fixes in the legacy Trading_System package."""
import types

import numpy as np
import pandas as pd
import pytest

engine = pytest.importorskip("Trading_System.engine")
execution = pytest.importorskip("Trading_System.execution")
minute_store = pytest.importorskip("Trading_System.minute_store")
alpaca_data = pytest.importorskip("Trading_System.alpaca_data")
system_mod = pytest.importorskip("Trading_System.system")

from Trading_System.engine import Allocation, Backtester, Position, Strategy  # noqa: E402


def mk_index(n):
    days = pd.bdate_range("2026-01-05", periods=n // 390 + 2, tz="America/New_York")
    idx = []
    for d in days:
        idx.extend(pd.date_range(d + pd.Timedelta(hours=9, minutes=30), periods=390, freq="min"))
    return pd.DatetimeIndex(idx[:n]).tz_convert("UTC")


def mk_df(close: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for s in close.columns:
        for f in ["open", "high", "low", "close"]:
            cols[(f, s)] = close[s]
        cols[("volume", s)] = pd.Series(1000.0, index=close.index)
    df = pd.DataFrame(cols)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


BASE = {"Volatility_Measuring_Lookback_Period": 60, "Number_of_Portfolio_Allocations": 2,
        "Method_Of_Wieghting": "Assets Volume", "Emergency_System_Enabled": False,
        "rebalance_minutes": 10, "universe_models": []}


@pytest.fixture(scope="module")
def trends():
    rng = np.random.default_rng(0)
    n = 300
    drifts = {"UP_STRONG": 0.002, "UP_MED": 0.001, "FLAT": 0.0, "DOWN_MED": -0.001, "DOWN_STRONG": -0.002}
    close = pd.DataFrame({s: 100 * np.exp(np.cumsum(d + 0.0003 * rng.standard_normal(n))) for s, d in drifts.items()},
                         index=mk_index(n))
    close["SHORT_HIST"] = np.nan
    close.iloc[-30:, close.columns.get_loc("SHORT_HIST")] = 50 * np.exp(np.cumsum(0.01 * np.ones(30)))
    return close


def test_metrics_count_picks_strongest(trends):
    strat = Strategy("t", dict(BASE, Selected_method_of_Momuntum_Metric="Metrics Count"))
    scores = strat._compute_momentum(mk_df(trends))
    assert "SHORT_HIST" not in scores.index
    alloc = strat.generate_allocation(mk_df(trends))
    assert [p.symbol for p in alloc.positions] == ["UP_STRONG", "UP_MED"]
    assert all(p.last_close is not None and p.last_close > 0 for p in alloc.positions)


def test_hl_ratio_scale_invariant(trends):
    s = trends["UP_MED"]
    v1 = Strategy._high_low_diff_ratio_static(s, 22, 132, 22)
    for k in (10, 0.01):
        assert Strategy._high_low_diff_ratio_static(s * k, 22, 132, 22) == pytest.approx(v1, rel=1e-6, abs=1e-9)


def test_backtester_no_lookahead_on_random_walk():
    rng = np.random.default_rng(42)
    n, k = 1600, 6
    rw = pd.DataFrame(100 * np.exp(np.cumsum(0.001 * rng.standard_normal((n, k)), axis=0)), index=mk_index(n),
                      columns=[f"S{i}" for i in range(k)])
    df = mk_df(rw)
    bt = Backtester()
    bt.run(Strategy("rw", dict(BASE, Selected_method_of_Momuntum_Metric="Metrics Combind")), df)
    r = np.array(bt.rebalance_returns[1:])
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    assert abs(t) < 3.5, f"zero-edge data shows t={t:.2f}: look-ahead suspected"
    bt2 = Backtester()
    bt2.run(Strategy("rw", dict(BASE, Selected_method_of_Momuntum_Metric="Metrics Combind", costs={"c2": 0.0005})), df)
    assert np.mean(bt2.rebalance_returns[1:]) < r.mean()
    full = Strategy("x", dict(BASE, Selected_method_of_Momuntum_Metric="Metrics Combind")).generate_allocation(df)
    lat = Strategy("x", dict(BASE, Selected_method_of_Momuntum_Metric="Metrics Combind")).latest_allocation(df)
    assert [(p.symbol, round(p.weight, 8)) for p in full.positions] == [(p.symbol, round(p.weight, 8)) for p in lat.positions]


# ---- execution ---------------------------------------------------------------
class FakeClient:
    is_open = True

    def get_clock(self):
        return types.SimpleNamespace(is_open=self.is_open)

    def get_orders(self, filter=None):
        return []


class FakeTrader:
    def __init__(self, client=None):
        self.client = client or FakeClient()
        self.orders = []

    def submit_market_order(self, **kw):
        self.orders.append(kw)
        return types.SimpleNamespace(id=len(self.orders))

    def cancel_order_safe(self, oid):
        return True


class FakeAccount:
    def __init__(self):
        self.n = 0

    def get_account(self):
        self.n += 1
        return types.SimpleNamespace(equity="10000", buying_power="20000", cash="10000",
                                     cash_withdrawable="10000", memoposts="0")


ALLOC = Allocation(date=pd.Timestamp("2026-01-06 15:00", tz="UTC"),
                   positions=[Position("AAA", 0.5, 100.0), Position("BBB", 0.5, 50.0)])


def test_router_reconciles_and_is_idempotent():
    tr, acct = FakeTrader(), FakeAccount()
    router = execution.ExecutionRouter(tr, acct, {"trading": {"rebalanceThresholdShares": 1, "roundQtyDecimals": 0}})
    router.rebalance(ALLOC, {"OLD": 7.0})
    assert any(o["symbol"] == "OLD" and o["side"] == "sell" and o["qty"] == 7.0 for o in tr.orders)
    assert {(o["symbol"], o["qty"]) for o in tr.orders if o["side"] == "buy"} == {("AAA", 50.0), ("BBB", 100.0)}
    assert all(o.get("client_order_id") for o in tr.orders)
    assert acct.n == 1
    n = len(tr.orders)
    router.rebalance(ALLOC, {"AAA": 50.0, "BBB": 100.0})
    assert len(tr.orders) == n


def test_router_skips_missing_price_and_equity():
    tr = FakeTrader()
    router = execution.ExecutionRouter(tr, FakeAccount(), {})
    s = router.rebalance(Allocation(date=pd.Timestamp("2026-01-06 15:10", tz="UTC"),
                                    positions=[Position("CCC", 1.0, None)]), {})
    assert not tr.orders and s["skipped"] == 1
    s = execution.ExecutionRouter(FakeTrader(), types.SimpleNamespace(get_account=lambda: None), {}).rebalance(ALLOC, {})
    assert s["submitted"] == 0


def test_market_hours_guard_fails_closed():
    closed = FakeClient()
    closed.is_open = False
    tr = FakeTrader(closed)
    execution.ExecutionRouter(tr, FakeAccount(), {}).rebalance(ALLOC, {})
    assert not tr.orders

    class Broken(FakeClient):
        def get_clock(self):
            raise RuntimeError("network down")

    r = execution.ExecutionRouter(FakeTrader(Broken()), None, {})._within_market_hours()
    assert isinstance(r, bool)
    assert execution.ExecutionRouter(FakeTrader(), None, {})._within_market_hours() is True


# ---- data ----------------------------------------------------------------------
def test_rth_filter_and_store(tmp_path):
    ix = pd.DatetimeIndex(["2026-01-06 14:29", "2026-01-06 14:30", "2026-01-06 20:59",
                           "2026-01-06 21:00", "2026-01-10 15:00"], tz="UTC")
    kept = minute_store.filter_rth(pd.DataFrame({"x": range(5)}, index=ix)).index
    assert list(kept) == [ix[1], ix[2]]
    st = minute_store.MinuteStore(str(tmp_path))
    bar = {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]}
    st.upsert_symbol("AAA", pd.DataFrame(bar, index=ix[1:2]))
    st.upsert_symbol("BBB", pd.DataFrame(bar, index=ix[:1]))
    u = [{"symbol": "AAA"}, {"symbol": "BBB"}]
    assert len(st.load_universe(u)) < len(st.load_universe(u, rth_only=False))
    assert st.clear_all() >= 2


def test_bars_request_passes_feed_and_adjustment():
    if alpaca_data.StockBarsRequest is None:
        pytest.skip("alpaca-py not installed")
    cap = {}

    class FakeDataClient:
        def get_stock_bars(self, req):
            cap["req"] = req
            return types.SimpleNamespace(df=pd.DataFrame())

    ld = alpaca_data.AlpacaMinuteDataLoader(None, None, feed="iex", adjustment="all")
    ld.client = FakeDataClient()
    ld.load_minute_bars([{"symbol": "SPY"}], "2026-01-05T14:30:00+00:00", None, feed="sip")
    assert "sip" in str(cap["req"].feed).lower()
    assert "all" in str(cap["req"].adjustment).lower()


def test_merge_bars_no_duplicates():
    rng = np.random.default_rng(1)
    rw = pd.DataFrame(100 + rng.standard_normal((4, 2)), index=mk_index(4), columns=["S0", "S1"])
    a1, a2 = mk_df(rw.iloc[:3]), mk_df(rw.iloc[2:4] * 2)
    m = system_mod._merge_bars(a1, a2)
    assert len(m) == 4 and not m.index.duplicated().any()
    assert m[("close", "S0")].iloc[2] == a2[("close", "S0")].iloc[0]
