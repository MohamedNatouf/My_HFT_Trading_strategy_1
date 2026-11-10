"""Tests for the infrastructure layer: cost model, open execution, drift band, group caps,
turnover budget, OPG orders, alerts, re-sync, scheduler checks, P&L vs backtest, stream hook."""
import asyncio
import json
from datetime import date, datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import random_walk_prices
from quant.backtest import cost_model_bps, cost_sensitivity, run_backtest
from quant.config import AlpacaCredentials, Settings
from quant.execution import PositionInfo, Quote, RebalanceConfig, Rebalancer
from quant.ops.alerts import Alerter, NullAlerter
from quant.portfolio import RiskLimits, apply_risk


def monthly_targets(prices, w):
    t = pd.DataFrame(np.nan, prices.index, prices.columns)
    me = prices.index.to_series().dt.to_period("M") != prices.index.to_series().shift(-1).dt.to_period("M")
    for s, v in w.items():
        t.loc[me.to_numpy(), s] = v
    return t


# --- costs / backtester ------------------------------------------------------------
def test_cost_series_and_frame_match_scalar():
    p = random_walk_prices(["A", "B"], n=400)
    t = monthly_targets(p, {"A": 0.6, "B": 0.4})
    r0 = run_backtest(p, t, cost_bps=5.0)
    r1 = run_backtest(p, t, cost_bps=pd.Series({"A": 5.0, "B": 5.0}))
    r2 = run_backtest(p, t, cost_bps=pd.DataFrame(5.0, p.index, p.columns))
    assert np.allclose(r0.returns, r1.returns) and np.allclose(r0.returns, r2.returns)
    r3 = run_backtest(p, t, cost_bps=pd.Series({"A": 50.0, "B": 0.0}))
    assert r3.costs.sum() > 0 and r3.returns.sum() < run_backtest(p, t, cost_bps=0.0).returns.sum()


def test_cost_model_is_causal_and_tiered():
    p = random_walk_prices(["SPY", "ZZZZ"], n=300)
    c = cost_model_bps(p, use_measured=False)
    assert (c["ZZZZ"] > c["SPY"]).all()
    p2 = p.copy()
    p2.iloc[200:] *= np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.05, (100, 2)), axis=0))
    c2 = cost_model_bps(p2, use_measured=False)
    pd.testing.assert_frame_equal(c.iloc[:200], c2.iloc[:200])   # future shock never changes the past


def test_cost_sensitivity_monotone():
    p = random_walk_prices(["A", "B"], n=600, drift=0.0004)
    t = pd.DataFrame(np.nan, p.index, p.columns)
    t.iloc[::2] = [[1.0, 0.0]]
    t.iloc[1::2] = [[0.0, 1.0]]        # flip every day: very cost sensitive
    sens = cost_sensitivity(p, t, 5.0, multipliers=(0, 1, 5, 20))
    assert sens["sharpe"].is_monotonic_decreasing
    assert 0 <= sens.attrs["breakeven_multiplier"] < 20


def test_open_execution_split_returns():
    idx = pd.bdate_range("2024-01-01", periods=4)
    close = pd.DataFrame({"A": [100.0, 110.0, 121.0, 121.0]}, idx)
    opens = pd.DataFrame({"A": [100.0, 105.0, 115.0, 121.0]}, idx)
    t = pd.DataFrame({"A": [1.0, np.nan, np.nan, np.nan]}, idx)     # decide day0, trade day1 open
    r = run_backtest(close, t, cost_bps=0.0, lag=1, open_prices=opens, start=idx[0])
    # day1: flat overnight (0%), then long open->close 105->110
    assert r.returns.iloc[1] == pytest.approx(110 / 105 - 1)
    assert r.returns.iloc[2] == pytest.approx(121 / 110 - 1)
    with pytest.raises(ValueError):
        run_backtest(close, t, lag=0, open_prices=opens)


def test_trade_band_skips_small_rebalances_but_not_exits():
    p = random_walk_prices(["A", "B"], n=300, vol=0.002)
    t = pd.DataFrame(np.nan, p.index, p.columns)
    t.iloc[10:] = [[0.5, 0.5]]           # daily re-targeting to 50/50
    t.iloc[200] = [1.0, 0.0]             # exit B
    t.iloc[201:] = np.nan
    no_band = run_backtest(p, t, cost_bps=0.0)
    band = run_backtest(p, t, cost_bps=0.0, trade_band=0.01)
    assert band.turnover.sum() < no_band.turnover.sum()
    assert band.weights.iloc[-1]["B"] == 0.0


# --- risk -------------------------------------------------------------------------
def test_group_caps_and_turnover_budget():
    syms = ["SPY", "QQQ", "IWM", "TLT"]
    p = random_walk_prices(syms, n=200)
    t = pd.DataFrame(np.nan, p.index, syms)
    t.iloc[100] = [0.3, 0.3, 0.3, 0.1]
    t.iloc[150] = [0.0, 0.0, 0.0, 1.0]
    lim = RiskLimits(max_position_weight=1.0, target_vol=None, group_caps={"us_equity": 0.6}, max_turnover=0.5)
    out = apply_risk(p, t, lim)
    r1 = out.iloc[100]
    assert r1[["SPY", "QQQ", "IWM"]].sum() == pytest.approx(0.6)
    assert r1["TLT"] == pytest.approx(0.1)
    r2 = out.iloc[150]
    assert (r2 - r1).abs().sum() == pytest.approx(0.5)              # budget binds
    # live anchor: holdings already close to the target -> no budget scaling
    out2 = apply_risk(p, t, lim, current_weights={"TLT": 0.9, "SPY": 0.05}, current_at=p.index[150])
    assert out2.iloc[150]["TLT"] == pytest.approx(1.0, abs=0.01)


# --- rebalancer: opening auction --------------------------------------------------
class MiniBroker:
    def __init__(self, prices, positions=None, equity=10_000.0):
        self.prices, self._pos, self._eq = prices, dict(positions or {}), equity
        self.submitted, self.calendar_days = [], []
        self.orders = []
        self.acct = SimpleNamespace(equity=str(equity), last_equity=str(equity))

    def equity(self):
        return self._eq

    def buying_power(self):
        return self._eq

    def account(self):
        return self.acct

    def positions(self):
        return {s: PositionInfo(s, q, q * self.prices[s], self.prices[s]) for s, q in self._pos.items()}

    def latest_prices(self, symbols):
        return {s: self.prices[s] for s in symbols}

    def latest_trades(self, symbols):
        now = pd.Timestamp.now(tz="UTC")
        return {s: (self.prices[s], now) for s in symbols}

    def latest_quotes(self, symbols):
        return {s: Quote(self.prices[s] * 0.999, self.prices[s] * 1.001) for s in symbols}

    def fractionable(self, s):
        return True

    def open_orders(self):
        return []

    def recent_orders(self, after=None, limit=500):
        return list(self.orders)

    def cancel_order(self, oid):
        pass

    def order_exists(self, cid):
        return any(o.client_order_id == cid for o in self.submitted)

    def submit(self, spec):
        self.submitted.append(spec)
        return SimpleNamespace(id="x")

    def close_all(self):
        self._pos.clear()
        return []

    def calendar(self, start, end):
        days = pd.bdate_range(start, end)
        return [SimpleNamespace(date=d.date(), close=datetime(d.year, d.month, d.day, 16, 0)) for d in days]


def test_moo_orders_use_opg_whole_shares_and_ref_price():
    b = MiniBroker({"AAA": 33.0})
    r = Rebalancer(b, RebalanceConfig(order_style="moo"), tag="t")
    plan = r.plan({"AAA": 0.5}, {"AAA"}, day="20260102")
    o = plan.orders[0]
    assert o.tif == "opg" and o.order_type == "market" and o.qty == int(o.qty) and o.ref_price == 33.0
    res = r.execute(plan, dry_run=True)
    assert res[0]["ref_price"] == 33.0


# --- alerts ----------------------------------------------------------------------
def test_alerter_dedupes_logs_and_survives_channel_failure(tmp_path):
    sent = []

    def post(url, **kw):
        if "fail" in url:
            raise RuntimeError("boom")
        sent.append((url, kw))

    env = {"ALERT_NTFY_TOPIC": "abc", "ALERT_WEBHOOK_URL": "https://fail.example", "ALERT_MIN_LEVEL": "warning"}
    a = Alerter(tmp_path, env=env, sender=post)
    assert set(a.channels) == {"ntfy", "webhook"}
    assert a.send("Kill switch", "x", level="critical", key="k") == ["ntfy"]
    assert a.send("Kill switch", "x", level="critical", key="k") == []          # deduped
    assert a.send("fyi", "y", level="info") == []                               # below min level
    recs = [json.loads(line) for line in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    assert len(recs) == 3 and recs[1]["suppressed"] is True
    assert sent[0][0] == "https://ntfy.sh/abc"


def test_email_and_telegram_channels(tmp_path):
    got = []
    env = {"ALERT_SMTP_HOST": "smtp.x", "ALERT_EMAIL_TO": "me@x", "ALERT_TELEGRAM_BOT_TOKEN": "t",
           "ALERT_TELEGRAM_CHAT_ID": "1"}
    a = Alerter(tmp_path, env=env, sender=lambda url, **kw: got.append(url))
    assert set(a.send("t", "m", level="critical")) == {"email", "telegram"}
    assert "smtp" in got and any("api.telegram.org" in g for g in got)


# --- daily runner: re-sync, kill switch, scheduler ----------------------------------
def settings_for(tmp_path, session="close"):
    raw = json.loads(open("config/quant.json").read())
    raw["execution"]["session"] = session
    return Settings(raw=raw, alpaca=AlpacaCredentials("k", "s", True, "iex"), path=tmp_path / "q.json")


def order(cid, status, symbol="SPY", side="buy"):
    return SimpleNamespace(client_order_id=cid, status=SimpleNamespace(value=status), symbol=symbol,
                           side=SimpleNamespace(value=side), qty="1", filled_qty="1", filled_avg_price="500",
                           id="oid-" + cid, updated_at="2026-11-10T21:00:00Z")


def test_resync_backfills_missed_events_once_and_alerts_rejections(tmp_path):
    from quant.live.daily import DailyRunner

    b = MiniBroker({"SPY": 500.0}, positions={"SPY": 3})
    b.orders = [order("q-core-20261110-SPY-b-1", "filled"), order("q-core-20261110-QQQ-b-2", "rejected", "QQQ"),
                order("someone-else", "filled")]
    al = NullAlerter()
    r = DailyRunner(settings_for(tmp_path), dry_run=True, state_dir=tmp_path, broker=b, alerter=al)
    snap = r.resync("test")
    assert snap["backfilled"] == 2 and snap["positions"] == {"SPY": 3}
    assert any("rejected" in a["title"] for a in al.records)
    assert r.resync("again")["backfilled"] == 0                     # journaled -> not repeated
    assert not r.needs_resync
    r.on_stream_connect(is_reconnect=True)
    assert any("reconnected" in a["title"] for a in al.records)


def test_kill_switch_flattens_and_alerts(tmp_path):
    from quant.live.daily import DailyRunner

    b = MiniBroker({"SPY": 500.0}, positions={"SPY": 3})
    b.acct = SimpleNamespace(equity="9000", last_equity="10000")
    al = NullAlerter()
    r = DailyRunner(settings_for(tmp_path), dry_run=False, state_dir=tmp_path, broker=b, alerter=al)
    assert r.kill_switch_tripped()
    assert (tmp_path / "HALT").exists() and not b._pos
    assert al.records[-1]["level"] == "critical"
    assert r.run_once() == []                                        # HALT blocks trading


def test_trade_update_rejection_alerts(tmp_path):
    from quant.live.daily import DailyRunner

    al = NullAlerter()
    r = DailyRunner(settings_for(tmp_path), state_dir=tmp_path, broker=MiniBroker({"SPY": 1.0}), alerter=al)
    r.on_trade_update({"event": "fill", "symbol": "SPY"})
    r.on_trade_update({"event": "rejected", "symbol": "SPY", "client_order_id": "c"})
    assert len(al.records) == 1


def clock(ts, is_open, next_open, next_close):
    return SimpleNamespace(timestamp=pd.Timestamp(ts, tz="America/New_York"), is_open=is_open,
                           next_open=pd.Timestamp(next_open, tz="America/New_York"),
                           next_close=pd.Timestamp(next_close, tz="America/New_York"))


def test_scheduler_due_and_missed_run(tmp_path):
    from quant.live.daily import DailyRunner

    al = NullAlerter()
    r = DailyRunner(settings_for(tmp_path), state_dir=tmp_path, broker=MiniBroker({"SPY": 1.0}), alerter=al)
    assert r._due(clock("2026-11-10 15:30", True, "2026-11-11 09:30", "2026-11-10 16:00")) is None
    assert r._due(clock("2026-11-10 15:41", True, "2026-11-11 09:30", "2026-11-10 16:00")) == date(2026, 11, 10)
    r._save_state(last_run="2026-11-10")
    assert r._due(clock("2026-11-10 15:45", True, "2026-11-11 09:30", "2026-11-10 16:00")) is None
    # missed: after the close on a session day without a run
    r._save_state(last_run="2026-11-09")
    assert r.check_missed_run(clock("2026-11-10 16:30", False, "2026-11-11 09:30", "2026-11-11 16:00"))
    assert al.records[-1]["title"] == "Missed rebalance"
    ro = DailyRunner(settings_for(tmp_path / "o", "open"), state_dir=tmp_path / "o", broker=MiniBroker({"SPY": 1.0}),
                     alerter=al)
    assert ro.rebalancer.cfg.order_style == "moo"
    assert ro._due(clock("2026-11-11 08:50", False, "2026-11-11 09:30", "2026-11-11 16:00")) == date(2026, 11, 11)
    assert ro._due(clock("2026-11-11 07:00", False, "2026-11-11 09:30", "2026-11-11 16:00")) is None


def test_staleness_check():
    from quant.live.daily import check_staleness

    now = pd.Timestamp("2026-11-10 20:00", tz="UTC")
    trades = {"A": (1.0, now - pd.Timedelta(minutes=5)), "B": (1.0, now - pd.Timedelta(minutes=90))}
    assert check_staleness(trades, ["A", "B", "C"], now, 30) == ["B", "C"]


def test_compute_targets_end_to_end_with_fake_broker(tmp_path):
    from quant.live.daily import build_sleeves, compute_targets

    s = settings_for(tmp_path)
    syms = sorted({x for st, _ in build_sleeves(s.raw) for x in st.symbols})
    prices = random_walk_prices(syms, n=700, drift=0.0003, start="2024-01-01")
    day = prices.index[-1]
    b = MiniBroker({x: float(prices[x].iloc[-1]) for x in syms})
    row = compute_targets(s, b, day.date(), (day + pd.offsets.BDay(1)).date(), provisional=True,
                          prices=prices.iloc[:-1], alerter=NullAlerter())
    assert set(row.index) == set(syms)
    if row.notna().any():
        assert row.fillna(0).sum() <= 1.0 + 1e-9 and row.fillna(0).max() <= 0.35 + 1e-9
    # open session: the official close for decision_day must exist
    from quant.live.daily import StaleDataError
    with pytest.raises(StaleDataError):
        compute_targets(s, b, (day + pd.offsets.BDay(1)).date(), (day + pd.offsets.BDay(2)).date(),
                        provisional=False, prices=prices)


# --- P&L vs backtest ---------------------------------------------------------------
def test_actual_returns_exclude_deposits_and_compare():
    from quant.ops.pnl import actual_daily_returns, compare

    days = pd.bdate_range("2026-11-02", periods=4, tz="America/New_York")
    hist = SimpleNamespace(timestamp=[int(d.timestamp()) for d in days],
                           equity=[10000, 10100, 20200, 20000],    # deposit of 10k on day 3
                           profit_loss=[0, 100, 100, -200])
    a = actual_daily_returns(hist)
    assert a.iloc[1] == pytest.approx(0.01) and a.iloc[2] == pytest.approx(100 / 20100)
    e = pd.Series([0.0, 0.01, 0.004, -0.01], index=a.index)
    c = compare(a, e)
    assert c.stats["days"] == 4 and abs(c.stats["mean_diff_bps_per_day"]) < 5


def test_order_slippage_vs_decision_and_auction():
    from quant.ops.pnl import order_slippage, slippage_summary

    journal = [{"event_type": "rebalance", "day": "2026-11-10",
                "orders": [{"client_order_id": "q-core-1", "ref_price": 100.0, "tif": "cls", "qty": 10,
                            "est_notional": 1000}]}]
    od = {"client_order_id": "q-core-1", "status": "filled", "filled_avg_price": "100.2", "filled_qty": "10",
          "side": "buy", "filled_at": "2026-11-10T21:00:05Z", "symbol": "SPY", "time_in_force": "cls"}
    closes = pd.DataFrame({"SPY": [100.1]}, index=[pd.Timestamp("2026-11-10")])
    s = order_slippage([od, {**od, "client_order_id": "other"}], journal, closes)
    assert len(s) == 1
    assert s.iloc[0]["slip_vs_decision_bps"] == pytest.approx(20.0)
    assert s.iloc[0]["slip_vs_auction_bps"] == pytest.approx((100.2 / 100.1 - 1) * 1e4)
    summ = slippage_summary(s)
    assert summ["orders"] == 1 and summ["slip_vs_decision_cost_usd"] == pytest.approx(2.004)


# --- stream reconnect hook -----------------------------------------------------------
def test_supervised_stream_reports_reconnects(monkeypatch):
    pytest.importorskip("alpaca")
    from alpaca.trading.stream import TradingStream

    from quant.live.stream import make_supervised_stream_class

    async def fake_start(self):
        return None

    monkeypatch.setattr(TradingStream, "_start_ws", fake_start)
    calls = []
    cls = make_supervised_stream_class()
    s = cls("k", "s", paper=True, on_connect=calls.append)

    async def go():
        await s._start_ws()
        await s._start_ws()
    asyncio.run(go())
    assert calls == [False, True]


# --- data: Yahoo dates must not shift ------------------------------------------------
def test_yahoo_dates_are_session_dates(monkeypatch, tmp_path):
    yf = pytest.importorskip("yfinance")
    from quant.data import BarCache, load_daily_closes

    idx = pd.DatetimeIndex(["2026-11-04", "2026-11-05", "2026-11-06"])   # naive, like yfinance
    raw = pd.DataFrame({"Open": [1.0, 2, 3], "High": [1.0, 2, 3], "Low": [1.0, 2, 3], "Close": [10.0, 20, 30],
                        "Volume": [1, 1, 1]}, idx)
    monkeypatch.setattr(yf, "download", lambda *a, **k: raw)
    c = load_daily_closes(["SPY"], "2026-11-01", "2026-11-07", source="yahoo", cache=BarCache(tmp_path))
    assert list(c.index) == list(idx) and c["SPY"].tolist() == [10.0, 20.0, 30.0]
