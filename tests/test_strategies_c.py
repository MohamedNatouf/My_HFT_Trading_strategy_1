"""Tests for the item-C research code: reversal strategies, option pricing, put-spread runner."""
import json
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import random_walk_prices
from quant.config import AlpacaCredentials, Settings
from quant.strategies.options import (PutSellConfig, bs_put, implied_vol_put, monthly_expiries, occ_symbol,
                                      simulate_put_selling, strike_for_delta)
from quant.strategies.reversal import SectorReversal, StockReversal


def test_sector_reversal_buys_laggards():
    p = random_walk_prices(["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "SPY"], n=60)
    p.iloc[-5:, 0] *= 0.9          # XLB lagged badly over the last week
    w = SectorReversal(n=3, rebalance="D").target_weights(p).iloc[-1]
    assert w["XLB"] > 0 and w.sum() == pytest.approx(1.0)


def test_stock_reversal_dollar_neutral():
    uni = {f"S{i}": ("XLK" if i % 2 else "XLF") for i in range(40)}
    st = StockReversal(universe=uni, rebalance="D")
    p = random_walk_prices(st.symbols, n=80)
    w = st.target_weights(p).iloc[-1]
    assert w.sum() == pytest.approx(0.0, abs=1e-12) and w.abs().sum() == pytest.approx(1.0)


def test_bs_and_iv_roundtrip():
    price = float(bs_put(500, 480, 30 / 365, 0.04, 0.013, 0.22))
    assert implied_vol_put(price, 500, 480, 30 / 365, 0.04, 0.013) == pytest.approx(0.22, abs=1e-4)
    k = strike_for_delta(500, 21 / 252, 0.04, 0.013, lambda K: 0.2, -0.20)
    assert 450 < k < 500
    assert occ_symbol("SPY", "2026-12-18", 600) == "SPY261218P00600000"
    assert [d.day for d in monthly_expiries("2026-01-01", "2026-03-31")] == [16, 20, 20]


def test_put_spread_loss_is_bounded():
    idx = pd.bdate_range("2020-01-01", periods=120)
    spx = pd.Series(3000.0, idx)
    spx.iloc[30:] = 2000.0                                    # -33% crash inside the first cycle
    vix = pd.Series(20.0, idx)
    vix.iloc[30:] = 60.0
    r = simulate_put_selling(spx, vix, pd.Series(0.0, idx), PutSellConfig(risk_per_trade=0.05, vix_stop=None))
    eq = (1 + r["ret"]).cumprod()
    assert eq.min() > 1 - 0.05 - 0.01                         # never loses more than the risk budget (+costs)


# --- put spread runner with fake Alpaca clients ---------------------------------------
def snap(delta, bid, ask):
    return SimpleNamespace(greeks=SimpleNamespace(delta=delta), latest_quote=SimpleNamespace(bid_price=bid, ask_price=ask))


class FakeTrading:
    def __init__(self, positions=()):
        self.positions = list(positions)

    def get_all_positions(self):
        return self.positions

    def get_account(self):
        return SimpleNamespace(equity="100000")


class FakeOptData:
    def __init__(self, chain):
        self.chain = chain

    def get_option_chain(self, req):
        return self.chain

    def get_option_snapshot(self, req):
        return self.chain


def settings(tmp_path):
    raw = json.loads(open("config/quant.json").read())
    return Settings(raw=raw, alpaca=AlpacaCredentials("k", "s", True, "iex"), path=tmp_path / "q.json")


def chain_for(exp="261218"):
    return {f"SPY{exp}P00600000": snap(-0.35, 9.0, 9.2), f"SPY{exp}P00570000": snap(-0.16, 4.0, 4.1),
            f"SPY{exp}P00510000": snap(-0.06, 1.0, 1.05), f"SPY{exp}P00500000": snap(-0.05, 0.9, 0.95)}


def test_runner_opens_sized_credit_spread(tmp_path):
    from quant.live.options import PutSpreadRunner

    r = PutSpreadRunner(settings(tmp_path), dry_run=True, trading=FakeTrading(), option_data=FakeOptData(chain_for()),
                        vix_fn=lambda: 18.0, spot_fn=lambda: 620.0, state_dir=tmp_path)
    out = r.run_once(date(2026, 11, 12))
    assert out["action"] == "open"
    assert out["pick"]["short"]["strike"] == 570 and out["pick"]["long"]["strike"] == 510
    o = out["order"]
    assert o["limit_price"] < 0                                  # credit = negative limit
    assert o["legs"][0][1:] == ("sell", "sell_to_open")
    max_loss = 60 - (4.05 - 1.025)
    assert out["contracts"] == max(1, int(0.03 * 100000 / (max_loss * 100)))


def test_runner_skips_high_or_unknown_vix_and_closes_near_expiry(tmp_path):
    from quant.live.options import PutSpreadRunner

    s = settings(tmp_path)
    r = PutSpreadRunner(s, trading=FakeTrading(), option_data=FakeOptData(chain_for()), vix_fn=lambda: 45.0,
                        spot_fn=lambda: 620.0, state_dir=tmp_path)
    assert r.run_once(date(2026, 11, 12))["action"] == "skip"
    r.vix_fn = lambda: None
    assert r.run_once(date(2026, 11, 12))["action"] == "skip"
    pos = [SimpleNamespace(symbol="SPY261218P00570000", qty="-2"), SimpleNamespace(symbol="SPY261218P00510000", qty="2")]
    r2 = PutSpreadRunner(s, trading=FakeTrading(pos), option_data=FakeOptData(chain_for()), vix_fn=lambda: 18.0,
                         spot_fn=lambda: 620.0, state_dir=tmp_path)
    assert r2.run_once(date(2026, 12, 1))["action"] == "hold"
    out = r2.run_once(date(2026, 12, 17))
    assert out["action"] == "close" and out["order"]["qty"] == 2 and out["order"]["limit_price"] > 0
    assert ("SPY261218P00570000", "buy", "buy_to_close") in out["order"]["legs"]


# --- news sentiment ---------------------------------------------------------------------
def test_lexicon_and_session_assignment():
    from quant.strategies.news import assign_sessions, lexicon_score

    assert lexicon_score("Apple beats estimates, raises guidance") > 0.5
    assert lexicon_score("Boeing downgraded after FAA probe") < -0.3
    assert lexicon_score("Microsoft to hold annual meeting") == 0
    sessions = pd.bdate_range("2026-11-02", "2026-11-10")
    news = pd.DataFrame({"created_at": ["2026-11-03T13:00:00Z",    # 08:00 ET Tue -> Tue (before 09:15)
                                        "2026-11-03T15:00:00Z",    # 10:00 ET Tue -> Wed for open variant
                                        "2026-11-06T22:00:00Z"]})  # Fri 17:00 ET -> Mon
    s_open = assign_sessions(news, sessions, "09:15")
    assert list(s_open.dt.day) == [3, 4, 9]
    s_close = assign_sessions(news, sessions, "15:30")
    assert list(s_close.dt.day) == [3, 3, 9]


def test_close_variant_has_no_lookahead():
    from quant.strategies.news import sentiment_returns

    idx = pd.bdate_range("2026-01-05", periods=6)
    closes = pd.DataFrame({"AAA": [100, 110, 110, 110, 110, 110.0], "SPY": [100.0] * 6}, idx)
    scores = pd.DataFrame({"AAA": [np.nan, 0.9, np.nan, np.nan, np.nan, np.nan]}, idx)
    r = sentiment_returns(scores, closes, closes, "close", threshold=0.3, hold=1, cost_bps=0.0)
    assert r["gross"].abs().sum() == 0      # the +10% happened ON the signal day, before the close entry


# --- opening-range breakout ---------------------------------------------------------------
def _minutes(highs, lows, closes, opens=None):
    idx = pd.date_range("2026-01-05 09:35", periods=len(highs), freq="min", tz="America/New_York")
    return pd.DataFrame({"open": opens or closes, "high": highs, "low": lows, "close": closes}, idx)


def test_orb_long_breakout_and_stop():
    from quant.strategies.orb import ORBConfig, simulate_day

    f5 = pd.DataFrame({"open": [10.0], "high": [10.5], "low": [9.9], "close": [10.4], "rv": [3.0]}, index=["AAA"])
    feats = pd.DataFrame({"atr": [1.0]}, index=["AAA"])
    win = _minutes([10.4, 10.6, 11.0, 11.2], [10.3, 10.46, 10.8, 11.0], [10.4, 10.55, 10.9, 11.1], [10.4, 10.45, 10.85, 11.05])
    r, tr = simulate_day(f5, {"AAA": win}, feats, ORBConfig(cost_bps=0.0))
    assert tr[0]["how"] == "close" and tr[0]["entry"] == 10.5 and tr[0]["exit"] == 11.1 and r > 0
    lose = _minutes([10.4, 10.6, 10.5], [10.3, 10.35, 10.2], [10.4, 10.45, 10.3], [10.4, 10.45, 10.4])
    _, tr = simulate_day(f5, {"AAA": lose}, feats, ORBConfig(cost_bps=0.0))
    assert tr[0]["how"] == "stop" and tr[0]["exit"] == pytest.approx(10.4)   # same bar -> stop (conservative)
    doji = f5.assign(close=10.0)
    assert simulate_day(doji, {"AAA": win}, feats, ORBConfig())[1] == []
