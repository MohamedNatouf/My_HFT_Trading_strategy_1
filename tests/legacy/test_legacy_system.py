"""Smoke tests for the legacy live loop (no network: broker and execution are faked)."""
import asyncio
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

system_mod = pytest.importorskip("Trading_System.system")
from Trading_System.engine import Backtester  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SYMS = ["AAA", "BBB", "CCC", "DDD"]


@pytest.fixture
def cfg_path(tmp_path):
    cfg = json.loads((ROOT / "config" / "config.json").read_text())
    cfg["universe"] = {"models": SYMS}
    cfg["instruments"] = []
    cfg["emergency"] = {}
    cfg["strategy"]["Method_Of_Wieghting"] = "Assets Volume"
    cfg["strategy"]["rebalance_minutes"] = 5
    cfg["data"] = {"storeRoot": str(tmp_path / "store"), "marketHoursOnly": True}
    cfg["alpaca"] = {"apiKeyId": "", "apiSecretKey": "", "paper": True}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return str(p)


@pytest.fixture
def bars():
    n = 400
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("min"), periods=n, freq="min")
    rng = np.random.default_rng(1)
    close = pd.DataFrame(100 * np.exp(np.cumsum(0.001 * rng.standard_normal((n, 4)), axis=0)), index=idx, columns=SYMS)
    return pd.concat({"close": close, "open": close, "high": close, "low": close, "volume": close * 0 + 1}, axis=1)


def _system(cfg_path, positions):
    ts = system_mod.TradingSystem(cfg_path)
    calls = []
    ts.execution.rebalance = lambda alloc, pos: calls.append((alloc.date, dict(pos)))
    ts._fetch_live_positions = lambda: positions
    return ts, calls


def test_rebalances_once_per_new_allocation(cfg_path, bars):
    ts, calls = _system(cfg_path, {"AAA": 3.0})

    async def main():
        await ts._maybe_rebalance_live(bars, "t")
        await ts._maybe_rebalance_live(bars, "t")            # same bar
        await ts._maybe_rebalance_live(bars.iloc[:-1], "t")  # older bar
        nxt = bars.iloc[[-1]].set_axis([bars.index[-1] + pd.Timedelta(minutes=1)])
        await ts._maybe_rebalance_live(pd.concat([bars, nxt]), "t")  # < rebalance_minutes
    asyncio.run(main())
    assert len(calls) == 1 and calls[0][1] == {"AAA": 3.0}


def test_stale_bars_or_missing_positions_block_trading(cfg_path, bars):
    ts, calls = _system(cfg_path, {})
    asyncio.run(ts._maybe_rebalance_live(bars.set_axis(bars.index - pd.Timedelta(hours=3)), "t"))
    assert not calls
    ts, calls = _system(cfg_path, None)
    asyncio.run(ts._maybe_rebalance_live(bars, "t"))
    assert not calls


def test_report_and_flatten(cfg_path, bars):
    ts, _ = _system(cfg_path, {})
    ts.minute_df = bars
    rep = ts._compute_report(Backtester().run(ts.strategy, bars))
    assert rep.get("bars") and "turnover" in rep

    class FC:
        def cancel_orders(self):
            return [1, 2]

        def close_all_positions(self, cancel_orders=None):
            return [types.SimpleNamespace(symbol="AAA", status=200, order_id="x", body={})]

    ts.alpaca_trader.client = FC()
    assert len(asyncio.run(ts.flatten_and_exit())) == 1
    assert callable(ts.store.clear_all)
