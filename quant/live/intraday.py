"""Live runner for the noise-area intraday momentum strategy (one symbol, e.g. SPY).

* history: prior 14+ sessions of SIP minute bars (free plan OK: data is > 15 min old)
* today: real-time 1-minute bars from StockDataStream (feed from ALPACA_DATA_FEED;
  IEX on the free plan - its VWAP/volume differ slightly from the consolidated tape)
* decisions on the bars closing at 10:00, 10:30 ... 15:30 using the SAME ``noise_step``
  function as the backtest; target shares = position * leverage * book / price
* flat by ``flatten_minute`` (default 15:55) via close_position
"""
from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import Settings
from ..data import load_minute_bars
from ..execution import AlpacaBroker, OrderSpec, make_client_order_id
from ..strategies.intraday import DECISION_MINUTES, NoiseAreaMomentum, noise_step

logger = logging.getLogger(__name__)
NY = "America/New_York"


class NoiseAreaRunner:
    def __init__(self, settings: Settings, symbol: str = "SPY", dry_run: bool = True,
                 allocation_fraction: float = 1.0, flatten_minute: int = 385, **strategy_params):
        self.settings = settings
        self.symbol = symbol
        self.dry_run = dry_run
        self.alloc = allocation_fraction
        self.flatten_minute = flatten_minute
        self.strategy = NoiseAreaMomentum(**strategy_params)
        self.broker = AlpacaBroker.from_settings(settings)
        self.bands = None
        self.pos = 0
        self.bars: list[dict] = []
        self.flattened = False
        self.done_decisions: set[int] = set()

    def prepare_day(self, today_open: float) -> None:
        end = pd.Timestamp.now(tz=NY).normalize().tz_convert("UTC")
        start = end - pd.Timedelta(days=40)
        hist = load_minute_bars(self.symbol, start, end, settings=self.settings)
        self.bands = self.strategy.bands_for_today(hist, today_open)
        logger.info("bands ready: leverage=%.2f prev_close=%.2f", self.bands["leverage"], self.bands["prev_close"])

    def _vwap(self) -> float:
        df = pd.DataFrame(self.bars)
        tp = (df.high + df.low + df.close) / 3
        v = df.volume.sum()
        return float((tp * df.volume).sum() / v) if v > 0 else float("nan")

    def _target_shares(self, price: float) -> float:
        book = self.broker.equity() * self.alloc
        return self.pos * self.bands["leverage"] * book / price

    def _rebalance_to(self, target_qty: float, minute: int, price: float) -> None:
        cur = self.broker.positions().get(self.symbol)
        cur_qty = cur.qty if cur else 0.0
        delta = math.trunc(target_qty - cur_qty)   # whole shares (shorts cannot be fractional)
        if delta == 0:
            return
        side = "buy" if delta > 0 else "sell"
        day = pd.Timestamp.now(tz=NY).strftime("%Y%m%d")
        spec = OrderSpec(self.symbol, side, abs(delta), make_client_order_id("noise", f"{day}{minute:03d}", self.symbol, side),
                         order_type="market", tif="day", est_notional=abs(delta) * price, reason=f"pos={self.pos}")
        logger.info("noise order %s", spec)
        if not self.dry_run and not self.broker.order_exists(spec.client_order_id):
            self.broker.submit(spec)

    async def on_bar(self, bar) -> None:
        ts = pd.Timestamp(bar.timestamp).tz_convert(NY)
        m = ts.hour * 60 + ts.minute - 570
        if m < 0 or m >= 390:
            return
        rec = {"m": m, "open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close, "volume": bar.volume}
        self.bars.append(rec)
        if self.bands is None:
            await asyncio.to_thread(self.prepare_day, float(bar.open))
        if m >= self.flatten_minute and not self.flattened:
            self.pos = 0
            self.flattened = True
            logger.info("flatten at %s", ts)
            if not self.dry_run:
                try:
                    self.broker.close_position(self.symbol)
                except Exception as e:
                    logger.info("close_position: %s", e)
            return
        # decide on the first bar at/after each decision minute (a minute with no prints
        # produces no bar, so never require an exact match)
        due = [d for d in DECISION_MINUTES if d <= m and d not in self.done_decisions]
        if due and not self.flattened:
            self.done_decisions.update(due)   # bands evaluated at the current bar's minute m
            new = noise_step(self.pos, float(bar.close), float(self.bands["ub"][m]), float(self.bands["lb"][m]), self._vwap())
            if not self.strategy.allow_short and new < 0:
                new = 0
            logger.info("decision %s m=%d close=%.2f ub=%.2f lb=%.2f vwap=%.2f pos %d -> %d", ts.time(), m, bar.close,
                        self.bands["ub"][m], self.bands["lb"][m], self._vwap(), self.pos, new)
            self.pos = new
            await asyncio.to_thread(self._rebalance_to, self._target_shares(float(bar.close)), m, float(bar.close))

    def run(self) -> None:
        from alpaca.data.enums import DataFeed
        from alpaca.data.live import StockDataStream

        c = self.settings.alpaca
        stream = StockDataStream(c.key_id, c.secret, feed=DataFeed(c.data_feed))
        stream.subscribe_bars(self.on_bar, self.symbol)
        logger.info("noise-area runner started for %s (dry_run=%s)", self.symbol, self.dry_run)
        stream.run()
