"""Supervised ``trade_updates`` stream.

alpaca-py's TradingStream reconnects by itself, but any fill/cancel/reject that happened
while the socket was down is never replayed. ``start_trade_updates`` therefore calls
``on_connect(is_reconnect)`` after every successful (re)connect + auth, so the caller can
re-sync positions and backfill missed order events from the REST API before it trades.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)
TERMINAL = {"fill", "filled", "canceled", "rejected", "expired", "done_for_day", "replaced"}


def enum_val(x):
    """'filled' from OrderStatus.FILLED, plain values unchanged."""
    v = getattr(x, "value", x)
    return None if v is None else str(v)


def trade_update_record(data) -> dict:
    o = data.order
    return {
        "event": enum_val(data.event), "symbol": o.symbol, "side": enum_val(o.side), "qty": enum_val(o.qty),
        "filled_qty": enum_val(o.filled_qty), "filled_avg_price": enum_val(o.filled_avg_price),
        "fill_price": enum_val(getattr(data, "price", None)), "fill_qty": enum_val(getattr(data, "qty", None)),
        "client_order_id": o.client_order_id, "order_id": str(o.id), "status": enum_val(o.status),
        "event_ts": str(getattr(data, "timestamp", "") or ""),
    }


def make_supervised_stream_class():
    from alpaca.trading.stream import TradingStream

    class SupervisedTradingStream(TradingStream):
        """TradingStream that reports every (re)connection."""

        def __init__(self, *a, on_connect: Callable[[bool], None] | None = None, **kw):
            super().__init__(*a, **kw)
            self.connects = 0
            self._on_connect = on_connect

        async def _start_ws(self):  # called by alpaca-py on every connection attempt
            await super()._start_ws()
            self.connects += 1
            if self._on_connect is not None:
                is_reconnect = self.connects > 1
                loop = asyncio.get_running_loop()
                try:
                    await loop.run_in_executor(None, self._on_connect, is_reconnect)
                except Exception:  # never let the callback kill the stream
                    logger.exception("on_connect callback failed")

    return SupervisedTradingStream


def start_trade_updates(settings, journal: Path, on_event: Callable[[dict], None] | None = None,
                        on_connect: Callable[[bool], None] | None = None):
    """Run the trade_updates websocket in a daemon thread. Returns the stream (or None)."""
    try:
        cls = make_supervised_stream_class()
    except Exception as e:  # pragma: no cover - alpaca-py missing
        logger.warning("TradingStream unavailable: %s", e)
        return None
    c = settings.alpaca
    stream = cls(c.key_id, c.secret, paper=c.paper, on_connect=on_connect)

    async def handler(data):
        rec = trade_update_record(data)
        logger.info("trade_update %s", rec)
        with Path(journal).open("a") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "event_type": "trade_update",
                                **rec}) + "\n")
        if on_event is not None:
            try:
                on_event(rec)
            except Exception:
                logger.exception("trade update callback failed")

    stream.subscribe_trade_updates(handler)
    threading.Thread(target=stream.run, name="trade-updates", daemon=True).start()
    return stream
