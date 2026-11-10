"""Daily close rebalancer.

Once per session, ``run_minutes_before_close`` minutes before the close (default 20, i.e.
15:40 ET, safely before Alpaca's market-on-close cutoff):

1. kill switch: if today's P&L < -max_daily_loss -> flatten managed book, write HALT file
2. load adjusted daily closes (Alpaca SIP history), append today's latest trade as a
   provisional close, and append the NEXT session date (from the Alpaca calendar) as an
   empty row so period-end rebalancing knows whether today is the last session of the
   week/month - with no access to future prices
3. compute sleeve targets with the exact same strategy code used in backtests,
   apply risk limits, and reconcile against broker positions
4. submit idempotent orders (MOC by default; whole shares) or log them in dry-run
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Settings
from ..data import load_daily_closes
from ..execution import AlpacaBroker, RebalanceConfig, Rebalancer
from ..portfolio import RiskLimits, apply_risk, combine_sleeves
from ..strategies import build_daily

logger = logging.getLogger(__name__)
NY = "America/New_York"


def build_sleeves(cfg: dict):
    sleeves = []
    for s in cfg.get("sleeves", []):
        strat = build_daily(s["strategy"], **(s.get("params") or {}))
        sleeves.append((strat, float(s.get("weight", 1.0))))
    if not sleeves:
        raise ValueError("config has no sleeves")
    total = sum(w for _, w in sleeves)
    if total > 1.0 + 1e-9:
        raise ValueError(f"sleeve weights sum to {total:.2f} > 1")
    return sleeves


def compute_today_targets(settings: Settings, broker: AlpacaBroker, today: date, next_session: date) -> pd.Series:
    cfg = settings.raw
    sleeves = build_sleeves(cfg)
    symbols = sorted({s for st, _ in sleeves for s in st.symbols})
    warm = max(st.warmup_days for st, _ in sleeves)
    start = pd.Timestamp(today) - pd.Timedelta(days=int(warm * 1.6) + 30)
    hist = load_daily_closes(symbols, start, pd.Timestamp(today) - pd.Timedelta(days=1), source="alpaca", settings=settings)
    hist = hist[hist.index < pd.Timestamp(today)]
    last = broker.latest_prices(symbols)
    hist.loc[pd.Timestamp(today)] = pd.Series(last)
    hist.loc[pd.Timestamp(next_session)] = np.nan          # calendar-only row
    targets = combine_sleeves(hist, sleeves)
    targets = apply_risk(hist, targets, RiskLimits(**(cfg.get("risk") or {})))
    row = targets.loc[pd.Timestamp(today)]
    return row


class DailyRunner:
    def __init__(self, settings: Settings, dry_run: bool = True, state_dir: str | Path = "state"):
        self.settings = settings
        self.dry_run = dry_run
        self.broker = AlpacaBroker.from_settings(settings)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "daily_runner.json"
        self.halt_file = self.state_dir / "HALT"
        self.journal = self.state_dir / "journal.jsonl"
        ex = dict(settings.raw.get("execution") or {})
        self.run_before_close = int(ex.pop("run_minutes_before_close", 20))
        self.max_daily_loss = float((settings.raw.get("risk_controls") or {}).get("max_daily_loss", 0.03))
        self.rebalancer = Rebalancer(self.broker, RebalanceConfig(**ex), tag=settings.raw.get("book", "daily"))

    # --- state -------------------------------------------------------------------
    def _state(self) -> dict:
        return json.loads(self.state_file.read_text()) if self.state_file.exists() else {}

    def _save_state(self, **kw) -> None:
        st = self._state()
        st.update(kw)
        self.state_file.write_text(json.dumps(st, indent=2, default=str))

    def _log(self, event: str, **kw) -> None:
        with self.journal.open("a") as f:
            f.write(json.dumps({"ts": datetime.utcnow().isoformat(), "event": event, **kw}, default=str) + "\n")

    # --- controls ----------------------------------------------------------------
    def kill_switch_tripped(self) -> bool:
        if self.halt_file.exists():
            logger.error("HALT file present (%s) - refusing to trade. Delete it after review.", self.halt_file)
            return True
        acct = self.broker.account()
        eq, last_eq = float(acct.equity), float(acct.last_equity or 0)
        if last_eq > 0 and eq / last_eq - 1 < -self.max_daily_loss:
            logger.error("Daily loss %.2f%% breaches limit %.2f%% - flattening", (eq / last_eq - 1) * 100, self.max_daily_loss * 100)
            if not self.dry_run:
                self.broker.close_all()
            self.halt_file.write_text(f"tripped {datetime.utcnow().isoformat()} equity={eq} last_equity={last_eq}\n")
            self._log("kill_switch", equity=eq, last_equity=last_eq)
            return True
        return False

    def _next_session(self, today: date) -> date:
        cal = self.broker.calendar(today + timedelta(days=1), today + timedelta(days=10))
        return pd.Timestamp(cal[0].date).date()

    def run_once(self) -> list[dict]:
        today = pd.Timestamp.now(tz=NY).date()
        if self.kill_switch_tripped():
            return []
        nxt = self._next_session(today)
        targets = compute_today_targets(self.settings, self.broker, today, nxt)
        if targets.isna().all():
            logger.info("No sleeve rebalances today (%s)", today)
            self._save_state(last_run=str(today))
            return []
        targets = targets.fillna(0.0)
        managed = set(targets.index)
        logger.info("Targets %s: %s", today, {k: round(v, 4) for k, v in targets.items() if v})
        plan = self.rebalancer.plan(targets.to_dict(), managed, day=today.strftime("%Y%m%d"))
        for s, why in plan.skipped:
            logger.info("skip %s: %s", s, why)
        results = self.rebalancer.execute(plan, dry_run=self.dry_run)
        for r in results:
            logger.info("order %s", r)
        self._log("rebalance", day=str(today), targets=targets.to_dict(), orders=results, dry_run=self.dry_run)
        self._save_state(last_run=str(today))
        return results

    def run_forever(self) -> None:
        stream = start_trade_updates_logger(self.settings, self.journal)
        try:
            while True:
                clock = self.broker.clock()
                now = pd.Timestamp(clock.timestamp)
                if not clock.is_open:
                    wait = min(1800.0, max(30.0, (pd.Timestamp(clock.next_open) - now).total_seconds()))
                    logger.info("Market closed; next open %s; sleeping %.0fs", clock.next_open, wait)
                    time.sleep(wait)
                    continue
                to_close = (pd.Timestamp(clock.next_close) - now).total_seconds() / 60
                today = now.tz_convert(NY).date()
                if to_close <= self.run_before_close and self._state().get("last_run") != str(today):
                    if to_close < 11:
                        logger.warning("Too close to the close (%.1f min) for MOC orders; switching to limit", to_close)
                        self.rebalancer.cfg.order_style = "limit"
                    self.run_once()
                time.sleep(30)
        finally:
            if stream is not None:
                stream.stop()


def start_trade_updates_logger(settings: Settings, journal: Path):
    """Log order lifecycle events (fill, partial_fill, canceled, rejected...) from the
    Trading API ``trade_updates`` websocket in a background thread."""
    try:
        import threading

        from alpaca.trading.stream import TradingStream
    except Exception as e:  # pragma: no cover
        logger.warning("TradingStream unavailable: %s", e)
        return None
    c = settings.alpaca
    stream = TradingStream(c.key_id, c.secret, paper=c.paper)

    async def on_update(data):
        o = data.order
        rec = {"event": str(data.event), "symbol": o.symbol, "side": str(o.side), "qty": str(o.qty),
               "filled_qty": str(o.filled_qty), "filled_avg_price": str(o.filled_avg_price),
               "client_order_id": o.client_order_id, "status": str(o.status)}
        logger.info("trade_update %s", rec)
        with journal.open("a") as f:
            f.write(json.dumps({"ts": datetime.utcnow().isoformat(), "event": "trade_update", **rec}) + "\n")

    stream.subscribe_trade_updates(on_update)
    threading.Thread(target=stream.run, name="trade-updates", daemon=True).start()
    return stream
