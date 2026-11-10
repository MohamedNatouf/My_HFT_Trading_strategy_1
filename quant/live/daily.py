"""Daily rebalancer (close or open auction).

``execution.session = "close"`` (default) - once per session, ``run_minutes_before_close``
minutes before the close (default 20 -> 15:40 ET, before Alpaca's MOC cutoff):

1. re-sync: positions, open orders and any order events missed while the stream was down
2. kill switch: if today's P&L < -max_daily_loss -> flatten, write HALT file, alert
3. load adjusted daily closes (fresh SIP history), append today's latest trade as a
   provisional close (stale prices -> alert; mostly stale -> abort) and the NEXT session
   date (Alpaca calendar) as an empty row, so period-end rebalancing knows whether today
   is the last session of the week/month without seeing any future price
4. same strategy code as the backtest -> risk limits (caps, group caps, vol target,
   turnover budget measured against live holdings) -> reconcile against broker positions
5. idempotent orders (MOC, whole shares) or a dry-run log; errors/rejections -> alert

``execution.session = "open"`` - runs ``run_minutes_before_open`` (default 45) minutes
before the open, decides on the previous session's official close and submits
opening-auction (OPG) orders. Alpaca accepts OPG orders between 19:00 and 09:28 ET.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Settings
from ..data import load_daily_closes
from ..execution import AlpacaBroker, RebalanceConfig, Rebalancer
from ..ops.alerts import Alerter
from ..portfolio import RiskLimits, apply_risk, combine_sleeves
from ..strategies import build_daily
from .stream import TERMINAL, enum_val, start_trade_updates

logger = logging.getLogger(__name__)
NY = "America/New_York"
ALERT_EVENTS = {"rejected", "expired"}


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


class StaleDataError(RuntimeError):
    pass


def check_staleness(trades: dict, symbols: list[str], now: pd.Timestamp, max_age_min: float) -> list[str]:
    stale = []
    for s in symbols:
        if s not in trades:
            stale.append(s)
            continue
        ts = pd.Timestamp(trades[s][1])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        if (now - ts).total_seconds() / 60 > max_age_min:
            stale.append(s)
    return stale


def compute_targets(settings: Settings, broker, decision_day: date, next_session: date,
                    provisional: bool = True, current_weights: dict | None = None,
                    alerter: Alerter | None = None, prices: pd.DataFrame | None = None) -> pd.Series:
    """Target weights decided at the close of ``decision_day``.

    provisional=True: today's close is not known yet -> use the latest trade (close session).
    provisional=False: ``decision_day``'s official close must be in the history (open session).
    """
    cfg = settings.raw
    sleeves = build_sleeves(cfg)
    symbols = sorted({s for st, _ in sleeves for s in st.symbols})
    warm = max(st.warmup_days for st, _ in sleeves)
    day = pd.Timestamp(decision_day)
    if prices is None:
        start = day - pd.Timedelta(days=int(warm * 1.6) + 30)
        end = day - pd.Timedelta(days=1) if provisional else day + pd.Timedelta(days=1)
        prices = load_daily_closes(symbols, start, end, source="alpaca", settings=settings, refresh=True)
    hist = prices[prices.index < day] if provisional else prices[prices.index <= day]
    if provisional:
        ctl = cfg.get("risk_controls") or {}
        trades = broker.latest_trades(symbols)
        stale = check_staleness(trades, symbols, pd.Timestamp.now(tz="UTC"), float(ctl.get("stale_minutes", 30)))
        if stale and alerter is not None:
            alerter.send("Stale prices", f"{len(stale)}/{len(symbols)} symbols without a recent trade: {stale}",
                         level="warning", key=f"stale-{decision_day}")
        if len(stale) > float(ctl.get("max_stale_fraction", 0.5)) * len(symbols):
            raise StaleDataError(f"{len(stale)}/{len(symbols)} prices stale - refusing to trade")
        hist.loc[day] = pd.Series({s: p for s, (p, _) in trades.items()})
    elif hist.empty or hist.index[-1] != day:
        raise StaleDataError(f"official close for {decision_day} not available yet (last={hist.index[-1] if len(hist) else None})")
    hist = hist.reindex(columns=symbols)
    hist.loc[pd.Timestamp(next_session)] = np.nan          # calendar-only row
    targets = combine_sleeves(hist, sleeves)
    targets = apply_risk(hist, targets, RiskLimits(**(cfg.get("risk") or {})),
                         current_weights=current_weights, current_at=day)
    return targets.loc[day]


# backwards-compatible name
def compute_today_targets(settings, broker, today, next_session, **kw):
    return compute_targets(settings, broker, today, next_session, provisional=True, **kw)


class DailyRunner:
    def __init__(self, settings: Settings, dry_run: bool = True, state_dir: str | Path = "state",
                 broker=None, alerter: Alerter | None = None):
        self.settings = settings
        self.dry_run = dry_run
        self.broker = broker or AlpacaBroker.from_settings(settings)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "daily_runner.json"
        self.halt_file = self.state_dir / "HALT"
        self.journal = self.state_dir / "journal.jsonl"
        self.alerter = alerter or Alerter(self.state_dir)
        ex = dict(settings.raw.get("execution") or {})
        self.session = ex.pop("session", "close")
        if self.session not in ("close", "open"):
            raise ValueError("execution.session must be 'close' or 'open'")
        self.run_before_close = int(ex.pop("run_minutes_before_close", 20))
        self.run_before_open = int(ex.pop("run_minutes_before_open", 45))
        if self.session == "open":
            ex["order_style"] = "moo"
        self.max_daily_loss = float((settings.raw.get("risk_controls") or {}).get("max_daily_loss", 0.03))
        self.tag = settings.raw.get("book", "daily")
        self.rebalancer = Rebalancer(self.broker, RebalanceConfig(**ex), tag=self.tag)
        self.needs_resync = True

    # --- state -------------------------------------------------------------------
    def _state(self) -> dict:
        try:
            return json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        except json.JSONDecodeError:
            return {}

    def _save_state(self, **kw) -> None:
        st = self._state()
        st.update(kw)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=2, default=str))
        tmp.replace(self.state_file)

    def _log(self, event: str, **kw) -> None:
        with self.journal.open("a") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "event_type": event, **kw},
                               default=str) + "\n")

    # --- re-sync (item: broker is the source of truth) ------------------------------
    def _journaled_terminal(self) -> set[tuple[str, str]]:
        seen = set()
        if not self.journal.exists():
            return seen
        for line in self.journal.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event_type") in ("trade_update", "trade_update_backfill") and r.get("client_order_id"):
                seen.add((r["client_order_id"], str(r.get("status"))))
        return seen

    def resync(self, reason: str = "manual") -> dict:
        """Refresh positions/open orders from the broker and backfill order events that the
        websocket missed (e.g. during a disconnect). Raises if the broker is unreachable."""
        positions = self.broker.positions()
        open_orders = self.broker.open_orders()
        st = self._state()
        after = st.get("last_resync")
        after = pd.Timestamp(after) - pd.Timedelta(minutes=5) if after else pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)
        seen = self._journaled_terminal()
        backfilled = []
        prefix = f"q-{self.tag[:16]}-"
        for o in self.broker.recent_orders(after=after.to_pydatetime()):
            cid = getattr(o, "client_order_id", "") or ""
            status = enum_val(getattr(o, "status", None))
            if not cid.startswith(prefix) or (cid, status) in seen:
                continue
            rec = {"symbol": o.symbol, "side": enum_val(o.side), "qty": enum_val(o.qty),
                   "filled_qty": enum_val(o.filled_qty), "filled_avg_price": enum_val(o.filled_avg_price),
                   "client_order_id": cid, "order_id": str(o.id), "status": status,
                   "event_ts": str(getattr(o, "updated_at", "") or "")}
            self._log("trade_update_backfill", **rec)
            backfilled.append(rec)
            if status in ALERT_EVENTS:
                self.alerter.send(f"Order {status}: {o.symbol}", json.dumps(rec, default=str), level="warning",
                                  key=f"order-{cid}-{status}")
        snap = {"reason": reason, "positions": {s: p.qty for s, p in positions.items()},
                "open_orders": [{"symbol": getattr(o, "symbol", None), "id": str(getattr(o, "id", "")),
                                 "client_order_id": getattr(o, "client_order_id", None)} for o in open_orders],
                "backfilled": len(backfilled)}
        (self.state_dir / "positions_snapshot.json").write_text(json.dumps(snap, indent=2, default=str))
        self._save_state(last_resync=pd.Timestamp.now(tz="UTC").isoformat())
        self._log("resync", **snap)
        self.needs_resync = False
        logger.info("resync (%s): %d positions, %d open orders, %d backfilled events",
                    reason, len(positions), len(open_orders), len(backfilled))
        return snap

    def on_stream_connect(self, is_reconnect: bool) -> None:
        if is_reconnect:
            self.needs_resync = True
            self.alerter.send("Trade stream reconnected", "re-syncing positions and missed order events",
                              level="info", key="stream-reconnect")
            try:
                self.resync("stream reconnect")
            except Exception as e:
                self.alerter.send("Re-sync after reconnect failed", str(e), level="warning", key="resync-failed")

    def on_trade_update(self, rec: dict) -> None:
        if rec.get("event") in ALERT_EVENTS:
            self.alerter.send(f"Order {rec['event']}: {rec.get('symbol')}", json.dumps(rec, default=str),
                              level="warning", key=f"order-{rec.get('client_order_id')}-{rec['event']}")

    # --- controls ----------------------------------------------------------------
    def kill_switch_tripped(self) -> bool:
        if self.halt_file.exists():
            logger.error("HALT file present (%s) - refusing to trade. Delete it after review.", self.halt_file)
            self.alerter.send("Trading halted", f"{self.halt_file} present; delete it after review",
                              level="warning", key=f"halt-{date.today()}")
            return True
        acct = self.broker.account()
        eq, last_eq = float(acct.equity), float(acct.last_equity or 0)
        if last_eq > 0 and eq / last_eq - 1 < -self.max_daily_loss:
            msg = f"daily P&L {(eq / last_eq - 1):.2%} breaches -{self.max_daily_loss:.0%}; equity={eq:,.2f}"
            logger.error("Kill switch: %s - flattening", msg)
            if not self.dry_run:
                self.broker.close_all()
            self.halt_file.write_text(f"tripped {datetime.now(timezone.utc).isoformat()} equity={eq} last_equity={last_eq}\n")
            self._log("kill_switch", equity=eq, last_equity=last_eq)
            self.alerter.send("KILL SWITCH tripped", msg + (" - positions closed" if not self.dry_run else " (dry run)"),
                              level="critical", key="kill-switch", force=True)
            return True
        return False

    def _sessions(self, start: date, end: date) -> list[date]:
        return [pd.Timestamp(c.date).date() for c in self.broker.calendar(start, end)]

    def _next_session(self, day: date) -> date:
        return self._sessions(day + timedelta(days=1), day + timedelta(days=10))[0]

    def _prev_session(self, day: date) -> date:
        return [d for d in self._sessions(day - timedelta(days=10), day) if d < day][-1]

    def current_weights(self, managed: set[str]) -> dict[str, float]:
        book = self.broker.equity() * self.rebalancer.cfg.allocation_fraction
        if book <= 0:
            return {}
        return {s: p.market_value / book for s, p in self.broker.positions().items() if s in managed}

    # --- one rebalance -----------------------------------------------------------
    def run_once(self, exec_session: date | None = None) -> list[dict]:
        """Rebalance for execution session ``exec_session`` (default: today for close mode,
        the next session for open mode)."""
        now_ny = pd.Timestamp.now(tz=NY)
        today = now_ny.date()
        if self.session == "close":
            exec_day = exec_session or today
            decision_day, nxt, provisional = exec_day, self._next_session(exec_day), True
        else:
            exec_day = exec_session or (today if now_ny.hour < 10 and self._sessions(today, today) else self._next_session(today))
            decision_day, nxt, provisional = self._prev_session(exec_day), exec_day, False
        self.resync("pre-trade")
        if self.kill_switch_tripped():
            return []
        sleeves = build_sleeves(self.settings.raw)
        managed = {s for st, _ in sleeves for s in st.symbols}
        cw = self.current_weights(managed)
        try:
            targets = compute_targets(self.settings, self.broker, decision_day, nxt, provisional=provisional,
                                      current_weights=cw, alerter=self.alerter)
        except StaleDataError as e:
            self.alerter.send("Rebalance aborted: stale data", str(e), level="critical", key=f"stale-abort-{exec_day}")
            raise
        if targets.isna().all():
            logger.info("No sleeve rebalances for %s", exec_day)
            self._save_state(last_run=str(exec_day))
            return []
        targets = targets.fillna(0.0)
        managed |= set(targets.index)
        logger.info("Targets %s (decided %s): %s", exec_day, decision_day, {k: round(v, 4) for k, v in targets.items() if v})
        plan = self.rebalancer.plan(targets.to_dict(), managed, day=exec_day.strftime("%Y%m%d"))
        for s, why in plan.skipped:
            logger.info("skip %s: %s", s, why)
        results = self.rebalancer.execute(plan, dry_run=self.dry_run)
        errors = [r for r in results if str(r.get("status", "")).startswith(("error", "skipped_no_buying_power"))]
        for r in results:
            logger.info("order %s", r)
        if errors:
            self.alerter.send(f"{len(errors)} order(s) failed", json.dumps(errors, default=str)[:1500],
                              level="critical", key=f"order-errors-{exec_day}")
        self._log("rebalance", day=str(exec_day), decision_day=str(decision_day), session=self.session,
                  targets=targets.to_dict(), current_weights=cw, orders=results, dry_run=self.dry_run)
        self._save_state(last_run=str(exec_day))
        return results

    # --- scheduler ---------------------------------------------------------------
    def _due(self, clock) -> date | None:
        """Execution session that should be traded now, or None."""
        now = pd.Timestamp(clock.timestamp)
        last = self._state().get("last_run")
        if self.session == "close":
            if not clock.is_open:
                return None
            to_close = (pd.Timestamp(clock.next_close) - now).total_seconds() / 60
            today = now.tz_convert(NY).date()
            return today if to_close <= self.run_before_close and last != str(today) else None
        if clock.is_open:
            return None
        to_open = (pd.Timestamp(clock.next_open) - now).total_seconds() / 60
        sess = pd.Timestamp(clock.next_open).tz_convert(NY).date()
        return sess if 3 <= to_open <= self.run_before_open and last != str(sess) else None

    def check_missed_run(self, clock) -> str | None:
        """Alert if a session that should have been traded was not."""
        now = pd.Timestamp(clock.timestamp).tz_convert(NY)
        today = now.date()
        last = self._state().get("last_run")
        if self.session == "close":
            if clock.is_open or not self._sessions(today, today):
                return None
            close_t = _session_close(self.broker.calendar(today, today)[0], today)
            missed = now > close_t + pd.Timedelta(minutes=5) and last != str(today)
        else:
            missed = clock.is_open and last != str(today) and bool(self._sessions(today, today))
        if missed:
            msg = f"no {self.session}-auction rebalance recorded for {today} (last_run={last})"
            self.alerter.send("Missed rebalance", msg, level="critical", key=f"missed-{today}")
            return msg
        return None

    def run_forever(self) -> None:
        stream = start_trade_updates(self.settings, self.journal, on_event=self.on_trade_update,
                                     on_connect=self.on_stream_connect)
        if stream is None:
            self.alerter.send("Trade stream unavailable", "fills will only be backfilled at re-sync", level="warning")
        failures: dict[str, int] = {}
        try:
            while True:
                try:
                    clock = self.broker.clock()
                    self._save_state(heartbeat=pd.Timestamp.now(tz="UTC").isoformat(), session=self.session)
                    due = self._due(clock)
                    if due is not None and failures.get(str(due), 0) < 3:
                        now = pd.Timestamp(clock.timestamp)
                        if self.session == "close":
                            to_close = (pd.Timestamp(clock.next_close) - now).total_seconds() / 60
                            if to_close < 11:
                                logger.warning("Too close to the close (%.1f min) for MOC; switching to limit", to_close)
                                self.rebalancer.cfg.order_style = "limit"
                        try:
                            self.run_once(due)
                        except Exception as e:
                            failures[str(due)] = failures.get(str(due), 0) + 1
                            logger.exception("rebalance failed")
                            self.alerter.send("Rebalance crashed", f"{type(e).__name__}: {e} "
                                              f"(attempt {failures[str(due)]}/3)", level="critical",
                                              key=f"crash-{due}-{failures[str(due)]}")
                    else:
                        self.check_missed_run(clock)
                    if self.needs_resync and clock.is_open:
                        self.resync("pending")
                    sleep = 30.0
                    if not clock.is_open:
                        to_open = (pd.Timestamp(clock.next_open) - pd.Timestamp(clock.timestamp)).total_seconds()
                        sleep = 30.0 if self.session == "open" and to_open < (self.run_before_open + 5) * 60 \
                            else min(900.0, max(30.0, to_open - (self.run_before_open + 5) * 60))
                    time.sleep(sleep)
                except KeyboardInterrupt:
                    raise
                except Exception as e:  # network blips: alert (deduped) and keep going
                    logger.exception("scheduler loop error")
                    self.alerter.send("Scheduler error", f"{type(e).__name__}: {e}", level="warning", key="loop-error")
                    time.sleep(60)
        finally:
            if stream is not None:
                stream.stop()


def _session_close(cal, day: date) -> pd.Timestamp:
    """Session close as a NY timestamp from an alpaca Calendar row (datetime or 'HH:MM')."""
    c = getattr(cal, "close", None)
    if isinstance(c, datetime):
        t = pd.Timestamp(c)
    else:
        t = pd.Timestamp(f"{day} {str(c or '16:00')[:5]}")
    return t.tz_localize(NY) if t.tzinfo is None else t.tz_convert(NY)


def watchdog(settings: Settings, state_dir: str | Path = "state", broker=None, alerter: Alerter | None = None,
             max_heartbeat_age_min: float = 10.0) -> list[str]:
    """Out-of-process check (schedule every 15 min with Windows Task Scheduler / cron):
    runner heartbeat, missed rebalances and a HALT file. Returns the problems found."""
    runner = DailyRunner(settings, dry_run=True, state_dir=state_dir, broker=broker, alerter=alerter)
    problems = []
    clock = runner.broker.clock()
    st = runner._state()
    hb = st.get("heartbeat")
    if clock.is_open or runner.session == "open":
        age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(hb)).total_seconds() / 60 if hb else float("inf")
        if age > max_heartbeat_age_min:
            msg = f"daily runner heartbeat is {age:.0f} min old (last={hb})"
            runner.alerter.send("Daily runner not running", msg, level="critical", key="heartbeat")
            problems.append(msg)
    m = runner.check_missed_run(clock)
    if m:
        problems.append(m)
    if runner.halt_file.exists():
        problems.append(f"HALT file present: {runner.halt_file.read_text().strip()}")
        runner.alerter.send("Trading halted", problems[-1], level="warning", key=f"halt-{date.today()}")
    return problems
