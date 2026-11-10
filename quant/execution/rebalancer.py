"""Reconciling, idempotent rebalancer.

target weights + live broker positions -> minimal set of orders.

Guarantees:
* The broker is the source of truth for positions (never an in-memory guess).
* Only symbols in ``managed`` are ever traded (other holdings in the account are left alone).
* Small deltas below ``min_trade_notional`` / ``drift_band`` are skipped (turnover control).
* Sells are planned before buys; buys are scaled down to available buying power.
* Every order has a deterministic ``client_order_id`` derived from (tag, day, symbol, side),
  so re-running the same rebalance (crash, restart, duplicate trigger) cannot double-trade.
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field

from .broker import Broker, OrderSpec

logger = logging.getLogger(__name__)


def make_client_order_id(tag: str, day: str, symbol: str, side: str) -> str:
    h = hashlib.sha1(f"{tag}|{day}|{symbol}|{side}".encode()).hexdigest()[:10]
    return f"q-{tag[:16]}-{day}-{symbol}-{side[0]}-{h}"[:128]


@dataclass
class RebalanceConfig:
    allocation_fraction: float = 1.0     # share of account equity this book may use
    min_trade_notional: float = 25.0
    drift_band: float = 0.01             # skip if |delta| < drift_band * book equity
    order_style: str = "limit"           # 'market' | 'limit' | 'moc'
    limit_slippage_bps: float = 10.0     # marketable-limit offset through the quote
    max_order_notional: float | None = None
    whole_shares_only: bool = False


@dataclass
class RebalancePlan:
    orders: list[OrderSpec] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    book_equity: float = 0.0


class Rebalancer:
    def __init__(self, broker: Broker, cfg: RebalanceConfig | None = None, tag: str = "book"):
        self.broker = broker
        self.cfg = cfg or RebalanceConfig()
        self.tag = tag

    def plan(self, targets: dict[str, float], managed: set[str], day: str,
             prices: dict[str, float] | None = None) -> RebalancePlan:
        cfg = self.cfg
        equity = self.broker.equity()
        book = equity * cfg.allocation_fraction
        positions = {s: p for s, p in self.broker.positions().items() if s in managed}
        symbols = sorted(set(targets) | set(positions))
        px = dict(prices or {})
        missing = [s for s in symbols if not px.get(s)]
        if missing:
            px.update(self.broker.latest_prices(missing))
        plan = RebalancePlan(book_equity=book)
        whole = cfg.whole_shares_only or cfg.order_style == "moc"
        for s in symbols:
            price = px.get(s)
            if not price or not math.isfinite(price) or price <= 0:
                plan.skipped.append((s, "no price"))
                continue
            w = float(targets.get(s, 0.0) or 0.0)
            cur_qty = positions[s].qty if s in positions else 0.0
            if w == 0.0 and cur_qty == 0.0:
                continue  # nothing held, nothing wanted
            target_val = w * book
            cur_val = cur_qty * price
            delta_val = target_val - cur_val
            full_exit = w == 0.0 and cur_qty != 0.0
            if not full_exit and abs(delta_val) < max(cfg.min_trade_notional, cfg.drift_band * book):
                plan.skipped.append((s, f"delta ${delta_val:,.2f} below threshold"))
                continue
            if cfg.max_order_notional:
                delta_val = math.copysign(min(abs(delta_val), cfg.max_order_notional), delta_val)
            qty = abs(cur_qty) if full_exit else abs(delta_val) / price
            frac_ok = (not whole) and self.broker.fractionable(s)
            if not frac_ok and not full_exit:
                qty = math.floor(qty)
            if full_exit and not frac_ok and abs(cur_qty) != int(abs(cur_qty)) and cfg.order_style == "moc":
                qty = math.floor(qty)  # fractional remainder must go via a day order later
            if qty <= 0:
                plan.skipped.append((s, "rounds to zero shares"))
                continue
            side = "sell" if (full_exit and cur_qty > 0) or (not full_exit and delta_val < 0) else "buy"
            plan.orders.append(OrderSpec(
                symbol=s, side=side, qty=round(qty, 9),
                client_order_id=make_client_order_id(self.tag, day, s, side),
                tif="cls" if cfg.order_style == "moc" else "day",
                order_type="limit" if cfg.order_style == "limit" else "market",
                est_notional=qty * price,
                reason="exit" if full_exit else f"target {w:.2%}",
            ))
        plan.orders.sort(key=lambda o: 0 if o.side == "sell" else 1)
        return plan

    def _price_limits(self, plan: RebalancePlan) -> None:
        lim = [o for o in plan.orders if o.order_type == "limit"]
        if not lim:
            return
        quotes = self.broker.latest_quotes([o.symbol for o in lim])
        off = self.cfg.limit_slippage_bps / 1e4
        for o in lim:
            q = quotes.get(o.symbol)
            if q is None or q.mid <= 0:
                o.order_type, o.limit_price = "market", None
                continue
            ref = q.ask if (o.side == "buy" and q.ask > 0) else (q.bid if q.bid > 0 else q.mid)
            o.limit_price = ref * (1 + off) if o.side == "buy" else ref * (1 - off)

    def execute(self, plan: RebalancePlan, dry_run: bool = True) -> list[dict]:
        self._price_limits(plan)
        # cancel stale open orders on managed symbols so we never stack orders
        symbols = {o.symbol for o in plan.orders}
        if not dry_run:
            for od in self.broker.open_orders():
                if getattr(od, "symbol", None) in symbols:
                    logger.info("cancel stale open order %s %s", od.symbol, od.id)
                    self.broker.cancel_order(str(od.id))
        results = []
        bp = None
        for o in plan.orders:
            rec = {"symbol": o.symbol, "side": o.side, "qty": o.qty, "type": o.order_type, "tif": o.tif,
                   "limit": o.limit_price, "est_notional": round(o.est_notional, 2),
                   "client_order_id": o.client_order_id, "reason": o.reason}
            if dry_run:
                rec["status"] = "dry_run"
                results.append(rec)
                continue
            if self.broker.order_exists(o.client_order_id):
                rec["status"] = "duplicate_skipped"
                results.append(rec)
                continue
            if o.side == "buy":
                bp = self.broker.buying_power() if bp is None else bp
                if o.est_notional > bp:
                    if bp < self.cfg.min_trade_notional:
                        rec["status"] = "skipped_no_buying_power"
                        results.append(rec)
                        continue
                    scale = bp / o.est_notional
                    o.qty = round(o.qty * scale, 9) if o.qty != int(o.qty) else math.floor(o.qty * scale)
                    o.est_notional *= scale
                    rec["qty"] = o.qty
                bp -= o.est_notional
            try:
                resp = self.broker.submit(o)
                rec["status"] = "submitted"
                rec["order_id"] = str(getattr(resp, "id", ""))
            except Exception as e:  # keep going; log every failure explicitly
                rec["status"] = f"error: {e}"
                logger.error("order failed %s: %s", o.client_order_id, e)
            results.append(rec)
        return results
