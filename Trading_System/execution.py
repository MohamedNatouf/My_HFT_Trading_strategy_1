# Execution router using Alpaca trading client. Maps allocations to Alpaca orders.
from typing import Dict, Any, Optional
import logging
import math
import pandas as pd
from .alpaca_trading import AlpacaTrader
from .alpaca_account import AlpacaAccountClient
from .engine import Allocation

logger = logging.getLogger(__name__)


def _make_client_order_id(alloc_ts, symbol: str, side: str) -> str:
    """Deterministic per-allocation order id (<=128 chars) so the same allocation
    can never be submitted twice (Alpaca rejects duplicate client_order_id)."""
    try:
        ts = pd.Timestamp(alloc_ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert('UTC')
        ts_s = f"{ts:%Y%m%d%H%M}"
    except Exception:
        ts_s = "na"
    return f"legacy-{ts_s}-{symbol}-{side}"[:128]


class ExecutionRouter:
    def __init__(self, trader: AlpacaTrader, account_client: AlpacaAccountClient, config: Dict[str, Any]):
        self.trader = trader
        self.account = account_client
        self.cfg = config or {}
        tc = (self.cfg.get('trading') or {})
        self.default_tif = tc.get('defaultTif', 'day')
        self.extended_hours = bool(tc.get('extendedHours', False))
        self.use_notional = bool(tc.get('useNotional', False))
        self.min_buy_notional = float(tc.get('minBuyNotional', 1.0))
        self.round_qty_dp = int(tc.get('roundQtyDecimals', 9))
        self.round_notional_dp = int(tc.get('roundNotionalDecimals', 2))
        # Minimum absolute share delta required before placing an order.
        # For fractional/notional trading we default this to 0 so small
        # allocations can still generate trades.
        self.rebalance_threshold = float(tc.get('rebalanceThresholdShares', 1))
        if self.use_notional and self.rebalance_threshold >= 1.0:
            self.rebalance_threshold = 0.0
        # Minimum dollar size of an adjustment; prevents churning tiny drift trades
        # every rebalance (especially in notional mode where threshold is 0 shares).
        self.min_trade_notional = float(tc.get('minTradeNotional', max(1.0, self.min_buy_notional)))
        # Optional cap on fraction of account equity this strategy can use (0-1)
        self.max_allocation_fraction = float(tc.get('maxAllocationFraction', 1.0))
        # Optional per-symbol max portfolio weight (0-1)
        self.max_symbol_weight = float(tc.get('maxSymbolWeight', 1.0))
        # When True, only place live orders during regular US market hours.
        self.market_hours_only = bool(tc.get('marketHoursOnly', True))

    def _within_market_hours(self) -> bool:
        """Return True only if the US equity market is open now.

        Prefers Alpaca's clock (handles holidays/half-days). Falls back to a
        09:30-16:00 America/New_York weekday check. Fails CLOSED (returns
        False) if neither can be determined.
        """
        client = getattr(self.trader, 'client', None)
        if client is not None and hasattr(client, 'get_clock'):
            try:
                clock = client.get_clock()
                is_open = getattr(clock, 'is_open', None)
                if is_open is not None:
                    return bool(is_open)
            except Exception as e:
                logger.warning("Alpaca get_clock failed (%s); falling back to local RTH computation", e)
        try:
            now_ny = pd.Timestamp.now(tz='America/New_York')
            if now_ny.weekday() >= 5:  # Sat/Sun
                return False
            mins = now_ny.hour * 60 + now_ny.minute
            return (9 * 60 + 30) <= mins < (16 * 60)
        except Exception as e:
            logger.warning("Market-hours check failed (%s); failing closed (no trading)", e)
            return False

    def _cancel_open_orders(self, symbols) -> None:
        """Best-effort cancel of open Alpaca orders for the given symbols so new
        orders are not stacked on top of still-pending ones."""
        trader_client = getattr(self.trader, 'client', None)
        if trader_client is None or not symbols:
            return
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            open_orders = trader_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
        except Exception as e:
            logger.warning("Failed to list open orders for cancel: %s", e)
            return
        cancelled = 0
        failed = 0
        for ord in open_orders or []:
            sym = getattr(ord, 'symbol', None)
            oid = getattr(ord, 'id', None)
            if not sym or sym not in symbols or not oid:
                continue
            try:
                ok = self.trader.cancel_order_safe(str(oid))
            except Exception as e:
                logger.error("cancel_order_safe raised for %s: %s", oid, e)
                ok = False
            if ok:
                cancelled += 1
            else:
                failed += 1
        if cancelled or failed:
            logger.info("Pre-rebalance cancel open orders summary: cancelled=%d failed=%d", cancelled, failed)

    @staticmethod
    def _buying_power_from(acc) -> Optional[float]:
        if not acc:
            return None
        bp = float(getattr(acc, 'buying_power', 0) or 0)
        # Some paper-account states report buying_power=0 with positive cash.
        if bp <= 0:
            cash = float(getattr(acc, 'cash', 0) or 0)
            if cash > 0:
                logger.info("Using cash fallback for buying power: bp=%.2f cash=%.2f", bp, cash)
                bp = cash
        return bp

    @staticmethod
    def _log_settlement(acc) -> None:
        if not acc:
            return
        cash_wdr = float(getattr(acc, 'cash_withdrawable', 0) or 0)
        cash = float(getattr(acc, 'cash', 0) or 0)
        memoposts = float(getattr(acc, 'memoposts', 0) or 0)
        if cash_wdr < 0 and (cash + memoposts) >= 0:
            logger.info("Settlement pending (cash_withdrawable<0, cash+memoposts>=0); continuing with trade decisions")

    def rebalance(self, allocation: Allocation, current_positions: Dict[str, float]) -> Dict[str, int]:
        """Move live positions toward ``allocation``.

        ``current_positions`` must be the broker's live positions {symbol: qty}.
        Positions not in the allocation are closed; held names are adjusted by
        delta, so calling this twice with the same allocation (and positions
        already at target) submits no orders. Returns a summary dict.
        """
        summary = {"submitted": 0, "failed": 0, "skipped": 0, "closed": 0}
        current_positions = dict(current_positions or {})
        alloc_targets = [(p.symbol, p.weight) for p in allocation.positions]
        logger.info("execution rebalance ts=%s emergency=%s targets=%s", allocation.date, allocation.emergency, alloc_targets)
        if not allocation.positions:
            logger.warning("Empty allocation at ts=%s; not trading (refusing to liquidate on an empty signal)", allocation.date)
            return summary
        if self.market_hours_only and not self._within_market_hours():
            logger.info("Skipping rebalance: market closed (marketHoursOnly=True)")
            return summary

        # ONE account fetch per rebalance (equity for sizing, buying power, settlement log).
        acc = self.account.get_account() if self.account else None
        self._log_settlement(acc)
        equity = float(getattr(acc, 'equity', 0) or 0) if acc else 0.0
        target_equity = equity * self.max_allocation_fraction if equity > 0 else None
        bp_available = self._buying_power_from(acc)
        if target_equity is None:
            logger.warning("Account equity unavailable; target sizing impossible -> allocation symbols will be skipped (close-outs still processed)")
        else:
            logger.info("Rebalance sizing using equity=%.2f max_allocation_fraction=%.3f target_equity=%.2f",
                        equity, self.max_allocation_fraction, target_equity)

        alloc_syms = {p.symbol for p in allocation.positions if getattr(p, "symbol", None)}
        # Cancel stale open orders for every symbol we may touch.
        self._cancel_open_orders(alloc_syms | set(current_positions.keys()))

        staged: list = []
        # Phase 1: close positions no longer in the allocation.
        for sym, curr_qty_raw in current_positions.items():
            if sym in alloc_syms:
                continue
            try:
                curr_qty = float(curr_qty_raw or 0.0)
            except Exception:
                continue
            if not math.isfinite(curr_qty) or abs(curr_qty) <= 0.0:
                continue
            side = "sell" if curr_qty > 0 else "buy"
            logger.info("Close-out position symbol=%s qty=%.9f side=%s (not in new allocation)", sym, abs(curr_qty), side)
            staged.append({"symbol": sym, "price": None, "delta": -curr_qty, "action": "close", "side": side, "close": True})

        # Phase 2: size targets for allocation symbols.
        for pos in allocation.positions:
            price = getattr(pos, 'last_close', None)
            try:
                price = float(price) if price is not None else None
            except Exception:
                price = None
            if price is not None and (not math.isfinite(price) or price <= 0):
                price = None
            if price is None or target_equity is None:
                logger.warning("Skip symbol=%s: %s unavailable; cannot size order",
                               pos.symbol, "price" if price is None else "equity")
                summary["skipped"] += 1
                continue
            eff_weight = float(pos.weight)
            if eff_weight > self.max_symbol_weight:
                logger.info("Capping weight for %s from %.4f to %.4f based on maxSymbolWeight",
                            pos.symbol, eff_weight, self.max_symbol_weight)
                eff_weight = self.max_symbol_weight
            target_notional = eff_weight * target_equity
            target_qty = target_notional / price
            curr_qty = float(current_positions.get(pos.symbol, 0) or 0.0)
            delta = target_qty - curr_qty
            if not (math.isfinite(target_qty) and math.isfinite(delta)):
                logger.info("Skip symbol=%s due to non-finite sizing target_qty=%s delta=%s", pos.symbol, target_qty, delta)
                summary["skipped"] += 1
                continue
            if abs(delta) < self.rebalance_threshold or abs(delta) * price < self.min_trade_notional:
                logger.info("Skip symbol=%s delta=%.6f (%.2f$) below thresholds (shares=%.6f, notional=%.2f)",
                            pos.symbol, delta, abs(delta) * price, self.rebalance_threshold, self.min_trade_notional)
                continue
            action = "open" if abs(curr_qty) <= 0.0 else ("increase" if delta > 0 else "decrease")
            side = "buy" if delta > 0 else "sell"
            logger.info("Rebalance action symbol=%s action=%s curr_qty=%.6f target_qty=%.6f delta=%.6f side=%s price=%.4f",
                        pos.symbol, action, curr_qty, target_qty, delta, side, price)
            staged.append({"symbol": pos.symbol, "price": price, "delta": delta, "action": action, "side": side, "close": False})

        # Execute sells first to free buying power, then buys.
        for side in ("sell", "buy"):
            for order in staged:
                if order["side"] != side:
                    continue
                symbol = order["symbol"]
                price = order["price"]
                qty = abs(order["delta"])
                coid = _make_client_order_id(allocation.date, symbol, side)
                kwargs: Dict[str, Any] = dict(symbol=symbol, side=side, tif=self.default_tif,
                                              extended_hours=self.extended_hours, client_order_id=coid)
                if order["close"]:
                    # Full close-out: always by exact quantity.
                    kwargs["qty"] = qty
                    desc = f"qty={qty:.9f}"
                elif self.use_notional:
                    notional = round(qty * price, self.round_notional_dp)
                    if target_equity is not None and notional > target_equity:
                        notional = round(target_equity, self.round_notional_dp)
                    if not math.isfinite(notional) or notional <= 0:
                        summary["skipped"] += 1
                        continue
                    if side == "buy":
                        notional = max(notional, self.min_buy_notional)
                        if notional < 1.0:
                            logger.info("Skipping buy below $1 notional for %s", symbol)
                            summary["skipped"] += 1
                            continue
                        if bp_available is not None and notional > bp_available:
                            logger.info("Insufficient buying power for notional %.2f on %s (bp=%.2f)", notional, symbol, bp_available)
                            summary["skipped"] += 1
                            continue
                    kwargs["notional"] = notional
                    desc = f"notional={notional:.2f}"
                else:
                    qty_rounded = float(f"{qty:.{self.round_qty_dp}f}")
                    if not math.isfinite(qty_rounded) or qty_rounded <= 0:
                        logger.info("Skip %s %s: qty rounds to zero (%.9f, dp=%d)", side, symbol, qty, self.round_qty_dp)
                        summary["skipped"] += 1
                        continue
                    if side == "buy" and bp_available is not None and qty_rounded * price > bp_available:
                        logger.info("Insufficient buying power for qty %s on %s (approx notional %.2f bp=%.2f)",
                                    qty_rounded, symbol, qty_rounded * price, bp_available)
                        summary["skipped"] += 1
                        continue
                    kwargs["qty"] = qty_rounded
                    desc = f"qty={qty_rounded:.9f}"
                try:
                    resp = self.trader.submit_market_order(**kwargs)
                except Exception as e:
                    logger.error("Order submit raised for %s %s: %s", side, symbol, e)
                    resp = None
                if resp is None:
                    summary["failed"] += 1
                    logger.warning("Alpaca did not return order for symbol=%s side=%s %s client_order_id=%s; see alpaca_trading logs",
                                   symbol, side, desc, coid)
                    continue
                summary["submitted"] += 1
                if order["close"]:
                    summary["closed"] += 1
                if side == "buy" and bp_available is not None and price is not None:
                    bp_available -= kwargs.get("notional", (kwargs.get("qty") or 0) * price)
                logger.info("Order submitted symbol=%s side=%s action=%s %s client_order_id=%s order_id=%s",
                            symbol, side, order["action"], desc, coid, getattr(resp, 'id', None))

        logger.info("Rebalance order summary: submitted=%d (closeouts=%d) failed=%d skipped=%d",
                    summary["submitted"], summary["closed"], summary["failed"], summary["skipped"])
        return summary
