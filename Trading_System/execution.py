# Execution router using Alpaca trading client. Maps allocations to Alpaca orders.
from typing import List, Dict, Any
import logging
from .alpaca_trading import AlpacaTrader
from .alpaca_account import AlpacaAccountClient
from .engine import Allocation

logger = logging.getLogger(__name__)

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
        self.rebalance_threshold = float(tc.get('rebalanceThresholdShares', 1))

    def _buying_power_ok(self, needed: float) -> bool:
        acc = self.account.get_account() if self.account else None
        if not acc:
            logger.warning("Account state unavailable; proceeding without BP check")
            return True
        bp = float(getattr(acc, 'buying_power', 0) or 0)
        return bp >= needed

    def _settlement_safe(self) -> bool:
        acc = self.account.get_account() if self.account else None
        if not acc:
            return True
        # Avoid relying on withdrawable cash intra-day; allow trading if buying_power is positive.
        cash_wdr = float(getattr(acc, 'cash_withdrawable', 0) or 0)
        cash = float(getattr(acc, 'cash', 0) or 0)
        memoposts = float(getattr(acc, 'memoposts', 0) or 0)
        # Strategy-side: do not block, but log if cash_withdrawable is negative while cash/memoposts indicate pending settlement.
        if cash_wdr < 0 and (cash + memoposts) >= 0:
            logger.info("Settlement pending (cash_withdrawable<0, cash+memoposts>=0); continuing with trade decisions")
        return True

    def rebalance(self, allocation: Allocation, current_positions: dict):
        logger.info("execution rebalance ts=%s emergency=%s positions=%s", allocation.date, allocation.emergency, [(p.symbol, p.weight) for p in allocation.positions])
        if not self._settlement_safe():
            return
        for pos in allocation.positions:
            target_qty = pos.weight * 100  # Placeholder mapping; ideally compute from account equity
            curr_qty = current_positions.get(pos.symbol, 0)
            delta = target_qty - curr_qty
            if abs(delta) < self.rebalance_threshold:
                logger.debug("skip symbol=%s delta=%.6f threshold=%.6f", pos.symbol, delta, self.rebalance_threshold)
                continue
            side = "buy" if delta > 0 else "sell"
            qty = abs(delta)
            logger.debug("order symbol=%s side=%s qty=%.6f", pos.symbol, side, qty)
            if self.use_notional:
                price = getattr(pos, 'last_close', None)
                if price is None:
                    logger.info("No price for notional calculation; falling back to qty")
                    self.trader.submit_market_order(symbol=pos.symbol, qty=qty, side=side, tif=self.default_tif, extended_hours=self.extended_hours)
                    continue
                notional = round(qty * float(price), self.round_notional_dp)
                if side == 'buy' and notional < self.min_buy_notional:
                    notional = self.min_buy_notional
                if side == 'buy' and notional < 1.0:
                    logger.info("Skipping buy below $1 notional for %s", pos.symbol)
                    continue
                if not self._buying_power_ok(notional):
                    logger.info("Insufficient buying power for notional %s on %s", notional, pos.symbol)
                    continue
                self.trader.submit_market_order(symbol=pos.symbol, side=side, tif=self.default_tif, notional=notional, extended_hours=self.extended_hours)
            else:
                qty = float(f"{qty:.{self.round_qty_dp}f}")
                if side == 'buy' and not self._buying_power_ok(qty):
                    logger.info("Insufficient buying power for qty %s on %s", qty, pos.symbol)
                    continue
                self.trader.submit_market_order(symbol=pos.symbol, qty=qty, side=side, tif=self.default_tif, extended_hours=self.extended_hours)
