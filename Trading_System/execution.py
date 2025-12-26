# Execution router using Alpaca trading client. Maps allocations to Alpaca orders.
from typing import List
from .alpaca_trading import AlpacaTrader
from .engine import Allocation

class ExecutionRouter:
    def __init__(self, trader: AlpacaTrader, default_tif: str = 'day'):
        self.trader = trader
        self.default_tif = default_tif

    def rebalance(self, allocation: Allocation, current_positions: dict):
        # Simple difference-based orders: target weights vs current positions
        for pos in allocation.positions:
            target_qty = pos.weight * 100  # TODO: map weights to shares/notional using account buying power
            curr_qty = current_positions.get(pos.symbol, 0)
            delta = target_qty - curr_qty
            if abs(delta) < 1:
                continue
            side = "buy" if delta > 0 else "sell"
            self.trader.submit_market_order(symbol=pos.symbol, qty=abs(delta), side=side, tif=self.default_tif)
