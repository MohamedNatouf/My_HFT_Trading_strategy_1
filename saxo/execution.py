# Execution router using FIX client. Maps allocations to orders.
from typing import List
from .fix_client import FixClient
from .engine import Allocation

class ExecutionRouter:
    def __init__(self, fix_client: FixClient, account: str):
        self.fix = fix_client
        self.account = account

    def rebalance(self, allocation: Allocation, current_positions: dict):
        # Simple difference-based orders: target weights vs current
        for pos in allocation.positions:
            target_qty = pos.weight * 100000  # scale factor placeholder
            curr_qty = current_positions.get(pos.symbol, 0)
            delta = target_qty - curr_qty
            if abs(delta) < 1:
                continue
            side = "BUY" if delta > 0 else "SELL"
            self.fix.send_market_order(symbol=pos.symbol, side=side, qty=abs(delta), account=self.account)
