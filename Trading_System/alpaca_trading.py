import logging
from typing import Optional

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.models import Order
except Exception:
    TradingClient = None
    MarketOrderRequest = None
    LimitOrderRequest = None
    OrderSide = None
    TimeInForce = None
    Order = None

logger = logging.getLogger(__name__)

# Lazy import to avoid circular at module import time
def _mk_account_client(api_key: Optional[str], api_secret: Optional[str], paper: bool):
    try:
        from .alpaca_account import AlpacaAccountClient
        return AlpacaAccountClient(api_key, api_secret, paper)
    except Exception:
        return None

class AlpacaTrader:
    def __init__(self, api_key: Optional[str], api_secret: Optional[str], paper: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        if TradingClient is None:
            logger.warning("alpaca-py not installed; Alpaca trading unavailable")
            self.client = None
        else:
            try:
                self.client = TradingClient(api_key, api_secret, paper=paper)
            except Exception as e:
                logger.error("Failed to init TradingClient: %s", e)
                self.client = None
        # Account client for overnight eligibility checks
        self.account_client = _mk_account_client(api_key, api_secret, paper)

    def _tif_enum(self, tif: str) -> TimeInForce:
        try:
            return TimeInForce[tif.upper()]
        except Exception:
            return TimeInForce.DAY

    def _side_enum(self, side: str) -> OrderSide:
        return OrderSide.BUY if str(side).lower() == 'buy' else OrderSide.SELL

    def _round_qty(self, qty: float) -> float:
        return float(f"{qty:.9f}")

    def _round_notional(self, notional: float) -> float:
        return float(f"{notional:.2f}")

    def _validate_overnight(self, symbol: str, extended_hours: bool) -> bool:
        if not extended_hours:
            return True
        if not self.account_client:
            logger.warning("Overnight validation unavailable; proceeding without check")
            return True
        ok = self.account_client.is_overnight_tradable(symbol)
        if not ok:
            logger.error("Symbol %s not eligible or halted for overnight; reject extended-hours order", symbol)
        return ok

    def submit_market_order(self, symbol: str, qty: Optional[float] = None, side: str = 'buy', tif: str = 'day', notional: Optional[float] = None, extended_hours: bool = False):
        """Submit market order. Enforces Alpaca constraints: buy min $1 notional, qty<=9dp, notional<=2dp, fractional day-only."""
        if self.client is None or MarketOrderRequest is None:
            return None
        if not self._validate_overnight(symbol, extended_hours):
            return None
        tif_enum = self._tif_enum(tif)
        side_enum = self._side_enum(side)
        # Fractional/notional orders supported only for 'day'
        if (qty is not None or notional is not None) and tif_enum is not TimeInForce.DAY:
            logger.warning("Fractional/notional supported only with TIF=day; forcing day")
            tif_enum = TimeInForce.DAY
        order_kwargs = { 'symbol': symbol, 'side': side_enum, 'time_in_force': tif_enum }
        if notional is not None:
            notional = self._round_notional(notional)
            if side_enum == OrderSide.BUY and notional < 1.0:
                logger.error("Buy orders require minimum $1 notional; got %.2f", notional)
                return None
            order_kwargs['notional'] = notional
        else:
            if qty is None:
                logger.error("Either qty or notional must be provided")
                return None
            qty = self._round_qty(qty)
            order_kwargs['qty'] = qty
        # extended hours flag when outside RTH
        if extended_hours:
            order_kwargs['extended_hours'] = True
        try:
            order = MarketOrderRequest(**order_kwargs)
            return self.client.submit_order(order)
        except Exception as e:
            logger.error("submit_market_order failed: %s", e)
            return None

    def submit_limit_order(self, symbol: str, qty: Optional[float], side: str, limit_price: float, tif: str = 'day', extended_hours: bool = False):
        if self.client is None or LimitOrderRequest is None:
            return None
        if not self._validate_overnight(symbol, extended_hours):
            return None
        tif_enum = self._tif_enum(tif)
        side_enum = self._side_enum(side)
        order_kwargs = { 'symbol': symbol, 'side': side_enum, 'time_in_force': tif_enum, 'limit_price': float(limit_price) }
        if qty is None:
            logger.error("Limit orders require qty; notional limit is not supported")
            return None
        qty = self._round_qty(qty)
        # For buy limit orders with fractional qty, ensure day tif
        if tif_enum is not TimeInForce.DAY:
            logger.warning("Fractional qty orders should use TIF=day; forcing day")
            tif_enum = TimeInForce.DAY
            order_kwargs['time_in_force'] = tif_enum
        order_kwargs['qty'] = qty
        if extended_hours:
            order_kwargs['extended_hours'] = True
        try:
            order = LimitOrderRequest(**order_kwargs)
            return self.client.submit_order(order)
        except Exception as e:
            logger.error("submit_limit_order failed: %s", e)
            return None

    def cancel_order_safe(self, order_id: str) -> bool:
        """Cancel only if order is open; returns True on 204 or already not open (treated as False)."""
        if self.client is None:
            return False
        try:
            ord = self.client.get_order_by_id(order_id)
            if getattr(ord, 'status', '').lower() in {'new', 'accepted', 'pending_new', 'open'}:
                self.client.cancel_order(order_id)
                return True
            logger.info("Order %s not open; status=%s", order_id, getattr(ord, 'status', ''))
            return False
        except Exception as e:
            logger.error("cancel_order_safe failed: %s", e)
            return False
