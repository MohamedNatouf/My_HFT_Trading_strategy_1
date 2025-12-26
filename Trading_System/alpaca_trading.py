import logging
from typing import Optional

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except Exception:
    TradingClient = None
    MarketOrderRequest = None
    LimitOrderRequest = None
    OrderSide = None
    TimeInForce = None

logger = logging.getLogger(__name__)

class AlpacaTrader:
    def __init__(self, api_key: Optional[str], api_secret: Optional[str], paper: bool = True):
        if TradingClient is None:
            logger.warning("alpaca-py not installed; Alpaca trading unavailable")
            self.client = None
        else:
            try:
                self.client = TradingClient(api_key, api_secret, paper=paper)
            except Exception as e:
                logger.error("Failed to init TradingClient: %s", e)
                self.client = None

    def submit_market_order(self, symbol: str, qty: float, side: str, tif: str = 'day'):
        if self.client is None or MarketOrderRequest is None:
            return None
        tif_enum = TimeInForce[tif.upper()] if hasattr(TimeInForce, tif.upper()) else TimeInForce.DAY
        side_enum = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        order = MarketOrderRequest(symbol=symbol, qty=qty, side=side_enum, time_in_force=tif_enum)
        try:
            return self.client.submit_order(order)
        except Exception as e:
            logger.error("submit_market_order failed: %s", e)
            return None

    def submit_limit_order(self, symbol: str, qty: float, side: str, limit_price: float, tif: str = 'day'):
        if self.client is None or LimitOrderRequest is None:
            return None
        tif_enum = TimeInForce[tif.upper()] if hasattr(TimeInForce, tif.upper()) else TimeInForce.DAY
        side_enum = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        order = LimitOrderRequest(symbol=symbol, qty=qty, side=side_enum, limit_price=limit_price, time_in_force=tif_enum)
        try:
            return self.client.submit_order(order)
        except Exception as e:
            logger.error("submit_limit_order failed: %s", e)
            return None
