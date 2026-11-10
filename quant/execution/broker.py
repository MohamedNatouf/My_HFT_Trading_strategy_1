"""Thin wrapper over alpaca-py's TradingClient + data client.

Only Trading API endpoints are used (no Broker API). Order/fill events come from the
``trade_updates`` websocket via :class:`alpaca.trading.stream.TradingStream`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    symbol: str
    qty: float
    market_value: float
    price: float


@dataclass
class Quote:
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.ask or self.bid


@dataclass
class OrderSpec:
    symbol: str
    side: str                 # 'buy' | 'sell'
    qty: float
    client_order_id: str
    order_type: str = "market"   # 'market' | 'limit'
    tif: str = "day"             # 'day' | 'cls' | 'opg'
    limit_price: float | None = None
    est_notional: float = 0.0
    reason: str = ""
    ref_price: float = 0.0       # price used for sizing (decision price) -> slippage reports


class Broker(Protocol):
    def equity(self) -> float: ...
    def buying_power(self) -> float: ...
    def positions(self) -> dict[str, PositionInfo]: ...
    def is_open(self) -> bool: ...
    def latest_quotes(self, symbols: list[str]) -> dict[str, Quote]: ...
    def latest_prices(self, symbols: list[str]) -> dict[str, float]: ...
    def fractionable(self, symbol: str) -> bool: ...
    def open_orders(self) -> list: ...
    def recent_orders(self, after=None, limit: int = 500) -> list: ...
    def latest_trades(self, symbols: list[str]) -> dict: ...
    def cancel_order(self, order_id: str) -> None: ...
    def order_exists(self, client_order_id: str) -> bool: ...
    def submit(self, spec: OrderSpec): ...
    def close_all(self) -> list: ...


class AlpacaBroker:
    def __init__(self, key_id: str, secret: str, paper: bool = True, data_feed: str = "iex"):
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        self.paper = paper
        self.trading = TradingClient(key_id, secret, paper=paper)
        self.data = StockHistoricalDataClient(key_id, secret)
        self.data_feed = data_feed
        self._asset_cache: dict[str, object] = {}

    @classmethod
    def from_settings(cls, settings) -> "AlpacaBroker":
        c = settings.alpaca
        if not c.available:
            raise RuntimeError("Alpaca credentials missing: set ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in .env")
        return cls(c.key_id, c.secret, paper=c.paper, data_feed=c.data_feed)

    # --- account ---------------------------------------------------------------
    def account(self):
        return self.trading.get_account()

    def equity(self) -> float:
        return float(self.account().equity)

    def buying_power(self) -> float:
        return float(self.account().buying_power)

    def positions(self) -> dict[str, PositionInfo]:
        out = {}
        for p in self.trading.get_all_positions():
            qty = float(p.qty)
            mv = float(p.market_value or 0.0)
            price = float(p.current_price or (mv / qty if qty else 0.0))
            out[p.symbol] = PositionInfo(p.symbol, qty, mv, price)
        return out

    def clock(self):
        return self.trading.get_clock()

    def is_open(self) -> bool:
        return bool(self.clock().is_open)

    def calendar(self, start, end):
        from alpaca.trading.requests import GetCalendarRequest

        return self.trading.get_calendar(GetCalendarRequest(start=start, end=end))

    def fractionable(self, symbol: str) -> bool:
        a = self._asset_cache.get(symbol)
        if a is None:
            a = self.trading.get_asset(symbol)
            self._asset_cache[symbol] = a
        return bool(getattr(a, "fractionable", False))

    # --- market data -----------------------------------------------------------
    def latest_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestQuoteRequest

        if not symbols:
            return {}
        q = self.data.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbols, feed=DataFeed(self.data_feed)))
        return {s: Quote(float(v.bid_price or 0), float(v.ask_price or 0)) for s, v in q.items()}

    def latest_trades(self, symbols: list[str]) -> dict[str, tuple[float, "object"]]:
        """symbol -> (price, timestamp UTC) of the latest trade on the configured feed."""
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestTradeRequest

        if not symbols:
            return {}
        t = self.data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbols, feed=DataFeed(self.data_feed)))
        return {s: (float(v.price), v.timestamp) for s, v in t.items()}

    def latest_prices(self, symbols: list[str]) -> dict[str, float]:
        return {s: p for s, (p, _) in self.latest_trades(symbols).items()}

    # --- orders ----------------------------------------------------------------
    def open_orders(self) -> list:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        return list(self.trading.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)))

    def recent_orders(self, after=None, limit: int = 500) -> list:
        """Closed (filled / canceled / rejected / expired) orders, newest first."""
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        return list(self.trading.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=after,
                                                                    limit=limit, direction="desc")))

    def portfolio_history(self, start=None, end=None, period: str | None = None):
        """Daily equity / P&L history (``profit_loss`` excludes deposits and withdrawals)."""
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        if start is not None and end is None:
            from datetime import datetime, timezone

            end = datetime.now(timezone.utc)   # without an end Alpaca returns only 1 month from start
        req = GetPortfolioHistoryRequest(timeframe="1D", start=start, end=end, period=period if start is None else None)
        return self.trading.get_portfolio_history(req)

    def cancel_order(self, order_id: str) -> None:
        self.trading.cancel_order_by_id(order_id)

    def order_exists(self, client_order_id: str) -> bool:
        try:
            self.trading.get_order_by_client_id(client_order_id)
            return True
        except Exception:
            return False

    def submit(self, spec: OrderSpec):
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        common = dict(
            symbol=spec.symbol,
            qty=round(spec.qty, 9),
            side=OrderSide.BUY if spec.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce(spec.tif),
            client_order_id=spec.client_order_id,
        )
        if spec.order_type == "limit":
            req = LimitOrderRequest(limit_price=round(spec.limit_price, 2), **common)
        else:
            req = MarketOrderRequest(**common)
        return self.trading.submit_order(req)

    def close_all(self) -> list:
        return list(self.trading.close_all_positions(cancel_orders=True))

    def close_position(self, symbol: str):
        return self.trading.close_position(symbol)
