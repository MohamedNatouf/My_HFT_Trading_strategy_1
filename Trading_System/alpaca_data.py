import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import pandas as pd

from .minute_store import filter_rth

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except Exception:
    StockHistoricalDataClient = None
    StockBarsRequest = None
    TimeFrame = None

try:
    from alpaca.data.enums import DataFeed, Adjustment
except Exception:
    DataFeed = None
    Adjustment = None

logger = logging.getLogger(__name__)


def _resolve_feed(feed: Optional[str]):
    """Map 'iex'|'sip'|... to alpaca-py DataFeed enum (string fallback)."""
    if feed is None or feed == "":
        return None
    val = str(getattr(feed, "value", feed)).lower()
    if DataFeed is not None:
        try:
            return DataFeed(val)
        except Exception:
            logger.warning("Unknown data feed %r; passing raw string", feed)
    return val


def _resolve_adjustment(adjustment: Optional[str]):
    """Map 'all'|'raw'|'split'|'dividend' to alpaca-py Adjustment enum (string fallback)."""
    if adjustment is None or adjustment == "":
        return None
    val = str(getattr(adjustment, "value", adjustment)).lower()
    if Adjustment is not None:
        try:
            return Adjustment(val)
        except Exception:
            logger.warning("Unknown adjustment %r; passing raw string", adjustment)
    return val


def _parse_dt(value: Union[None, str, datetime, pd.Timestamp]) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    return pd.Timestamp(str(value)).to_pydatetime()


class AlpacaMinuteDataLoader:
    """
    Loads minute bars for a list of instruments using Alpaca Market Data API.
    Expects instruments as dicts containing 'symbol'.

    ``feed``: 'iex' (free plan) or 'sip' (paid; free plan rejects the latest 15 min).
    ``adjustment``: 'raw' | 'split' | 'dividend' | 'all'.
    ``rth_only``: keep only regular-trading-hours bars (09:30 <= t < 16:00 NY, weekdays).
    """
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, use_sandbox: bool = False,
                 feed: Optional[str] = None, adjustment: Optional[str] = None, rth_only: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_sandbox = use_sandbox
        self.feed = feed
        self.adjustment = adjustment
        self.rth_only = bool(rth_only)
        if StockHistoricalDataClient is None:
            logger.warning("alpaca-py not installed; Alpaca data unavailable")
            self.client = None
        else:
            # Trading accounts authenticate via headers in SDK when provided
            try:
                self.client = StockHistoricalDataClient(self.api_key, self.api_secret)
            except Exception as e:
                logger.warning("Failed to init StockHistoricalDataClient: %s", e)
                self.client = None

    def load_minute_bars(self, instruments: List[Dict[str, Any]], start, end,
                         feed: Optional[str] = None, adjustment: Optional[str] = None,
                         rth_only: Optional[bool] = None) -> pd.DataFrame:
        """Return MultiIndex (field, symbol) minute bars, or an EMPTY frame on any failure
        (errors are logged at ERROR level). Per-call feed/adjustment/rth_only override
        the constructor defaults."""
        if self.client is None:
            logger.error("Alpaca data client unavailable (alpaca-py missing or init failed); returning empty bars")
            return pd.DataFrame()
        symbols = [i['symbol'] for i in instruments]
        if not symbols:
            return pd.DataFrame()
        feed_v = _resolve_feed(feed if feed is not None else self.feed)
        adj_v = _resolve_adjustment(adjustment if adjustment is not None else self.adjustment)
        rth = self.rth_only if rth_only is None else bool(rth_only)
        try:
            req_kwargs: Dict[str, Any] = dict(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Minute,
                start=_parse_dt(start),
                end=_parse_dt(end),
            )
            if feed_v is not None:
                req_kwargs['feed'] = feed_v
            if adj_v is not None:
                req_kwargs['adjustment'] = adj_v
            req = StockBarsRequest(**req_kwargs)
        except Exception as e:
            logger.error("Failed to build StockBarsRequest (feed=%s adjustment=%s start=%s end=%s): %s",
                         feed_v, adj_v, start, end, e)
            return pd.DataFrame()
        try:
            bars = self.client.get_stock_bars(req)
            df = bars.df
        except Exception as e:
            logger.error("Alpaca minute bars load FAILED (feed=%s adjustment=%s symbols=%d start=%s end=%s): %s",
                         feed_v, adj_v, len(symbols), start, end, e)
            return pd.DataFrame()
        if df is None or df.empty:
            logger.warning("Alpaca returned no bars (feed=%s adjustment=%s symbols=%d start=%s end=%s)",
                           feed_v, adj_v, len(symbols), start, end)
            return pd.DataFrame()
        try:
            # Pivot to MultiIndex columns (price_type, symbol) as expected by engine
            frames = []
            for sym, sdf in df.groupby(level=0):
                # sdf index has (symbol, timestamp)
                sdf = sdf.droplevel(0)
                cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in sdf.columns]
                sub = sdf[cols].copy()
                sub.columns = pd.MultiIndex.from_tuples([(c, sym) for c in cols])
                frames.append(sub)
            if not frames:
                return pd.DataFrame()
            out = pd.concat(frames, axis=1).sort_index()
            if rth:
                out = filter_rth(out)
            return out
        except Exception as e:
            logger.error("Failed to reshape Alpaca bars: %s", e)
            return pd.DataFrame()
