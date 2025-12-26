import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except Exception:
    StockHistoricalDataClient = None
    StockBarsRequest = None
    TimeFrame = None

logger = logging.getLogger(__name__)

class AlpacaMinuteDataLoader:
    """
    Loads minute bars for a list of instruments using Alpaca Market Data API.
    Expects instruments as dicts containing 'symbol'.
    """
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, use_sandbox: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_sandbox = use_sandbox
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

    def load_minute_bars(self, instruments: List[Dict[str, Any]], start: Optional[str], end: Optional[str]) -> pd.DataFrame:
        if self.client is None:
            return pd.DataFrame()
        symbols = [i['symbol'] for i in instruments]
        # Parse dates
        start_dt = datetime.fromisoformat(start) if start else None
        end_dt = datetime.fromisoformat(end) if end else None
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
            start=start_dt,
            end=end_dt
        )
        try:
            bars = self.client.get_stock_bars(req)
            df = bars.df
            if df is None or df.empty:
                return pd.DataFrame()
            # Pivot to MultiIndex columns (price_type, symbol) as expected by engine
            out_cols = []
            frames = []
            for sym, sdf in df.groupby(level=0):
                # sdf index has (symbol, timestamp)
                sdf = sdf.droplevel(0)
                sdf = sdf.rename(columns={
                    'open': ('open', sym),
                    'high': ('high', sym),
                    'low': ('low', sym),
                    'close': ('close', sym),
                    'volume': ('volume', sym)
                })
                frames.append(sdf[[('open', sym), ('high', sym), ('low', sym), ('close', sym), ('volume', sym)]])
            if not frames:
                return pd.DataFrame()
            out = pd.concat(frames, axis=1)
            out.columns = pd.MultiIndex.from_tuples(out.columns)
            return out.sort_index()
        except Exception as e:
            logger.error("Alpaca minute bars load failed: %s", e)
            return pd.DataFrame()
