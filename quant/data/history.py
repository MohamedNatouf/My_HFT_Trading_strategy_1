"""Historical bars with a parquet cache.

* Alpaca is the primary source. On the free plan SIP history is allowed as long as the
  request ``end`` is at least 15 minutes old, so history requests use ``feed=sip`` and clip
  ``end`` to now-16min. (IEX is ~2% of volume and has gaps for thin ETFs - avoid it for
  research.)
* Yahoo (yfinance) is a research-only fallback for *daily* adjusted closes when no
  Alpaca keys are configured.
* Minute bars are filtered to regular trading hours (09:30 <= t < 16:00 America/New_York).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

NY = "America/New_York"


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular-trading-hours minute bars (weekday 09:30 <= t < 16:00 NY)."""
    if df.empty:
        return df
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    ny = idx.tz_convert(NY)
    minutes = ny.hour * 60 + ny.minute
    mask = (ny.day_of_week < 5) & (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
    return df.loc[mask]


class BarCache:
    """Parquet cache: ``<root>/<source>/<timeframe>/<SYMBOL>.parquet``."""

    def __init__(self, root: str | Path = "data/cache"):
        self.root = Path(root)

    def path(self, source: str, timeframe: str, symbol: str) -> Path:
        return self.root / source / timeframe / f"{symbol.replace('/', '_')}.parquet"

    def load(self, source: str, timeframe: str, symbol: str) -> pd.DataFrame:
        p = self.path(source, timeframe, symbol)
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()

    def save(self, source: str, timeframe: str, symbol: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        p = self.path(source, timeframe, symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        old = self.load(source, timeframe, symbol)
        if not old.empty:
            df = pd.concat([old[~old.index.isin(df.index)], df]).sort_index()
        df.to_parquet(p)


def _to_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


class AlpacaHistory:
    """Historical stock bars from Alpaca Market Data (alpaca-py)."""

    def __init__(self, key_id: str, secret: str, feed: str = "sip", adjustment: str = "all",
                 cache: BarCache | None = None):
        from alpaca.data.historical import StockHistoricalDataClient

        self.client = StockHistoricalDataClient(key_id, secret)
        self.feed = feed
        self.adjustment = adjustment
        self.cache = cache or BarCache()

    def _request(self, symbols: list[str], timeframe, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest

        # Free plan: SIP history must end >= 15 minutes ago.
        latest_allowed = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)
        end = min(end, latest_allowed)
        if end <= start:
            return pd.DataFrame()
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start.to_pydatetime(),
            end=end.to_pydatetime(),
            feed=DataFeed(self.feed),
            adjustment=Adjustment(self.adjustment),
        )
        bars = self.client.get_stock_bars(req)  # alpaca-py follows next_page_token
        df = bars.df
        if df is None or df.empty:
            return pd.DataFrame()
        return df

    def daily(self, symbols: Iterable[str], start, end=None, use_cache: bool = True) -> dict[str, pd.DataFrame]:
        from alpaca.data.timeframe import TimeFrame

        start_ts, end_ts = _to_utc(start), _to_utc(end or pd.Timestamp.now(tz="UTC"))
        out: dict[str, pd.DataFrame] = {}
        need: list[str] = []
        for s in symbols:
            cached = self.cache.load("alpaca", "1Day", s) if use_cache else pd.DataFrame()
            # Cache is "fresh" if it covers the start and reaches within 3 days of end.
            if not cached.empty and cached.index[0] <= start_ts + pd.Timedelta(days=5) and \
                    cached.index[-1] >= end_ts - pd.Timedelta(days=3):
                out[s] = cached.loc[start_ts:end_ts]
            else:
                need.append(s)
        if need:
            df = self._request(need, TimeFrame.Day, start_ts, end_ts)
            if not df.empty:
                for sym, sdf in df.groupby(level=0):
                    sdf = sdf.droplevel(0)[["open", "high", "low", "close", "volume"]]
                    self.cache.save("alpaca", "1Day", sym, sdf)
                    out[sym] = sdf
        return out

    def minute(self, symbol: str, start, end=None, rth_only: bool = True, use_cache: bool = True) -> pd.DataFrame:
        """Minute bars for one symbol, fetched month by month and cached."""
        from alpaca.data.timeframe import TimeFrame

        start_ts, end_ts = _to_utc(start), _to_utc(end or pd.Timestamp.now(tz="UTC"))
        cached = self.cache.load("alpaca", "1Min", symbol) if use_cache else pd.DataFrame()
        frames = [cached] if not cached.empty else []
        cur = start_ts
        while cur < end_ts:
            nxt = min(cur + pd.Timedelta(days=31), end_ts)
            have = (not cached.empty) and cached.index[0] <= cur and cached.index[-1] >= nxt - pd.Timedelta(days=4)
            if not have:
                logger.info("alpaca minute fetch %s %s -> %s", symbol, cur.date(), nxt.date())
                df = self._request([symbol], TimeFrame.Minute, cur, nxt)
                if not df.empty:
                    df = df.droplevel(0)[["open", "high", "low", "close", "volume", "vwap"]]
                    self.cache.save("alpaca", "1Min", symbol, df)
                    frames.append(df)
            cur = nxt
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames)
        out = out[~out.index.duplicated(keep="last")].sort_index().loc[start_ts:end_ts]
        return filter_rth(out) if rth_only else out


class YahooHistory:
    """Research-only fallback for daily adjusted bars (yfinance)."""

    def __init__(self, cache: BarCache | None = None):
        self.cache = cache or BarCache()

    def daily(self, symbols: Iterable[str], start, end=None, use_cache: bool = True) -> dict[str, pd.DataFrame]:
        import yfinance as yf

        symbols = list(symbols)
        start_ts = _to_utc(start)
        end_ts = _to_utc(end or pd.Timestamp.now(tz="UTC"))
        out: dict[str, pd.DataFrame] = {}
        need = []
        for s in symbols:
            cached = self.cache.load("yahoo", "1Day", s) if use_cache else pd.DataFrame()
            if not cached.empty and cached.index[0] <= start_ts + pd.Timedelta(days=5) and \
                    cached.index[-1] >= end_ts - pd.Timedelta(days=4):
                out[s] = cached.loc[start_ts:end_ts]
            else:
                need.append(s)
        if need:
            raw = yf.download(need, start=start_ts.date().isoformat(), end=(end_ts + pd.Timedelta(days=1)).date().isoformat(),
                              interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=False)
            for s in need:
                try:
                    sdf = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
                except KeyError:
                    logger.warning("yahoo: no data for %s", s)
                    continue
                sdf = sdf.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(how="all")
                if sdf.empty:
                    continue
                # yfinance daily dates are exchange-local session dates: localise to New York,
                # NOT UTC (UTC midnight = previous evening in NY -> every bar one day early)
                ix = pd.to_datetime(sdf.index)
                sdf.index = (ix.tz_localize(NY) if ix.tz is None else ix.tz_convert(NY)).tz_convert("UTC")
                self.cache.save("yahoo", "1Day", s, sdf)
                out[s] = sdf
        return out


def _closes_panel(bars: dict[str, pd.DataFrame], symbols: list[str], field: str = "close") -> pd.DataFrame:
    cols = {}
    for s in symbols:
        if s in bars and not bars[s].empty and field in bars[s]:
            c = bars[s][field].copy()
            # normalise to a date index (daily bars)
            c.index = pd.DatetimeIndex(c.index).tz_convert(NY).normalize().tz_localize(None)
            cols[s] = c[~c.index.duplicated(keep="last")]
    missing = [s for s in symbols if s not in cols]
    if missing:
        logger.warning("no daily data for: %s", missing)
    # always return every requested column (all-NaN if unavailable) so callers can index safely
    return pd.DataFrame(cols).sort_index().reindex(columns=symbols)


def load_daily_closes(symbols: Iterable[str], start, end=None, source: str = "auto",
                      settings=None, cache: BarCache | None = None, field: str = "close",
                      refresh: bool = False) -> pd.DataFrame:
    """Daily split/dividend-adjusted closes, columns=symbols, index=naive dates.

    ``source``: 'alpaca', 'yahoo' or 'auto' (alpaca if credentials exist, else yahoo).
    ``field``: 'close' (default), 'open', 'high', 'low' or 'volume'.
    ``refresh``: ignore the cache freshness window (live trading must see yesterday's bar;
    the cache only guarantees data within 3 days of ``end``).
    """
    symbols = list(dict.fromkeys(symbols))
    creds = getattr(settings, "alpaca", None) if settings is not None else None
    if source == "auto":
        source = "alpaca" if (creds is not None and creds.available) else "yahoo"
    if source == "alpaca":
        if creds is None or not creds.available:
            raise RuntimeError("Alpaca credentials not configured (.env)")
        hist = AlpacaHistory(creds.key_id, creds.secret, feed="sip", cache=cache)
        bars = hist.daily(symbols, start, end, use_cache=not refresh)
    else:
        bars = YahooHistory(cache=cache).daily(symbols, start, end, use_cache=not refresh)
    return _closes_panel(bars, symbols, field)


def load_minute_bars(symbol: str, start, end=None, settings=None, cache: BarCache | None = None) -> pd.DataFrame:
    creds = getattr(settings, "alpaca", None) if settings is not None else None
    if creds is None or not creds.available:
        raise RuntimeError("Minute history requires Alpaca credentials (.env)")
    return AlpacaHistory(creds.key_id, creds.secret, feed="sip", cache=cache).minute(symbol, start, end)
