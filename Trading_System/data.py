# Minute data loader utilities for backtesting
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

@dataclass
class BacktestConfig:
    start: str
    end: str
    interval: str = "1m"
    source: str = "yahoo"
    api_key: Optional[str] = None
    output_size: str = "compact"  # compact or full
    rapidapi_host: str = "alpha-vantage.p.rapidapi.com"
    rapidapi_key: Optional[str] = None

class MinuteDataLoader:
    def __init__(self, universe: List[Dict]):
        # universe entries: { symbol, assetType, ... }
        self.universe = universe

    @staticmethod
    def _symbol_to_yahoo(symbol: str) -> str:
        # Map common FX like EURUSD -> EURUSD=X; otherwise return as-is
        if symbol.isalpha() and len(symbol) in (6,7):
            return f"{symbol}=X"
        return symbol

    @staticmethod
    def _split_fx_symbol(symbol: str) -> Tuple[str, str]:
        # EURUSD -> (EUR, USD)
        if len(symbol) >= 6 and symbol.isalpha():
            return symbol[:3], symbol[3:6]
        parts = symbol.replace("/", "").upper()
        return parts[:3], parts[3:6]

    def _yf_download_chunked(self, tickers: List[str], start: str, end: str, interval: str) -> pd.DataFrame:
        logger.info("Yahoo download start: tickers=%s interval=%s start=%s end=%s", tickers, interval, start, end)
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        frames = []
        # Define max days per request for 1m
        max_days = 7 if interval == "1m" else 60
        cur_start = start_ts
        empty_windows = 0
        while cur_start < end_ts:
            cur_end = min(cur_start + pd.Timedelta(days=max_days), end_ts)
            logger.debug("Yahoo window: %s -> %s", cur_start, cur_end)
            try:
                df = yf.download(
                    tickers=tickers,
                    start=cur_start.to_pydatetime(),
                    end=cur_end.to_pydatetime(),
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    group_by='ticker'
                )
            except Exception as e:
                logger.warning("Yahoo window failed %s -> %s err=%s", cur_start, cur_end, e)
                df = pd.DataFrame()
            if df is not None and not df.empty:
                logger.debug("Yahoo window rows=%d cols=%d", len(df), len(df.columns))
                frames.append(df)
            else:
                empty_windows += 1
                logger.warning("Yahoo returned empty frame for window %s -> %s", cur_start, cur_end)
            cur_start = cur_end
            if empty_windows >= 3:
                logger.warning("Yahoo multiple empty windows encountered; continuing without retries")
        if not frames:
            logger.warning("Yahoo returned no data across all windows")
            return pd.DataFrame()
        # Concatenate along index
        data = pd.concat(frames).sort_index()
        # Drop duplicates that may appear at chunk joins
        data = data[~data.index.duplicated(keep='last')]
        logger.info("Yahoo combined shape: %s", data.shape)
        return data

    def _load_yahoo(self, cfg: BacktestConfig) -> pd.DataFrame:
        if not HAS_YF:
            raise RuntimeError("yfinance not installed. Please install yfinance to use Yahoo minute data.")
        tickers = [self._symbol_to_yahoo(u['symbol']) for u in self.universe]
        logger.info("Loading Yahoo data for %d tickers", len(tickers))
        data = self._yf_download_chunked(tickers, cfg.start, cfg.end, cfg.interval)
        # Normalize to MultiIndex (PriceType, Symbol)
        # yfinance returns columns like ('EURUSD=X','Close') or for single ticker just OHLC
        if isinstance(data.columns, pd.MultiIndex):
            # pivot to PriceType first level
            panels = {}
            for price_type in ["Open","High","Low","Close","Volume"]:
                # select second level == price_type
                try:
                    sub = data.xs(price_type, axis=1, level=1)
                    panels[price_type] = sub
                except Exception:
                    logger.debug("Price type missing in Yahoo data: %s", price_type)
                    continue
            if not panels:
                logger.warning("Yahoo returned no OHLCV panels")
                return pd.DataFrame()
            # align and build MultiIndex
            all_syms = sorted({s for df in panels.values() for s in df.columns})
            # Drop symbols with no data across panels
            sym_nonempty = []
            for s in all_syms:
                any_nonempty = any((s in df.columns and df[s].dropna().shape[0] > 0) for df in panels.values())
                if any_nonempty:
                    sym_nonempty.append(s)
            if len(sym_nonempty) < len(all_syms):
                dropped = sorted(set(all_syms) - set(sym_nonempty))
                logger.warning("Dropping symbols with no minute data: %s", dropped)
            reindexed = {k: v.reindex(columns=sym_nonempty) for k,v in panels.items()}
            arrays = []
            frames = []
            for price_type, sub in reindexed.items():
                arrays.extend([(price_type, s) for s in sub.columns])
                frames.append(sub)
            combined = pd.concat(frames, axis=1)
            combined.columns = pd.MultiIndex.from_tuples(arrays)
            combined = combined.sort_index()
            logger.info("Yahoo normalized shape: %s", combined.shape)
            na_pct = combined.isna().mean().mean()
            logger.debug("Yahoo NaN overall pct: %.4f", na_pct)
            return combined.dropna(how='all')
        else:
            symbol = self.universe[0]['symbol'] if self.universe else ""
            if data.empty:
                logger.warning("Yahoo single-ticker data empty for %s", symbol)
                return pd.DataFrame()
            df = data[["Open","High","Low","Close","Volume"]]
            arrays = []
            for p in ["Open","High","Low","Close","Volume"]:
                arrays.append((p, symbol))
            df.columns = pd.MultiIndex.from_tuples(arrays)
            logger.info("Yahoo single normalized shape: %s", df.shape)
            return df

    def _load_alpha_vantage_rapidapi(self, cfg: BacktestConfig) -> pd.DataFrame:
        """
        Equity/FX intraday via RapidAPI Alpha Vantage.
        Uses headers x-rapidapi-host and x-rapidapi-key.
        Interval: 1min, 5min, 15min, 30min, 60min. Output size: compact|full.
        Parses "Time Series (<interval>)" where interval matches.
        Converts strings to numeric, timestamps to timezone-aware UTC.
        """
        import requests
        host = cfg.rapidapi_host or "alpha-vantage.p.rapidapi.com"
        key = cfg.rapidapi_key or cfg.api_key
        if not key:
            raise RuntimeError("RapidAPI key required. Set BacktestConfig.rapidapi_key or api_key.")
        frames = []
        for u in self.universe:
            symbol = u['symbol']
            interval = cfg.interval if cfg.interval in ("1min","5min","15min","30min","60min") else "1min"
            params = {
                "datatype": "json",
                "output_size": cfg.output_size if cfg.output_size in ("compact","full") else "compact",
                "interval": interval,
            }
            base, quote = self._split_fx_symbol(symbol)
            is_fx = symbol.isalpha() and len(symbol.replace("/", "")) >= 6 and base and quote
            if is_fx:
                params.update({"function": "FX_INTRADAY", "from_symbol": base, "to_symbol": quote})
            else:
                params.update({"function": "TIME_SERIES_INTRADAY", "symbol": symbol})
            headers = {
                "x-rapidapi-host": host,
                "x-rapidapi-key": key,
            }
            url = "https://alpha-vantage.p.rapidapi.com/query"
            r = requests.get(url, headers=headers, params=params, timeout=30)
            j = r.json()
            series_key = f"Time Series ({interval})" if params["function"] == "TIME_SERIES_INTRADAY" else f"Time Series FX ({interval})"
            if series_key not in j:
                logger.warning("RapidAPI Alpha Vantage returned no series for %s function=%s", symbol, params["function"])
                continue
            ts = pd.DataFrame.from_dict(j[series_key], orient='index')
            idx = pd.to_datetime(ts.index)
            idx = idx.tz_localize("US/Eastern", ambiguous='infer').tz_convert("UTC")
            ts.index = idx
            cols_map = {
                '1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'
            }
            ts = ts.rename(columns=cols_map)
            if 'Volume' not in ts.columns:
                ts['Volume'] = np.nan
            if cfg.start:
                ts = ts[ts.index >= pd.to_datetime(cfg.start).tz_localize("UTC")]
            if cfg.end:
                ts = ts[ts.index <= pd.to_datetime(cfg.end).tz_localize("UTC")]
            for c in ['Open','High','Low','Close','Volume']:
                if c in ts.columns:
                    ts[c] = pd.to_numeric(ts[c], errors='coerce')
            sub = ts[['Open','High','Low','Close','Volume']]
            sub.columns = pd.MultiIndex.from_tuples([(c, symbol) for c in sub.columns])
            frames.append(sub)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, axis=1).sort_index()
        return df

    def _load_alpha_vantage_fx(self, cfg: BacktestConfig) -> pd.DataFrame:
        import requests
        api_key = cfg.api_key
        logger.info("Alpha Vantage (direct) loader start interval=%s api_key_present=%s", cfg.interval, bool(api_key))
        if not api_key:
            raise RuntimeError("Alpha Vantage api_key required in config.backtest.apiKey or alphaVantage.apiKey")
        frames = []
        for u in self.universe:
            symbol = u['symbol']
            base, quote = self._split_fx_symbol(symbol)
            interval = cfg.interval if cfg.interval in ("1min","5min","15min","30min","60min") else "1min"
            url = (
                "https://www.alphavantage.co/query?function=FX_INTRADAY"
                f"&from_symbol={base}&to_symbol={quote}&interval={interval}&outputsize=full&apikey={api_key}"
            )
            r = requests.get(url, timeout=30)
            j = r.json()
            key = f"Time Series FX ({interval})"
            ts = None
            if key in j:
                ts = pd.DataFrame.from_dict(j[key], orient='index')
            else:
                url2 = (
                    "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
                    f"&symbol={symbol}&interval={interval}&outputsize=full&apikey={api_key}"
                )
                r2 = requests.get(url2, timeout=30)
                j2 = r2.json()
                key2 = f"Time Series ({interval})"
                if key2 in j2:
                    ts = pd.DataFrame.from_dict(j2[key2], orient='index')
                else:
                    try:
                        headers = {
                            "x-rapidapi-host": cfg.rapidapi_host or "alpha-vantage.p.rapidapi.com",
                            "x-rapidapi-key": api_key,
                        }
                        params = {
                            "function": "TIME_SERIES_INTRADAY",
                            "symbol": symbol,
                            "interval": interval,
                            "output_size": cfg.output_size if cfg.output_size in ("compact","full") else "compact",
                            "datatype": "json",
                        }
                        url_ra = "https://alpha-vantage.p.rapidapi.com/query"
                        rr = requests.get(url_ra, headers=headers, params=params, timeout=30)
                        jr = rr.json()
                        sk = f"Time Series ({interval})"
                        if sk in jr:
                            ts = pd.DataFrame.from_dict(jr[sk], orient='index')
                        else:
                            logger.warning("Alpha Vantage returned no time series for %s", symbol)
                            continue
                    except Exception:
                        logger.warning("Alpha Vantage returned no time series for %s", symbol)
                        continue
            ts.index = pd.to_datetime(ts.index)
            ts = ts.sort_index()
            cols_map = {
                '1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'
            }
            ts = ts.rename(columns=cols_map)
            if 'Volume' not in ts.columns:
                ts['Volume'] = np.nan
            if cfg.start:
                ts = ts[ts.index >= pd.to_datetime(cfg.start)]
            if cfg.end:
                ts = ts[ts.index <= pd.to_datetime(cfg.end)]
            for c in ['Open','High','Low','Close','Volume']:
                if c in ts.columns:
                    ts[c] = pd.to_numeric(ts[c], errors='coerce')
            sub = ts[['Open','High','Low','Close','Volume']]
            sub.columns = pd.MultiIndex.from_tuples([(c, symbol) for c in sub.columns])
            frames.append(sub)
        if not frames:
            logger.warning("Alpha Vantage returned no data for any symbol")
            return pd.DataFrame()
        df = pd.concat(frames, axis=1).sort_index()
        return df

    def load(self, cfg: BacktestConfig) -> pd.DataFrame:
        logger.info("MinuteDataLoader.load source=%s interval=%s api_key_present=%s", cfg.source, cfg.interval, bool(cfg.api_key))
        src = cfg.source.lower()
        if src == "yahoo":
            return self._load_yahoo(cfg)
        elif src in ("alpha_vantage", "alphavantage"):
            return self._load_alpha_vantage_fx(cfg)
        elif src in ("alpha_vantage_rapidapi", "alphavantage_rapidapi", "rapidapi"):
            return self._load_alpha_vantage_rapidapi(cfg)
        else:
            raise NotImplementedError(f"Source {cfg.source} not supported yet.")
