import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular-trading-hours bars: weekdays, 09:30 <= t < 16:00 America/New_York.

    Naive indexes are assumed to be UTC. Exchange holidays are not special-cased
    (there are no bars on them anyway).
    """
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    ny = idx.tz_convert('America/New_York')
    mins = ny.hour * 60 + ny.minute
    mask = (mins >= 9 * 60 + 30) & (mins < 16 * 60) & (ny.day_of_week < 5)
    return df[mask]

class MinuteStore:
    """
    Local minute-bar storage using Parquet per symbol.
    Layout:
      data/minute/{symbol}.parquet
    Each file contains columns: open, high, low, close, volume, indexed by UTC minute timestamp.
    """
    def __init__(self, root: str = "data/minute"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # Detect available engines
        self.engine = None
        try:
            import pyarrow  # noqa: F401
            self.engine = 'pyarrow'
        except Exception:
            try:
                import fastparquet  # noqa: F401
                self.engine = 'fastparquet'
            except Exception:
                self.engine = None
                logger.warning("Parquet engine not available (pyarrow/fastparquet missing); store will be disabled")

    def path_for(self, symbol: str) -> Path:
        return self.root / f"{symbol}.parquet"

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        if self.engine is None:
            return pd.DataFrame(columns=["open","high","low","close","volume"])  # disabled
        p = self.path_for(symbol)
        if not p.exists():
            return pd.DataFrame(columns=["open","high","low","close","volume"]).astype({
                "open": float, "high": float, "low": float, "close": float, "volume": float
            })
        try:
            df = pd.read_parquet(p, engine=self.engine)
            for c in ["open","high","low","close","volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            df.index = pd.to_datetime(df.index, utc=True)
            return df.sort_index()
        except Exception as e:
            logger.error("Failed to read parquet for %s: %s", symbol, e)
            return pd.DataFrame(columns=["open","high","low","close","volume"]).astype(float)

    def upsert_symbol(self, symbol: str, df: pd.DataFrame) -> int:
        """Merge incoming minute bars for symbol into store, by index. Returns rows added."""
        if self.engine is None:
            return 0
        if df is None or df.empty:
            return 0
        df = df.copy()
        # Normalize schema
        cols_map = {"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}
        for k,v in cols_map.items():
            if k in df.columns and v not in df.columns:
                df[v] = df[k]
        df = df[[c for c in ["open","high","low","close","volume"] if c in df.columns]]
        df.index = pd.to_datetime(df.index, utc=True)
        cur = self.load_symbol(symbol)
        before = len(cur)
        parts = []
        if not cur.empty:
            parts.append(cur[~cur.index.isin(df.index)])
        parts.append(df)
        merged = pd.concat(parts).sort_index()
        try:
            merged.to_parquet(self.path_for(symbol), engine=self.engine)
        except Exception as e:
            logger.error("Failed to write parquet for %s: %s", symbol, e)
        return max(0, len(merged) - before)

    def clear_all(self) -> int:
        """Delete every *.parquet file under the store root. Returns files removed."""
        removed = 0
        for f in self.root.glob('*.parquet'):
            try:
                f.unlink()
                removed += 1
            except Exception as e:
                logger.error("Failed to delete %s: %s", f, e)
        logger.info("MinuteStore.clear_all removed=%d root=%s", removed, self.root)
        return removed

    def load_universe(self, instruments: List[Dict[str, Any]], rth_only: bool = True) -> pd.DataFrame:
        frames = []
        for inst in instruments:
            sym = inst.get('symbol')
            df = self.load_symbol(sym)
            if df is None or df.empty:
                continue
            # Convert to MultiIndex columns
            sub = pd.DataFrame({
                ('open', sym): df['open'],
                ('high', sym): df['high'],
                ('low', sym): df['low'],
                ('close', sym): df['close'],
                ('volume', sym): df['volume'],
            })
            frames.append(sub)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=1, sort=True).sort_index()
        out.columns = pd.MultiIndex.from_tuples(out.columns)
        if rth_only:
            out = filter_rth(out)
        return out

    def latest_timestamp(self, symbol: str) -> Optional[pd.Timestamp]:
        df = self.load_symbol(symbol)
        if df is None or df.empty:
            return None
        return df.index[-1]
