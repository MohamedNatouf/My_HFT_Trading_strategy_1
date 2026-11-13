#!/usr/bin/env python3
r"""
Verify minute-store activity and price changes for specific symbols/timestamps using
only local Parquet files under data/minute.

Usage (PowerShell):
  & "venv\Scripts\python.exe" scripts\verify_store_activity.py --symbols CXSE,XSVM,PGJ --timestamps 2025-12-08T18:35:00
"""
import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd
import sys

STORE_ROOT = Path('data/minute')

def read_symbol(symbol: str) -> pd.DataFrame:
    p = STORE_ROOT / f"{symbol}.parquet"
    if not p.exists():
        print(f"WARN: store file missing for {symbol}: {p}")
        return pd.DataFrame(columns=["open","high","low","close","volume"])  
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        print(f"ERROR: failed reading {p}: {e}")
        return pd.DataFrame(columns=["open","high","low","close","volume"])  
    cols_map = {"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}
    for k,v in cols_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]
    df = df[[c for c in ["open","high","low","close","volume"] if c in df.columns]]
    try:
        if df.index.tz is None:
            df.index = pd.to_datetime(df.index).tz_localize('UTC')
        else:
            df.index = pd.to_datetime(df.index).tz_convert('UTC')
    except Exception:
        df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()

def resample_5m(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    for c in ["open","high","low","close","volume"]:
        if c not in df.columns:
            df[c] = pd.NA
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    return df.resample('5T', closed='right', label='left').agg(agg)

def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    idx = df.index
    hrs_ok = ((idx.hour > 14) | ((idx.hour == 14) & (idx.minute >= 30))) & (idx.hour < 21)
    wk_ok = idx.dayofweek < 5
    return df.loc[hrs_ok & wk_ok]

def assemble(symbols: List[str]) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        raw = read_symbol(sym)
        rs = resample_5m(raw)
        rth = filter_rth(rs)
        if rth is None or rth.empty:
            continue
        sub = pd.DataFrame({
            ("close", sym): rth['close'],
            ("volume", sym): rth['volume']
        })
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out

def analyze(df: pd.DataFrame, timestamps: Optional[List[pd.Timestamp]] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    close = df.xs('close', axis=1, level=0)
    volume = df.xs('volume', axis=1, level=0)
    changed = (close != close.shift(1))
    any_changed = changed.any(axis=1)
    any_vol = (volume.fillna(0) > 0).any(axis=1)
    summary = pd.DataFrame({
        'any_close_changed': any_changed,
        'any_volume_gt0': any_vol
    }, index=df.index)
    if timestamps:
        idx_set = set(timestamps)
        summary = summary.loc[[ts for ts in summary.index if ts in idx_set]]
    return summary

def parse_ts_list(ts_list: Optional[str]) -> Optional[List[pd.Timestamp]]:
    if not ts_list:
        return None
    out = []
    for s in ts_list.split(','):
        s = s.strip()
        if not s:
            continue
        out.append(pd.to_datetime(s).tz_convert('UTC'))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', type=str, required=True)
    ap.add_argument('--start', type=str, default=None)
    ap.add_argument('--end', type=str, default=None)
    ap.add_argument('--timestamps', type=str, default=None)
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    df = assemble(symbols)
    if df.empty:
        print('No data found in store for given symbols')
        sys.exit(0)
    if args.start:
        df = df[df.index >= pd.to_datetime(args.start).tz_localize('UTC')]
    if args.end:
        df = df[df.index <= pd.to_datetime(args.end).tz_localize('UTC')]
    ts_list = parse_ts_list(args.timestamps)
    summary = analyze(df, timestamps=ts_list)
    if summary.empty:
        print('No rows match the filters; try adjusting date range or timestamps.')
        sys.exit(0)
    print('UTC\tany_close_changed\tany_volume_gt0')
    for ts, row in summary.iterrows():
        print(f"{ts}\t{bool(row['any_close_changed'])}\t{bool(row['any_volume_gt0'])}")

if __name__ == '__main__':
    main()
