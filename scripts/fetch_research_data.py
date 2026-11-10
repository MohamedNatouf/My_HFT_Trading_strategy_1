"""Download (free) research data from Alpaca into data/cache/research/.

    python scripts/fetch_research_data.py sp100 --start 2016-01-01      # daily bars S&P 100 + sector ETFs
    python scripts/fetch_research_data.py news  --start 2022-01-01      # Benzinga headlines via Alpaca News API
    python scripts/fetch_research_data.py orb   --start 2022-06-01      # stocks-in-play scan data (SIP, >15 min old)

Everything uses the free plan: historical SIP bars older than 15 minutes and the News API.
Requests are throttled below the ~200 req/min limit and every step is resumable (cached).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.config import load_settings  # noqa: E402
from quant.data.universe import SECTOR_ETFS, SP100  # noqa: E402

OUT = ROOT / "data" / "cache" / "research"
NY = "America/New_York"
log = logging.getLogger("fetch")


class Throttle:
    """Thread-safe request pacer (requests are latency-bound, so callers run in parallel)."""

    def __init__(self, per_min: int = 180):
        import threading

        self.dt = 60.0 / per_min
        self.next = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            slot = max(now, self.next)
            self.next = slot + self.dt
        if slot > now:
            time.sleep(slot - now)


def with_retry(fn, *a, tries=5, **kw):
    for i in range(tries):
        try:
            return fn(*a, **kw)
        except Exception as e:  # 429 / transient network errors
            wait = 5 * (i + 1)
            log.warning("request failed (%s); retry in %ss", e, wait)
            time.sleep(wait)
    raise RuntimeError("giving up after retries")


def clients(settings):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.historical.news import NewsClient
    from alpaca.trading.client import TradingClient

    c = settings.alpaca
    return (StockHistoricalDataClient(c.key_id, c.secret), NewsClient(c.key_id, c.secret),
            TradingClient(c.key_id, c.secret, paper=c.paper))


def bars(data, symbols, timeframe, start, end, thr: Throttle, adjustment="all"):
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest

    thr.wait()
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=timeframe, start=start, end=end,
                           feed=DataFeed.SIP, adjustment=Adjustment(adjustment), limit=None)
    df = with_retry(data.get_stock_bars, req).df
    return df if df is not None else pd.DataFrame()


# ---------------------------------------------------------------------------------
def cmd_sp100(args, settings):
    from alpaca.data.timeframe import TimeFrame

    data, _, _ = clients(settings)
    syms = sorted(set(SP100) | set(SECTOR_ETFS) | {"SPY"})
    end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)
    frames = [bars(data, syms[i:i + 50], TimeFrame.Day, pd.Timestamp(args.start, tz="UTC"), end, Throttle())
              for i in range(0, len(syms), 50)]
    df = pd.concat(frames)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "sp100_daily.parquet")
    print(f"sp100 daily: {df.index.get_level_values(0).nunique()} symbols, {len(df)} rows")


def cmd_news(args, settings):
    """Alpaca News API (Benzinga), raw REST so paging and rate are under our control."""
    import requests

    c = settings.alpaca
    hdr = {"APCA-API-KEY-ID": c.key_id, "APCA-API-SECRET-KEY": c.secret}
    url = "https://data.alpaca.markets/v1beta1/news"
    syms = ",".join(sorted(SP100))
    ndir = OUT / "news"
    ndir.mkdir(parents=True, exist_ok=True)
    months = pd.period_range(pd.Timestamp(args.start), pd.Timestamp.now(), freq="M")
    thr = Throttle(args.rate)
    for m in months:
        path = ndir / f"{m}.parquet"
        current = m == pd.Timestamp.now().to_period("M")
        if path.exists() and not current:
            continue
        rows, token = [], None
        start = m.start_time.tz_localize("UTC")
        end = min(m.end_time.tz_localize("UTC"), pd.Timestamp.now(tz="UTC"))
        while True:
            thr.wait()
            params = {"symbols": syms, "start": start.isoformat(), "end": end.isoformat(), "limit": 50,
                      "sort": "asc", "include_content": "false"}
            if token:
                params["page_token"] = token

            def get():
                r = requests.get(url, headers=hdr, params=params, timeout=30)
                r.raise_for_status()
                return r.json()

            js = with_retry(get)
            rows += [{"id": n["id"], "created_at": n["created_at"], "updated_at": n["updated_at"],
                      "headline": n.get("headline", ""), "summary": n.get("summary", ""),
                      "symbols": ",".join(n.get("symbols", [])), "source": n.get("source", "")}
                     for n in js.get("news", [])]
            token = js.get("next_page_token")
            if not token:
                break
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.drop_duplicates("id")
        df.to_parquet(path)
        print(f"news {m}: {len(df)} articles", flush=True)


def cmd_orb(args, settings):
    """Point-in-time stocks-in-play scan data:
    1) active US common stocks (ETF-like names removed), daily bars for all of them
    2) per day: first 5-minute bar for every liquid name (price>5, 14d avg volume>1M, ATR>0.5)
    3) per day: 1-minute bars for the top-N by relative opening volume (candidates)
    """
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

    data, _, trading = clients(settings)
    odir = OUT / "orb"
    (odir / "first5").mkdir(parents=True, exist_ok=True)
    (odir / "minute").mkdir(parents=True, exist_ok=True)
    thr = Throttle(args.rate)
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)

    # 1) universe + daily bars
    dpath = odir / "daily.parquet"
    if not dpath.exists():
        assets = trading.get_all_assets(GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY))
        bad = ("ETF", "ETN", "FUND", "TRUST", "ISHARES", "PROSHARES", "DIREXION", "SPDR", "INVESCO", "VANGUARD",
               "WISDOMTREE", "2X", "3X", "BULL", "BEAR", "WARRANT", "RIGHTS", "UNITS", "PREFERRED", "DEPOSITARY")
        syms = sorted(a.symbol for a in assets
                      if a.tradable and str(getattr(a.exchange, "value", a.exchange)) in ("NYSE", "NASDAQ")
                      and a.symbol.isalpha() and not any(b in (a.name or "").upper() for b in bad))
        log.info("universe: %d symbols", len(syms))
        frames = []
        for i in range(0, len(syms), 400):
            frames.append(bars(data, syms[i:i + 400], TimeFrame.Day, start - pd.Timedelta(days=40), end, thr,
                               adjustment="split"))
            log.info("daily %d/%d", i + 400, len(syms))
        pd.concat(frames).to_parquet(dpath)
    daily = pd.read_parquet(dpath)
    close = daily["close"].unstack(0)
    vol = daily["volume"].unstack(0)
    hi, lo = daily["high"].unstack(0), daily["low"].unstack(0)
    for f in (close, vol, hi, lo):
        f.index = pd.DatetimeIndex(f.index).tz_convert(NY).normalize().tz_localize(None)
    prev_c = close.shift(1)
    tr = pd.concat([hi - lo, (hi - prev_c).abs(), (lo - prev_c).abs()]).groupby(level=0).max()
    atr = tr.rolling(14).mean().shift(1)            # known before today's open
    avgvol = vol.rolling(14).mean().shift(1)
    eligible = (prev_c > 5) & (avgvol > 1e6) & (atr > 0.5)
    days = [d for d in close.index if d >= start.tz_convert(None).normalize()]
    log.info("%d days, avg eligible/day=%.0f", len(days), eligible.loc[days].sum(axis=1).mean())

    # 2) first 5-minute bar of every eligible name, per day (parallel: the API is latency-bound)
    from concurrent.futures import ThreadPoolExecutor

    tf5 = TimeFrame(5, TimeFrameUnit.Minute)

    def first5_day(d):
        p = odir / "first5" / f"{d.date()}.parquet"
        if p.exists():
            return
        names = list(eligible.columns[eligible.loc[d].fillna(False).to_numpy()])
        if not names:
            return
        t0 = pd.Timestamp(f"{d.date()} 09:30", tz=NY).tz_convert("UTC")
        frames = [bars(data, names[i:i + 200], tf5, t0, t0 + pd.Timedelta(seconds=299), thr, adjustment="split")
                  for i in range(0, len(names), 200)]
        frames = [f for f in frames if not f.empty]
        (pd.concat(frames) if frames else pd.DataFrame()).to_parquet(p)

    with ThreadPoolExecutor(args.workers) as ex:
        for k, _ in enumerate(ex.map(first5_day, days)):
            if k % 50 == 0:
                log.info("first5 %d/%d", k, len(days))
    log.info("first-5-minute bars done")

    # 3) candidates: top-N relative opening volume -> minute bars that day
    f5 = []
    for d in days:
        p = odir / "first5" / f"{d.date()}.parquet"
        if p.exists():
            x = pd.read_parquet(p)
            if not x.empty:
                x = x.reset_index()
                t0 = pd.Timestamp(f"{d.date()} 09:30", tz=NY).tz_convert("UTC")
                x = x[pd.to_datetime(x["timestamp"], utc=True) == t0]     # only the 09:30-09:35 bar
                x["day"] = d
                f5.append(x)
    f5 = pd.concat(f5)
    f5.to_parquet(odir / "first5_all.parquet")
    v5 = f5.pivot_table(index="day", columns="symbol", values="volume")
    rv = v5 / v5.rolling(14, min_periods=10).mean().shift(1)

    def minute_day(d):
        p = odir / "minute" / f"{d.date()}.parquet"
        if p.exists() or d not in rv.index:
            return
        r = rv.loc[d].dropna()
        r = r[r >= 1.0].sort_values(ascending=False).head(args.top)
        if r.empty:
            return
        t0 = pd.Timestamp(f"{d.date()} 09:30", tz=NY).tz_convert("UTC")
        bars(data, list(r.index), TimeFrame.Minute, t0, t0 + pd.Timedelta(minutes=390) - pd.Timedelta(seconds=1),
             thr, adjustment="split").to_parquet(p)

    with ThreadPoolExecutor(args.workers) as ex:
        for k, _ in enumerate(ex.map(minute_day, days)):
            if k % 100 == 0:
                log.info("minute %d/%d", k, len(days))
    pd.DataFrame({"atr": atr.stack(), "avgvol": avgvol.stack(), "prev_close": prev_c.stack()}).to_parquet(odir / "daily_features.parquet")
    print("orb data complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sp100", "news", "orb"])
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--top", type=int, default=30, help="orb: candidates per day")
    ap.add_argument("--rate", type=int, default=150, help="max requests per minute")
    ap.add_argument("--workers", type=int, default=12, help="parallel requests (orb)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("alpaca").setLevel(logging.WARNING)
    settings = load_settings()
    {"sp100": cmd_sp100, "news": cmd_news, "orb": cmd_orb}[args.cmd](args, settings)


if __name__ == "__main__":
    main()
