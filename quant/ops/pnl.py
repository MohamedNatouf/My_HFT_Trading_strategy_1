"""Paper/live P&L versus backtest, and per-order slippage.

* actual daily returns: Alpaca portfolio history (``profit_loss`` / start-of-day equity,
  so deposits and withdrawals are excluded)
* expected daily returns: the SAME configured book (sleeves + risk + cost model) run
  through ``run_backtest`` over the same dates. Live trading decides ~20 min before the
  close and fills in the closing auction, so the comparable backtest is lag=0 on closes
  (for ``session: open`` it is lag=1 executed at the next open). Scaled by
  ``allocation_fraction`` (the rest of the account is assumed to sit in cash).
* slippage: every filled order whose client_order_id belongs to this book, fill price vs
  (a) the decision/reference price recorded in the journal and (b) that day's official close
  (or open for opening-auction orders). Positive bps = cost.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NY = "America/New_York"


# --- actual -----------------------------------------------------------------------
def actual_daily_returns(history) -> pd.Series:
    """From an alpaca-py PortfolioHistory (or any object with timestamp/equity/profit_loss)."""
    ts = pd.to_datetime(pd.Series(history.timestamp), unit="s", utc=True).dt.tz_convert(NY).dt.normalize().dt.tz_localize(None)
    eq = pd.Series(history.equity, dtype=float).to_numpy()
    pl = pd.Series(history.profit_loss, dtype=float).to_numpy()
    base = eq - pl
    r = np.where((base > 0) & np.isfinite(pl), pl / np.where(base > 0, base, 1), np.nan)
    s = pd.Series(r, index=pd.DatetimeIndex(ts), name="actual")
    s = s[~s.index.duplicated(keep="last")]
    # drop leading days before the account had any equity
    return s[(pd.Series(eq, index=s.index) > 0).to_numpy()].dropna()


# --- expected ---------------------------------------------------------------------
def expected_daily_returns(settings, start, end=None, source: str = "auto", prices=None, opens=None) -> pd.Series:
    from ..backtest import cost_model_bps, run_backtest
    from ..data import load_daily_closes
    from ..live.daily import build_sleeves
    from ..portfolio import RiskLimits, apply_risk, combine_sleeves

    cfg = settings.raw
    sleeves = build_sleeves(cfg)
    symbols = sorted({s for st, _ in sleeves for s in st.symbols})
    warm = max(st.warmup_days for st, _ in sleeves)
    start = pd.Timestamp(start)
    if prices is None:
        prices = load_daily_closes(symbols, start - pd.Timedelta(days=int(warm * 1.6) + 60), end,
                                   source=source, settings=settings)
    targets = apply_risk(prices, combine_sleeves(prices, sleeves), RiskLimits(**(cfg.get("risk") or {})))
    ex = cfg.get("execution") or {}
    costs = cost_model_bps(prices)
    band = float(ex.get("drift_band", 0.0))
    if ex.get("session", "close") == "open":
        if opens is None:
            opens = load_daily_closes(symbols, prices.index[0], end, source=source, settings=settings, field="open")
        res = run_backtest(prices, targets, cost_bps=costs, lag=1, open_prices=opens, trade_band=band)
    else:
        res = run_backtest(prices, targets, cost_bps=costs, lag=0, trade_band=band)
    frac = float(ex.get("allocation_fraction", 1.0))
    return (res.returns * frac).loc[start:].rename("expected")


@dataclass
class Comparison:
    daily: pd.DataFrame
    stats: dict


def compare(actual: pd.Series, expected: pd.Series) -> Comparison:
    df = pd.concat([actual.rename("actual"), expected.rename("expected")], axis=1, join="inner").dropna()
    df["diff"] = df["actual"] - df["expected"]
    df["cum_actual"] = (1 + df["actual"]).cumprod() - 1
    df["cum_expected"] = (1 + df["expected"]).cumprod() - 1
    n = len(df)
    stats = {
        "days": n,
        "cum_actual": float(df["cum_actual"].iloc[-1]) if n else np.nan,
        "cum_expected": float(df["cum_expected"].iloc[-1]) if n else np.nan,
        "tracking_error_ann": float(df["diff"].std(ddof=1) * np.sqrt(252)) if n > 2 else np.nan,
        "mean_diff_bps_per_day": float(df["diff"].mean() * 1e4) if n else np.nan,
        "correlation": float(df["actual"].corr(df["expected"])) if n > 2 else np.nan,
        "worst_day_diff_bps": float(df["diff"].abs().max() * 1e4) if n else np.nan,
    }
    return Comparison(df, stats)


# --- slippage ---------------------------------------------------------------------
def read_journal(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def decision_prices(journal: list[dict]) -> dict[str, dict]:
    """client_order_id -> {ref_price, tif, day} from rebalance records."""
    refs = {}
    for rec in journal:
        if (rec.get("event_type") or rec.get("event")) != "rebalance":
            continue
        for o in rec.get("orders", []):
            cid = o.get("client_order_id")
            if cid:
                ref = o.get("ref_price") or (o["est_notional"] / o["qty"] if o.get("qty") else None)
                refs[cid] = {"ref_price": ref, "tif": o.get("tif"), "day": rec.get("day")}
    return refs


def order_slippage(orders: list, journal: list[dict], closes: pd.DataFrame | None = None,
                   opens: pd.DataFrame | None = None, prefix: str = "q-") -> pd.DataFrame:
    """One row per filled order of this book. ``orders``: alpaca Order objects or dicts."""
    from ..live.stream import enum_val

    refs = decision_prices(journal)
    rows = []
    for o in orders:
        g = (lambda k: o.get(k)) if isinstance(o, dict) else (lambda k: getattr(o, k, None))
        cid = g("client_order_id") or ""
        if not cid.startswith(prefix) or enum_val(g("status")) not in ("filled", "partially_filled"):
            continue
        fill = float(g("filled_avg_price") or 0)
        qty = float(g("filled_qty") or 0)
        if fill <= 0 or qty <= 0:
            continue
        side = enum_val(g("side"))
        sign = 1.0 if side == "buy" else -1.0
        day = _ny_day(g("filled_at") or g("submitted_at"))
        tif = enum_val(g("time_in_force")) or refs.get(cid, {}).get("tif")
        ref = refs.get(cid, {}).get("ref_price")
        bench_tbl = opens if tif == "opg" else closes
        bench = None
        sym = g("symbol")
        if bench_tbl is not None and sym in bench_tbl.columns and day in bench_tbl.index:
            bench = float(bench_tbl.at[day, sym])
        rows.append({
            "day": day, "symbol": sym, "side": side, "qty": qty, "fill": fill, "notional": fill * qty, "tif": tif,
            "ref_price": ref, "auction_price": bench,
            "slip_vs_decision_bps": sign * (fill / ref - 1) * 1e4 if ref else np.nan,
            "slip_vs_auction_bps": sign * (fill / bench - 1) * 1e4 if bench else np.nan,
            "client_order_id": cid,
        })
    return pd.DataFrame(rows)


def _ny_day(ts):
    t = pd.Timestamp(ts) if ts is not None else pd.NaT
    if t is pd.NaT:
        return pd.NaT
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(NY).normalize().tz_localize(None)


def slippage_summary(slip: pd.DataFrame) -> dict:
    if slip.empty:
        return {"orders": 0}
    w = slip["notional"]
    out = {"orders": int(len(slip)), "notional": float(w.sum())}
    for col in ("slip_vs_decision_bps", "slip_vs_auction_bps"):
        v = slip[col]
        ok = v.notna()
        if ok.any():
            out[col.replace("_bps", "_mean_bps")] = float(v[ok].mean())
            out[col.replace("_bps", "_wavg_bps")] = float((v[ok] * w[ok]).sum() / w[ok].sum())
            out[col.replace("_bps", "_cost_usd")] = float((v[ok] / 1e4 * w[ok]).sum())
    return out
