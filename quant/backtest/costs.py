"""Per-symbol transaction cost model + cost-sensitivity analysis.

cost_bps(symbol, day) = half_spread_bps(symbol) + impact_coef * daily_vol_bps(symbol, day) + fee_bps

* half-spread: measured from live quotes when available (``data/cache/spreads.json``,
  written by ``scripts/run_quant.py spreads``), otherwise a liquidity-tier default.
  ETF closing-auction fills (MOC) usually beat the quoted spread, so this is conservative.
* volatility slippage: a small fraction of the trailing daily volatility (causal:
  uses returns up to t only). Captures wider spreads / worse fills in turbulent markets.
* Alpaca charges no commission on US equities; ``fee_bps`` covers SEC/TAF fees (~0.3 bp on sells).

All numbers are per unit of one-way traded notional, the same unit as ``run_backtest``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Typical quoted half-spreads in bps (2024-2026 averages, rounded up).
TIER1 = {  # > $1bn/day, 1-cent spreads
    "SPY", "QQQ", "IVV", "VOO", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "AGG", "HYG", "LQD",
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB", "SHY", "BIL", "SHV", "VEU", "VNQ",
}
TIER2 = {  # liquid but wider
    "RSP", "DBC", "EWJ", "EWY", "EWT", "FXI", "KWEB", "IJT", "IJS", "ICLN", "TIP", "BND", "VWO",
    "IAU", "SLV", "USO", "MTUM", "QUAL", "USMV", "VTV", "VUG", "IWD", "IWF", "MDY", "DIA",
}
TIER_BPS = {1: 0.75, 2: 2.5, 3: 6.0}
SPREAD_CACHE = Path("data/cache/spreads.json")


def default_half_spread_bps(symbol: str) -> float:
    if symbol in TIER1:
        return TIER_BPS[1]
    if symbol in TIER2:
        return TIER_BPS[2]
    return TIER_BPS[3]


def load_measured_spreads(path: str | Path = SPREAD_CACHE) -> dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return {k: float(v) for k, v in json.loads(p.read_text()).get("half_spread_bps", {}).items()}
    except Exception:
        return {}


def half_spread_bps(symbols, overrides: dict[str, float] | None = None, use_measured: bool = True) -> pd.Series:
    measured = load_measured_spreads() if use_measured else {}
    overrides = overrides or {}
    out = {}
    for s in symbols:
        if s in overrides:
            out[s] = float(overrides[s])
        elif s in measured:
            # never trust a single snapshot below the tier floor/2 (quotes can be momentarily locked)
            out[s] = max(measured[s], default_half_spread_bps(s) / 2)
        else:
            out[s] = default_half_spread_bps(s)
    return pd.Series(out, dtype=float)


def cost_model_bps(prices: pd.DataFrame, impact_coef: float = 0.03, vol_lookback: int = 20,
                   fee_bps: float = 0.2, half_spreads: pd.Series | dict | None = None,
                   use_measured: bool = True) -> pd.DataFrame:
    """Day x symbol cost matrix in bps (causal)."""
    hs = half_spread_bps(prices.columns, use_measured=use_measured) if half_spreads is None \
        else pd.Series(half_spreads, dtype=float).reindex(prices.columns)
    hs = hs.fillna(pd.Series({s: default_half_spread_bps(s) for s in prices.columns}))
    vol = prices.pct_change(fill_method=None).rolling(vol_lookback, min_periods=5).std() * 1e4
    vol = vol.fillna(vol.expanding().median()).fillna(100.0)   # 1%/day before enough history
    return vol.mul(impact_coef).add(hs, axis=1).add(fee_bps)


def cost_sensitivity(prices: pd.DataFrame, targets: pd.DataFrame, base_cost, multipliers=(0, 0.5, 1, 2, 3, 5),
                     **bt_kwargs) -> pd.DataFrame:
    """Re-run the backtest with costs scaled by each multiplier.

    Returns one row per multiplier plus ``breakeven_multiplier`` in ``.attrs``: the cost
    multiple at which the net Sharpe reaches zero (linear interpolation; inf if never).
    """
    from .engine import run_backtest

    rows = []
    for k in multipliers:
        cost = base_cost * k
        r = run_backtest(prices, targets, cost_bps=cost, **bt_kwargs)
        rows.append({"cost_x": k, "sharpe": r.stats["sharpe"], "cagr": r.stats["cagr"],
                     "max_drawdown": r.stats["max_drawdown"],
                     "avg_cost_bps_per_trade": float((r.costs.sum() / max(r.turnover.sum(), 1e-12)) * 1e4),
                     "cost_drag_per_year": float(r.costs.mean() * 252)})
    df = pd.DataFrame(rows).set_index("cost_x")
    be = np.inf
    s = df["sharpe"].to_numpy()
    xs = df.index.to_numpy(dtype=float)
    for i in range(1, len(s)):
        if s[i - 1] > 0 >= s[i]:
            be = xs[i - 1] + (xs[i] - xs[i - 1]) * s[i - 1] / (s[i - 1] - s[i])
            break
    if s[0] <= 0:
        be = 0.0
    df.attrs["breakeven_multiplier"] = be
    return df
