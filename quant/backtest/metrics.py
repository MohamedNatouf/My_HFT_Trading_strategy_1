"""Performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(returns: pd.Series) -> float:
    eq = (1 + returns.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd and sd > 0 else float("nan")


def summarize(returns: pd.Series, periods_per_year: int = 252, turnover: pd.Series | None = None,
              weights: pd.DataFrame | None = None) -> dict:
    r = returns.dropna()
    if r.empty:
        return {}
    n = len(r)
    years = n / periods_per_year
    total = float((1 + r).prod() - 1)
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan")
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    downside = r[r < 0].std(ddof=1)
    mdd = max_drawdown(r)
    out = {
        "start": str(r.index[0].date()) if hasattr(r.index[0], "date") else str(r.index[0]),
        "end": str(r.index[-1].date()) if hasattr(r.index[-1], "date") else str(r.index[-1]),
        "years": round(years, 2),
        "total_return": total,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe(r, periods_per_year),
        "sortino": float(r.mean() / downside * np.sqrt(periods_per_year)) if downside and downside > 0 else float("nan"),
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 and np.isfinite(cagr) else float("nan"),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurt()),  # excess
        "hit_rate": float((r > 0).sum() / max(1, (r != 0).sum())),
        "best_day": float(r.max()),
        "worst_day": float(r.min()),
    }
    if turnover is not None and len(turnover):
        out["turnover_per_year"] = float(turnover.sum() / years) if years > 0 else float("nan")
    if weights is not None and len(weights):
        gross = weights.abs().sum(axis=1)
        out["avg_gross_exposure"] = float(gross.mean())
        out["time_in_market"] = float((gross > 1e-9).mean())
    return out


def format_stats(stats: dict) -> str:
    pct = {"total_return", "cagr", "vol", "max_drawdown", "hit_rate", "best_day", "worst_day",
           "avg_gross_exposure", "time_in_market"}
    parts = []
    for k, v in stats.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.2%}" if k in pct else f"{k}={v:.2f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)
