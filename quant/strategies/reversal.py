"""Short-term reversal candidates (research, item 12).

SectorReversal - weekly: buy the sector ETFs that lagged SPY most over ``lookback`` days
    (optionally short the leaders). Liquid, cheap to trade, long history.
StockReversal  - large caps (S&P 100): rank by the ``lookback``-day return MINUS the return of
    the stock's own sector ETF (industry-adjusted reversal, Da/Liu/Schaumburg 2014 style),
    long the worst ``frac``, short the best ``frac``, equal weight, dollar neutral.

Both are causal (row t uses closes up to t) and traded at the close (or next close).
Long-short weights: longs sum to +gross/2, shorts to -gross/2 (net 0) when ``long_short``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.universe import SECTOR_ETFS, SP100
from .base import DailyStrategy, trailing_return

ORIGINAL_SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]   # since 1998


def _rank_select(score: pd.DataFrame, n: int, ascending: bool) -> pd.DataFrame:
    r = score.rank(axis=1, ascending=ascending, method="first")
    return (r <= n) & score.notna()


@dataclass
class SectorReversal(DailyStrategy):
    name: str = "sector_reversal"
    rebalance: str = "W"
    sectors: list = field(default_factory=lambda: list(ORIGINAL_SECTORS))
    benchmark: str = "SPY"
    lookback: int = 5
    n: int = 3
    long_short: bool = False
    gross: float = 1.0

    @property
    def symbols(self) -> list[str]:
        return list(self.sectors) + [self.benchmark]

    @property
    def warmup_days(self) -> int:
        return self.lookback + 2

    def signal_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        sec = prices[self.sectors]
        rel = trailing_return(sec, self.lookback).sub(trailing_return(prices[self.benchmark], self.lookback), axis=0)
        avail = rel.notna().sum(axis=1)
        n = np.minimum(self.n, (avail // 2).clip(lower=0)).to_numpy()
        w = pd.DataFrame(0.0, prices.index, prices.columns)
        rk_lo = rel.rank(axis=1, ascending=True, method="first")
        rk_hi = rel.rank(axis=1, ascending=False, method="first")
        longs = rk_lo.le(n, axis=0) & rel.notna()
        shorts = rk_hi.le(n, axis=0) & rel.notna()
        cnt = longs.sum(axis=1).replace(0, np.nan)
        side = self.gross / 2 if self.long_short else self.gross
        w[self.sectors] = longs.astype(float).div(cnt, axis=0).fillna(0.0) * side
        if self.long_short:
            w[self.sectors] -= shorts.astype(float).div(cnt, axis=0).fillna(0.0) * side
        return w


@dataclass
class StockReversal(DailyStrategy):
    name: str = "stock_reversal"
    rebalance: str = "W"
    universe: dict = field(default_factory=lambda: dict(SP100))
    lookback: int = 5
    frac: float = 0.1
    residual: str = "sector"      # 'sector' | 'market' | 'none'
    long_short: bool = True
    gross: float = 1.0
    skip_days: int = 0            # 1 = skip the most recent day (bid-ask bounce control)

    @property
    def symbols(self) -> list[str]:
        etfs = sorted(set(self.universe.values())) if self.residual == "sector" else []
        return sorted(self.universe) + etfs + (["SPY"] if self.residual == "market" else [])

    @property
    def warmup_days(self) -> int:
        return self.lookback + self.skip_days + 2

    def signal_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        stocks = sorted(self.universe)
        p = prices.shift(self.skip_days) if self.skip_days else prices
        r = trailing_return(p[stocks], self.lookback)
        if self.residual == "sector":
            etf_r = trailing_return(p[sorted(set(self.universe.values()))], self.lookback)
            bench = pd.DataFrame({s: etf_r[self.universe[s]] for s in stocks})
            r = r - bench
        elif self.residual == "market":
            r = r.sub(trailing_return(p["SPY"], self.lookback), axis=0)
        n = (r.notna().sum(axis=1) * self.frac).round().astype(int).clip(lower=1)
        rk_lo = r.rank(axis=1, ascending=True, method="first")
        rk_hi = r.rank(axis=1, ascending=False, method="first")
        longs = rk_lo.le(n, axis=0) & r.notna()
        shorts = rk_hi.le(n, axis=0) & r.notna()
        side = self.gross / 2 if self.long_short else self.gross
        w = pd.DataFrame(0.0, prices.index, prices.columns)
        w[stocks] = longs.astype(float).div(n, axis=0) * side
        if self.long_short:
            w[stocks] -= shorts.astype(float).div(n, axis=0) * side
        w[r.notna().sum(axis=1) < 20] = 0.0
        return w
