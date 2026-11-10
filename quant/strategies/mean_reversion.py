"""Connors RSI(2) short-term mean reversion (Connors & Alvarez, 2008).

Long an index ETF when it is in a long-term uptrend (close > 200-day SMA) and short-term
oversold (RSI(2) < entry). Exit when close > 5-day SMA. Each symbol gets 1/N of the sleeve.
Designed to be decided from a price snapshot shortly before the close and executed with
market-on-close orders; backtest with lag=1 (next close) as the conservative baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import DailyStrategy, rsi


@dataclass
class RSI2MeanReversion(DailyStrategy):
    name: str = "rsi2"
    rebalance: str = "D"
    assets: list = field(default_factory=lambda: ["SPY", "QQQ"])
    rsi_period: int = 2
    entry: float = 10.0
    trend_sma: int = 200
    exit_sma: int = 5

    @property
    def symbols(self) -> list[str]:
        return list(self.assets)

    @property
    def warmup_days(self) -> int:
        return self.trend_sma + 1

    def signal_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        r = rsi(prices, self.rsi_period)
        trend = prices > prices.rolling(self.trend_sma).mean()
        exit_ = prices > prices.rolling(self.exit_sma).mean()
        entry = trend & (r < self.entry)
        # state machine per asset: in position from an entry until an exit signal
        pos = np.zeros(prices.shape)
        e, x = entry.to_numpy(), exit_.to_numpy()
        for j in range(prices.shape[1]):
            inpos = False
            for t in range(len(prices)):
                if inpos and x[t, j]:
                    inpos = False
                if not inpos and e[t, j]:
                    inpos = True
                pos[t, j] = 1.0 if inpos else 0.0
        w = pd.DataFrame(pos / len(self.assets), prices.index, prices.columns)
        return w
