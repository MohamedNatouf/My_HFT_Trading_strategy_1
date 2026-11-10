"""Opening-range breakout on "stocks in play" (item 14).

Zarattini, Barbon & Aziz (2024), "A Profitable Day Trading Strategy for the U.S. Equity Market":
* universe each day (known before the open): price > $5, 14-day avg volume > 1M, ATR(14) > $0.50
* stocks in play: relative volume of the first 5-minute bar (vs its 14-day average) >= 1,
  top ``top_n`` by relative volume
* direction from the first 5-minute candle: up -> buy-stop at its high, down -> sell-stop at
  its low, doji -> no trade
* stop loss 10% of ATR(14) from the entry; otherwise exit at the close (last minute bar)
* risk ``risk_per_trade`` of equity per trade, total notional capped at ``max_leverage``

Fill model (deliberately conservative):
* entry when a minute bar trades through the level; fill = max(level, bar open) (+ slippage
  via ``cost_bps``). ``entry_delay=1``: fill at the CLOSE of the bar after the trigger
  (delayed-entry robustness test)
* stop: fill = min(stop, bar open) for longs (gaps through the stop are not forgiven)
* if entry and stop are both touched in the entry bar, assume the stop was hit
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NY = "America/New_York"


@dataclass
class ORBConfig:
    top_n: int = 20
    min_rv: float = 1.0
    stop_atr: float = 0.10
    risk_per_trade: float = 0.01
    max_leverage: float = 4.0
    cost_bps: float = 1.0          # per side, on notional
    entry_delay: int = 0


def simulate_day(first5: pd.DataFrame, minutes: dict[str, pd.DataFrame], feats: pd.DataFrame,
                 cfg: ORBConfig) -> tuple[float, list[dict]]:
    """first5: index symbol, cols open/high/low/close/rv. minutes: symbol -> minute bars
    (NY time index) from 09:35 on. feats: index symbol, col atr. Returns (day return, trades)."""
    cand = first5[first5["rv"] >= cfg.min_rv].sort_values("rv", ascending=False).head(cfg.top_n)
    trades = []
    per_trade_cap = cfg.max_leverage / max(cfg.top_n, 1)
    c = cfg.cost_bps / 1e4
    for sym, fb in cand.iterrows():
        if fb["close"] == fb["open"] or sym not in minutes or sym not in feats.index:
            continue
        atr = float(feats.at[sym, "atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue
        side = 1 if fb["close"] > fb["open"] else -1
        level = fb["high"] if side > 0 else fb["low"]
        m = minutes[sym]
        o, h, lo_, cl = (m[k].to_numpy() for k in ("open", "high", "low", "close"))
        trig = np.where(h >= level)[0] if side > 0 else np.where(lo_ <= level)[0]
        if len(trig) == 0:
            continue
        i = int(trig[0])
        if cfg.entry_delay:
            i += cfg.entry_delay
            if i >= len(m):
                continue
            entry = cl[i]
            start_check = i + 1
        else:
            entry = max(level, o[i]) if side > 0 else min(level, o[i])
            start_check = i
        stop = entry - side * cfg.stop_atr * atr
        exit_px, how = cl[-1], "close"
        for j in range(start_check, len(m)):
            if (side > 0 and lo_[j] <= stop) or (side < 0 and h[j] >= stop):
                exit_px = min(stop, o[j]) if side > 0 else max(stop, o[j])
                if j == i and cfg.entry_delay == 0:
                    exit_px = stop
                how = "stop"
                break
        risk_per_share = cfg.stop_atr * atr
        notional_frac = min(cfg.risk_per_trade * entry / risk_per_share, per_trade_cap)
        r = side * (exit_px / entry - 1) - 2 * c
        trades.append({"symbol": sym, "side": side, "entry": entry, "exit": exit_px, "how": how,
                       "weight": notional_frac, "ret": r, "pnl": notional_frac * r})
    return float(sum(t["pnl"] for t in trades)), trades
