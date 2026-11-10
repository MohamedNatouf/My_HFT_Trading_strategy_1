"""Intraday strategies on 1-minute RTH bars of a single liquid ETF (SPY / QQQ).

NoiseAreaMomentum - Zarattini, Aziz & Barbon (2024), "Beat the Market: An Effective
    Intraday Momentum Strategy for S&P500 ETF (SPY)", SFI WP 24-97.
    * sigma(d, m) = mean over the previous 14 days of |close(m)/open - 1| at minute-of-day m
    * Upper band = max(open, prev_close) * (1 + sigma); lower = min(open, prev_close) * (1 - sigma)
    * decisions only every 30 min (10:00, 10:30, ... 15:30): long above the upper band,
      short below the lower band; exit a long when price < max(upper band, VWAP)
      (mirror for shorts); always flat at the close
    * leverage = min(max_leverage, target_daily_vol / trailing 14-day daily vol)
    Published 2007-2024 net Sharpe ~1.3; no independent out-of-sample since publication,
    so treat everything after May 2024 as the out-of-sample test.
    Our replication (Alpaca SIP SPY 2018-2026, 1bp cost): Sharpe ~1.0 pre-June-2024 but
    NEGATIVE (-0.4..-0.5) since publication and negative at 3bp cost -> paper only.

LastHalfHourMomentum - Gao, Han, Li & Zhou (2018), "Market Intraday Momentum", JFE.
    The return from the previous close to 10:00 sets the direction of a position held
    from 15:30 to the close. Our replication (SPY 2018-2026): Sharpe -1.06 -> do not trade.

Bar convention: Alpaca minute bars are stamped with the bar START; minute index m (0..389)
is the bar starting at 09:30+m, whose close is the price at 09:31+m. The decision
"at 10:00" therefore uses the close of bar m=29.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NY = "America/New_York"
MINUTES = 390


def minute_matrix(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pivot RTH minute bars to day x minute-of-day matrices (open/close/vwap/volume)."""
    idx = pd.DatetimeIndex(bars.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    ny = idx.tz_convert(NY)
    m = (ny.hour * 60 + ny.minute - 570).to_numpy()
    keep = (m >= 0) & (m < MINUTES)
    df = bars.loc[keep].copy()
    df["_day"] = ny[keep].normalize().tz_localize(None)
    df["_m"] = m[keep]
    out = {}
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = df.pivot_table(index="_day", columns="_m", values=col, aggfunc="last").reindex(columns=range(MINUTES))
    out["close"] = out["close"].ffill(axis=1)
    out["volume"] = out["volume"].fillna(0.0)
    typical = (out["high"].ffill(axis=1) + out["low"].ffill(axis=1) + out["close"]) / 3
    pv = (typical * out["volume"]).cumsum(axis=1)
    cv = out["volume"].cumsum(axis=1).replace(0, np.nan)
    out["vwap"] = (pv / cv).ffill(axis=1)
    # day open = first available open
    out["day_open"] = out["open"].bfill(axis=1).iloc[:, 0]   # within-day, minute 0 (not cross-day)
    out["day_close"] = out["close"].ffill(axis=1).iloc[:, -1]
    return out


def noise_bands(mat: dict[str, pd.DataFrame], lookback_days: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    op = mat["day_open"]
    move = (mat["close"].div(op, axis=0) - 1).abs()
    sigma = move.shift(1).rolling(lookback_days, min_periods=lookback_days).mean()  # previous days only
    prev_close = mat["day_close"].shift(1)
    hi_ref = np.maximum(op, prev_close.fillna(op))
    lo_ref = np.minimum(op, prev_close.fillna(op))
    ub = (1 + sigma).mul(hi_ref, axis=0)
    lb = (1 - sigma).mul(lo_ref, axis=0)
    return ub, lb


def noise_step(pos: int, price: float, ub: float, lb: float, vwap: float) -> int:
    """One decision of the noise-area state machine. Shared by backtest and live."""
    if not np.isfinite(ub) or not np.isfinite(lb) or not np.isfinite(price):
        return 0
    stopped = 0
    if pos > 0 and price < max(ub, vwap if np.isfinite(vwap) else ub):
        pos, stopped = 0, 1
    elif pos < 0 and price > min(lb, vwap if np.isfinite(vwap) else lb):
        pos, stopped = 0, -1
    if pos == 0:
        # a stop-out may flip to the opposite side but never re-enter the same side
        if price > ub and stopped != 1:
            pos = 1
        elif price < lb and stopped != -1:
            pos = -1
    return pos


DECISION_MINUTES = list(range(29, 360, 30))  # closes at 10:00, 10:30, ..., 15:30


@dataclass
class NoiseAreaMomentum:
    name: str = "noise_area"
    lookback_days: int = 14
    target_daily_vol: float = 0.02
    max_leverage: float = 4.0
    allow_short: bool = True
    cost_bps: float = 1.0    # per unit of traded notional (SPY spread ~0.3bp + slippage)

    def leverage(self, mat) -> pd.Series:
        dret = mat["day_close"].pct_change(fill_method=None)
        vol = dret.shift(1).rolling(self.lookback_days, min_periods=self.lookback_days).std()
        return (self.target_daily_vol / vol).clip(upper=self.max_leverage)

    def backtest(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Daily results: columns ret (net), gross, trades, lev."""
        mat = minute_matrix(bars)
        ub, lb = noise_bands(mat, self.lookback_days)
        lev = self.leverage(mat)
        close, vwap = mat["close"].to_numpy(), mat["vwap"].to_numpy()
        U, L = ub.to_numpy(), lb.to_numpy()
        rows = []
        c = self.cost_bps / 1e4
        for i, day in enumerate(mat["close"].index):
            k = lev.iloc[i]
            if not np.isfinite(k):
                continue
            last = int(np.max(np.where(np.isfinite(close[i]))[0])) if np.isfinite(close[i]).any() else -1
            if last < DECISION_MINUTES[0]:
                continue
            pos, eq, trades, turnover = 0, 1.0, 0, 0.0
            dec = [m for m in DECISION_MINUTES if m < last] + [last]
            for j, m in enumerate(dec):
                if j > 0 and pos != 0:
                    eq *= 1 + pos * k * (close[i, m] / close[i, dec[j - 1]] - 1)
                new = 0 if m == last else noise_step(pos, close[i, m], U[i, m], L[i, m], vwap[i, m])
                if not self.allow_short and new < 0:
                    new = 0
                if new != pos:
                    turnover += abs(new - pos) * k
                    trades += 1
                    pos = new
            gross = eq - 1
            rows.append((day, gross - turnover * c, gross, trades, k))
        return pd.DataFrame(rows, columns=["day", "ret", "gross", "trades", "lev"]).set_index("day")

    # ---- live helpers -------------------------------------------------------
    def bands_for_today(self, history_bars: pd.DataFrame, today_open: float) -> dict:
        """Compute today's sigma curve / leverage from completed prior days."""
        mat = minute_matrix(history_bars)
        op = mat["day_open"]
        move = (mat["close"].div(op, axis=0) - 1).abs().tail(self.lookback_days)
        sigma = move.mean(axis=0).to_numpy()
        prev_close = float(mat["day_close"].iloc[-1])
        dret = mat["day_close"].pct_change(fill_method=None).tail(self.lookback_days)
        lev = float(min(self.max_leverage, self.target_daily_vol / dret.std()))
        ub = max(today_open, prev_close) * (1 + sigma)
        lb = min(today_open, prev_close) * (1 - sigma)
        return {"ub": ub, "lb": lb, "leverage": lev, "prev_close": prev_close}


@dataclass
class LastHalfHourMomentum:
    name: str = "last_half_hour"
    use_r12: bool = False
    cost_bps: float = 1.0

    def backtest(self, bars: pd.DataFrame) -> pd.DataFrame:
        mat = minute_matrix(bars)
        close = mat["close"]
        prev_close = mat["day_close"].shift(1)
        r1 = close[29] / prev_close - 1
        sig = np.sign(r1)
        if self.use_r12:
            sig = np.sign(r1 + (close[359] / close[329] - 1))
        r_last = mat["day_close"] / close[359] - 1   # 15:30 -> close
        gross = sig * r_last
        ret = gross - (sig != 0) * 2 * self.cost_bps / 1e4
        out = pd.DataFrame({"ret": ret, "gross": gross, "signal": sig}).dropna()
        return out
