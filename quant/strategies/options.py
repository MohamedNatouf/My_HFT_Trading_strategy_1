"""Volatility-risk-premium research (item 15): cash-secured puts and put credit spreads on SPY.

Free data only, so options are priced synthetically and the engine is validated against a
real index:

* underlying: S&P 500 index (^GSPC), dividend yield ``q``
* implied vol: VIX (30-day SPX implied vol) scaled by ``atm_scale`` (VIX sits above ATM IV
  because it integrates the skew) plus a linear put skew: iv(K) = atm + skew * ln(S/K)
* rate: 13-week T-bill (^IRX)
* validation: the ATM cash-secured put sleeve is compared with the CBOE PutWrite index (^PUT),
  which is built from real SPX option prices

Every position is marked to market daily (so crash drawdowns are visible, not hidden until
expiry). Costs: ``cost_bps`` of the underlying per option leg per transaction (SPY options
at 1-2 cents wide on a ~$600 underlying are 0.2-0.4 bp; the default 1 bp is conservative).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _ncdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2.0)))


def bs_put(S, K, T, r, q, iv):
    S, K, T, iv = (np.asarray(v, dtype=float) for v in (S, K, T, iv))
    T = np.maximum(T, 1e-8)
    iv = np.maximum(iv, 1e-4)
    d1 = (np.log(S / K) + (r - q + 0.5 * iv * iv) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    return K * np.exp(-r * T) * _ncdf(-d2) - S * np.exp(-q * T) * _ncdf(-d1)


def put_delta(S, K, T, r, q, iv):
    T = max(T, 1e-8)
    d1 = (math.log(S / K) + (r - q + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
    return -math.exp(-q * T) * float(_ncdf(-d1))


def strike_for_delta(S, T, r, q, iv_fn, target_delta):
    """Put strike whose (skew-adjusted) delta is ``target_delta`` (e.g. -0.20); bisection."""
    lo, hi = S * 0.5, S * 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        d = put_delta(S, mid, T, r, q, iv_fn(mid))
        if d < target_delta:   # too deep (more negative) -> lower strike
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


@dataclass
class PutSellConfig:
    kind: str = "spread"          # 'spread' | 'csp' (cash-secured put)
    dte: int = 21                 # trading days to expiry (~1 month)
    short_delta: float = -0.20    # 'spread': short-put delta; 'csp': ignored if atm=True
    atm: bool = False             # csp at the money (PutWrite replication)
    width: float = 0.05           # spread width as a fraction of S (long put = short strike - width*S)
    risk_per_trade: float = 0.05  # spread: max loss per cycle as a fraction of sleeve equity
    atm_scale: float = 0.90
    skew: float = 0.75            # iv add-on per unit of log-moneyness below the money
    q: float = 0.015
    cost_bps: float = 1.0         # per leg per transaction, bps of underlying
    vix_stop: float | None = None  # skip new entries when VIX > this level (None = always trade)


def simulate_put_selling(spx: pd.Series, vix: pd.Series, rate: pd.Series, cfg: PutSellConfig) -> pd.DataFrame:
    """Daily sleeve returns. Positions open on day t's close and are held to expiry
    (``dte`` trading days later), then rolled. Returns columns: ret, credit_yield, in_trade."""
    df = pd.concat([spx.rename("S"), vix.rename("V"), rate.rename("r")], axis=1).dropna()
    S, V, R = df["S"].to_numpy(), df["V"].to_numpy() / 100, df["r"].to_numpy()
    n = len(df)
    eq = 1.0
    rets = np.zeros(n)
    credit_y = np.full(n, np.nan)
    in_trade = np.zeros(n, dtype=bool)
    pos = None    # dict(K1, K2, units, expiry_idx, value)
    c = cfg.cost_bps / 1e4
    T_year = 252.0

    def ivf(s, atm):
        return lambda K: atm * cfg.atm_scale + cfg.skew * max(0.0, math.log(s / K))

    def value(p, i):
        T = max(p["exp"] - i, 0) / T_year
        iv1, iv2 = ivf(S[i], V[i])(p["K1"]), ivf(S[i], V[i])(p["K2"]) if p["K2"] else 0.0
        if T <= 0:
            v = max(p["K1"] - S[i], 0.0) - (max(p["K2"] - S[i], 0.0) if p["K2"] else 0.0)
        else:
            v = float(bs_put(S[i], p["K1"], T, R[i], cfg.q, iv1))
            if p["K2"]:
                v -= float(bs_put(S[i], p["K2"], T, R[i], cfg.q, iv2))
        return v

    for i in range(n):
        daily_cash = (1 + R[i - 1]) ** (1 / T_year) - 1 if i > 0 else 0.0
        pnl = 0.0
        if pos is not None:
            v = value(pos, i)
            pnl += pos["units"] * (pos["value"] - v)          # short the spread
            pos["value"] = v
            in_trade[i] = True
            if i >= pos["exp"]:
                legs = 2 if pos["K2"] else 1
                if v > 0:                                      # close ITM legs at intrinsic + costs
                    pnl -= pos["units"] * legs * c * S[i]
                pos = None
        new_eq = eq * (1 + daily_cash) + pnl
        rets[i] = new_eq / eq - 1 if eq > 0 else 0.0
        eq = new_eq
        if pos is None and i + cfg.dte < n and (cfg.vix_stop is None or V[i] * 100 <= cfg.vix_stop):
            T = cfg.dte / T_year
            iv_fn = ivf(S[i], V[i])
            if cfg.kind == "csp":
                K1 = S[i] if cfg.atm else strike_for_delta(S[i], T, R[i], cfg.q, iv_fn, cfg.short_delta)
                K2 = 0.0
                prem = float(bs_put(S[i], K1, T, R[i], cfg.q, iv_fn(K1)))
                units = eq / K1                                # fully collateralised
                legs = 1
            else:
                K1 = strike_for_delta(S[i], T, R[i], cfg.q, iv_fn, cfg.short_delta)
                K2 = K1 - cfg.width * S[i]
                prem = float(bs_put(S[i], K1, T, R[i], cfg.q, iv_fn(K1)) - bs_put(S[i], K2, T, R[i], cfg.q, iv_fn(K2)))
                max_loss = (K1 - K2) - prem
                units = cfg.risk_per_trade * eq / max(max_loss, 1e-9)
                legs = 2
            cost = units * legs * c * S[i]
            eq -= cost
            rets[i] = (1 + rets[i]) * (1 - cost / (eq + cost)) - 1
            pos = {"K1": K1, "K2": K2, "units": units, "exp": i + cfg.dte, "value": prem}
            credit_y[i] = units * prem / eq
    return pd.DataFrame({"ret": rets, "credit_yield": credit_y, "in_trade": in_trade}, index=df.index)


# --- calibration helpers (real option prices from Alpaca, 2024+) -----------------------
def occ_symbol(underlying: str, expiry, strike: float, right: str = "P") -> str:
    e = pd.Timestamp(expiry)
    return f"{underlying}{e:%y%m%d}{right}{int(round(strike * 1000)):08d}"


def implied_vol_put(price: float, S: float, K: float, T: float, r: float, q: float) -> float:
    """Bisection IV; NaN if the price is outside no-arbitrage bounds."""
    lo_bound = max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
    if not np.isfinite(price) or price <= lo_bound + 1e-6 or price >= K:
        return float("nan")
    lo, hi = 1e-3, 3.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if float(bs_put(S, K, T, r, q, mid)) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def monthly_expiries(start, end) -> list[pd.Timestamp]:
    """Third Friday of each month (standard monthly option expiration)."""
    out = []
    for m in pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M"):
        fridays = pd.date_range(m.start_time, m.end_time, freq="W-FRI")
        out.append(fridays[2])
    return out
