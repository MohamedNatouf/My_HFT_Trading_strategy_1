"""Research harness for the quant/ package.

Examples (repo root):
    python scripts/research.py daily                       # all daily strategies + portfolio
    python scripts/research.py daily --source yahoo --start 2005-01-01 --cost-bps 5
    python scripts/research.py overfit                     # PBO / DSR / walk-forward on momentum grid
    python scripts/research.py intraday --symbol SPY --start 2019-01-01   # needs Alpaca keys
    python scripts/research.py costs                       # per-symbol cost model, cost sensitivity,
                                                           # risk-layer variants, close vs open execution

Every number is net of ``--cost-bps`` per unit of one-way turnover, with next-close
execution (lag=1) unless stated otherwise.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.backtest import (cost_model_bps, cost_sensitivity, deflated_sharpe_ratio, probability_of_backtest_overfitting,  # noqa: E402
                            run_backtest, sharpe, summarize, walk_forward)
from quant.config import load_settings  # noqa: E402
from quant.data import load_daily_closes  # noqa: E402
from quant.portfolio import RiskLimits, apply_risk, combine_sleeves  # noqa: E402
from quant.strategies import DAILY_STRATEGIES, build_daily  # noqa: E402
from quant.strategies.momentum import MultiFactorMomentum  # noqa: E402

COLS = ["cagr", "vol", "sharpe", "sortino", "max_drawdown", "calmar", "turnover_per_year", "time_in_market"]


def fmt_table(rows: dict[str, dict]) -> str:
    df = pd.DataFrame(rows).T.reindex(columns=["start", "end"] + COLS)
    out = df.copy()
    for c in ("cagr", "vol", "max_drawdown", "time_in_market"):
        out[c] = df[c].map(lambda v: f"{v:6.1%}" if pd.notna(v) else "")
    for c in ("sharpe", "sortino", "calmar", "turnover_per_year"):
        out[c] = df[c].map(lambda v: f"{v:5.2f}" if pd.notna(v) else "")
    return out.to_string()


def benchmark(prices: pd.DataFrame, weights: dict[str, float], cost_bps: float, start) -> dict:
    t = pd.DataFrame(np.nan, prices.index, list(weights))
    month_end = prices.index.to_series().dt.to_period("M") != prices.index.to_series().shift(-1).dt.to_period("M")
    for s, w in weights.items():
        t.loc[month_end.to_numpy(), s] = w
    res = run_backtest(prices[list(weights)], t, cost_bps=cost_bps, lag=1, start=start)
    return res


def cmd_daily(args, settings):
    cfg = settings.raw
    strategies = {name: build_daily(name) for name in DAILY_STRATEGIES}
    symbols = sorted({s for st in strategies.values() for s in st.symbols} | {"SPY", "AGG"})
    prices = load_daily_closes(symbols, args.start, args.end, source=args.source, settings=settings)
    print(f"loaded {prices.shape[1]} symbols, {prices.index[0].date()} -> {prices.index[-1].date()}")
    rows, rets = {}, {}
    # common evaluation start: after the longest warm-up of all strategies, once core ETFs exist
    for name, st in strategies.items():
        t = st.target_weights(prices[st.symbols])
        for lag in ([1, 0] if name == "rsi2" else [1]):
            res = run_backtest(prices[st.symbols], t, cost_bps=args.cost_bps, lag=lag)
            key = name if lag == 1 else f"{name} (MOC, lag0)"
            rows[key] = res.stats
            rets[key] = res.returns
    sleeves = [(build_daily(s["strategy"], **(s.get("params") or {})), float(s["weight"])) for s in cfg["sleeves"]]
    port_syms = sorted({s for st, _ in sleeves for s in st.symbols})
    pt = combine_sleeves(prices[port_syms], sleeves)
    pt_risk = apply_risk(prices[port_syms], pt, RiskLimits(**cfg.get("risk", {})))
    for key, tt in (("PORTFOLIO (raw)", pt), ("PORTFOLIO (vol-target+caps)", pt_risk)):
        res = run_backtest(prices[port_syms], tt, cost_bps=args.cost_bps, lag=1)
        rows[key], rets[key] = res.stats, res.returns
    common_start = max(r.index[0] for r in rets.values())
    for key, w in (("BENCH SPY", {"SPY": 1.0}), ("BENCH 60/40 SPY/AGG", {"SPY": 0.6, "AGG": 0.4})):
        res = benchmark(prices, w, args.cost_bps, start=common_start)
        rows[key], rets[key] = res.stats, res.returns
    print(f"\n=== Full sample (each from its own start), cost={args.cost_bps}bp, next-close execution ===")
    print(fmt_table(rows))
    common = {k: summarize(r.loc[common_start:]) for k, r in rets.items()}
    print(f"\n=== Common period {common_start.date()} -> {prices.index[-1].date()} ===")
    print(fmt_table(common))
    oos_start = pd.Timestamp("2024-06-01")
    recent = {k: summarize(r.loc[oos_start:]) for k, r in rets.items() if r.index[-1] > oos_start}
    print(f"\n=== Recent period {oos_start.date()} -> end (true out-of-sample for this code) ===")
    print(fmt_table(recent))
    corr = pd.DataFrame(rets).loc[common_start:].corr().round(2)
    print("\n=== Daily return correlations (common period) ===")
    print(corr.to_string())
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rets).to_csv(out / "daily_strategy_returns.csv")
    pd.DataFrame(rows).T.to_csv(out / "daily_strategy_stats.csv")
    print(f"\nSaved reports/daily_strategy_returns.csv and reports/daily_strategy_stats.csv")


def cmd_overfit(args, settings):
    base = MultiFactorMomentum()
    prices = load_daily_closes(base.symbols, args.start, args.end, source=args.source, settings=settings)
    grid = [dict(top_k=k, rebalance=r, regime_sma=s, abs_mom_days=a)
            for k, r, s, a in itertools.product([2, 3, 4, 6], ["W", "M"], [150, 200], [63, 126, 252])]
    print(f"grid size = {len(grid)} configurations")
    trial_rets = {}
    for g in grid:
        st = MultiFactorMomentum(**g)
        res = run_backtest(prices, st.target_weights(prices), cost_bps=args.cost_bps, lag=1)
        trial_rets[json.dumps(g)] = res.returns
    M = pd.DataFrame(trial_rets).dropna()
    srs = M.apply(lambda r: r.mean() / r.std(ddof=1))
    best = srs.idxmax()
    print(f"best in-sample config: {best}  annual Sharpe={srs[best] * np.sqrt(252):.2f}")
    print(f"median config annual Sharpe={srs.median() * np.sqrt(252):.2f}")
    dsr = deflated_sharpe_ratio(M[best], n_trials=len(grid), trial_sharpes=srs.values)
    print(f"Deflated Sharpe Ratio (prob. true SR>0 after {len(grid)} trials) = {dsr['dsr']:.3f}")
    pbo = probability_of_backtest_overfitting(M, n_splits=10)
    print(f"Probability of Backtest Overfitting (CSCV, 10 splits) = {pbo['pbo']:.2%}")

    def make(h, params):
        return MultiFactorMomentum(**params).target_weights(h)

    def evaluate(h, t):
        return run_backtest(h, t, cost_bps=args.cost_bps, lag=1, start=h.index[0]).returns

    wf_grid = grid[::4]  # strided sample spanning every dimension of the grid
    wf = walk_forward(prices, make, wf_grid, evaluate, train_days=252 * 4, test_days=252)
    r = wf["oos_returns"]
    print(f"Walk-forward OOS ({r.index[0].date()} -> {r.index[-1].date()}): Sharpe={sharpe(r):.2f} "
          f"CAGR={summarize(r)['cagr']:.1%} maxDD={summarize(r)['max_drawdown']:.1%}")
    for s in wf["selections"]:
        print("  ", s["test_start"], s["params"], f"train SR={s['train_sharpe']:.2f}")


def cmd_intraday(args, settings):
    from quant.data import load_minute_bars
    from quant.strategies.intraday import LastHalfHourMomentum, NoiseAreaMomentum

    bars = load_minute_bars(args.symbol, args.start, args.end, settings=settings)
    print(f"{args.symbol}: {len(bars)} RTH minute bars, {bars.index[0]} -> {bars.index[-1]}")
    rows = {}
    for lev in (1.0, 4.0):
        na = NoiseAreaMomentum(max_leverage=lev, cost_bps=args.cost_bps)
        d = na.backtest(bars)
        d.index = pd.to_datetime(d.index)
        rows[f"noise_area maxlev={lev}"] = summarize(d["ret"])
        rows[f"noise_area maxlev={lev}"]["trades_per_day"] = d["trades"].mean()
    g = LastHalfHourMomentum(cost_bps=args.cost_bps).backtest(bars)
    g.index = pd.to_datetime(g.index)
    rows["last_half_hour (Gao)"] = summarize(g["ret"])
    df = pd.DataFrame(rows).T
    print(df[["start", "end", "cagr", "vol", "sharpe", "max_drawdown"] + (["trades_per_day"] if "trades_per_day" in df else [])].to_string())


def cmd_costs(args, settings):
    cfg = settings.raw
    sleeves = [(build_daily(s["strategy"], **(s.get("params") or {})), float(s["weight"])) for s in cfg["sleeves"]]
    strategies = {name: build_daily(name) for name in DAILY_STRATEGIES}
    symbols = sorted({s for st in strategies.values() for s in st.symbols} | {s for st, _ in sleeves for s in st.symbols})
    prices = load_daily_closes(symbols, args.start, args.end, source=args.source, settings=settings)
    opens = load_daily_closes(symbols, args.start, args.end, source=args.source, settings=settings, field="open")
    costs = cost_model_bps(prices)
    print("=== cost model: median one-way cost per symbol (bps) ===")
    print(costs.median().round(2).sort_values().to_string())

    print("\n=== cost sensitivity (cost model x multiplier), next-close execution ===")
    summary = {}
    port_syms = sorted({s for st, _ in sleeves for s in st.symbols})
    risk = RiskLimits(**cfg.get("risk", {}))
    pt = apply_risk(prices[port_syms], combine_sleeves(prices[port_syms], sleeves), risk)
    items = {name: (st.symbols, st.target_weights(prices[st.symbols])) for name, st in strategies.items()}
    items["PORTFOLIO (config)"] = (port_syms, pt)
    for name, (syms, t) in items.items():
        sens = cost_sensitivity(prices[syms], t, costs[syms], multipliers=(0, 1, 2, 3, 5, 10), lag=1)
        summary[name] = {"sharpe@0": sens.loc[0, "sharpe"], "sharpe@1x": sens.loc[1, "sharpe"],
                         "sharpe@3x": sens.loc[3, "sharpe"], "sharpe@10x": sens.loc[10, "sharpe"],
                         "avg_cost_bps": sens.loc[1, "avg_cost_bps_per_trade"],
                         "cost_drag/yr": sens.loc[1, "cost_drag_per_year"],
                         "breakeven_x": sens.attrs["breakeven_multiplier"]}
    print(pd.DataFrame(summary).T.round(3).to_string())

    print("\n=== risk-layer variants (portfolio, cost model, next close) ===")
    base = {k: v for k, v in cfg.get("risk", {}).items() if k not in ("group_caps", "max_turnover")}
    gc = cfg.get("risk", {}).get("group_caps") or {"us_equity": 0.6, "intl_equity": 0.35, "bonds": 0.6,
                                                   "commodities": 0.25, "real_estate": 0.15}
    variants = {
        "caps + vol target": RiskLimits(**base),
        "+ group caps": RiskLimits(**base, group_caps=gc),
        "+ turnover 0.5": RiskLimits(**base, max_turnover=0.5),
        "+ turnover 0.3": RiskLimits(**base, max_turnover=0.3),
        "+ group caps + turnover 0.5": RiskLimits(**base, group_caps=gc, max_turnover=0.5),
    }
    raw = combine_sleeves(prices[port_syms], sleeves)
    rows = {}
    for name, lim in variants.items():
        res = run_backtest(prices[port_syms], apply_risk(prices[port_syms], raw, lim), cost_bps=costs[port_syms], lag=1)
        rows[name] = res.stats
    band = float(cfg.get("execution", {}).get("drift_band", 0.01))
    res = run_backtest(prices[port_syms], pt, cost_bps=costs[port_syms], lag=1, trade_band=band)
    rows[f"config risk + live drift band {band:.0%}"] = res.stats
    print(fmt_table(rows))

    print("\n=== execution timing (portfolio, config risk, cost model) ===")
    rows = {}
    for name, kw in (("close t (MOC, lag0)", dict(lag=0)), ("next close (lag1)", dict(lag=1)),
                     ("next open (OPG)", dict(lag=1, open_prices=opens[port_syms]))):
        rows[name] = run_backtest(prices[port_syms], pt, cost_bps=costs[port_syms], **kw).stats
    print(fmt_table(rows))


def _alpaca_research_closes(name: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "cache" / "research" / name)
    c = df["close"].unstack(0)
    c.index = pd.DatetimeIndex(c.index).tz_convert("America/New_York").normalize().tz_localize(None)
    return c[~c.index.duplicated(keep="last")].sort_index()


def _grid_report(prices, make, grid, costs_bps, label, lag=1):
    """Backtest every config at each cost level; return (table, trial returns at the first cost)."""
    rows, trials = {}, {}
    for g in grid:
        st = make(g)
        t = st.target_weights(prices[st.symbols])
        key = json.dumps(g)
        for c in costs_bps:
            r = run_backtest(prices[st.symbols], t, cost_bps=c, lag=lag)
            rows[(key, c)] = {"sharpe": r.stats["sharpe"], "cagr": r.stats["cagr"], "vol": r.stats["vol"],
                              "max_dd": r.stats["max_drawdown"], "turnover/yr": r.stats["turnover_per_year"]}
            if c == costs_bps[0]:
                trials[key] = r.returns
    tab = pd.DataFrame(rows).T
    tab.index.names = ["config", "cost_bps"]
    print(f"\n=== {label}: {len(grid)} configs x costs {costs_bps} (lag={lag}) ===")
    print(tab.unstack("cost_bps")["sharpe"].round(2).sort_values(costs_bps[-1], ascending=False).head(12).to_string())
    return tab, pd.DataFrame(trials).dropna()


def _overfit_line(M: pd.DataFrame, label: str) -> dict:
    srs = M.apply(lambda r: r.mean() / r.std(ddof=1))
    best = srs.idxmax()
    dsr = deflated_sharpe_ratio(M[best], n_trials=M.shape[1], trial_sharpes=srs.values)
    try:
        pbo = probability_of_backtest_overfitting(M, n_splits=10) if M.shape[1] >= 4 else {"pbo": float("nan")}
    except ValueError:
        pbo = {"pbo": float("nan")}
    out = {"best": best, "best_sharpe": srs[best] * np.sqrt(252), "median_sharpe": srs.median() * np.sqrt(252),
           "dsr": dsr["dsr"], "pbo": pbo["pbo"]}
    print(f"{label}: best={best} SR={out['best_sharpe']:.2f} median SR={out['median_sharpe']:.2f} "
          f"DSR={out['dsr']:.3f} PBO={out['pbo']:.0%}")
    return out


def cmd_reversal(args, settings):
    from quant.strategies.reversal import ORIGINAL_SECTORS, SectorReversal, StockReversal

    # --- A: sector ETFs, long history from Yahoo --------------------------------
    sec = load_daily_closes(ORIGINAL_SECTORS + ["SPY"], args.start, args.end, source="yahoo", settings=settings)
    grid_a = [dict(lookback=lb, n=n, long_short=ls, rebalance=rb)
              for lb in (5, 10, 21) for n in (2, 3) for ls in (False, True) for rb in ("W",)]
    tab_a, M_a = _grid_report(sec, lambda g: SectorReversal(**g), grid_a, (5.0, 10.0), "Sector reversal (9 SPDRs)")
    ew = run_backtest(sec[ORIGINAL_SECTORS], pd.DataFrame(1 / 9, sec.index, ORIGINAL_SECTORS).where(
        pd.Series(np.arange(len(sec)) % 21 == 0, sec.index), np.nan), cost_bps=5.0)
    print(f"benchmark equal-weight sectors (monthly): Sharpe={ew.stats['sharpe']:.2f} CAGR={ew.stats['cagr']:.1%}")
    # A long-only sector book mostly earns the sector beta; the reversal *signal* is judged on
    # (a) long-short configs and (b) long-only returns in excess of equal-weight sectors.
    ew_r = ew.returns.reindex(M_a.index).fillna(0)
    signal = pd.DataFrame({k: (M_a[k] - ew_r) if '"long_short": false' in k else M_a[k] for k in M_a.columns})
    sig_sr = signal.apply(lambda r: r.mean() / r.std(ddof=1) * np.sqrt(252))
    print("signal Sharpe @5bp (long-only = excess vs equal-weight sectors):")
    print(sig_sr.round(2).sort_values(ascending=False).to_string())
    oa = _overfit_line(signal, "Sector reversal signal overfit")
    # same adjustment at 10bp
    sig10 = {}
    for g in grid_a:
        st = SectorReversal(**g)
        r = run_backtest(sec[st.symbols], st.target_weights(sec[st.symbols]), cost_bps=10.0).returns
        x = r - ew_r.reindex(r.index).fillna(0) if not g["long_short"] else r
        sig10[json.dumps(g)] = x.mean() / x.std(ddof=1) * np.sqrt(252)
    oa["signal_best10"] = max(sig10.values())

    # --- B: S&P 100 stocks, Alpaca SIP daily since 2016 (survivorship-biased) ----------
    px = _alpaca_research_closes("sp100_daily.parquet")
    grid_b = [dict(lookback=lb, residual=res, rebalance=rb, skip_days=sk)
              for lb in (1, 5, 21) for res in ("sector", "market", "none") for rb in ("D", "W") for sk in (0, 1)
              if not (lb == 1 and sk == 1) and not (lb == 21 and rb == "D")]
    tab_b, M_b = _grid_report(px, lambda g: StockReversal(**g), grid_b, (5.0, 10.0), "S&P 100 stock reversal (L/S)")
    ob = _overfit_line(M_b, "Stock reversal overfit")
    recent = M_b.loc["2024-01-01":]
    print("since 2024 (5bp) top configs Sharpe:",
          recent.apply(lambda r: r.mean() / r.std(ddof=1) * np.sqrt(252)).sort_values().tail(5).round(2).to_dict())

    # --- go / no-go -----------------------------------------------------------------
    for label, tab, ov in (("sector", tab_a, oa), ("stock", tab_b, ob)):
        best10 = ov.get("signal_best10", tab.xs(10.0, level="cost_bps")["sharpe"].max())
        go = best10 > 0.3 and ov["pbo"] < 0.5
        print(f"GO/NO-GO {label}: best Sharpe@10bp={best10:.2f}, PBO={ov['pbo']:.0%} -> {'GO (paper)' if go else 'NO-GO'}")
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    pd.concat({"sector": tab_a, "stock": tab_b}).to_csv(out / "reversal_grid.csv")


def _option_calibration(settings, vix: pd.Series, rate: pd.Series, q: float = 0.013):
    """Real SPY put prices (Alpaca, free, 2024+) -> implied vols at entry (~21 sessions before
    each monthly expiry) -> fit iv = a * VIX + b * ln(S/K) for K <= S. Also returns the
    real-price data for a direct spread backtest."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    from quant.strategies.options import implied_vol_put, monthly_expiries, occ_symbol

    c = settings.alpaca
    sc = StockHistoricalDataClient(c.key_id, c.secret)
    oc = OptionHistoricalDataClient(c.key_id, c.secret)
    spy = sc.get_stock_bars(StockBarsRequest(symbol_or_symbols="SPY", timeframe=TimeFrame.Day,
                                             start=pd.Timestamp("2024-01-01", tz="UTC"), feed=DataFeed.SIP,
                                             adjustment=Adjustment.RAW,
                                             end=pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16))).df
    S = spy["close"].droplevel(0)
    S.index = pd.DatetimeIndex(S.index).tz_convert("America/New_York").normalize().tz_localize(None)
    days = S.index
    rows = []
    for exp in monthly_expiries("2024-03-01", days[-1]):
        exp_day = days[days <= exp][-1] if (days <= exp).any() else None
        if exp_day is None or exp_day < exp - pd.Timedelta(days=3):
            continue
        i_exp = days.get_loc(exp_day)
        if i_exp < 21:
            continue
        entry = days[i_exp - 21]
        s0 = float(S[entry])
        strikes = sorted({int(round(s0 * m)) for m in np.arange(0.78, 1.001, 0.01)})
        syms = [occ_symbol("SPY", exp, k) for k in strikes]
        try:
            b = oc.get_option_bars(OptionBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame.Day,
                                                     start=pd.Timestamp(entry, tz="UTC"),
                                                     end=pd.Timestamp(entry, tz="UTC") + pd.Timedelta(days=1))).df
        except Exception:
            continue
        if b is None or b.empty:
            continue
        T = (exp - entry).days / 365.0
        r = float(rate.asof(entry)) if not np.isnan(rate.asof(entry)) else 0.04
        for (sym, _), row in b.iterrows():
            k = int(sym[-8:]) / 1000
            iv = implied_vol_put(float(row["close"]), s0, k, T, r, q)
            rows.append({"entry": entry, "expiry": exp, "S0": s0, "ST": float(S[exp_day]), "K": k, "price": float(row["close"]),
                         "volume": float(row["volume"]), "iv": iv, "vix": float(vix.asof(entry)) / 100, "T": T, "r": r})
    d = pd.DataFrame(rows).dropna(subset=["iv"])
    d = d[d["volume"] > 0]
    d["lm"] = np.log(d["S0"] / d["K"])
    X = np.column_stack([d["vix"], d["lm"]])
    a, b = np.linalg.lstsq(X, d["iv"].to_numpy(), rcond=None)[0]
    resid = d["iv"] - X @ np.array([a, b])
    print(f"\n=== calibration on real SPY puts (Alpaca, {d['entry'].min().date()} -> {d['entry'].max().date()}, "
          f"{len(d)} quotes, {d['expiry'].nunique()} expiries) ===")
    print(f"iv = {a:.3f} * VIX + {b:.3f} * ln(S/K)   (residual sd {resid.std():.3f}); "
          f"model defaults were 0.90 / 0.75")
    return float(a), float(b), d


def _real_spread_backtest(d: pd.DataFrame, short_m: float, width: float, risk: float, cost_bps: float) -> pd.Series:
    """Per-cycle returns of a put spread priced with REAL entry closes and settled at expiry.
    short_m: short strike as fraction of spot (e.g. 0.95); width: fraction of spot."""
    out = {}
    for exp, g in d.groupby("expiry"):
        s0, st = g["S0"].iloc[0], g["ST"].iloc[0]
        g = g.set_index("K")["price"]
        k1 = min(g.index, key=lambda k: abs(k - s0 * short_m))
        k2 = min(g.index, key=lambda k: abs(k - (k1 - width * s0)))
        if k2 >= k1:
            continue
        credit = g[k1] - g[k2] - 2 * cost_bps / 1e4 * s0
        payoff = max(k1 - st, 0) - max(k2 - st, 0)
        if payoff > 0:
            credit -= 2 * cost_bps / 1e4 * s0
        max_loss = (k1 - k2) - credit
        out[exp] = risk * (credit - payoff) / max_loss
    return pd.Series(out)


def cmd_putspread(args, settings):
    from quant.strategies.options import PutSellConfig, simulate_put_selling

    px = load_daily_closes(["^GSPC", "^VIX", "^IRX", "^PUT", "SPY"], "1995-06-01", args.end, source="yahoo",
                           settings=settings)
    spx, vix, rate = px["^GSPC"], px["^VIX"], px["^IRX"] / 100
    cash = ((1 + rate) ** (1 / 252) - 1).shift(1).fillna(0.0)

    def excess_stats(ret: pd.Series) -> dict:
        x = ret - cash.reindex(ret.index).fillna(0.0)
        st = summarize(ret)
        st["excess_cagr"] = float((1 + x).prod() ** (252 / max(len(x), 1)) - 1)
        st["sharpe"] = float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.std() > 0 else float("nan")
        return st

    calib = None
    if settings.alpaca.available:
        try:
            calib = _option_calibration(settings, vix, rate)
        except Exception as e:
            print(f"(option calibration skipped: {e})")
    atm_scale, skew = (calib[0], calib[1]) if calib else (0.90, 0.75)

    cost = args.cost_bps if args.cost_bps != 5.0 else 1.0

    def cfg(**g):
        return PutSellConfig(**{"atm_scale": atm_scale, "skew": skew, "cost_bps": cost, **g})

    # --- validation 1: synthetic ATM cash-secured puts vs CBOE PutWrite index (real prices) ---
    csp = simulate_put_selling(spx, vix, rate, cfg(kind="csp", atm=True, cost_bps=0.0))["ret"]
    put_idx = px["^PUT"].pct_change(fill_method=None).dropna()
    both = pd.concat([csp.rename("synthetic"), put_idx.rename("cboe_put")], axis=1).dropna()
    m = (1 + both).resample("ME").prod() - 1
    print("\n=== validation: synthetic ATM cash-secured put (calibrated) vs CBOE PutWrite (^PUT) ===")
    print(f"monthly correlation = {m.corr().iloc[0, 1]:.3f}   (Sharpe below is in excess of T-bills)")
    print(fmt_table({k: excess_stats(both[k]) for k in both}))

    # --- validation 2: put spreads on REAL SPY option prices since 2024 ----------------------
    if calib:
        d = calib[2]
        print("\n=== put spreads priced with real SPY option closes (entry ~21 sessions before expiry) ===")
        real = {}
        for sm in (0.97, 0.95, 0.93, 0.90):
            for w in (0.03, 0.05):
                cyc = _real_spread_backtest(d, sm, w, 0.03, cost)
                real[f"short {sm:.0%} spot, width {w:.0%}"] = {
                    "cycles": len(cyc), "win_rate": float((cyc > 0).mean()),
                    "sum_ret_per_year": float(cyc.sum() / max(len(cyc), 1) * 12),
                    "worst_cycle": float(cyc.min())}
        print(pd.DataFrame(real).T.round(4).to_string())
        syn = {}
        for dlt in (-0.10, -0.20, -0.30):
            r = simulate_put_selling(spx, vix, rate, cfg(kind="spread", short_delta=dlt, width=0.05, risk_per_trade=0.03))
            x = r["ret"].loc[d["entry"].min():] - cash.reindex(r.index).loc[d["entry"].min():].fillna(0)
            syn[f"synthetic delta {dlt}, width 5%"] = {"excess_ret_per_year": float(x.mean() * 252)}
        print(pd.DataFrame(syn).T.round(4).to_string())

    # --- grid of defined-risk put credit spreads + cash-secured puts ----------------------
    crises = {"2008": ("2008-09-01", "2009-03-31"), "2020": ("2020-02-15", "2020-04-30"),
              "2022": ("2022-01-01", "2022-12-31")}
    grid = [dict(kind="spread", short_delta=dl, width=w, risk_per_trade=r, vix_stop=v)
            for dl in (-0.10, -0.20, -0.30) for w in (0.03, 0.05, 0.10) for r in (0.03, 0.05) for v in (None, 40)]
    grid += [dict(kind="csp", short_delta=dl, vix_stop=None) for dl in (-0.20, -0.30)] + [dict(kind="csp", atm=True)]
    rows, trials = {}, {}
    start = pd.Timestamp(args.start if args.start != "2005-01-01" else "1997-01-01")
    for g in grid:
        r = simulate_put_selling(spx, vix, rate, cfg(**g))
        ret = r["ret"].loc[start:]
        st = excess_stats(ret)
        row = {"excess_sharpe": st["sharpe"], "cagr": st["cagr"], "excess_cagr": st["excess_cagr"], "vol": st["vol"],
               "max_dd": st["max_drawdown"], "avg_credit": r["credit_yield"].mean()}
        for k, (a_, b_) in crises.items():
            eqc = (1 + ret.loc[a_:b_]).cumprod()
            row[f"dd_{k}"] = float((eqc / eqc.cummax() - 1).min())
        rows[json.dumps(g)] = row
        trials[json.dumps(g)] = ret - cash.reindex(ret.index).fillna(0)
    tab = pd.DataFrame(rows).T
    pd.set_option("display.width", 220)
    print(f"\n=== put selling grid ({start.date()} -> end), calibrated iv={atm_scale:.2f}*VIX+{skew:.2f}*ln(S/K), "
          f"cost {cost}bp/leg, daily MTM ===")
    print(tab.sort_values("excess_sharpe", ascending=False).round(3).to_string())
    M = pd.DataFrame(trials).dropna()
    ov = _overfit_line(M[[k for k in M if '"spread"' in k]], "Put-spread grid overfit (excess returns)")
    budget = 0.10
    ok = tab[(tab["dd_2020"] > -budget) & (tab["dd_2022"] > -budget) & (tab["dd_2008"] > -2 * budget)]
    print(f"\nconfigs inside the risk budget (DD 2020 & 2022 > -{budget:.0%}, 2008 > -{2 * budget:.0%}): {len(ok)}")
    if len(ok):
        print(ok.sort_values("excess_sharpe", ascending=False).head(8).round(3).to_string())
    best_ok = ok["excess_sharpe"].max() if len(ok) else float("nan")
    spy = excess_stats(px["SPY"].pct_change(fill_method=None).loc[start:].dropna())
    print(f"reference SPY buy&hold: excess Sharpe={spy['sharpe']:.2f} maxDD={spy['max_drawdown']:.1%}")
    go = len(ok) > 0 and best_ok > 0.4 and ov["pbo"] < 0.5
    print(f"GO/NO-GO put spreads: best in-budget excess Sharpe={best_ok:.2f}, PBO={ov['pbo']:.0%} -> "
          f"{'GO (paper, small size)' if go else 'NO-GO'}")
    tab.to_csv(ROOT / "reports" / "putspread_grid.csv")


def _sr(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if len(x) > 20 and x.std() > 0 else float("nan")


def cmd_news(args, settings):
    from quant.data.universe import SP100
    from quant.strategies.news import daily_scores, lexicon_score, sentiment_returns

    ndir = ROOT / "data" / "cache" / "research" / "news"
    news = pd.concat([pd.read_parquet(p) for p in sorted(ndir.glob("*.parquet"))]).drop_duplicates("id")
    raw = pd.read_parquet(ROOT / "data" / "cache" / "research" / "sp100_daily.parquet")
    closes, opens = _alpaca_research_closes("sp100_daily.parquet"), raw["open"].unstack(0)
    opens.index = pd.DatetimeIndex(opens.index).tz_convert("America/New_York").normalize().tz_localize(None)
    opens = opens[~opens.index.duplicated(keep="last")].reindex(closes.index)
    t0 = pd.to_datetime(news["created_at"], utc=True).min().tz_convert(None).normalize()
    closes, opens = closes.loc[t0:], opens.loc[t0:]
    uni = sorted(SP100)
    news["score"] = [lexicon_score(h, s) for h, s in zip(news["headline"], news["summary"].fillna(""))]
    print(f"{len(news)} articles {news['created_at'].min()} -> {news['created_at'].max()}; "
          f"non-neutral {(news['score'] != 0).mean():.0%}")
    rows, trials = {}, {}
    for variant, cutoff in (("open", "09:15"), ("close", "15:30")):
        sc = daily_scores(news, closes.index, uni, cutoff)
        for thr in (0.2, 0.4, 0.6):
            for hold in ((1,) if variant == "open" else (1, 3, 5)):
                for cost in (5.0, 10.0):
                    r = sentiment_returns(sc, closes, opens, variant, threshold=thr, hold=hold, cost_bps=cost)
                    key = f"{variant} thr={thr} hold={hold}"
                    ret = r["ret"].loc[closes.index[1]:]
                    rec = ret.loc["2023-01-01":]
                    mo = (1 + rec).resample("ME").prod() - 1
                    rows[(key, cost)] = {"sharpe_all": _sr(ret), "sharpe_2023+": _sr(rec),
                                         "ann_ret_2023+": float(rec.mean() * 252), "pos_months_2023+": float((mo > 0).mean()),
                                         "gross_sharpe": _sr(r["gross"]), "avg_names": float(r["names"].mean())}
                    if cost == 5.0:
                        trials[key] = ret
    tab = pd.DataFrame(rows).T
    tab.index.names = ["config", "cost_bps"]
    pd.set_option("display.width", 220)
    print(tab.round(3).to_string())
    ov = _overfit_line(pd.DataFrame(trials).dropna(), "News grid overfit")
    t5 = tab.xs(5.0, level="cost_bps")
    best = t5["sharpe_2023+"].idxmax()
    b = t5.loc[best]
    go = b["sharpe_2023+"] > 0.5 and b["pos_months_2023+"] >= 0.55 and ov["pbo"] < 0.5 and \
        tab.loc[(best, 10.0), "sharpe_2023+"] > 0
    print(f"GO/NO-GO news: best={best} Sharpe2023+={b['sharpe_2023+']:.2f} pos months={b['pos_months_2023+']:.0%} "
          f"@10bp={tab.loc[(best, 10.0), 'sharpe_2023+']:.2f} PBO={ov['pbo']:.0%} -> {'GO (paper)' if go else 'NO-GO'}")
    tab.to_csv(ROOT / "reports" / "news_grid.csv")


def cmd_orb(args, settings):
    from quant.strategies.orb import ORBConfig, simulate_day

    odir = ROOT / "data" / "cache" / "research" / "orb"
    f5 = pd.read_parquet(odir / "first5_all.parquet")
    feats = pd.read_parquet(odir / "daily_features.parquet")
    v5 = f5.pivot_table(index="day", columns="symbol", values="volume")
    rv = v5 / v5.rolling(14, min_periods=10).mean().shift(1)
    f5 = f5.set_index(["day", "symbol"])
    days = sorted(pd.Timestamp(p.stem) for p in (odir / "minute").glob("*.parquet"))
    print(f"ORB: {len(days)} days with candidate minute bars ({days[0].date()} -> {days[-1].date()})")
    cache = {}
    for d in days:
        m = pd.read_parquet(odir / "minute" / f"{d.date()}.parquet")
        if m.empty:
            continue
        m = m.reset_index()
        m["ts"] = pd.to_datetime(m["timestamp"], utc=True).dt.tz_convert("America/New_York")
        m = m[(m["ts"].dt.hour * 60 + m["ts"].dt.minute) >= 9 * 60 + 35]
        mins = {s: g.set_index("ts")[["open", "high", "low", "close"]] for s, g in m.groupby("symbol")}
        fd = f5.loc[d] if d in f5.index.get_level_values(0) else None
        if fd is None:
            continue
        fd = fd.assign(rv=rv.loc[d].reindex(fd.index)) if d in rv.index else None
        if fd is None:
            continue
        ft = feats.loc[d] if d in feats.index.get_level_values(0) else pd.DataFrame(columns=["atr"])
        cache[d] = (fd, mins, ft)
    rows, rets = {}, {}
    for cost in (1.0, 3.0, 5.0):
        for delay, intrabar in ((0, "pessimistic"), (0, "neutral"), (1, "pessimistic")):
            cfg = ORBConfig(cost_bps=cost, entry_delay=delay, intrabar=intrabar)
            out, ntr, wins = {}, 0, 0
            for d, (fd, mins, ft) in cache.items():
                r, tr = simulate_day(fd, mins, ft, cfg)
                out[d] = r
                ntr += len(tr)
                wins += sum(t["ret"] > 0 for t in tr)
            s = pd.Series(out).sort_index()
            key = f"cost={cost}bp delay={delay}" + (" neutral" if intrabar == "neutral" else "")
            rets[key] = s
            rows[key] = {"sharpe": _sr(s), "ann_ret": float(s.mean() * 252), "max_dd": summarize(s)["max_drawdown"],
                         "sharpe_2024+": _sr(s.loc["2024-01-01":]), "trades/day": ntr / max(len(s), 1),
                         "win_rate": wins / max(ntr, 1)}
    tab = pd.DataFrame(rows).T
    print(tab.round(3).to_string())
    k0 = "cost=3.0bp delay=0 neutral"   # judged on the fair fill model; pessimistic shown alongside
    go = tab.loc[k0, "sharpe"] > 0.5 and tab.loc["cost=3.0bp delay=1", "sharpe"] > 0 and tab.loc[k0, "sharpe_2024+"] > 0
    print(f"GO/NO-GO ORB: Sharpe@3bp(neutral)={tab.loc[k0, 'sharpe']:.2f}, "
          f"delayed@3bp={tab.loc['cost=3.0bp delay=1', 'sharpe']:.2f}, 2024+={tab.loc[k0, 'sharpe_2024+']:.2f}"
          f" -> {'GO (paper only)' if go else 'NO-GO'}")
    pd.DataFrame(rets).to_csv(ROOT / "reports" / "orb_returns.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["daily", "overfit", "intraday", "costs", "reversal", "news", "orb", "putspread"])
    ap.add_argument("--config", default="config/quant.json")
    ap.add_argument("--source", default="auto", choices=["auto", "alpaca", "yahoo"])
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--symbol", default="SPY")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    settings = load_settings(args.config)
    cmds = {"daily": cmd_daily, "overfit": cmd_overfit, "intraday": cmd_intraday, "costs": cmd_costs,
            "reversal": cmd_reversal, "putspread": cmd_putspread, "news": cmd_news, "orb": cmd_orb}
    cmds[args.cmd](args, settings)


if __name__ == "__main__":
    main()
