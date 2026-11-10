"""Research harness for the quant/ package.

Examples (repo root):
    python scripts/research.py daily                       # all daily strategies + portfolio
    python scripts/research.py daily --source yahoo --start 2005-01-01 --cost-bps 5
    python scripts/research.py overfit                     # PBO / DSR / walk-forward on momentum grid
    python scripts/research.py intraday --symbol SPY --start 2019-01-01   # needs Alpaca keys

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

from quant.backtest import (deflated_sharpe_ratio, probability_of_backtest_overfitting,  # noqa: E402
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["daily", "overfit", "intraday"])
    ap.add_argument("--config", default="config/quant.json")
    ap.add_argument("--source", default="auto", choices=["auto", "alpaca", "yahoo"])
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--symbol", default="SPY")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    settings = load_settings(args.config)
    {"daily": cmd_daily, "overfit": cmd_overfit, "intraday": cmd_intraday}[args.cmd](args, settings)


if __name__ == "__main__":
    main()
