"""Live / paper runners for the quant/ package.

    python scripts/run_quant.py check                    # read-only Alpaca connectivity check
    python scripts/run_quant.py daily --once             # compute today's targets, DRY RUN
    python scripts/run_quant.py daily --once --execute   # ...and submit orders (paper by default)
    python scripts/run_quant.py daily --execute          # scheduler: rebalance 20 min before close
    python scripts/run_quant.py intraday --execute       # noise-area SPY intraday runner
    python scripts/run_quant.py flatten --execute        # cancel orders + close all positions
    python scripts/run_quant.py report --start 2026-11-01   # paper P&L vs backtest + slippage per order
    python scripts/run_quant.py watchdog                 # heartbeat / missed-run / HALT check (schedule every 15 min)
    python scripts/run_quant.py alert-test               # send a test alert to every configured channel
    python scripts/run_quant.py spreads                  # measure live half-spreads -> data/cache/spreads.json

Paper vs live is decided ONLY by ALPACA_ENV in .env (live also needs LIVE_TRADING_CONFIRM=yes).
Without --execute nothing is ever sent to the broker.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.config import load_settings  # noqa: E402


def setup_logging(name: str) -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"logs/quant_{name}.log")],
    )


def cmd_check(settings, args):
    from quant.execution import AlpacaBroker

    b = AlpacaBroker.from_settings(settings)
    a = b.account()
    clock = b.clock()
    print(f"environment : {'PAPER' if settings.alpaca.paper else 'LIVE'}")
    print(f"account     : status={a.status} equity={a.equity} buying_power={a.buying_power} cash={a.cash}")
    print(f"clock       : open={clock.is_open} next_open={clock.next_open} next_close={clock.next_close}")
    pos = b.positions()
    print(f"positions   : {len(pos)} -> " + ", ".join(f"{p.symbol}:{p.qty:g}" for p in pos.values()))
    print(f"open orders : {len(b.open_orders())}")
    print(f"latest SPY  : {b.latest_prices(['SPY'])} (feed={settings.alpaca.data_feed})")


def cmd_report(settings, args):
    import pandas as pd

    from quant.data import load_daily_closes
    from quant.execution import AlpacaBroker
    from quant.live.daily import build_sleeves
    from quant.ops.pnl import (actual_daily_returns, compare, expected_daily_returns, order_slippage,
                               read_journal, slippage_summary)

    b = AlpacaBroker.from_settings(settings)
    journal = read_journal(Path(args.state_dir) / "journal.jsonl")
    first_live = next((r.get("day") for r in journal
                       if (r.get("event_type") or r.get("event")) == "rebalance" and not r.get("dry_run")), None)
    if args.start:
        start = pd.Timestamp(args.start)
    elif first_live:
        start = pd.Timestamp(first_live)          # first real (paper) rebalance
    else:
        start = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize() - pd.Timedelta(days=30)
    hist = b.portfolio_history(start=start.tz_localize("America/New_York").to_pydatetime())
    actual = actual_daily_returns(hist)
    expected = expected_daily_returns(settings, start, source="alpaca")
    cmp_ = compare(actual, expected)
    syms = sorted({s for st, _ in build_sleeves(settings.raw) for s in st.symbols})
    closes = load_daily_closes(syms, start - pd.Timedelta(days=5), source="alpaca", settings=settings, refresh=True)
    opens = load_daily_closes(syms, start - pd.Timedelta(days=5), source="alpaca", settings=settings, field="open")
    tag = settings.raw.get("book", "daily")
    orders = b.recent_orders(after=start.tz_localize("America/New_York").to_pydatetime())
    slip = order_slippage(orders, journal, closes, opens,
                          prefix=f"q-{tag[:16]}-")
    ss = slippage_summary(slip)
    out = Path("reports/paper")
    out.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d")
    cmp_.daily.to_csv(out / f"{stamp}_pnl_vs_backtest.csv")
    slip.to_csv(out / f"{stamp}_slippage.csv", index=False)
    lines = [f"# Paper vs backtest ({start.date()} -> today)", "", "## P&L", ""]
    def fmt(k, v):
        if not isinstance(v, float):
            return f"- {k}: {v}"
        if k.startswith(("cum_", "tracking_error")):
            return f"- {k}: {v:.2%}"
        return f"- {k}: {v:,.2f}"

    lines += [fmt(k, v) for k, v in cmp_.stats.items()]
    lines += ["", "## Slippage (positive = cost)", ""] + [fmt(k, v) for k, v in ss.items()]
    if settings.raw.get("intraday") and Path("logs/quant_intraday.log").exists():
        lines += ["", "> note: an intraday sleeve log exists; if it traded this account, actual P&L includes it."]
    (out / f"{stamp}_report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    if not cmp_.daily.empty:
        print("\nlast 10 days (bps):")
        print((cmp_.daily[["actual", "expected", "diff"]].tail(10) * 1e4).round(1).to_string())
    print(f"\nwritten to {out}/")


def cmd_spreads(settings, args):
    import json
    import statistics
    import time

    from quant.execution import AlpacaBroker
    from quant.live.daily import build_sleeves

    b = AlpacaBroker.from_settings(settings)
    if not b.is_open():
        print("market closed: quotes are not representative; run during regular hours")
        return
    syms = sorted({s for st, _ in build_sleeves(settings.raw) for s in st.symbols})
    samples: dict[str, list[float]] = {s: [] for s in syms}
    for _ in range(args.samples):
        for s, q in b.latest_quotes(syms).items():
            if q.bid > 0 and q.ask > q.bid:
                samples[s].append((q.ask - q.bid) / 2 / q.mid * 1e4)
        time.sleep(args.interval)
    hs = {s: round(statistics.median(v), 3) for s, v in samples.items() if v}
    path = Path("data/cache/spreads.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "feed": settings.alpaca.data_feed,
                                "samples": args.samples, "half_spread_bps": hs}, indent=2))
    for s, v in sorted(hs.items(), key=lambda kv: kv[1]):
        print(f"{s:6s} {v:7.2f} bp")
    print(f"saved {path} (used by quant.backtest.costs.cost_model_bps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "daily", "intraday", "flatten", "report", "watchdog", "alert-test",
                                    "spreads"])
    ap.add_argument("--config", default="config/quant.json")
    ap.add_argument("--execute", action="store_true", help="actually send orders (default: dry run)")
    ap.add_argument("--once", action="store_true", help="daily: run one rebalance now instead of scheduling")
    ap.add_argument("--start", help="report: first day (default: 30 days ago)")
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--samples", type=int, default=10, help="spreads: quote snapshots")
    ap.add_argument("--interval", type=float, default=30.0, help="spreads: seconds between snapshots")
    args = ap.parse_args()
    setup_logging(args.cmd)
    settings = load_settings(args.config)
    log = logging.getLogger("run_quant")
    log.info("env=%s execute=%s", "paper" if settings.alpaca.paper else "LIVE", args.execute)

    if args.cmd == "check":
        cmd_check(settings, args)
    elif args.cmd == "daily":
        from quant.live.daily import DailyRunner

        r = DailyRunner(settings, dry_run=not args.execute, state_dir=args.state_dir)
        r.run_once() if args.once else r.run_forever()
    elif args.cmd == "intraday":
        from quant.live.intraday import NoiseAreaRunner

        icfg = settings.raw.get("intraday", {})
        NoiseAreaRunner(settings, symbol=icfg.get("symbol", "SPY"), dry_run=not args.execute,
                        allocation_fraction=float(icfg.get("allocation_fraction", 0.5)),
                        **(icfg.get("params") or {})).run()
    elif args.cmd == "report":
        cmd_report(settings, args)
    elif args.cmd == "spreads":
        cmd_spreads(settings, args)
    elif args.cmd == "watchdog":
        from quant.live.daily import watchdog

        problems = watchdog(settings, state_dir=args.state_dir)
        print("OK" if not problems else "\n".join(problems))
        sys.exit(1 if problems else 0)
    elif args.cmd == "alert-test":
        from quant.ops.alerts import Alerter

        a = Alerter(args.state_dir)
        if not a.channels:
            print("no alert channel configured: set ALERT_NTFY_TOPIC / ALERT_WEBHOOK_URL / "
                  "ALERT_TELEGRAM_* / ALERT_SMTP_* in .env (see .env.example)")
        print("delivered to:", a.send("Test alert", "quant alerting works", level="critical", force=True))
    elif args.cmd == "flatten":
        from quant.execution import AlpacaBroker

        b = AlpacaBroker.from_settings(settings)
        if not args.execute:
            print("dry run: would close", list(b.positions()))
        else:
            for r in b.close_all():
                print(r)


if __name__ == "__main__":
    main()
