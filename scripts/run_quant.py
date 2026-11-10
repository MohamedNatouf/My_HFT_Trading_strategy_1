"""Live / paper runners for the quant/ package.

    python scripts/run_quant.py check                    # read-only Alpaca connectivity check
    python scripts/run_quant.py daily --once             # compute today's targets, DRY RUN
    python scripts/run_quant.py daily --once --execute   # ...and submit orders (paper by default)
    python scripts/run_quant.py daily --execute          # scheduler: rebalance 20 min before close
    python scripts/run_quant.py intraday --execute       # noise-area SPY intraday runner
    python scripts/run_quant.py flatten --execute        # cancel orders + close all positions

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "daily", "intraday", "flatten"])
    ap.add_argument("--config", default="config/quant.json")
    ap.add_argument("--execute", action="store_true", help="actually send orders (default: dry run)")
    ap.add_argument("--once", action="store_true", help="daily: run one rebalance now instead of scheduling")
    args = ap.parse_args()
    setup_logging(args.cmd)
    settings = load_settings(args.config)
    log = logging.getLogger("run_quant")
    log.info("env=%s execute=%s", "paper" if settings.alpaca.paper else "LIVE", args.execute)

    if args.cmd == "check":
        cmd_check(settings, args)
    elif args.cmd == "daily":
        from quant.live.daily import DailyRunner

        r = DailyRunner(settings, dry_run=not args.execute)
        r.run_once() if args.once else r.run_forever()
    elif args.cmd == "intraday":
        from quant.live.intraday import NoiseAreaRunner

        icfg = settings.raw.get("intraday", {})
        NoiseAreaRunner(settings, symbol=icfg.get("symbol", "SPY"), dry_run=not args.execute,
                        allocation_fraction=float(icfg.get("allocation_fraction", 0.5)),
                        **(icfg.get("params") or {})).run()
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
