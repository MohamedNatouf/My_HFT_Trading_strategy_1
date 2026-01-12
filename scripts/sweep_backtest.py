import argparse
import itertools
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

# Ensure repository root is on sys.path so `Trading_System` can be imported when running this script directly
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Trading_System.config_loader import ConfigLoader
from Trading_System.system import TradingSystem

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("sweep_backtest")


def override_strategy(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(cfg))  # deep copy
    out['strategy'] = {**out.get('strategy', {}), **overrides}
    return out


def run_once(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Create a temporary config file
    tmp_path = Path("tmp_config.json")
    tmp_path.write_text(json.dumps(cfg))
    try:
        ts = TradingSystem(str(tmp_path))
        ts.config['mode'] = 'backtest'
        allocs = asyncio_run(ts.run_backtest())
        report = ts._compute_report(allocs)
        return report or {}
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


def asyncio_run(coro):
    import asyncio
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # nested loop case
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep backtester (minute data)")
    parser.add_argument("--config", default="config/config.json")
    parser.add_argument("--output", default="sweeps.csv")
    parser.add_argument("--max", type=int, default=0, help="Limit number of combinations (0=all)")
    args = parser.parse_args()

    base_cfg = ConfigLoader(args.config).load()

    # Define parameter grid from strategy; users can extend this list
    grid = {
        'Volatility_Measuring_Lookback_Period': [30, 60, 120],
        'rebalance_minutes': [5, 15, 30, 60],
        'Selected_method_of_Momuntum_Metric': ["Metrics Combind", "Metrics Count"],
        'Method_Of_Wieghting': ["Assets Inverse Volatility", "Efficient Frontier Optimization", "Volume & Inverse Volume"],
        'Number_of_Portfolio_Allocations': [3, 5, 8],
        'Emergency_System_Enabled': [False, True],
        'Equity_Lookback_Period': [30, 55, 90],
        'Bond_Lookback_Period': [60, 110, 180],
        'Equity_Signal_Wait_Period': [2, 3, 5],
        'Bond_Signal_Wait_Period': [1, 2, 3],
        'Equity_N_Bond_Hold_Period_Status': [True, False],
        'Equity_N_Bond_Hold_Period': [0, 3, 5],
        'Rebalancing_Method': ["active", "1", "2"],
        'minutes_per_day': [390],
        'Volatility_Weight_Strength': [0.5, 0.75, 1.0],
    }

    keys = list(grid.keys())
    values = list(grid.values())
    combos = itertools.product(*values)

    rows: List[Dict[str, Any]] = []
    count = 0
    for combo in combos:
        overrides = {k: v for k, v in zip(keys, combo)}
        cfg = override_strategy(base_cfg, overrides)
        logger.info("Run combo %d overrides=%s", count+1, overrides)
        rep = run_once(cfg)
        if rep:
            rep_row = {**overrides, **rep}
            rows.append(rep_row)
        count += 1
        if args.max and count >= args.max:
            break

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)
        logger.info("Saved sweep results to %s rows=%d", args.output, len(df))
    else:
        logger.warning("No results produced")


if __name__ == '__main__':
    sys.exit(main())
