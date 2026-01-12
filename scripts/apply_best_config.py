import argparse
import json
import sys
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("apply_best_config")


def choose_best(df: pd.DataFrame) -> pd.Series:
    # Score: prioritize sharpe, then cum_return_minute, penalize max_drawdown and std_ret
    score = (
        (df['sharpe'].fillna(0)) * 1.0 +
        (df['cum_return_minute'].fillna(0)) * 0.5 -
        (df['max_drawdown'].fillna(0).abs()) * 0.5 -
        (df['std_ret'].fillna(0)) * 0.2
    )
    idx = int(score.idxmax())
    return df.iloc[idx]


def main():
    parser = argparse.ArgumentParser(description="Apply best strategy params from sweep CSV to config.json")
    parser.add_argument("--sweeps", required=True, help="Path to sweeps CSV")
    parser.add_argument("--config", default="config/config.json", help="Path to config JSON to update")
    parser.add_argument("--out", default=None, help="Optional output path (if not overwriting config)")
    args = parser.parse_args()

    df = pd.read_csv(args.sweeps)
    if df.empty:
        logger.error("Sweeps CSV is empty")
        return 2
    best = choose_best(df)
    logger.info("Best row: %s", best.to_dict())

    cfg = json.loads(Path(args.config).read_text())
    strat = cfg.get('strategy', {})
    # Update strategy keys present in the sweeps
    for k in df.columns:
        if k in strat:
            strat[k] = best[k]
    cfg['strategy'] = strat

    out_path = Path(args.out) if args.out else Path(args.config)
    out_path.write_text(json.dumps(cfg, indent=2))
    logger.info("Wrote updated config to %s", out_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
