import asyncio
import logging
import argparse
import os
import sys

# Ensure repository root is on sys.path when running from anywhere
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Trading_System.system import TradingSystem

# Initialize structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(description="Run minute-data backtest")
    parser.add_argument('--config', default='config/config.json', help='Path to config JSON')
    parser.add_argument('--mode', default='backtest', help='Mode: backtest or live')
    # Backtest overrides
    parser.add_argument('--source', help='Backtest data source (alpaca, alphaVantage, yahoo)')
    parser.add_argument('--interval', help='Backtest interval, e.g., 1Min')
    parser.add_argument('--start', help='Backtest start date/time ISO8601')
    parser.add_argument('--end', help='Backtest end date/time ISO8601')
    parser.add_argument('--feed', help='Market data feed for live/ws or backtest metadata (iex/sip)')
    parser.add_argument('--output_size', help='Output size for AV historical (compact/full)')
    args = parser.parse_args()

    logger.info("Starting backtest script with config=%s", args.config)
    ts = TradingSystem(args.config)
    # Apply mode
    ts.config['mode'] = args.mode.lower()
    # Apply overrides into config
    bt = ts.config.setdefault('backtest', {})
    if args.source:
        bt['source'] = args.source
    if args.interval:
        bt['interval'] = args.interval
    if args.start is not None:
        bt['start'] = args.start
    if args.end is not None:
        bt['end'] = args.end
    if args.output_size:
        bt['output_size'] = args.output_size
    if args.feed:
        bt['feed'] = args.feed

    if ts.config['mode'] == 'backtest':
        allocs = await ts.run_backtest()
        logger.info("Backtest finished with %d allocation windows", len(allocs) if allocs else 0)
        if allocs:
            last = allocs[-1]
            logger.info("Date: %s", last.date)
            logger.info("Positions:")
            for p in last.positions:
                logger.info("  %s: %.4f", p.symbol, p.weight)
            logger.info("Emergency: %s", last.emergency)
    else:
        await ts.run()

if __name__ == "__main__":
    asyncio.run(main())
