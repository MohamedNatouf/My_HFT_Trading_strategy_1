import asyncio
import logging
from saxo.system import TradingSystem

# Initialize structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting backtest script")
    ts = TradingSystem("config/saxo_config.json")
    allocs = await ts.run()
    logger.info("Backtest finished with %d allocation windows", len(allocs) if allocs else 0)
    if allocs:
        logger.info("Backtest produced %d allocations. Last allocation:",
                    len(allocs))
        last = allocs[-1]
        logger.info("Date: %s", last.date)
        logger.info("Positions:")
        for p in last.positions:
            logger.info("  %s: %.4f", p.symbol, p.weight)
        logger.info("Emergency: %s", last.emergency)

if __name__ == "__main__":
    asyncio.run(main())
