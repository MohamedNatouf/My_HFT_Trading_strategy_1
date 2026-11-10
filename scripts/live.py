import argparse
import asyncio
import logging
import os
import sys
import signal

# Ensure repository root is on sys.path when running from scripts/
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from logging_conf import setup_logging
from Trading_System.system import TradingSystem


async def main(args: argparse.Namespace) -> None:
    # Configure logging with a per-run id
    run_id = args.run_id or "live"
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(run_id=run_id, level=level)

    logger = logging.getLogger(__name__)
    logger.info("Starting live trading run_id=%s flatten_on_exit=%s", run_id, args.flatten_on_exit)

    system = TradingSystem(config_path=args.config)
    if args.clear_store:
        logger.info("Clearing local minute-bar store before starting live run")
        try:
            removed = system.store.clear_all()
            logger.info("Minute store cleared; files_removed=%d", removed)
        except Exception as e:
            logger.error("Failed to clear minute store: %s", e)
    # Force live mode regardless of config file to ensure forward streaming and trading
    try:
        system.config['mode'] = 'live'
        logging.getLogger(__name__).info("Live runner overriding config mode to 'live'")
    except Exception:
        pass

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    # Signal handlers (Ctrl+C etc.): only set the stop event (thread-safe);
    # the actual shutdown/flatten happens below in the main coroutine.
    def _handle_signal(sig, frame):
        logger.info("Received signal %s; scheduling shutdown", sig)
        loop.call_soon_threadsafe(stop_event.set)

    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _handle_signal)
        except (ValueError, OSError):
            # Not all platforms support setting SIGTERM in this context
            pass

    # Run the system in live mode (run() dispatches to run_stream when mode != backtest)
    run_task = asyncio.create_task(system.run())

    # Wait until either the run finishes or a shutdown is requested
    done, pending = await asyncio.wait({run_task, asyncio.create_task(stop_event.wait())}, return_when=asyncio.FIRST_COMPLETED)

    # If stop_event fired first, cancel the run task
    if stop_event.is_set() and not run_task.done():
        logger.info("Cancelling live run task after shutdown request")
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            logger.info("Live run task cancelled")
    for t in pending:
        t.cancel()

    # Flatten only after the trading loop has stopped, so no new orders race the close-out.
    if args.flatten_on_exit:
        logger.info("Flatten-on-exit enabled; flattening positions before exit")
        try:
            await system.flatten_and_exit()
        except Exception as e:
            logger.error("Error during flatten_and_exit: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live trading runner for TradingSystem")
    parser.add_argument("--config", default="config/config.json", help="Path to config JSON")
    parser.add_argument("--run-id", default=None, help="Run identifier used for log file naming")
    parser.add_argument("--clear-store", action="store_true", help="Clear local minute-bar store before starting live run")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    parser.add_argument("--flatten-on-exit", action="store_true", help="Flatten all strategy positions on shutdown")

    args = parser.parse_args()
    # Run main, with optional clearing of local minute store handled inside
    asyncio.run(main(args))
