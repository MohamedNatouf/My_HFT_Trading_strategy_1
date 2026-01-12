import argparse
import sys
import logging
from typing import Optional, Tuple
from pathlib import Path

# Ensure repository root is on sys.path so `Trading_System` can be imported when running this script directly
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Trading_System.config_loader import ConfigLoader
from Trading_System.alpaca_account import AlpacaAccountClient
try:
    # Import the captured import error from the module for diagnostics
    from Trading_System.alpaca_account import IMPORT_ERR as ACCT_IMPORT_ERR
except Exception:
    ACCT_IMPORT_ERR = None

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.models import Position
    ALPACA_PY_AVAILABLE = True
except Exception:
    TradingClient = None
    Position = None
    ALPACA_PY_AVAILABLE = False

# Try to import top-level alpaca to capture version or import errors for better debug
try:
    import alpaca as _alpaca_pkg
    ALPACA_VERSION = getattr(_alpaca_pkg, "__version__", "?")
    ALPACA_TOP_IMPORT_ERR = None
except Exception as _e:
    _alpaca_pkg = None
    ALPACA_VERSION = None
    ALPACA_TOP_IMPORT_ERR = _e

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("alpaca_check")


def _get_keys_from_args_or_config(args) -> Tuple[Optional[str], Optional[str], bool]:
    if args.api_key and args.api_secret:
        paper = not args.live  # default to paper unless --live
        return args.api_key, args.api_secret, paper
    cfg = ConfigLoader(args.config).load()
    alp = cfg.get("alpaca", {})
    return alp.get("apiKeyId"), alp.get("apiSecretKey"), bool(alp.get("paper", True))


def load_keys_from_config(config_path: str):
    cfg = ConfigLoader(config_path).load()
    alp = cfg.get("alpaca", {})
    return alp.get("apiKeyId"), alp.get("apiSecretKey"), bool(alp.get("paper", True))


def check_api_connectivity(api_key: Optional[str], api_secret: Optional[str], paper: bool) -> bool:
    if not ALPACA_PY_AVAILABLE:
        logger.error("alpaca-py is not installed or failed to import. Install with: pip install alpaca-py")
        if ALPACA_TOP_IMPORT_ERR:
            logger.error("alpaca import error: %s", ALPACA_TOP_IMPORT_ERR)
        if ACCT_IMPORT_ERR:
            logger.error("alpaca-py import error detail: %s", ACCT_IMPORT_ERR)
        return False
    acct_client = AlpacaAccountClient(api_key, api_secret, paper)
    acct = acct_client.get_account()
    if acct is None:
        env = "paper" if paper else "live"
        logger.error("Failed to get account. Check API keys, environment (%s vs keys), network, or package installation.", env)
        if ACCT_IMPORT_ERR:
            logger.error("alpaca-py import error detail: %s", ACCT_IMPORT_ERR)
        return False
    logger.info("Connected: account id=%s status=%s", getattr(acct, "id", "?"), getattr(acct, "status", "?"))
    return True


def summarize_account(acct) -> dict:
    return {
        "account_number": getattr(acct, "account_number", None),
        "status": getattr(acct, "status", None),
        "trading_blocked": bool(getattr(acct, "trading_blocked", False)),
        "transfers_blocked": bool(getattr(acct, "transfers_blocked", False)),
        "account_blocked": bool(getattr(acct, "account_blocked", False)),
        "equity": float(getattr(acct, "equity", 0) or 0),
        "cash": float(getattr(acct, "cash", 0) or 0),
        "long_market_value": float(getattr(acct, "long_market_value", 0) or 0),
        "short_market_value": float(getattr(acct, "short_market_value", 0) or 0),
        "buying_power": float(getattr(acct, "buying_power", 0) or 0),
        "multiplier": getattr(acct, "multiplier", None),
        "created_at": str(getattr(acct, "created_at", "")),
    }


def _status_active(status_obj) -> bool:
    # Handle AccountStatus enum or string
    val = getattr(status_obj, "value", None)
    if isinstance(val, str):
        return val.upper() == "ACTIVE"
    s = str(status_obj)
    # e.g., "AccountStatus.ACTIVE"
    if ".ACTIVE" in s or s.upper() == "ACTIVE":
        return True
    return False


def check_positions(api_key: Optional[str], api_secret: Optional[str], paper: bool):
    if not ALPACA_PY_AVAILABLE:
        logger.error("alpaca-py is not installed or failed to import. Install with: pip install alpaca-py")
        if ALPACA_TOP_IMPORT_ERR:
            logger.error("alpaca import error: %s", ALPACA_TOP_IMPORT_ERR)
        return []
    try:
        client = TradingClient(api_key, api_secret, paper=paper)
        positions = client.get_all_positions()
        out = []
        for p in positions or []:
            out.append({
                "symbol": getattr(p, "symbol", ""),
                "qty": float(getattr(p, "qty", 0) or 0),
                "avg_entry_price": float(getattr(p, "avg_entry_price", 0) or 0),
                "market_value": float(getattr(p, "market_value", 0) or 0),
                "unrealized_pl": float(getattr(p, "unrealized_pl", 0) or 0),
                "side": "long" if float(getattr(p, "qty", 0) or 0) >= 0 else "short",
            })
        return out
    except Exception as e:
        logger.error("Failed to get positions: %s", e)
        return []


def readiness_checks(api_key: Optional[str], api_secret: Optional[str], paper: bool) -> int:
    if not ALPACA_PY_AVAILABLE:
        logger.error("alpaca-py is not installed or failed to import. Install with: pip install alpaca-py")
        if ALPACA_TOP_IMPORT_ERR:
            logger.error("alpaca import error: %s", ALPACA_TOP_IMPORT_ERR)
        if ACCT_IMPORT_ERR:
            logger.error("alpaca-py import error detail: %s", ACCT_IMPORT_ERR)
        return 2
    acct_client = AlpacaAccountClient(api_key, api_secret, paper)
    acct = acct_client.get_account()
    if acct is None:
        logger.error("Account retrieval failed.")
        return 2

    info = summarize_account(acct)
    logger.info("Account summary: %s", info)

    issues = []
    if not _status_active(info.get("status")):
        issues.append(f"Account status not ACTIVE: {info['status']}")
    if info["trading_blocked"]:
        issues.append("Trading is blocked")
    if info["account_blocked"]:
        issues.append("Account is blocked")
    if info["equity"] <= 0:
        issues.append("Equity is non-positive")
    if info["cash"] < 1.0:
        issues.append("Cash below $1; buy notional minimum may fail")

    positions = check_positions(api_key, api_secret, paper)
    if positions:
        logger.info("Open positions (%d):", len(positions))
        for pos in positions:
            logger.info(" - %s qty=%s mv=$%.2f upl=$%.2f", pos["symbol"], pos["qty"], pos["market_value"], pos["unrealized_pl"])
    else:
        logger.info("No open positions.")

    if issues:
        for i in issues:
            logger.error("Issue: %s", i)
        return 1
    logger.info("Account appears ready for trading.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Check Alpaca account readiness (API, status, positions)")
    parser.add_argument("--config", default="config/config.json", help="Path to config JSON")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--api-key", dest="api_key", default=None, help="Override apiKeyId from config")
    parser.add_argument("--api-secret", dest="api_secret", default=None, help="Override apiSecretKey from config")
    parser.add_argument("--live", action="store_true", help="Use live environment (default is paper)")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("python_exe=%s", sys.executable)
        logger.debug("python_version=%s", sys.version.replace("\n", " "))
        logger.debug("repo_root=%s", REPO_ROOT)
        logger.debug("sys.path[0:3]=%s", sys.path[0:3])
        logger.debug("alpaca_version=%s", ALPACA_VERSION if ALPACA_VERSION else "<import-failed>")
        if ALPACA_TOP_IMPORT_ERR:
            logger.debug("alpaca top-level import error: %s", ALPACA_TOP_IMPORT_ERR)
        if ACCT_IMPORT_ERR:
            logger.debug("alpaca-py import error detail: %s", ACCT_IMPORT_ERR)

    api_key, api_secret, paper = _get_keys_from_args_or_config(args)
    env = "live" if args.live else ("paper" if paper else "live")
    if args.verbose:
        logger.debug("env_selected=%s", env)

    if not api_key or not api_secret:
        logger.error("Missing API keys (from args or config)")
        return 2

    if not check_api_connectivity(api_key, api_secret, paper=(env == "paper")):
        return 2

    code = readiness_checks(api_key, api_secret, paper=(env == "paper"))
    return code


if __name__ == "__main__":
    sys.exit(main())
