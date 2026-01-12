import logging
from typing import Optional

IMPORT_ERR = None

try:
    from alpaca.trading.client import TradingClient
except Exception as e:
    TradingClient = None
    IMPORT_ERR = e

logger = logging.getLogger(__name__)

class AlpacaAccountClient:
    def __init__(self, api_key: Optional[str], api_secret: Optional[str], paper: bool = True):
        if TradingClient is None:
            if IMPORT_ERR is not None:
                logger.warning("alpaca-py import failed; account client unavailable: %s", IMPORT_ERR)
            else:
                logger.warning("alpaca-py not installed; account client unavailable")
            self.client = None
        else:
            try:
                self.client = TradingClient(api_key, api_secret, paper=paper)
            except Exception as e:
                logger.error("Failed to init TradingClient: %s", e)
                self.client = None

    def get_account(self):
        if self.client is None:
            return None
        try:
            return self.client.get_account()
        except Exception as e:
            logger.error("get_account failed: %s", e)
            return None

    def is_overnight_tradable(self, symbol: str) -> bool:
        """Check asset flags for overnight trading/halt."""
        if self.client is None:
            return False
        try:
            # Prefer simple get_asset call to avoid request/enums import brittleness
            a = self.client.get_asset(symbol)
            return bool(getattr(a, 'overnight_tradable', False)) and not bool(getattr(a, 'overnight_halted', False))
        except Exception as e:
            logger.error("asset check failed: %s", e)
            return False
