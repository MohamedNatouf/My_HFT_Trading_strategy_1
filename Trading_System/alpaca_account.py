import logging
from typing import Optional

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.models import Account
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass
except Exception:
    TradingClient = None
    Account = None
    GetAssetsRequest = None
    AssetClass = None

logger = logging.getLogger(__name__)

class AlpacaAccountClient:
    def __init__(self, api_key: Optional[str], api_secret: Optional[str], paper: bool = True):
        if TradingClient is None:
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
        if self.client is None or GetAssetsRequest is None:
            return False
        try:
            req = GetAssetsRequest(symbol=symbol, asset_class=AssetClass.US_EQUITY)
            assets = self.client.get_assets(req)
            if not assets:
                return False
            a = assets[0]
            return bool(getattr(a, 'overnight_tradable', False)) and not bool(getattr(a, 'overnight_halted', False))
        except Exception as e:
            logger.error("asset check failed: %s", e)
            return False
