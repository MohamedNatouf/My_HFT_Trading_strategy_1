from dataclasses import dataclass
from typing import Optional

@dataclass
class AlpacaConfig:
    api_key_id: str
    api_secret_key: str
    paper: bool = True
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None
    # Market data feed selection for live streaming
    feed: str = "iex"  # or "sip"
    # Account/monitoring settings
    account_poll_seconds: int = 30

    # Optional RapidAPI Alpha Vantage config for intraday minute data (legacy backfill)
    rapidapi_host: str = "alpha-vantage.p.rapidapi.com"
    rapidapi_key: Optional[str] = None
