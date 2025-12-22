from dataclasses import dataclass
from typing import Optional

@dataclass
class SaxoConfig:
    fix_host: str
    fix_port: int
    fix_sender_comp_id: str
    fix_target_comp_id: str
    fix_username: str
    fix_password: str
    websocket_url: str
    websocket_token: str

    venue: str = "SAXO"
    # Optional RapidAPI Alpha Vantage config for intraday minute data
    rapidapi_host: str = "alpha-vantage.p.rapidapi.com"
    rapidapi_key: Optional[str] = None
