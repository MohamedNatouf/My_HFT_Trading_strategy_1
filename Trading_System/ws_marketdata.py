# Saxo OpenAPI WebSocket client and minute aggregation (deprecated - retained only for reference)
# This module is deprecated; live data is sourced from Alpaca.
# Classes are kept minimal to avoid import errors if referenced elsewhere.

from typing import Callable, Dict, Any
import pandas as pd

class SaxoWSClient:
    def __init__(self, *args, **kwargs):
        pass
    async def connect(self):
        pass
    async def rest_subscribe_quotes(self, *args, **kwargs):
        return []
    async def stream(self, on_message: Callable[[dict], None]):
        pass
    async def close(self):
        pass

class MinuteAggregator:
    def __init__(self):
        self.frames = []
    def on_quote(self, msg: Dict[str, Any]):
        pass
    def finalize_minute(self) -> pd.DataFrame:
        return pd.DataFrame()
