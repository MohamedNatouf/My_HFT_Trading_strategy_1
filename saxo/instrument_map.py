# Mapping between strategy symbols and Saxo instruments (UIC/AssetType)
from typing import Dict, Any, List

class InstrumentMap:
    def __init__(self, instruments: List[Dict[str, Any]]):
        self.by_symbol = {i['symbol']: i for i in instruments}
        self.by_uic = {i['uic']: i for i in instruments}

    def to_symbol(self, uic: int) -> str:
        return self.by_uic[uic]['symbol']

    def to_uic(self, symbol: str) -> int:
        return self.by_symbol[symbol]['uic']

    def asset_type(self, symbol: str) -> str:
        return self.by_symbol[symbol]['assetType']
