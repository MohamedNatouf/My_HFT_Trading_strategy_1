# Mapping between strategy symbols and instruments
from typing import Dict, Any, List, Optional

class InstrumentMap:
    def __init__(self, instruments: List[Dict[str, Any]]):
        self.by_symbol: Dict[str, Dict[str, Any]] = {i.get('symbol'): i for i in instruments if i.get('symbol')}
        # Build by_uic only for entries that have a UIC
        self.by_uic: Dict[int, Dict[str, Any]] = {}
        for i in instruments:
            uic = i.get('uic')
            if uic is not None:
                self.by_uic[int(uic)] = i

    def to_symbol(self, uic: int) -> Optional[str]:
        return (self.by_uic.get(uic) or {}).get('symbol')

    def to_uic(self, symbol: str) -> Optional[int]:
        entry = self.by_symbol.get(symbol)
        return entry.get('uic') if entry else None

    def asset_type(self, symbol: str) -> Optional[str]:
        entry = self.by_symbol.get(symbol)
        return entry.get('assetType') if entry else None
