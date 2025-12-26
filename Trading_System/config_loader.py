import json
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, path: str = "config/config.json"):
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        with self.path.open() as f:
            return json.load(f)
