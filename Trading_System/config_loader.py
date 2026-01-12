import json
import os
from pathlib import Path
from typing import Dict, Any

try:
    from dotenv import load_dotenv  # optional
    _HAS_DOTENV = True
except Exception:
    load_dotenv = None
    _HAS_DOTENV = False

class ConfigLoader:
    def __init__(self, path: str = "config/config.json"):
        self.path = Path(path)
        # Load .env from repo root if available
        if _HAS_DOTENV:
            # Try repository root and config directory
            load_dotenv(dotenv_path=Path(".env"), override=False)
            if self.path.parent.exists():
                load_dotenv(dotenv_path=self.path.parent / ".env", override=False)

    def load(self) -> Dict[str, Any]:
        with self.path.open() as f:
            cfg = json.load(f)
        # Env overrides for Alpaca keys and environment
        alp = cfg.setdefault("alpaca", {})
        alp_env_key = os.getenv("ALPACA_API_KEY_ID")
        alp_env_secret = os.getenv("ALPACA_API_SECRET_KEY")
        alp_env_mode = os.getenv("ALPACA_ENV", "paper").lower()
        if alp_env_key:
            alp["apiKeyId"] = alp_env_key
        if alp_env_secret:
            alp["apiSecretKey"] = alp_env_secret
        if alp_env_mode in ("paper", "live"):
            alp["paper"] = (alp_env_mode == "paper")
        # Data store toggles
        data = cfg.setdefault("data", {})
        env_store_enabled = os.getenv("MINUTE_STORE_ENABLED")
        if env_store_enabled is not None:
            data["storeEnabled"] = env_store_enabled.lower() in ("1", "true", "yes", "on")
        env_store_root = os.getenv("MINUTE_STORE_ROOT")
        if env_store_root:
            data["storeRoot"] = env_store_root
        return cfg
