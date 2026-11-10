"""Settings loading.

Secrets come ONLY from the environment (optionally a local ``.env``). The JSON file
holds strategy / risk / execution parameters and must never contain keys.
Trading against a live account requires BOTH ``ALPACA_ENV=live`` and
``LIVE_TRADING_CONFIRM=yes`` so a stray config can never route real orders.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # optional dependency
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


@dataclass(frozen=True)
class AlpacaCredentials:
    key_id: str | None
    secret: str | None
    paper: bool
    data_feed: str  # "iex" (free) or "sip" (Algo Trader Plus) for real-time data

    @property
    def available(self) -> bool:
        return bool(self.key_id and self.secret)


@dataclass
class Settings:
    raw: dict[str, Any]
    alpaca: AlpacaCredentials
    path: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name) or {})


class LiveTradingNotConfirmed(RuntimeError):
    pass


def load_env(dotenv_path: str | os.PathLike | None = None) -> None:
    if load_dotenv is None:
        return
    load_dotenv(dotenv_path or Path(".env"), override=False)


def credentials_from_env() -> AlpacaCredentials:
    env = (os.getenv("ALPACA_ENV") or "paper").strip().lower()
    if env not in ("paper", "live"):
        raise ValueError(f"ALPACA_ENV must be 'paper' or 'live', got {env!r}")
    paper = env == "paper"
    if not paper and (os.getenv("LIVE_TRADING_CONFIRM") or "").strip().lower() != "yes":
        raise LiveTradingNotConfirmed(
            "ALPACA_ENV=live requires LIVE_TRADING_CONFIRM=yes in the environment."
        )
    return AlpacaCredentials(
        key_id=os.getenv("ALPACA_API_KEY_ID") or None,
        secret=os.getenv("ALPACA_API_SECRET_KEY") or None,
        paper=paper,
        data_feed=(os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower(),
    )


_SECRET_HINTS = ("apikeyid", "apisecretkey", "secret", "password", "apikey", "token", "webhook")


def _assert_no_secrets(obj: Any, trail: str = "") -> None:
    """Refuse configs that embed non-empty secret-looking values."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower().replace("_", "")
            if any(h in key for h in _SECRET_HINTS) and isinstance(v, str) and v.strip():
                raise ValueError(
                    f"Config contains a secret at '{trail}{k}'. Move it to .env (see .env.example)."
                )
            _assert_no_secrets(v, f"{trail}{k}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_secrets(v, f"{trail}{i}.")


def load_settings(path: str | os.PathLike | None = "config/quant.json") -> Settings:
    load_env()
    raw: dict[str, Any] = {}
    p = Path(path) if path else None
    if p is not None and p.exists():
        raw = json.loads(p.read_text())
        _assert_no_secrets(raw)
    return Settings(raw=raw, alpaca=credentials_from_env(), path=p)
