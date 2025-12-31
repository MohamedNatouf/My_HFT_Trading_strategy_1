import asyncio
import logging
import time
from typing import Optional, Callable, Dict, Any
import requests

logger = logging.getLogger(__name__)

class AlpacaSSEClient:
    """
    Minimal SSE client for Alpaca Broker API event streams.
    Supports optional OAuth2 client-credentials to auth against authx and use Bearer token.
    If no client_id/client_secret are provided, falls back to Basic auth using api_key:api_secret.
    """
    def __init__(
        self,
        api_key: Optional[str],
        api_secret: Optional[str],
        sandbox: bool = True,
        oauth_client_id: Optional[str] = None,
        oauth_client_secret: Optional[str] = None,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self.oauth_client_id = oauth_client_id
        self.oauth_client_secret = oauth_client_secret
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0
        self.session = requests.Session()
        # Base URLs
        self.base = "https://broker-api.sandbox.alpaca.markets" if sandbox else "https://broker-api.alpaca.markets"
        self.authx = "https://authx.sandbox.alpaca.markets" if sandbox else "https://authx.alpaca.markets"

    def _ensure_token(self):
        """Fetch or refresh OAuth2 token. Valid for 15 minutes."""
        if not self.oauth_client_id or not self.oauth_client_secret:
            # No OAuth configured
            return None
        now = time.time()
        if self._access_token and now < self._token_expiry - 30:
            return self._access_token
        try:
            r = self.session.post(
                f"{self.authx}/v1/oauth2/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.oauth_client_id,
                    "client_secret": self.oauth_client_secret,
                },
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            self._access_token = data.get("access_token")
            expires_in = data.get("expires_in", 899)
            self._token_expiry = now + float(expires_in)
            logger.info("Obtained Alpaca access token; expires_in=%s", expires_in)
            return self._access_token
        except Exception as e:
            logger.error("OAuth token fetch failed: %s", e)
            return None

    def _headers(self) -> Dict[str, str]:
        # Prefer Bearer if token available
        tok = self._ensure_token()
        if tok:
            return {"Authorization": f"Bearer {tok}"}
        # Fallback to Basic using api key/secret
        if self.api_key and self.api_secret:
            from base64 import b64encode
            basic = b64encode(f"{self.api_key}:{self.api_secret}".encode()).decode()
            return {"Authorization": f"Basic {basic}"}
        return {}

    async def _stream(self, url: str, on_event: Callable[[Dict[str, Any]], None], params: Optional[Dict[str, str]] = None):
        """Basic SSE reader using requests streaming; runs in thread via asyncio.to_thread."""
        headers = self._headers()
        backoff = 1
        # Reconnect loop
        while True:
            try:
                with self.session.get(url, headers=headers, params=params or {}, stream=True, timeout=30) as resp:
                    resp.raise_for_status()
                    buf = ""
                    for raw in resp.iter_lines(decode_unicode=True):
                        if raw is None:
                            continue
                        line = raw.strip()
                        if not line:
                            # message boundary
                            txt = buf.strip()
                            buf = ""
                            if not txt:
                                continue
                            try:
                                import json
                                # Some SSE servers prefix with 'data: ' lines; normalize
                                if txt.startswith("data: "):
                                    txt = txt[6:]
                                evt = json.loads(txt)
                                on_event(evt)
                            except Exception:
                                logger.debug("Non-JSON SSE chunk: %s", txt[:200])
                            continue
                        if line.startswith(":heartbeat"):
                            # heartbeat; ignore
                            continue
                        if line.startswith("data:") or line.startswith("event:") or line.startswith("id:"):
                            buf += line + "\n"
                        # else: ignore comments or unexpected
                backoff = 1
            except Exception as e:
                logger.warning("SSE stream error on %s: %s; reconnecting in %ss", url, e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def subscribe_orders(self, on_event: Callable[[Dict[str, Any]], None], since_id: Optional[str] = None, since_ulid: Optional[str] = None):
        # Trades SSE (orders/executions)
        url = f"{self.base}/v1/events/trade/updates"
        params: Dict[str, str] = {}
        if since_ulid:
            params['since_ulid'] = since_ulid
        elif since_id:
            params['since_id'] = since_id
        await self._stream(url, on_event, params)

    async def subscribe_journals(self, on_event: Callable[[Dict[str, Any]], None], since_id: Optional[str] = None, since_ulid: Optional[str] = None):
        url = f"{self.base}/v1/events/journal/updates"
        params: Dict[str, str] = {}
        if since_ulid:
            params['since_ulid'] = since_ulid
        elif since_id:
            params['since_id'] = since_id
        await self._stream(url, on_event, params)

    async def subscribe_transfers(self, on_event: Callable[[Dict[str, Any]], None], since_id: Optional[str] = None, since_ulid: Optional[str] = None):
        url = f"{self.base}/v1/events/transfer/updates"
        params: Dict[str, str] = {}
        if since_ulid:
            params['since_ulid'] = since_ulid
        elif since_id:
            params['since_id'] = since_id
        await self._stream(url, on_event, params)

    async def subscribe_nta(self, on_event: Callable[[Dict[str, Any]], None], since_id: Optional[str] = None, since_ulid: Optional[str] = None):
        # Non-trade activities notifications
        url = f"{self.base}/v1/events/nta"
        params: Dict[str, str] = {}
        if since_ulid:
            params['since_ulid'] = since_ulid
        elif since_id:
            params['since_id'] = since_id
        await self._stream(url, on_event, params)
