# Saxo OpenAPI WebSocket streaming client with REST subscription and minute aggregation per OpenAPI docs.
import asyncio
import json
import aiohttp
import websockets
from typing import Callable, Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import pandas as pd
import numpy as np

CONTROL_HEARTBEAT = "_heartbeat"
CONTROL_DISCONNECT = "_disconnect"
CONTROL_RESET = "_resetsubscriptions"

class SaxoWSClient:
    def __init__(self, base_stream_url: str, token: str, context_id: str = "ctx-1"):
        # base_stream_url: e.g. https://sim-streaming.saxobank.com/sim/oapi/streaming/ws/connect
        self.base_stream_url = base_stream_url
        self.token = token
        self.context_id = context_id
        self._conn = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_message_id: Optional[int] = None
        self._subs_locations: Dict[str, str] = {}  # refId -> subscription location URL

    async def connect(self):
        # Per docs, ContextId must be query param; token via Authorization header preferred
        headers = {"Authorization": f"BEARER {self.token}"}
        # Use query param for contextId, and optionally last message id on reconnect
        url = f"{self.base_stream_url}?contextId={self.context_id}"
        if self._last_message_id is not None:
            url += f"&messageid={self._last_message_id}"
        self._conn = await websockets.connect(url, extra_headers=headers)
        self._session = aiohttp.ClientSession(headers={"Authorization": f"BEARER {self.token}", "Content-Type": "application/json"})

    async def authorize(self, authorize_url: str):
        # PUT /streaming/ws/authorize?contextid={contextid}
        assert self._session is not None
        async with self._session.put(f"{authorize_url}?contextid={self.context_id}") as resp:
            if resp.status != 202:
                txt = await resp.text()
                raise RuntimeError(f"WS authorize failed: {resp.status} {txt}")

    async def rest_subscribe_quotes(self, gateway_base_url: str, instruments: List[Dict[str, Any]], reference_id_prefix: str, refresh_rate_ms: int = 1000) -> List[str]:
        # POST /trade/v1/prices/subscriptions with AssetType/Uic
        ref_ids: List[str] = []
        assert self._session is not None
        for inst in instruments:
            ref_id = f"{reference_id_prefix}-{inst['uic']}"
            payload = {
                "Arguments": {
                    "AssetType": inst["assetType"],
                    "Uic": int(inst["uic"]) ,
                    # "FieldGroups": ["Quote"],
                },
                "ContextId": self.context_id,
                "ReferenceId": ref_id,
                "RefreshRate": refresh_rate_ms
            }
            url = f"{gateway_base_url}/trade/v1/prices/subscriptions"
            async with self._session.post(url, json=payload) as resp:
                if resp.status not in (200, 201):
                    txt = await resp.text()
                    raise RuntimeError(f"Subscription failed: {resp.status} {txt}")
                # read Location header for deletion later
                loc = resp.headers.get("Location")
                if loc:
                    self._subs_locations[ref_id] = loc
            ref_ids.append(ref_id)
        return ref_ids

    async def rest_delete_subscription(self, ref_id: str):
        # DELETE {location} or gateway/subscriptions/{context}/{ref}
        assert self._session is not None
        loc = self._subs_locations.get(ref_id)
        if not loc:
            return
        async with self._session.delete(loc) as resp:
            # No content expected
            if resp.status not in (200, 202, 204):
                txt = await resp.text()
                raise RuntimeError(f"Delete subscription failed: {resp.status} {txt}")
        del self._subs_locations[ref_id]

    async def stream(self, on_message: Callable[[dict], None], on_control: Optional[Callable[[dict], None]] = None):
        # Receive streaming frames; payload is JSON messages or control messages
        while True:
            frame = await self._conn.recv()
            # Saxo frames may be binary; assume text for simplicity; real impl should parse header framing
            try:
                data = json.loads(frame)
            except Exception:
                # ignore non-JSON payloads (protobuf not supported here)
                continue
            # Messages can be arrays or envelopes with Data
            if isinstance(data, dict) and data.get("ReferenceId"):
                ref = data["ReferenceId"]
                if ref.startswith("_"):
                    # control
                    if on_control:
                        on_control(data)
                    # handle resets locally
                    if ref == CONTROL_RESET:
                        targets = data.get("TargetReferenceIds", []) or list(self._subs_locations.keys())
                        for rid in targets:
                            await self.rest_delete_subscription(rid)
                    elif ref == CONTROL_DISCONNECT:
                        # disconnect and raise
                        await self.close()
                        raise ConnectionError("Server requested disconnect")
                    elif ref == CONTROL_HEARTBEAT:
                        pass
                else:
                    on_message(data)
            elif isinstance(data, dict) and 'Data' in data:
                for item in data['Data']:
                    # each item has ReferenceId/Body
                    on_message(item)
            # Update last message id if present
            msg_id = data.get('MessageId') if isinstance(data, dict) else None
            if isinstance(msg_id, int):
                self._last_message_id = msg_id

    async def close(self):
        if self._conn:
            await self._conn.close()
        if self._session:
            await self._session.close()


class MinuteAggregator:
    """
    Build 1-minute OHLCV bars per ReferenceId from Saxo price updates.
    Output DataFrame index by minute (UTC), columns as MultiIndex (PriceType, ReferenceId).
    PriceType in [Open, High, Low, Close, Volume]. Volume is synthetic (count of ticks) if not provided.
    """
    def __init__(self):
        self.current_minute: Optional[pd.Timestamp] = None
        # state: ref_id -> [open, high, low, close, volume]
        self.state: Dict[str, List[float]] = {}
        self.frames: List[pd.DataFrame] = []

    @staticmethod
    def _extract_price(msg: Dict[str, Any]) -> Tuple[str, Optional[float], pd.Timestamp]:
        # Expect message with keys: ReferenceId, Body, Timestamp (optional)
        ref = msg.get("ReferenceId")
        body = msg.get("Body", msg)
        # Saxo FX price body typically has Bid and Ask; take mid
        price = None
        if isinstance(body, dict):
            if "Mid" in body and isinstance(body["Mid"], (int, float)):
                price = float(body["Mid"])
            elif "LastTraded" in body and isinstance(body["LastTraded"], (int, float)):
                price = float(body["LastTraded"])
            elif "Bid" in body and "Ask" in body and all(isinstance(body[k], (int, float)) for k in ("Bid", "Ask")):
                price = (float(body["Bid"]) + float(body["Ask"])) / 2.0
            elif "Price" in body and isinstance(body["Price"], (int, float)):
                price = float(body["Price"]) 
        ts_raw = body.get("Time") if isinstance(body, dict) else None
        if ts_raw:
            try:
                ts = pd.to_datetime(ts_raw, utc=True)
            except Exception:
                ts = pd.Timestamp.utcnow().tz_localize("UTC")
        else:
            ts = pd.Timestamp.utcnow().tz_localize("UTC")
        # Truncate to minute
        ts_min = ts.floor('T')
        return ref, price, ts_min

    def _ensure_minute(self, ts_min: pd.Timestamp):
        if self.current_minute is None:
            self.current_minute = ts_min
            return
        if ts_min > self.current_minute:
            # finalize current minute frame
            if self.state:
                cols = []
                data_open = []; data_high = []; data_low = []; data_close = []; data_vol = []
                for ref_id, ohlcv in self.state.items():
                    cols.append(ref_id)
                    o, h, l, c, v = ohlcv
                    data_open.append(o)
                    data_high.append(h)
                    data_low.append(l)
                    data_close.append(c)
                    data_vol.append(v)
                arrays = [
                    ("Open", rid) for rid in cols
                ] + [
                    ("High", rid) for rid in cols
                ] + [
                    ("Low", rid) for rid in cols
                ] + [
                    ("Close", rid) for rid in cols
                ] + [
                    ("Volume", rid) for rid in cols
                ]
                values = data_open + data_high + data_low + data_close + data_vol
                df = pd.DataFrame([values], index=[self.current_minute])
                df.columns = pd.MultiIndex.from_tuples(arrays)
                self.frames.append(df)
            # reset for new minute
            self.state = {}
            self.current_minute = ts_min

    def on_quote(self, msg: Dict[str, Any]):
        ref, price, ts_min = self._extract_price(msg)
        if ref is None or price is None:
            return
        self._ensure_minute(ts_min)
        bar = self.state.get(ref)
        if bar is None:
            self.state[ref] = [price, price, price, price, 1.0]
        else:
            # update OHLCV
            bar[1] = max(bar[1], price)
            bar[2] = min(bar[2], price)
            bar[3] = price
            bar[4] += 1.0

    def finalize_minute(self) -> pd.DataFrame:
        if not self.frames:
            return pd.DataFrame()
        df = pd.concat(self.frames)
        self.frames = []
        return df
