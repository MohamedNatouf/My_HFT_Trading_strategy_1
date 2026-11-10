"""SPY put credit spread runner (item 15, PAPER ONLY until validated).

Rules (from ``scripts/research.py putspread``; robust region, not the grid maximum):
* once a day (e.g. 10:00 ET): if no spread is open and VIX <= ``vix_stop``, open a put credit
  spread on the monthly expiry 25-45 calendar days out: short the put closest to
  ``short_delta`` (greeks from Alpaca option snapshots, free indicative feed), long the put
  ``width`` x spot below it
* size: contracts = floor(risk_per_trade * book / (max_loss_per_spread * 100))
* close the spread ``close_dte`` calendar days before expiry (never hold into expiration /
  assignment); limit orders at the mid +/- ``limit_buffer``
* multi-leg (mleg) orders so both legs fill together; credits are NEGATIVE limit prices
  (Alpaca convention: credits negative, debits positive)

Needs options level 3 on the account (spreads). Everything is a dry run unless executed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from ..strategies.options import monthly_expiries

logger = logging.getLogger(__name__)
NY = "America/New_York"


@dataclass
class SpreadConfig:
    underlying: str = "SPY"
    short_delta: float = -0.20
    width: float = 0.10
    risk_per_trade: float = 0.03
    vix_stop: float | None = 40.0
    min_dte: int = 25
    max_dte: int = 45
    close_dte: int = 2
    allocation_fraction: float = 0.25   # share of account equity this sleeve may risk against
    limit_buffer: float = 0.02          # $ concession from the mid per spread
    min_credit: float = 0.05
    # PAPER ONLY: allow a single contract when sizing rounds to 0 but its max loss is below this
    # fraction of equity (lets small paper accounts validate execution of the robust wide spread)
    paper_one_contract_max_risk: float = 0.08


def parse_occ(sym: str) -> dict:
    """'SPY261218P00600000' -> underlying, expiry, right, strike."""
    root = sym[:-15]
    return {"underlying": root, "expiry": pd.Timestamp(datetime.strptime(sym[-15:-9], "%y%m%d")),
            "right": sym[-9], "strike": int(sym[-8:]) / 1000}


def _mid(snap) -> float | None:
    q = getattr(snap, "latest_quote", None)
    if q is None or not q.bid_price or not q.ask_price or q.ask_price < q.bid_price:
        return None
    return (float(q.bid_price) + float(q.ask_price)) / 2


def choose_expiry(today: date, cfg: SpreadConfig) -> pd.Timestamp | None:
    t = pd.Timestamp(today)
    for e in monthly_expiries(t, t + pd.Timedelta(days=cfg.max_dte + 31)):
        if cfg.min_dte <= (e - t).days <= cfg.max_dte:
            return e
    return None


def choose_spread(snapshots: dict, spot: float, cfg: SpreadConfig) -> dict | None:
    puts = []
    for sym, snap in snapshots.items():
        info = parse_occ(sym)
        g = getattr(snap, "greeks", None)
        mid = _mid(snap)
        if info["right"] != "P" or g is None or g.delta is None or mid is None:
            continue
        puts.append({"symbol": sym, "strike": info["strike"], "delta": float(g.delta), "mid": mid})
    if not puts:
        return None
    short = min(puts, key=lambda p: abs(p["delta"] - cfg.short_delta))
    target = short["strike"] - cfg.width * spot
    longs = [p for p in puts if p["strike"] < short["strike"]]
    if not longs:
        return None
    long = min(longs, key=lambda p: abs(p["strike"] - target))
    credit = short["mid"] - long["mid"]
    width_pts = short["strike"] - long["strike"]
    return {"short": short, "long": long, "credit": credit, "width_pts": width_pts,
            "max_loss": width_pts - credit}


def size_contracts(book: float, cfg: SpreadConfig, max_loss: float) -> int:
    if max_loss <= 0:
        return 0
    return int(math.floor(cfg.risk_per_trade * book / (max_loss * 100)))


def _cid(kind: str, day: str, legs: str) -> str:
    return f"opt-{kind}-{day}-" + hashlib.sha1(f"{kind}|{day}|{legs}".encode()).hexdigest()[:12]


class PutSpreadRunner:
    def __init__(self, settings, dry_run: bool = True, trading=None, option_data=None, vix_fn=None,
                 equity_fn=None, spot_fn=None, state_dir: str | Path = "state", alerter=None):
        self.cfg = SpreadConfig(**(settings.raw.get("options") or {}))
        self.dry_run = dry_run
        c = settings.alpaca
        if trading is None:
            from alpaca.trading.client import TradingClient
            trading = TradingClient(c.key_id, c.secret, paper=c.paper)
        if option_data is None:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            option_data = OptionHistoricalDataClient(c.key_id, c.secret)
        self.trading, self.data = trading, option_data
        self.vix_fn = vix_fn or _yahoo_vix
        self.equity_fn = equity_fn or (lambda: float(self.trading.get_account().equity))
        self.spot_fn = spot_fn
        self.settings = settings
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.journal = self.state_dir / "options_journal.jsonl"
        self.alerter = alerter

    def _log(self, event: str, **kw):
        with self.journal.open("a") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "event_type": event, **kw},
                               default=str) + "\n")

    def open_legs(self) -> list[dict]:
        out = []
        for p in self.trading.get_all_positions():
            sym = p.symbol
            if len(sym) > 15 and sym.startswith(self.cfg.underlying) and sym[-9] in "PC":
                out.append({"symbol": sym, "qty": float(p.qty), **parse_occ(sym)})
        return out

    def _spot(self) -> float:
        if self.spot_fn:
            return self.spot_fn()
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest
        c = self.settings.alpaca
        t = StockHistoricalDataClient(c.key_id, c.secret).get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=self.cfg.underlying))
        return float(t[self.cfg.underlying].price)

    def _submit(self, legs: list[tuple[str, str, str]], qty: int, limit: float, cid: str) -> dict:
        rec = {"legs": legs, "qty": qty, "limit_price": round(limit, 2), "client_order_id": cid}
        if self.dry_run:
            rec["status"] = "dry_run"
            return rec
        from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
        try:
            self.trading.get_order_by_client_id(cid)
            rec["status"] = "duplicate_skipped"
            return rec
        except Exception:
            pass
        req = LimitOrderRequest(
            qty=qty, order_class=OrderClass.MLEG, time_in_force=TimeInForce.DAY, limit_price=round(limit, 2),
            client_order_id=cid,
            legs=[OptionLegRequest(symbol=s, ratio_qty=1, side=OrderSide(side), position_intent=PositionIntent(intent))
                  for s, side, intent in legs])
        try:
            o = self.trading.submit_order(req)
            rec.update(status="submitted", order_id=str(o.id))
        except Exception as e:
            rec["status"] = f"error: {e}"
            if self.alerter:
                self.alerter.send("Option spread order failed", str(e), level="critical", key=f"opt-{cid}")
        return rec

    def run_once(self, today: date | None = None) -> dict:
        today = today or pd.Timestamp.now(tz=NY).date()
        day = today.strftime("%Y%m%d")
        cfg = self.cfg
        legs = self.open_legs()
        # 1) close an open spread near expiry
        if legs:
            exp = min(leg["expiry"] for leg in legs)
            dte = (exp - pd.Timestamp(today)).days
            if dte > cfg.close_dte:
                self._log("hold", legs=legs, dte=dte)
                return {"action": "hold", "dte": dte, "legs": legs}
            snaps = self._snapshots([leg["symbol"] for leg in legs])
            debit = 0.0
            close_legs = []
            for leg in legs:
                mid = _mid(snaps.get(leg["symbol"])) or 0.0
                if leg["qty"] < 0:      # short -> buy to close
                    debit += mid
                    close_legs.append((leg["symbol"], "buy", "buy_to_close"))
                else:
                    debit -= mid
                    close_legs.append((leg["symbol"], "sell", "sell_to_close"))
            qty = int(min(abs(leg["qty"]) for leg in legs))
            rec = self._submit(close_legs, qty, max(debit, 0.01) + cfg.limit_buffer,
                               _cid("close", day, ",".join(s for s, *_ in close_legs)))
            self._log("close", dte=dte, order=rec)
            return {"action": "close", "order": rec}
        # 2) open a new spread
        vix = self.vix_fn()
        if cfg.vix_stop is not None and (vix is None or vix > cfg.vix_stop):
            self._log("skip_vix", vix=vix)
            return {"action": "skip", "reason": f"VIX {vix} unavailable or > {cfg.vix_stop}"}
        exp = choose_expiry(today, cfg)
        if exp is None:
            return {"action": "skip", "reason": "no expiry in DTE window"}
        spot = self._spot()
        snaps = self._chain(exp, spot)
        pick = choose_spread(snaps, spot, cfg)
        if pick is None or pick["credit"] < cfg.min_credit:
            self._log("skip_chain", expiry=str(exp.date()), pick=pick)
            return {"action": "skip", "reason": "no acceptable spread", "pick": pick}
        book = self.equity_fn() * cfg.allocation_fraction
        n = size_contracts(book, cfg, pick["max_loss"])
        if n < 1 and self.settings.alpaca.paper and pick["max_loss"] * 100 <= cfg.paper_one_contract_max_risk * book:
            n = 1
            logger.warning("paper: 1 contract risks %.1f%% of the book (> %.0f%% budget)",
                           pick["max_loss"] * 100 / book * 100, cfg.risk_per_trade * 100)
        if n < 1:
            return {"action": "skip", "reason": f"book ${book:,.0f} too small for one spread", "pick": pick}
        limit = -(pick["credit"] - cfg.limit_buffer)           # negative = credit received
        open_legs = [(pick["short"]["symbol"], "sell", "sell_to_open"), (pick["long"]["symbol"], "buy", "buy_to_open")]
        rec = self._submit(open_legs, n, limit, _cid("open", day, ",".join(s for s, *_ in open_legs)))
        self._log("open", expiry=str(exp.date()), spot=spot, vix=vix, pick=pick, contracts=n, order=rec)
        return {"action": "open", "pick": pick, "contracts": n, "order": rec}

    # --- data -------------------------------------------------------------------
    def _snapshots(self, symbols: list[str]) -> dict:
        from alpaca.data.requests import OptionSnapshotRequest
        return dict(self.data.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=symbols)))

    def _chain(self, expiry: pd.Timestamp, spot: float) -> dict:
        from alpaca.trading.enums import ContractType
        from alpaca.data.requests import OptionChainRequest
        req = OptionChainRequest(underlying_symbol=self.cfg.underlying, type=ContractType.PUT,
                                 expiration_date=expiry.date(), strike_price_gte=round(spot * 0.70),
                                 strike_price_lte=round(spot * 1.0))
        return dict(self.data.get_option_chain(req))


def _yahoo_vix() -> float | None:
    try:
        import yfinance as yf
        h = yf.Ticker("^VIX").history(period="5d")
        return float(h["Close"].iloc[-1])
    except Exception as e:  # never trade blind on a missing VIX: caller treats None as unknown
        logger.warning("VIX unavailable: %s", e)
        return None
