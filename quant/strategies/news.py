"""News-sentiment research (item 13), free: Alpaca News API (Benzinga headlines) + a local
finance lexicon scorer. ``score_fn`` is pluggable, so an LLM scorer can be dropped in later
without touching the backtest.

Two timing variants (no look-ahead: every article is assigned by its ``created_at`` stamp):
* overnight -> open: articles from 16:00 ET (previous session) to 09:15 ET; enter at the open,
  exit at the close (Lopez-Lira & Tang timing)
* intraday -> close: articles from 16:00 ET (previous session) to 15:30 ET; enter at the close
  (MOC), hold ``hold`` sessions

Positions are long positive / short negative names, each hedged with SPY (market neutral).
"""
from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd

NY = "America/New_York"

# Loughran-McDonald-style word lists + event phrases common in Benzinga headlines.
POS_PHRASES = {
    "beats": 2, "beat estimates": 2, "tops estimates": 2, "above estimates": 2, "raises guidance": 3,
    "raises outlook": 3, "raises forecast": 3, "boosts guidance": 3, "record revenue": 2, "record quarter": 2,
    "upgrade": 2, "upgraded": 2, "upgrades": 2, "outperform": 1, "overweight": 1, "buy rating": 1,
    "price target raised": 2, "raises price target": 2, "buyback": 1, "repurchase": 1, "dividend increase": 2,
    "raises dividend": 2, "approval": 2, "approved": 2, "fda approves": 3, "wins contract": 2, "awarded": 1,
    "partnership": 1, "surges": 1, "soars": 1, "jumps": 1, "rallies": 1, "strong demand": 2, "accelerat": 1,
    "expands": 1, "exceeds": 2, "better-than-expected": 2, "better than expected": 2, "positive": 1,
}
NEG_PHRASES = {
    "misses": 2, "missed estimates": 2, "below estimates": 2, "lowers guidance": 3, "cuts guidance": 3,
    "cuts outlook": 3, "lowers outlook": 3, "weak guidance": 3, "downgrade": 2, "downgraded": 2, "downgrades": 2,
    "underperform": 1, "underweight": 1, "sell rating": 1, "price target cut": 2, "lowers price target": 2,
    "cuts price target": 2, "lawsuit": 1, "sued": 1, "probe": 1, "investigation": 1, "recall": 2, "fraud": 3,
    "sec charges": 3, "layoffs": 1, "job cuts": 1, "plunges": 1, "tumbles": 1, "sinks": 1, "falls": 1,
    "slumps": 1, "worse-than-expected": 2, "worse than expected": 2, "weak demand": 2, "delay": 1, "halts": 1,
    "suspends": 2, "bankruptcy": 3, "default": 2, "warning": 1, "warns": 2, "negative": 1, "decline": 1,
    "fda rejects": 3, "complete response letter": 2, "data breach": 2, "outage": 1, "antitrust": 1,
}
NEGATORS = re.compile(r"\b(not|no|fails? to|without|despite)\b")


def lexicon_score(headline: str, summary: str = "") -> float:
    """Score in [-1, 1]; headline counts double. 0 = neutral / no signal words."""
    def raw(text: str) -> float:
        t = (text or "").lower()
        pos = sum(w for p, w in POS_PHRASES.items() if p in t)
        neg = sum(w for p, w in NEG_PHRASES.items() if p in t)
        if NEGATORS.search(t) and (pos or neg):
            pos, neg = neg * 0.5, pos * 0.5   # crude negation flip, damped
        return pos - neg
    s = 2 * raw(headline) + raw(summary)
    return float(np.tanh(s / 4))


def assign_sessions(news: pd.DataFrame, sessions: pd.DatetimeIndex, cutoff: str) -> pd.Series:
    """Map each article to the session whose decision it can inform.

    cutoff '09:15' (open variant) or '15:30' (close variant): an article stamped after the
    cutoff on day d belongs to the next session. Articles after 16:00 belong to the next session.
    """
    ts = pd.to_datetime(news["created_at"], utc=True).dt.tz_convert(NY)
    day = ts.dt.normalize().dt.tz_localize(None)
    hh = ts.dt.hour * 60 + ts.dt.minute
    ch, cm = map(int, cutoff.split(":"))
    after = hh > ch * 60 + cm
    cand = day + pd.to_timedelta(after.astype(int), unit="D")
    # next session on/after the candidate date (handles weekends / holidays)
    pos = sessions.searchsorted(cand.to_numpy())
    pos = np.clip(pos, 0, len(sessions) - 1)
    out = pd.Series(sessions[pos], index=news.index)
    # open variant: articles between 15:30 and 16:00 of d still precede d+1 open -> fine
    return out.where(pos < len(sessions))


def daily_scores(news: pd.DataFrame, sessions: pd.DatetimeIndex, universe: list[str], cutoff: str,
                 score_fn: Callable[[str, str], float] = lexicon_score, max_symbols: int = 3) -> pd.DataFrame:
    """session x symbol mean sentiment of focused articles (<= max_symbols tickers)."""
    df = news.copy()
    df["syms"] = df["symbols"].fillna("").str.split(",")
    df = df[df["syms"].map(len) <= max_symbols]
    if "score" not in df:
        df["score"] = [score_fn(h, s) for h, s in zip(df["headline"], df["summary"].fillna(""))]
    df["session"] = assign_sessions(df, sessions, cutoff)
    df = df.explode("syms")
    df = df[df["syms"].isin(universe) & (df["score"] != 0)]
    return df.pivot_table(index="session", columns="syms", values="score", aggfunc="mean").reindex(
        index=sessions, columns=universe)


def sentiment_returns(scores: pd.DataFrame, closes: pd.DataFrame, opens: pd.DataFrame, variant: str,
                      threshold: float = 0.3, hold: int = 1, cost_bps: float = 5.0, max_names: int = 10,
                      hedge: str = "SPY") -> pd.DataFrame:
    """Daily market-neutral returns. Each signal gets weight 1/max(n_active, max_names) per side."""
    sig = np.sign(scores.where(scores.abs() >= threshold))
    sig = sig.reindex(index=closes.index).fillna(0.0)
    c = cost_bps / 1e4
    if variant == "open":
        r = (closes / opens - 1)
        n = sig.abs().sum(axis=1)
        w = sig.div(np.maximum(n, max_names), axis=0)
        hedge_w = -w.sum(axis=1)                       # hedge only the NET exposure with SPY
        gross = (w * r.reindex(columns=w.columns)).sum(axis=1) + hedge_w * r[hedge]
        tc = (w.abs().sum(axis=1) + hedge_w.abs()) * 2 * c   # in at the open, out at the close
        return pd.DataFrame({"ret": gross - tc, "gross": gross, "names": n})
    # close variant: enter at close of signal day, hold `hold` sessions (overlapping books)
    r = closes.pct_change(fill_method=None)
    book = sum(sig.shift(k + 1) for k in range(hold)).fillna(0.0) / hold
    n = book.abs().sum(axis=1)
    w = book.div(np.maximum(n, max_names), axis=0)
    hedge_w = -w.sum(axis=1)
    gross = (w * r.reindex(columns=w.columns)).sum(axis=1) + hedge_w * r[hedge]
    turn = w.diff().abs().sum(axis=1).fillna(0.0) + hedge_w.diff().abs().fillna(0.0)
    tc = turn * c
    return pd.DataFrame({"ret": gross - tc, "gross": gross, "names": n})
