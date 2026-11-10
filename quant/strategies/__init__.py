"""Strategy registry."""
from __future__ import annotations

from .allocation import GTAA, DualMomentum, TrendFollowing
from .base import DailyStrategy, rebalance_mask
from .intraday import LastHalfHourMomentum, NoiseAreaMomentum
from .mean_reversion import RSI2MeanReversion
from .momentum import MultiFactorMomentum
from .reversal import SectorReversal, StockReversal

DAILY_STRATEGIES: dict[str, type[DailyStrategy]] = {
    "multifactor_momentum": MultiFactorMomentum,
    "dual_momentum": DualMomentum,
    "gtaa": GTAA,
    "trend_following": TrendFollowing,
    "rsi2": RSI2MeanReversion,
}

# Research candidates: causality-tested, but not part of the default portfolio research run
# until they pass their go/no-go test (see scripts/research.py reversal|news|orb|putspread).
CANDIDATE_STRATEGIES: dict[str, type[DailyStrategy]] = {
    "sector_reversal": SectorReversal,
    "stock_reversal": StockReversal,
}

INTRADAY_STRATEGIES = {
    "noise_area": NoiseAreaMomentum,
    "last_half_hour": LastHalfHourMomentum,
}


def build_daily(name: str, **params) -> DailyStrategy:
    registry = {**DAILY_STRATEGIES, **CANDIDATE_STRATEGIES}
    try:
        cls = registry[name]
    except KeyError as e:
        raise KeyError(f"unknown strategy {name!r}; options: {sorted(registry)}") from e
    return cls(**params)


__all__ = [
    "DailyStrategy", "rebalance_mask", "MultiFactorMomentum", "DualMomentum", "GTAA",
    "TrendFollowing", "RSI2MeanReversion", "NoiseAreaMomentum", "LastHalfHourMomentum",
    "SectorReversal", "StockReversal",
    "DAILY_STRATEGIES", "CANDIDATE_STRATEGIES", "INTRADAY_STRATEGIES", "build_daily",
]
