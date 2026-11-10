"""Strategy registry."""
from __future__ import annotations

from .allocation import GTAA, DualMomentum, TrendFollowing
from .base import DailyStrategy, rebalance_mask
from .intraday import LastHalfHourMomentum, NoiseAreaMomentum
from .mean_reversion import RSI2MeanReversion
from .momentum import MultiFactorMomentum

DAILY_STRATEGIES: dict[str, type[DailyStrategy]] = {
    "multifactor_momentum": MultiFactorMomentum,
    "dual_momentum": DualMomentum,
    "gtaa": GTAA,
    "trend_following": TrendFollowing,
    "rsi2": RSI2MeanReversion,
}

INTRADAY_STRATEGIES = {
    "noise_area": NoiseAreaMomentum,
    "last_half_hour": LastHalfHourMomentum,
}


def build_daily(name: str, **params) -> DailyStrategy:
    try:
        cls = DAILY_STRATEGIES[name]
    except KeyError as e:
        raise KeyError(f"unknown strategy {name!r}; options: {sorted(DAILY_STRATEGIES)}") from e
    return cls(**params)


__all__ = [
    "DailyStrategy", "rebalance_mask", "MultiFactorMomentum", "DualMomentum", "GTAA",
    "TrendFollowing", "RSI2MeanReversion", "NoiseAreaMomentum", "LastHalfHourMomentum",
    "DAILY_STRATEGIES", "INTRADAY_STRATEGIES", "build_daily",
]
