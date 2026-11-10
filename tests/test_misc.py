import numpy as np
import pandas as pd
import pytest

from quant.backtest import deflated_sharpe_ratio, probability_of_backtest_overfitting
from quant.config import LiveTradingNotConfirmed, _assert_no_secrets, credentials_from_env
from quant.strategies.intraday import LastHalfHourMomentum, NoiseAreaMomentum, noise_step


# --- overfitting diagnostics -------------------------------------------------------
def test_pbo_high_for_pure_noise():
    rng = np.random.default_rng(0)
    M = pd.DataFrame(rng.standard_normal((1000, 30)) * 0.01)
    assert probability_of_backtest_overfitting(M, n_splits=8)["pbo"] > 0.3


def test_pbo_low_for_real_signal():
    rng = np.random.default_rng(1)
    M = pd.DataFrame(rng.standard_normal((1000, 10)) * 0.01)
    M[0] += 0.003
    assert probability_of_backtest_overfitting(M, n_splits=8)["pbo"] < 0.1


def test_dsr_penalises_many_trials():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.standard_normal(1000) * 0.01 + 0.0008)
    one = deflated_sharpe_ratio(r, n_trials=1)["dsr"]
    many = deflated_sharpe_ratio(r, n_trials=1000)["dsr"]
    assert one > many


# --- config safety -------------------------------------------------------------------
def test_secret_in_config_rejected():
    with pytest.raises(ValueError):
        _assert_no_secrets({"alpaca": {"apiSecretKey": "abc"}})
    _assert_no_secrets({"alpaca": {"apiSecretKey": ""}})


def test_live_requires_confirmation(monkeypatch):
    monkeypatch.setenv("ALPACA_ENV", "live")
    monkeypatch.delenv("LIVE_TRADING_CONFIRM", raising=False)
    with pytest.raises(LiveTradingNotConfirmed):
        credentials_from_env()
    monkeypatch.setenv("LIVE_TRADING_CONFIRM", "yes")
    assert credentials_from_env().paper is False


# --- intraday ------------------------------------------------------------------------
def test_noise_step_state_machine():
    assert noise_step(0, 101, 100.5, 99.5, 100.0) == 1          # breakout up
    assert noise_step(1, 100.8, 100.5, 99.5, 100.9) == 0        # below VWAP stop -> exit
    assert noise_step(0, 99.0, 100.5, 99.5, 100.0) == -1        # breakdown
    assert noise_step(-1, 99.2, 100.5, 99.5, 99.4) == -1        # still below stop -> hold
    assert noise_step(0, 100.0, np.nan, 99.5, 100.0) == 0


def _synthetic_minutes(days=40, seed=0, trend_days=()):
    rng = np.random.default_rng(seed)
    out = []
    price = 400.0
    for d, day in enumerate(pd.bdate_range("2025-01-02", periods=days)):
        start = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
        drift = 0.0004 if d in trend_days else 0.0
        steps = drift + 0.0006 * rng.standard_normal(390)
        closes = price * np.exp(np.cumsum(steps))
        idx = pd.date_range(start, periods=390, freq="1min").tz_convert("UTC")
        opens = np.r_[price, closes[:-1]]
        out.append(pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) * 1.0002,
                                 "low": np.minimum(opens, closes) * 0.9998, "close": closes,
                                 "volume": 1000.0}, idx))
        price = closes[-1] * (1 + 0.002 * rng.standard_normal())
    return pd.concat(out)


def test_noise_area_backtest_runs_and_profits_on_trend_days():
    bars = _synthetic_minutes(days=60, seed=3, trend_days=set(range(20, 60)))
    d = NoiseAreaMomentum(max_leverage=1.0, cost_bps=0.0).backtest(bars)
    assert len(d) >= 40 and d["trades"].max() >= 1
    assert d["ret"].mean() > 0     # strong one-way trend days should be captured
    g = LastHalfHourMomentum().backtest(bars)
    assert len(g) > 30


def test_noise_bands_use_only_prior_days():
    from quant.strategies.intraday import minute_matrix, noise_bands

    bars = _synthetic_minutes(days=30, seed=4)
    ub1, _ = noise_bands(minute_matrix(bars))
    # change the LAST day's prices: bands for that day must not change (sigma from prior days)
    last_day = pd.DatetimeIndex(bars.index).tz_convert("America/New_York").normalize().max()
    mod = bars.copy()
    mask = pd.DatetimeIndex(mod.index).tz_convert("America/New_York").normalize() == last_day
    mod.loc[mask, "close"] *= 1.05
    ub2, _ = noise_bands(minute_matrix(mod))
    sigma1 = ub1.iloc[-1] / ub1.iloc[-1].iloc[0]
    sigma2 = ub2.iloc[-1] / ub2.iloc[-1].iloc[0]
    assert np.allclose(sigma1.to_numpy(), sigma2.to_numpy(), equal_nan=True)
