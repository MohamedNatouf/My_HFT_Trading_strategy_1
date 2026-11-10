"""Backtest-overfitting diagnostics.

* Deflated Sharpe Ratio - Bailey & Lopez de Prado (2014): probability that the true
  Sharpe is > 0 after accounting for the number of trials, skew, kurtosis, sample length.
* Probability of Backtest Overfitting via CSCV - Bailey, Borwein, Lopez de Prado, Zhu (2017).
* Walk-forward parameter selection (train on a rolling window, trade the next window).
"""
from __future__ import annotations

import itertools
import math
from typing import Callable, Iterable

import numpy as np
import pandas as pd

EULER_GAMMA = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_ppf(p: float) -> float:
    # Acklam's rational approximation (sufficient accuracy for DSR thresholds)
    if not 0 < p < 1:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum (per-period) Sharpe among n_trials skill-less strategies."""
    if n_trials <= 1:
        return 0.0
    return math.sqrt(sharpe_variance) * (
        (1 - EULER_GAMMA) * _norm_ppf(1 - 1 / n_trials)
        + EULER_GAMMA * _norm_ppf(1 - 1 / (n_trials * math.e))
    )


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int, trial_sharpes: Iterable[float] | None = None,
                          sharpe_variance: float | None = None) -> dict:
    """DSR for the selected strategy's per-period returns.

    Provide either the per-period Sharpes of all trials (preferred) or their variance.
    """
    r = pd.Series(returns).dropna()
    T = len(r)
    sr = float(r.mean() / r.std(ddof=1))  # per-period
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0  # raw kurtosis
    if sharpe_variance is None:
        if trial_sharpes is not None:
            ts = np.asarray(list(trial_sharpes), dtype=float)
            sharpe_variance = float(np.nanvar(ts, ddof=1)) if len(ts) > 1 else 0.0
        else:
            sharpe_variance = 1.0 / T  # variance of SR estimator under the null
    sr0 = expected_max_sharpe(n_trials, sharpe_variance)
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    z = (sr - sr0) * math.sqrt(T - 1) / denom
    return {"sharpe_per_period": sr, "sr0_per_period": sr0, "dsr": _norm_cdf(z), "n_trials": n_trials, "T": T}


def probability_of_backtest_overfitting(trial_returns: pd.DataFrame, n_splits: int = 10) -> dict:
    """CSCV PBO. ``trial_returns``: columns = strategy configurations, rows = periods."""
    M = trial_returns.dropna(how="all").fillna(0.0).to_numpy()
    T, N = M.shape
    if N < 2:
        raise ValueError("need >= 2 trials")
    n_splits -= n_splits % 2
    blocks = np.array_split(np.arange(T), n_splits)

    def sr(x):
        sd = x.std(axis=0, ddof=1)
        sd[sd == 0] = np.nan
        return x.mean(axis=0) / sd

    logits = []
    for is_ids in itertools.combinations(range(n_splits), n_splits // 2):
        is_rows = np.concatenate([blocks[i] for i in is_ids])
        oos_rows = np.concatenate([blocks[i] for i in range(n_splits) if i not in is_ids])
        is_sr, oos_sr = sr(M[is_rows]), sr(M[oos_rows])
        best = int(np.nanargmax(is_sr))
        rank = (pd.Series(oos_sr).rank(method="average").iloc[best])  # 1..N
        w = rank / (N + 1)
        logits.append(math.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_combinations": len(logits),
            "median_logit": float(np.median(logits))}


def walk_forward(
    prices: pd.DataFrame,
    make_targets: Callable[[pd.DataFrame, dict], pd.DataFrame],
    param_grid: list[dict],
    evaluate: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    train_days: int = 252 * 5,
    test_days: int = 252,
) -> dict:
    """Rolling walk-forward. ``evaluate(prices, targets) -> daily returns``.

    Targets are always computed on the full price history up to the end of each test
    window (strategies are causal), and parameters are chosen on the train window only.
    """
    from .metrics import sharpe as _sharpe

    idx = prices.index
    oos = []
    chosen = []
    start = train_days
    while start < len(idx):
        end = min(start + test_days, len(idx))
        hist = prices.iloc[:end]
        best, best_sr = None, -np.inf
        for params in param_grid:
            r = evaluate(hist, make_targets(hist, params))
            s = _sharpe(r.loc[idx[start - train_days]:idx[start - 1]])
            if np.isfinite(s) and s > best_sr:
                best, best_sr = params, s
        if best is None:
            best = param_grid[0]
        r = evaluate(hist, make_targets(hist, best))
        oos.append(r.loc[idx[start]:idx[end - 1]])
        chosen.append({"test_start": str(idx[start].date()), "params": best, "train_sharpe": best_sr})
        start = end
    return {"oos_returns": pd.concat(oos) if oos else pd.Series(dtype=float), "selections": chosen}
