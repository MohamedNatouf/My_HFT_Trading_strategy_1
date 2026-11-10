# Legacy configs (reference only — not used by any code)

`grid.json` and `config_best.json` come from a lost later version of `Trading_System/`.
They reference features that **the current legacy engine does not implement**, so loading
them changes nothing (or silently ignores keys):

| Key | Status |
|---|---|
| `weighting.method` = erc / hrp / robust_mv / max_diversification / risk_parity_buckets / Efficient Frontier | not implemented (engine only has inverse-vol and volume weighting) |
| `weighting.cov_estimator` (ewma / ogk / ledoit_wolf), `ewma_lambda`, `l1/l2_penalty`, `turnover_penalty`, `robust_mean`, `expected_gamma_*`, `buckets` | not implemented |
| `costs.*` beyond `c2` | only `costs.c2` (per-rebalance turnover cost) is used by the legacy Backtester |
| stop-loss settings | not implemented |

The "best" config was selected from ~4.6 trading days of minute bars (Sharpe 13.7 with a −35% drawdown)
and is an overfitting artefact. API keys were removed from it.

For new work use `config/quant.json` and the `quant/` package: risk caps, group caps, turnover budget,
volatility targeting and a per-symbol cost model live in `quant/portfolio` and `quant/backtest/costs.py`.
