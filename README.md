# My_HFT_Trading_strategy_1

Two systems live in this repo:

| Package | Status | What it is |
|---|---|---|
| `quant/` | **new, recommended** | Alpaca-native framework: causal strategies → risk → reconciling, idempotent execution. Daily multi-strategy core + optional SPY intraday sleeve. |
| `Trading_System/` | legacy, bug-fixed | Original minute-bar cross-sectional ETF momentum. Kept for reference; its edge does not survive costs (see "Research results"). |

## Setup

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env      # then fill in your Alpaca keys
```

Secrets live **only** in `.env` (git-ignored). `quant` refuses to load a JSON config that contains keys.
Paper vs live is chosen only by `ALPACA_ENV`; live additionally requires `LIVE_TRADING_CONFIRM=yes`.

> The keys that used to be committed in `config/*.json` must be **rotated** in the Alpaca dashboard
> and purged from git history (`git filter-repo`), because they remain in old commits.

## quant/ layout

```
quant/config.py          settings (.env secrets, JSON params, live-trading guard)
quant/data/              Alpaca SIP history (free plan: data >15 min old), Yahoo daily fallback, parquet cache, RTH filter
quant/strategies/        multifactor_momentum (old_model rescued onto daily bars), dual_momentum, gtaa,
                         trend_following, rsi2, intraday noise_area (Zarattini 2024), last_half_hour (Gao 2018)
quant/portfolio/         sleeve combination, position caps, causal volatility targeting
quant/backtest/          next-bar execution with costs, metrics, Deflated Sharpe, PBO (CSCV), walk-forward
quant/execution/         AlpacaBroker (Trading API only) + Rebalancer (broker = source of truth,
                         deterministic client_order_id, sells first, drift bands, MOC/limit/market)
quant/live/              daily close rebalancer (kill switch, HALT file, trade_updates journal), intraday noise-area runner
tests/                   causality of every strategy, no-look-ahead backtester, rebalancer idempotency, DSR/PBO, config safety
```

Contract: a daily strategy is a pure function `prices -> target weights` where row t uses data up to t.
The same code runs in backtests and live; `tests/test_causality.py` enforces it.

## Commands

```powershell
python -m pytest tests -q                                   # 31 tests

# research (Yahoo needs no keys; Alpaca used automatically when .env has keys)
python scripts/research.py daily --source yahoo --start 2003-01-01
python scripts/research.py overfit --source yahoo           # PBO / DSR / walk-forward
python scripts/research.py intraday --symbol SPY --start 2018-01-01 --cost-bps 1   # Alpaca minute data

# trading (dry run unless --execute)
python scripts/run_quant.py check                           # read-only account/clock/data check
python scripts/run_quant.py daily --once                    # today's targets + planned orders (dry run)
python scripts/run_quant.py daily --execute                 # scheduler: rebalance 20 min before close (MOC)
python scripts/run_quant.py intraday --execute              # SPY noise-area intraday sleeve
python scripts/run_quant.py flatten --execute               # cancel orders + close all positions
```

Portfolio, risk and execution parameters: `config/quant.json`. State and an order/fill journal are written to `state/`
(`state/HALT` blocks all trading after a kill-switch trip until you delete it).

## Research results (2026-11-10, net of costs, next-close execution)

Daily, Yahoo adjusted closes, 2004-02 → 2026-11, 5 bp per unit turnover:

| | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|
| multifactor_momentum (old_model on daily bars) | 7.5% | 13.2% | 0.61 | −21% |
| dual_momentum | 8.5% | 15.5% | 0.61 | −34% |
| gtaa | 5.2% | 7.6% | 0.71 | −12% |
| trend_following | 5.3% | 10.0% | 0.56 | −27% |
| rsi2 (next close / MOC) | 2.9% / 3.4% | 6.7% | 0.46 / 0.55 | −14% |
| **Portfolio (4 sleeves)** | 6.3% | 8.9% | 0.73 | **−19%** |
| SPY buy & hold | 10.6% | 18.6% | 0.64 | −55% |
| 60/40 SPY/AGG | 7.8% | 11.2% | 0.73 | −36% |

The core portfolio matches 60/40 on Sharpe with about half the drawdown; it is a risk-control product, not an alpha machine.
Momentum grid (48 configs): PBO 31%, walk-forward OOS Sharpe 0.46 vs 0.60 median config → use fixed defaults, don't re-optimise.

Intraday SPY, Alpaca SIP minute bars 2018 → 2026-11, 1 bp cost:
* noise_area: Sharpe 0.98 (1x) / 1.12 (4x) before June 2024 but **−0.54 / −0.41 since publication**, and negative over the full sample at 3 bp. **Paper-trade only; do not run live.**
* last_half_hour (Gao): Sharpe −1.06. Dropped.

Caveats: ETF universe chosen with hindsight (mild survivorship bias); Yahoo data; lag-0 MOC results assume a 15:40 snapshot ≈ the close.

## Research process (do not skip)

hypothesis → backtest with costs and lag=1 → walk-forward + DSR/PBO on every grid you try →
1–3 months of paper trading compared against the backtest → small live size.

## Legacy Trading_System CLI

```powershell
python scripts\live.py --config config\config.json --run-id live_01 --log-level INFO --flatten-on-exit
python scripts\backtest.py --config config\config.json --start 2025-12-01 --end 2026-01-15 --log-level INFO
python scripts\sweep_backtest.py --config config\config.json --output sweeps.csv --max 50
python scripts\alpaca_check.py --config config\config.json
```

Legacy fixes applied: execution sizing from real prices and live broker positions (no more repeated buying every poll),
working market-hours guard (Alpaca clock), "Metrics Count" picking the best rather than the worst assets, look-ahead
removed from the backtester and from `bfill`, scale-free high/low factor, explicit data feed/adjustment, RTH filter,
TradingStream instead of Broker-API SSE, latest-only live computation off the event loop, `flatten_and_exit()` and
`MinuteStore.clear_all()` implemented, deterministic `client_order_id`s.
