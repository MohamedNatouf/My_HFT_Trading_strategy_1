"""quant - Alpaca-native research, backtesting and execution framework.

Layers (each importable on its own):
    quant.config      settings from JSON + environment (.env); secrets never live in JSON
    quant.data        historical bars (Alpaca / Yahoo fallback), parquet cache, RTH filtering
    quant.strategies  causal strategies: prices -> target weights (same code in backtest & live)
    quant.portfolio   sleeve combination, volatility targeting, position caps
    quant.backtest    next-bar execution backtester with costs, metrics, DSR / PBO, walk-forward
    quant.execution   Alpaca broker wrapper + reconciling, idempotent rebalancer
    quant.live        scheduled runners (daily close rebalance, intraday noise-area)
"""

__version__ = "0.1.0"
