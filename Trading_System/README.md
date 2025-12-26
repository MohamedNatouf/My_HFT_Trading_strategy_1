Saxo trading system skeleton (deprecated)

This project now uses Alpaca as the primary trading and market data provider.

- Trading via `alpaca-py` SDK
- Market data via Alpaca Market Data API (REST/WebSocket)
- Strategy engine capable of cross-sectional momentum on minute bars
- Backtester for minute rebalancing

Configuration: `config/saxo_config.json` includes Alpaca keys and backtest sources.
