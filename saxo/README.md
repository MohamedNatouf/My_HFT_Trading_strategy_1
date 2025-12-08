Saxo trading system skeleton

- FIX via quickfix (configure .cfg per Saxo docs)
- WebSocket streaming for minute data
- Strategy engine capable of cross-sectional momentum on minute bars
- Backtester for minute rebalancing

Next steps:
1. Extract parameters from old_model to match weights, lookbacks, emergency logic.
2. Implement proper Saxo streaming subscription and schema.
3. Implement instrument mapping (UIC, asset types) and symbol normalization.
4. Risk management, order sizing, and reconciliation.
