import pandas as pd
from pathlib import Path

root = Path(r"C:\Users\DELL\source\repos\My_HFT_Trading_strategy_1")
store_dir = root / "data" / "minute"

syms = ["MUB", "RPG"]
frames = {}
for s in syms:
    df = pd.read_parquet(store_dir / f"{s}.parquet")
    df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    idx = df.index
    if getattr(idx, 'tz', None) is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    idx = df.index
    rth = (((idx.hour > 14) | ((idx.hour == 14) & (idx.minute >= 30))) & (idx.hour < 21) & (idx.dayofweek < 5))
    df = df.loc[rth]
    rs = df.resample('5min', closed='right', label='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
    frames[s] = rs['close']

close = pd.concat(frames, axis=1)
close_filled = close.ffill().bfill()

base_ts = pd.Timestamp('2026-01-15 14:40:00', tz='UTC')
ts = pd.Timestamp('2026-01-15 15:45:00', tz='UTC')
base_pos = close_filled.index.get_indexer([base_ts], method='pad')[0]
ts_pos = close_filled.index.get_indexer([ts], method='bfill')[0]
base_label = close_filled.index[base_pos]
ts_label = close_filled.index[ts_pos]
prev_px = close_filled.loc[base_label]
px = close_filled.loc[ts_label]
ret_vec = (px / prev_px) - 1
w = pd.Series({'MUB': 0.962641, 'RPG': 0.037359})
port_ret = float((w * ret_vec).sum())
print({'base_label': str(base_label), 'ts_label': str(ts_label), 'prev_px': {k: float(v) for k, v in prev_px.to_dict().items()}, 'px': {k: float(v) for k, v in px.to_dict().items()}, 'ret_vec': {k: float(v) for k, v in ret_vec.to_dict().items()}, 'port_ret': float(port_ret)})
