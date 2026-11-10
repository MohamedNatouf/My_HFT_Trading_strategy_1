# End-to-end system wiring: config -> engine -> Alpaca data/trading
import asyncio
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from .config_loader import ConfigLoader
from .instrument_map import InstrumentMap
from .engine import Strategy, Backtester, Allocation, _extract_close_frame
from .data import MinuteDataLoader, BacktestConfig
from .alpaca_data import AlpacaMinuteDataLoader
from .alpaca_trading import AlpacaTrader
from .alpaca_account import AlpacaAccountClient
from .execution import ExecutionRouter
from .minute_store import MinuteStore, filter_rth

try:
    from alpaca.trading.stream import TradingStream
except Exception:
    TradingStream = None

logger = logging.getLogger(__name__)


def _merge_bars(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge two (field, symbol) MultiIndex bar frames by timestamp; ``new`` wins on overlap.
    Never produces duplicate index rows."""
    if old is None or old.empty:
        return new.sort_index() if new is not None else pd.DataFrame()
    if new is None or new.empty:
        return old
    out = new.combine_first(old)
    out = out[~out.index.duplicated(keep='last')]
    return out.sort_index()

class TradingSystem:
    def __init__(self, config_path: str = "config/config.json"):
        loader = ConfigLoader(config_path)
        self.config = loader.load()
        # Ensure instruments populated from universe if missing
        if not self.config.get('instruments'):
            models = (self.config.get('universe', {}) or {}).get('models', [])
            self.config['instruments'] = [{ 'symbol': s, 'assetType': 'equity' } for s in models]
        self.instrument_map = InstrumentMap(self.config['instruments'])
        self.minute_df = pd.DataFrame()
        # Local minute store
        store_root = (self.config.get('data', {}) or {}).get('storeRoot', 'data/minute')
        self.store = MinuteStore(store_root)
        # Merge emergency and universe tickers into strategy params
        strat_params = dict(self.config['strategy'])
        strat_params['emergency'] = self.config.get('emergency', {})
        strat_params['universe_models'] = (self.config.get('universe', {}) or {}).get('models', [])
        self.strategy = Strategy("CS-Momentum-Minute", strat_params)
        alpaca_cfg = self.config.get('alpaca', {})
        # Regular-trading-hours filter for all bar sources (config data.marketHoursOnly, default True)
        self.rth_only = bool((self.config.get('data', {}) or {}).get('marketHoursOnly', True))
        bt_cfg0 = self.config.get('backtest', {}) or {}
        self.alpaca_data = AlpacaMinuteDataLoader(
            alpaca_cfg.get('apiKeyId'), alpaca_cfg.get('apiSecretKey'), alpaca_cfg.get('paper', True),
            feed=bt_cfg0.get('feed'), adjustment=bt_cfg0.get('adjustment'), rth_only=self.rth_only,
        )
        self.alpaca_trader = AlpacaTrader(alpaca_cfg.get('apiKeyId'), alpaca_cfg.get('apiSecretKey'), alpaca_cfg.get('paper', True))
        # NOTE: alpaca_sse.AlpacaSSEClient targets the Broker API and is NOT used;
        # order/fill events come from alpaca-py TradingStream (see _run_trade_updates).
        self._trading_stream = None
        # Live-mode bookkeeping: never re-execute the same allocation timestamp.
        self._last_executed_alloc_ts: Optional[pd.Timestamp] = None
        self._last_computed_bar_ts: Optional[pd.Timestamp] = None
        self._last_cycle_empty = False
        self.alpaca_account = AlpacaAccountClient(alpaca_cfg.get('apiKeyId'), alpaca_cfg.get('apiSecretKey'), alpaca_cfg.get('paper', True))
        # Execution router for translating allocations to orders
        self.execution = ExecutionRouter(self.alpaca_trader, self.alpaca_account, self.config.get('live', {}))
        logger.info("TradingSystem initialized mode=%s source=%s", self.config.get('mode'), self.config.get('backtest',{}).get('source'))

    def _universe_instruments(self) -> List[Dict[str, Any]]:
        """Return instruments to load for data. Include trading universe plus emergency signal/active symbols so
        emergency logic and emergency allocations can work identically in backtest and live.
        Exclude non-Alpaca symbols (e.g., caret-prefixed indices like ^IRX)."""
        models_syms = set(self.config.get('universe', {}).get('models', []))
        emergency_cfg = self.config.get('emergency', {}) or {}
        extra_syms = set(sum([
            emergency_cfg.get('equity_signal', []),
            emergency_cfg.get('bond_signal', []),
            emergency_cfg.get('active', []),
            emergency_cfg.get('backup', []),
            # DO NOT include risk_free_reference here; symbols like ^IRX are invalid for Alpaca bars
        ], []))
        combined_syms = {s for s in (models_syms | extra_syms) if s and not str(s).startswith('^')}
        # Build instruments list from existing instruments merged with any missing emergency symbols
        existing = self.config.get('instruments', []) or []
        existing_syms = {i.get('symbol') for i in existing}
        merged: List[Dict[str, Any]] = []
        # include all existing entries that are in the combined set and not caret-prefixed
        for i in existing:
            sym = i.get('symbol')
            if sym in combined_syms and not str(sym).startswith('^'):
                merged.append(i)
        # add any missing symbols as equities by default
        for s in sorted(combined_syms - existing_syms):
            if str(s).startswith('^'):
                continue
            merged.append({'symbol': s, 'assetType': 'equity'})
        logger.info("Universe resolved count=%d symbols=%s", len(merged), [u['symbol'] for u in merged])
        return merged

    def _map_interval(self, src: str, interval: str) -> str:
        s = (src or '').lower()
        i = (interval or '').lower()
        if s in ('yahoo', 'alpha_vantage', 'alphavantage', 'alpha_vantage_rapidapi', 'alphavantage_rapidapi', 'rapidapi'):
            # yfinance and Alpha Vantage expect 1m|5m|15m|30m|60m
            return {'1min':'1m','5min':'5m','15min':'15m','30min':'30m','60min':'60m'}.get(i, i if i.endswith('m') else '1m')
        # Alpaca expects 1Min, 5Min, 15Min
        return {'1min':'1Min','5min':'5Min','15min':'15Min'}.get(i, interval)

    def _apply_store_update(self, df: pd.DataFrame):
        """Upsert fetched minute bars to local store."""
        if df is None or df.empty or not isinstance(df.columns, pd.MultiIndex):
            return
        try:
            symbols = df.columns.get_level_values(1).unique()
        except Exception:
            return
        for sym in symbols:
            try:
                sub = df.xs(sym, axis=1, level=1)
                # Normalize to lowercase columns
                sub = sub.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
                self.store.upsert_symbol(sym, sub)
            except Exception as e:
                logger.debug("Store upsert failed for %s: %s", sym, e)

    async def run(self) -> List[Allocation]:
        mode = self.config.get('mode', 'live').lower()
        logger.info("Run called mode=%s", mode)
        if mode == 'backtest':
            return await self.run_backtest()
        else:
            await self.run_stream()
            return []

    async def run_stream(self):
        logger.info("Starting live stream (Alpaca primary)")
        alpaca_cfg = self.config.get('alpaca', {})
        use_alpaca = bool(alpaca_cfg.get('apiKeyId') and alpaca_cfg.get('apiSecretKey'))
        if not use_alpaca:
            logger.error("Alpaca credentials missing. Provide 'alpaca.apiKeyId' and 'alpaca.apiSecretKey' in config.")
            return
        # Order/fill events via the Trading API websocket (TradingStream).
        trade_updates_task = self._start_trade_updates()
        acct_task = asyncio.create_task(self._monitor_account())
        stream_cfg = (self.config.get('live', {}) or {}).get('stream', {})
        mode = stream_cfg.get('mode', 'poll').lower()
        try:
            if mode == 'websocket':
                await self._run_stream_alpaca_ws()
            else:
                poll_secs = int(stream_cfg.get('pollSeconds', 10))
                await self._run_stream_alpaca_poll(poll_secs)
        finally:
            acct_task.cancel()
            await self._stop_trade_updates(trade_updates_task)

    # ------------------------------------------------------------------ trade updates
    def _start_trade_updates(self) -> Optional[asyncio.Task]:
        """Start alpaca-py TradingStream (trade_updates) in a worker thread."""
        if TradingStream is None:
            logger.warning("alpaca-py TradingStream unavailable; order/fill events will not be streamed")
            return None
        alpaca_cfg = self.config.get('alpaca', {})
        try:
            stream = TradingStream(alpaca_cfg.get('apiKeyId'), alpaca_cfg.get('apiSecretKey'), paper=bool(alpaca_cfg.get('paper', True)))
        except Exception as e:
            logger.error("Failed to create TradingStream: %s", e)
            return None

        async def _on_trade_update(data):
            try:
                event = str(getattr(data, 'event', '') or '')
                event = getattr(getattr(data, 'event', None), 'value', event)
                order = getattr(data, 'order', None)
                sym = getattr(order, 'symbol', None)
                side = getattr(getattr(order, 'side', None), 'value', getattr(order, 'side', None))
                coid = getattr(order, 'client_order_id', None)
                msg = ("TRADE_UPDATE event=%s symbol=%s side=%s qty=%s filled_qty=%s filled_avg_price=%s "
                       "price=%s position_qty=%s client_order_id=%s")
                args = (event, sym, side, getattr(order, 'qty', None), getattr(order, 'filled_qty', None),
                        getattr(order, 'filled_avg_price', None), getattr(data, 'price', None),
                        getattr(data, 'position_qty', None), coid)
                if event in ('rejected', 'canceled', 'expired'):
                    logger.warning(msg, *args)
                elif event in ('fill', 'partial_fill'):
                    logger.info(msg, *args)
                else:
                    logger.debug(msg, *args)
            except Exception as e:
                logger.debug("trade update logging failed: %s", e)

        stream.subscribe_trade_updates(_on_trade_update)
        self._trading_stream = stream
        logger.info("TradingStream trade_updates started (paper=%s)", bool(alpaca_cfg.get('paper', True)))
        return asyncio.create_task(asyncio.to_thread(stream.run))

    async def _stop_trade_updates(self, task: Optional[asyncio.Task]) -> None:
        stream = self._trading_stream
        if stream is not None:
            try:
                stream.stop()
            except Exception as e:
                logger.debug("TradingStream.stop failed: %s", e)
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
            except Exception:
                pass
        self._trading_stream = None

    async def _monitor_account(self):
        """Periodically read account fields and log settlement-aware state."""
        interval = int((self.config.get('live', {}) or {}).get('accountPollSeconds', 30))
        try:
            while True:
                acc = await asyncio.to_thread(self.alpaca_account.get_account)
                if acc:
                    logger.info(
                        "Account bp=%s regt_bp=%s dtbp=%s cash=%s cash_wdr=%s memoposts=%s eq=%s",
                        getattr(acc, 'buying_power', None),
                        getattr(acc, 'regt_buying_power', None),
                        getattr(acc, 'daytrading_buying_power', None),
                        getattr(acc, 'cash', None),
                        getattr(acc, 'cash_withdrawable', None),
                        getattr(acc, 'memoposts', None),
                        getattr(acc, 'equity', None),
                    )
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Account monitor cancelled")

    # ------------------------------------------------------------------ live helpers
    def _allowed_symbols(self) -> set:
        models_syms = set(self.config.get('universe', {}).get('models', []))
        emergency_cfg = self.config.get('emergency', {}) or {}
        signals_syms = set(sum([
            emergency_cfg.get('equity_signal', []),
            emergency_cfg.get('bond_signal', []),
            emergency_cfg.get('active', []),
        ], []))
        return models_syms | signals_syms

    def _live_feed(self) -> str:
        return str(((self.config.get('live', {}) or {}).get('stream', {}) or {}).get('feed', 'iex')).lower()

    def _live_adjustment(self) -> Optional[str]:
        stream_cfg = (self.config.get('live', {}) or {}).get('stream', {}) or {}
        return stream_cfg.get('adjustment') or (self.config.get('backtest', {}) or {}).get('adjustment')

    def _fetch_live_positions(self) -> Optional[Dict[str, float]]:
        """Return broker positions {symbol: signed qty}, or None if they cannot be fetched
        (callers must then NOT trade)."""
        client = getattr(self.alpaca_trader, 'client', None)
        if client is None:
            logger.error("Trading client unavailable; cannot fetch live positions")
            return None
        try:
            out: Dict[str, float] = {}
            for p in client.get_all_positions() or []:
                sym = getattr(p, 'symbol', None)
                if not sym:
                    continue
                qty = float(getattr(p, 'qty', 0) or 0)
                side = str(getattr(getattr(p, 'side', None), 'value', getattr(p, 'side', '')) or '').lower()
                if side == 'short' and qty > 0:
                    qty = -qty
                out[sym] = qty
            return out
        except Exception as e:
            logger.error("get_all_positions failed: %s", e)
            return None

    def _trim_live_df(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = max(2 * self.strategy.required_window(), 2000)
        return df.iloc[-keep:] if len(df) > keep else df

    async def _maybe_rebalance_live(self, minute_df: pd.DataFrame, source: str) -> None:
        """Compute the latest allocation (off the event loop) and execute it ONLY if it
        is a new allocation timestamp, at least rebalance_minutes after the last one."""
        if minute_df is None or minute_df.empty:
            return
        last_ts = minute_df.index[-1]
        if self._last_computed_bar_ts is not None and last_ts <= self._last_computed_bar_ts:
            return  # no new bar since last evaluation
        step = int(self.strategy.params.get('rebalance_minutes', self.strategy.rebalance_minutes))
        if self._last_executed_alloc_ts is not None and (last_ts - self._last_executed_alloc_ts) < pd.Timedelta(minutes=step):
            return
        self._last_computed_bar_ts = last_ts
        max_age = float((self.config.get('live', {}) or {}).get('maxBarAgeMinutes', 5))
        now = pd.Timestamp.now(tz='UTC')
        ts_utc = last_ts if last_ts.tzinfo is not None else last_ts.tz_localize('UTC')
        if (now - ts_utc) > pd.Timedelta(minutes=max_age):
            logger.info("%s: latest bar %s is older than %.0f min; not trading on stale data", source, last_ts, max_age)
            return
        alloc = await asyncio.to_thread(self.strategy.latest_allocation, minute_df)
        if alloc is None:
            return
        if self._last_executed_alloc_ts is not None and alloc.date <= self._last_executed_alloc_ts:
            logger.debug("%s: allocation ts=%s already executed; skipping", source, alloc.date)
            return
        positions = await asyncio.to_thread(self._fetch_live_positions)
        if positions is None:
            logger.warning("%s: live positions unavailable; skipping this rebalance", source)
            return
        logger.info("%s: executing NEW allocation ts=%s live_positions=%s", source, alloc.date, positions)
        # Mark before executing: an allocation is never re-submitted, even if execution partially fails.
        self._last_executed_alloc_ts = alloc.date
        await asyncio.to_thread(self.execution.rebalance, alloc, positions)

    async def _run_stream_alpaca_poll(self, poll_secs: int):
        logger.info("Using Alpaca polling for live data feed=%s rth_only=%s", self._live_feed(), self.rth_only)
        universe = self._universe_instruments()
        allowed = self._allowed_symbols()
        warmup_days = int(((self.config.get('live', {}) or {}).get('stream', {}) or {}).get('warmupDays', 5))
        try:
            while True:
                try:
                    if self.minute_df is None or self.minute_df.empty:
                        start = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=warmup_days)
                    else:
                        start = self.minute_df.index[-1] - pd.Timedelta(minutes=5)
                    df = await asyncio.to_thread(
                        self.alpaca_data.load_minute_bars, universe, start, None,
                        self._live_feed(), self._live_adjustment(), self.rth_only)
                    if df is None or df.empty:
                        # Do NOT fall back to the (possibly stale) local store for live trading.
                        if not self._last_cycle_empty:
                            logger.warning("Live data fetch returned no bars (feed=%s); skipping cycle, not trading on stale store data", self._live_feed())
                        self._last_cycle_empty = True
                    else:
                        self._last_cycle_empty = False
                        await asyncio.to_thread(self._apply_store_update, df)
                        df = df.loc[:, [c for c in df.columns if c[1] in allowed]]
                        self.minute_df = self._trim_live_df(_merge_bars(self.minute_df, df))
                        await self._maybe_rebalance_live(self.minute_df, "poll")
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error("Alpaca poll cycle failed: %s", e, exc_info=True)
                await asyncio.sleep(poll_secs)
        except asyncio.CancelledError:
            logger.info("Alpaca polling cancelled")

    async def _run_stream_alpaca_ws(self):
        """WebSocket streaming via alpaca-py StockDataStream. The stream runs in a worker
        thread; bars are handed to the main loop through an asyncio.Queue."""
        try:
            from alpaca.data.live import StockDataStream
        except Exception as e:
            logger.error("alpaca-py live streaming unavailable: %s", e)
            return
        try:
            from alpaca.data.enums import DataFeed
            feed = DataFeed(self._live_feed())
        except Exception:
            feed = self._live_feed()
        alpaca_cfg = self.config.get('alpaca', {})
        allowed = sorted(self._allowed_symbols())
        universe = self._universe_instruments()
        # Warm up history so the strategy has enough bars immediately.
        if self.minute_df is None or self.minute_df.empty:
            warmup_days = int(((self.config.get('live', {}) or {}).get('stream', {}) or {}).get('warmupDays', 5))
            start = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=warmup_days)
            seed = await asyncio.to_thread(self.alpaca_data.load_minute_bars, universe, start, None,
                                           self._live_feed(), self._live_adjustment(), self.rth_only)
            if seed is not None and not seed.empty:
                self.minute_df = self._trim_live_df(seed.loc[:, [c for c in seed.columns if c[1] in allowed]].sort_index())
            else:
                logger.warning("WebSocket warm-up returned no bars; strategy will wait for enough live bars")
        loop = asyncio.get_running_loop()
        backoff = 1
        while True:
            stream = None
            stream_task = None
            try:
                stream = StockDataStream(alpaca_cfg.get('apiKeyId'), alpaca_cfg.get('apiSecretKey'), feed=feed)
                queue: asyncio.Queue = asyncio.Queue()

                async def on_bar(bar):
                    loop.call_soon_threadsafe(queue.put_nowait, bar)

                stream.subscribe_bars(on_bar, *allowed)
                stream_task = asyncio.create_task(asyncio.to_thread(stream.run))
                while True:
                    get_task = asyncio.create_task(queue.get())
                    done, _ = await asyncio.wait({get_task, stream_task}, return_when=asyncio.FIRST_COMPLETED)
                    if get_task not in done:
                        get_task.cancel()
                        raise RuntimeError("StockDataStream stopped")
                    bar = get_task.result()
                    backoff = 1
                    ts = pd.to_datetime(bar.timestamp, utc=True).floor('min')
                    sym = bar.symbol
                    if sym not in allowed:
                        continue
                    if self.rth_only and filter_rth(pd.DataFrame(index=pd.DatetimeIndex([ts]), data={'x': [0]})).empty:
                        continue
                    row = pd.DataFrame({
                        ('open', sym): [bar.open],
                        ('high', sym): [bar.high],
                        ('low', sym): [bar.low],
                        ('close', sym): [bar.close],
                        ('volume', sym): [bar.volume]
                    }, index=pd.DatetimeIndex([ts]))
                    row.columns = pd.MultiIndex.from_tuples(row.columns)
                    prev_last = self.minute_df.index[-1] if (self.minute_df is not None and not self.minute_df.empty) else None
                    self.minute_df = self._trim_live_df(_merge_bars(self.minute_df, row))
                    try:
                        await asyncio.to_thread(self.store.upsert_symbol, sym, row.xs(sym, axis=1, level=1))
                    except Exception:
                        pass
                    # When the first bar of a NEW minute arrives, the previous minute is
                    # complete for all symbols: evaluate on bars strictly before ts.
                    if prev_last is not None and ts > prev_last:
                        complete = self.minute_df[self.minute_df.index < ts]
                        await self._maybe_rebalance_live(complete, "ws")
            except asyncio.CancelledError:
                logger.info("Alpaca websocket cancelled")
                if stream is not None:
                    stream.stop()
                return
            except Exception as e:
                logger.warning("WebSocket error: %s; reconnecting in %ss", e, backoff)
                if stream is not None:
                    try:
                        stream.stop()
                    except Exception:
                        pass
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def flatten_and_exit(self) -> List[Any]:
        """Cancel all open orders and close all positions at the broker; log per-symbol results."""
        client = getattr(self.alpaca_trader, 'client', None)
        if client is None:
            logger.error("flatten_and_exit: trading client unavailable; nothing done")
            return []
        try:
            cancelled = await asyncio.to_thread(client.cancel_orders)
            logger.info("flatten_and_exit: cancel_orders responses=%d", len(cancelled or []))
        except Exception as e:
            logger.error("flatten_and_exit: cancel_orders failed: %s", e)
        try:
            results = await asyncio.to_thread(client.close_all_positions, True)
        except Exception as e:
            logger.error("flatten_and_exit: close_all_positions failed: %s", e)
            return []
        for r in results or []:
            logger.info("flatten_and_exit: symbol=%s status=%s order_id=%s body=%s",
                        getattr(r, 'symbol', None), getattr(r, 'status', None),
                        getattr(r, 'order_id', None), getattr(r, 'body', None))
        return list(results or [])

    def _compute_report(self, allocs: List[Allocation]) -> Dict[str, Any]:
        if self.minute_df is None or self.minute_df.empty or not allocs:
            return {}
        # Close frame extraction (data uses lowercase 'close'); ffill only, no bfill.
        close = _extract_close_frame(self.minute_df).ffill()
        # Build weight time series from allocations (step-wise hold until next rebalance)
        w_ts = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        alloc_times: List[pd.Timestamp] = []
        for a in allocs:
            ts = pd.to_datetime(a.date)
            idx_pos = w_ts.index.get_indexer([ts], method='pad')
            if len(idx_pos) == 0 or idx_pos[0] == -1:
                continue
            ts = w_ts.index[idx_pos[0]]
            alloc_times.append(ts)
            for p in a.positions:
                if p.symbol in w_ts.columns:
                    w_ts.loc[ts:, p.symbol] = float(p.weight)
        rets = close.pct_change(fill_method=None).fillna(0.0)
        # Minute portfolio returns from the very start of the backtest window
        port_rets_min_full = (w_ts.shift(1).fillna(0.0) * rets).sum(axis=1)
        # Also compute per-rebalance next-bar returns
        per_reb_rets: List[float] = []
        per_reb_rows: List[Tuple[pd.Timestamp, float]] = []
        for ts in alloc_times:
            idx = w_ts.index.get_indexer([ts], method='pad')[0]
            if idx+1 >= len(w_ts.index):
                continue
            next_ts = w_ts.index[idx+1]
            r_next = rets.loc[next_ts]
            w = w_ts.loc[ts]
            per_ret = float((w * r_next).sum())
            per_reb_rets.append(per_ret)
            per_reb_rows.append((next_ts, per_ret))
        # Metrics based on minute series for the entire backtest
        minutes_per_year = 252 * self.strategy.minutes_per_day
        avg = float(port_rets_min_full.mean())
        std = float(port_rets_min_full.std())
        sharpe = (avg/std) * np.sqrt(minutes_per_year) if std > 0 else np.nan
        neg = port_rets_min_full[port_rets_min_full < 0]
        downside_std = float(neg.std())
        sortino = (avg/downside_std) * np.sqrt(minutes_per_year) if downside_std > 0 else np.nan
        equity_curve_full = (100.0 * (1.0 + port_rets_min_full).cumprod())
        dd_series = equity_curve_full / equity_curve_full.cummax() - 1.0
        max_dd = float(dd_series.min())
        # Turnover: sum of absolute weight changes at rebalance times
        turnover = 0.0
        prev_w = None
        trades: List[Tuple[pd.Timestamp, str, float, float]] = []
        for ts in alloc_times:
            w = w_ts.loc[ts]
            if prev_w is not None:
                dw = (w - prev_w).abs().sum()
                turnover += float(dw)
                idx = w_ts.index.get_indexer([ts], method='pad')[0]
                if idx+1 < len(w_ts.index):
                    next_ts = w_ts.index[idx+1]
                    r_next = rets.loc[next_ts]
                    for s in close.columns:
                        dwi = float(w.get(s, 0.0) - prev_w.get(s, 0.0))
                        if dwi != 0:
                            pnl = dwi * float(r_next.get(s, 0.0)) * 100.0
                            trades.append((ts, s, dwi, pnl))
            prev_w = w
        wins = sum(1 for (_, _, _, pnl) in trades if pnl > 0)
        losses = sum(1 for (_, _, _, pnl) in trades if pnl <= 0)
        avg_win = float(np.mean([pnl for (_, _, _, pnl) in trades if pnl > 0])) if wins > 0 else 0.0
        avg_loss = float(np.mean([pnl for (_, _, _, pnl) in trades if pnl <= 0])) if losses > 0 else 0.0
        hit_rate = wins / (wins + losses) if (wins + losses) > 0 else np.nan
        expectancy = avg_win * hit_rate + avg_loss * (1 - hit_rate) if not np.isnan(hit_rate) else np.nan
        # Rebalance-based cumulative return
        reb_equity = 100.0
        for r in per_reb_rets:
            reb_equity *= (1.0 + r)
        reb_cum = (reb_equity / 100.0) - 1.0
        report = {
            'bars': int(len(port_rets_min_full)),
            'sharpe': sharpe,
            'sortino': sortino,
            'avg_ret': avg,
            'std_ret': std,
            'max_drawdown': max_dd,
            'turnover': turnover,
            'cum_return_minute': float(equity_curve_full.iloc[-1] / 100.0 - 1.0),
            'cum_return_rebalance': float(reb_cum),
            'wins': wins,
            'losses': losses,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'hit_rate': hit_rate,
            'expectancy': expectancy,
        }
        logger.info(
            "REPORT    | bars=%d | sharpe=%.3f | sortino=%.3f\nRETURNS   | avg=%+.4f%% | std=%.4f%% | cum(minute)=%+.2f%% | cum(rebalance)=%+.2f%%\nDRAWDOWN  | maxDD=%+.2f%%\nTURNOVER  | %.4f\nTRADES    | hit=%.2f | wins=%d | losses=%d | avg_win=$%.4f | avg_loss=$%.4f | expectancy=$%.4f",
            report['bars'], report['sharpe'], report['sortino'], avg*100.0, std*100.0, report['cum_return_minute']*100.0, report['cum_return_rebalance']*100.0, report['max_drawdown']*100.0, report['turnover'], report['hit_rate'] if not np.isnan(report['hit_rate']) else np.nan, report['wins'], report['losses'], report['avg_win'], report['avg_loss'], report['expectancy']
        )
        for (dt, r) in per_reb_rows[:200]:
            logger.info("REBAL_RET | ts=%s | ret=%+.4f%%", dt, r*100.0)
        for (ts, s, dW, pnl) in trades[:20]:
            logger.info("TRADE     | ts=%s | symbol=%s | dW=%+.4f | pnl=$%.4f", ts, s, dW, pnl)
        return report

    async def run_backtest(self) -> List[Allocation]:
        bt_cfg = self.config.get('backtest', {})
        src = bt_cfg.get('source', 'alpaca').lower()
        ds = self.config.get('data_sources', {})
        output_size = bt_cfg.get('output_size', 'compact')
        interval = bt_cfg.get('interval', self.config.get('data', {}).get('timeframe', '1Min'))
        interval_mapped = self._map_interval(src, interval)
        logger.info("Backtest config source=%s interval=%s(mapped=%s)", src, interval, interval_mapped)
        bt = Backtester()
        universe_instruments = self._universe_instruments()
        if src == 'alpaca':
            # feed/adjustment read at call time so CLI overrides (scripts/backtest.py --feed) apply
            df = self.alpaca_data.load_minute_bars(
                universe_instruments, bt_cfg.get('start'), bt_cfg.get('end'),
                feed=bt_cfg.get('feed'), adjustment=bt_cfg.get('adjustment'), rth_only=self.rth_only,
            )
            # Cache fetched bars
            self._apply_store_update(df)
        else:
            loader = MinuteDataLoader(universe_instruments)
            bt_conf = BacktestConfig(
                start=bt_cfg.get('start'),
                end=bt_cfg.get('end'),
                interval=interval_mapped,
                source=src,
                api_key=(ds.get('alphaVantage', {}) or {}).get('apiKey'),
                output_size=output_size,
                rapidapi_key=None
            )
            df = loader.load(bt_conf)
            if self.rth_only:
                df = filter_rth(df)
            self._apply_store_update(df)
        # Fallback to local store if empty
        if df is None or df.empty:
            logger.warning("Backtest loader returned empty dataframe; falling back to local store")
            df = self.store.load_universe(universe_instruments, rth_only=self.rth_only)
        models_syms = set(self.config.get('universe', {}).get('models', []))
        emergency_cfg = self.config.get('emergency', {}) or {}
        signals_syms = set(sum([
            emergency_cfg.get('equity_signal', []),
            emergency_cfg.get('bond_signal', []),
            emergency_cfg.get('active', []),
        ], []))
        allowed = models_syms | signals_syms
        if df is None or df.empty:
            self.minute_df = pd.DataFrame()
            logger.warning("Backtest data still empty after store fallback")
        else:
            df = df.loc[:, [c for c in df.columns if c[1] in allowed]]
            logger.info("Backtest dataframe shape=%s columns_count=%d", df.shape, len(df.columns))
            na_pct = df.isna().mean().mean()
            logger.debug("Backtest dataframe NaN overall pct=%.4f", na_pct)
            self.minute_df = df
        logger.info("Running strategy on minute_df")
        allocs = bt.run(self.strategy, self.minute_df)
        logger.info("Strategy finished - allocation windows=%d", len(allocs) if allocs else 0)
        # Compute and log backtest report
        try:
            _ = self._compute_report(allocs)
        except Exception as e:
            logger.warning("Report computation failed: %s", e)
        return allocs

    def validate_overnight_asset(self, symbol: str) -> bool:
        """Helper to validate overnight eligibility before submitting extended-hours orders."""
        eligible = self.alpaca_account.is_overnight_tradable(symbol)
        if not eligible:
            logger.warning("Symbol %s not eligible for overnight trading or halted", symbol)
        return eligible
