# End-to-end system wiring: config -> engine -> Alpaca data/trading
import asyncio
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from .config_loader import ConfigLoader
from .instrument_map import InstrumentMap
from .engine import Strategy, Backtester, Allocation
from .data import MinuteDataLoader, BacktestConfig
from .alpaca_data import AlpacaMinuteDataLoader
from .alpaca_trading import AlpacaTrader
from .alpaca_sse import AlpacaSSEClient
from .alpaca_account import AlpacaAccountClient
from .execution import ExecutionRouter
from .minute_store import MinuteStore

logger = logging.getLogger(__name__)

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
        self.alpaca_data = AlpacaMinuteDataLoader(alpaca_cfg.get('apiKeyId'), alpaca_cfg.get('apiSecretKey'), alpaca_cfg.get('paper', True))
        self.alpaca_trader = AlpacaTrader(alpaca_cfg.get('apiKeyId'), alpaca_cfg.get('apiSecretKey'), alpaca_cfg.get('paper', True))
        self.alpaca_sse = AlpacaSSEClient(
            api_key=alpaca_cfg.get('apiKeyId'),
            api_secret=alpaca_cfg.get('apiSecretKey'),
            sandbox=alpaca_cfg.get('paper', True),
            oauth_client_id=alpaca_cfg.get('oauthClientId'),
            oauth_client_secret=alpaca_cfg.get('oauthClientSecret'),
        )
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
        # Launch SSE listeners concurrently with data stream
        async def _orders_evt(evt: Dict[str, Any]):
            logger.info("SSE order evt: %s", evt)
        async def _journals_evt(evt: Dict[str, Any]):
            logger.info("SSE journal evt: %s", evt)
        async def _transfers_evt(evt: Dict[str, Any]):
            logger.info("SSE transfer evt: %s", evt)
        async def _nta_evt(evt: Dict[str, Any]):
            logger.info("SSE nta evt: %s", evt)
        def orders_cb(evt: Dict[str, Any]):
            asyncio.create_task(_orders_evt(evt))
        def journals_cb(evt: Dict[str, Any]):
            asyncio.create_task(_journals_evt(evt))
        def transfers_cb(evt: Dict[str, Any]):
            asyncio.create_task(_transfers_evt(evt))
        def nta_cb(evt: Dict[str, Any]):
            asyncio.create_task(_nta_evt(evt))
        sse_tasks = [
            asyncio.create_task(self.alpaca_sse.subscribe_orders(orders_cb)),
            asyncio.create_task(self.alpaca_sse.subscribe_journals(journals_cb)),
            asyncio.create_task(self.alpaca_sse.subscribe_transfers(transfers_cb)),
            asyncio.create_task(self.alpaca_sse.subscribe_nta(nta_cb)),
        ]
        # Start account monitor task
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
            for t in sse_tasks:
                t.cancel()
            acct_task.cancel()

    async def _monitor_account(self):
        """Periodically read account fields and log settlement-aware state."""
        interval = int((self.config.get('live', {}) or {}).get('accountPollSeconds', 30))
        try:
            while True:
                acc = self.alpaca_account.get_account()
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

    async def _run_stream_alpaca_poll(self, poll_secs: int):
        logger.info("Using Alpaca polling for live data")
        universe = self._universe_instruments()
        try:
            while True:
                df = self.alpaca_data.load_minute_bars(universe, None, None)
                if df is not None and not df.empty:
                    # Cache to local store
                    self._apply_store_update(df)
                    models_syms = set(self.config.get('universe', {}).get('models', []))
                    emergency_cfg = self.config.get('emergency', {}) or {}
                    signals_syms = set(sum([
                        emergency_cfg.get('equity_signal', []),
                        emergency_cfg.get('bond_signal', []),
                        emergency_cfg.get('active', []),
                    ], []))
                    allowed = models_syms | signals_syms
                    # Filter to allowed (models + emergency) to align with backtest
                    df = df.loc[:, [c for c in df.columns if c[1] in allowed]]
                    self.minute_df = df.sort_index()
                else:
                    # Fallback to local store
                    self.minute_df = self.store.load_universe(universe)
                bt = Backtester()
                allocs = bt.run(self.strategy, self.minute_df)
                logger.info("Alpaca poll rebalancing allocations_count=%d", len(allocs) if allocs else 0)
                # translate latest allocation to orders
                if allocs:
                    latest = allocs[-1]
                    self.execution.rebalance(latest, current_positions={})
                await asyncio.sleep(poll_secs)
        except asyncio.CancelledError:
            logger.info("Alpaca polling cancelled")
        except Exception as e:
            logger.error("Alpaca poll failed: %s", e)

    async def _run_stream_alpaca_ws(self):
        """WebSocket streaming via alpaca-py StockDataStream to get bars in real time with feed selection and reconnect."""
        try:
            from alpaca.data.live import StockDataStream
        except Exception as e:
            logger.error("alpaca-py live streaming unavailable: %s", e)
            return
        alpaca_cfg = self.config.get('alpaca', {})
        feed = (self.config.get('live', {}).get('stream', {}).get('feed', 'iex')).lower()
        # Align symbols with backtest: models + emergency (equity/bond/active)
        models_syms = set(self.config.get('universe', {}).get('models', []))
        emergency_cfg = self.config.get('emergency', {}) or {}
        signals_syms = set(sum([
            emergency_cfg.get('equity_signal', []),
            emergency_cfg.get('bond_signal', []),
            emergency_cfg.get('active', []),
        ], []))
        allowed = sorted(models_syms | signals_syms)
        backoff = 1
        while True:
            try:
                stream = StockDataStream(alpaca_cfg.get('apiKeyId'), alpaca_cfg.get('apiSecretKey'), feed=feed)
                async def on_bar(bar):
                    ts = pd.to_datetime(bar.timestamp, utc=True).floor('T')
                    sym = bar.symbol
                    if sym not in allowed:
                        return
                    row = pd.DataFrame({
                        ('open', sym): [bar.open],
                        ('high', sym): [bar.high],
                        ('low', sym): [bar.low],
                        ('close', sym): [bar.close],
                        ('volume', sym): [bar.volume]
                    }, index=[ts])
                    row.columns = pd.MultiIndex.from_tuples(row.columns)
                    self.minute_df = pd.concat([self.minute_df, row]).sort_index()
                    # Filter columns to allowed set like backtest
                    self.minute_df = self.minute_df.loc[:, [c for c in self.minute_df.columns if c[1] in allowed]]
                    # Cache incremental bar to local store
                    try:
                        self.store.upsert_symbol(sym, row.xs(sym, axis=1, level=1).rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}))
                    except Exception:
                        pass
                    bt = Backtester()
                    allocs = bt.run(self.strategy, self.minute_df)
                    logger.info("Alpaca ws rebalancing allocations_count=%d", len(allocs) if allocs else 0)
                    if allocs:
                        latest = allocs[-1]
                        self.execution.rebalance(latest, current_positions={})
                # Subscribe only to allowed symbols
                stream.subscribe_bars(on_bar, *allowed)
                await stream.run()
                backoff = 1
            except asyncio.CancelledError:
                logger.info("Alpaca websocket cancelled")
                return
            except Exception as e:
                logger.warning("WebSocket error: %s; reconnecting in %ss", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    def _compute_report(self, allocs: List[Allocation]) -> Dict[str, Any]:
        if self.minute_df is None or self.minute_df.empty or not allocs:
            return {}
        # Close frame extraction
        if isinstance(self.minute_df.columns, pd.MultiIndex) and 'Close' in self.minute_df.columns.get_level_values(0):
            close = self.minute_df.xs('Close', axis=1, level=0)
        else:
            close = self.minute_df
        close = close.ffill().bfill()
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
            df = self.alpaca_data.load_minute_bars(universe_instruments, bt_cfg.get('start'), bt_cfg.get('end'))
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
            self._apply_store_update(df)
        # Fallback to local store if empty
        if df is None or df.empty:
            logger.warning("Backtest loader returned empty dataframe; falling back to local store")
            df = self.store.load_universe(universe_instruments)
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
