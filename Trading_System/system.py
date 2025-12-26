# End-to-end system wiring: config -> engine -> Alpaca data/trading
import asyncio
import pandas as pd
import logging
from typing import Dict, Any, List
from .config_loader import ConfigLoader
from .instrument_map import InstrumentMap
from .engine import Strategy, Backtester, Allocation
from .data import MinuteDataLoader, BacktestConfig
from .alpaca_data import AlpacaMinuteDataLoader
from .alpaca_trading import AlpacaTrader

logger = logging.getLogger(__name__)

class TradingSystem:
    def __init__(self, config_path: str = "config/config.json"):
        loader = ConfigLoader(config_path)
        self.config = loader.load()
        self.instrument_map = InstrumentMap(self.config['instruments'])
        self.minute_df = pd.DataFrame()
        self.strategy = Strategy("CS-Momentum-Minute", self.config['strategy'])
        alpaca_cfg = self.config.get('alpaca', {})
        self.alpaca_data = AlpacaMinuteDataLoader(alpaca_cfg.get('apiKeyId'), alpaca_cfg.get('apiSecretKey'), alpaca_cfg.get('paper', True))
        self.alpaca_trader = AlpacaTrader(alpaca_cfg.get('apiKeyId'), alpaca_cfg.get('apiSecretKey'), alpaca_cfg.get('paper', True))
        logger.info("TradingSystem initialized mode=%s source=%s", self.config.get('mode'), self.config.get('backtest',{}).get('source'))

    def _universe_instruments(self) -> List[Dict[str, Any]]:
        universe_syms = set(self.config.get('universe', {}).get('models', []))
        universe = [i for i in self.config['instruments'] if i['symbol'] in universe_syms]
        logger.info("Universe resolved count=%d symbols=%s", len(universe), [u['symbol'] for u in universe])
        return universe

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
        stream_cfg = (self.config.get('live', {}) or {}).get('stream', {})
        mode = stream_cfg.get('mode', 'poll').lower()
        if mode == 'websocket':
            await self._run_stream_alpaca_ws()
        else:
            poll_secs = int(stream_cfg.get('pollSeconds', 10))
            await self._run_stream_alpaca_poll(poll_secs)

    async def _run_stream_alpaca_poll(self, poll_secs: int):
        logger.info("Using Alpaca polling for live data")
        universe = self._universe_instruments()
        try:
            while True:
                df = self.alpaca_data.load_minute_bars(universe, None, None)
                if df is not None and not df.empty:
                    universe_syms = set(self.config.get('universe', {}).get('models', []))
                    df = df.loc[:, [c for c in df.columns if c[1] in universe_syms]]
                    self.minute_df = df.sort_index()
                    bt = Backtester()
                    allocs = bt.run(self.strategy, self.minute_df)
                    logger.info("Alpaca poll rebalancing allocations_count=%d", len(allocs) if allocs else 0)
                await asyncio.sleep(poll_secs)
        except asyncio.CancelledError:
            logger.info("Alpaca polling cancelled")
        except Exception as e:
            logger.error("Alpaca poll failed: %s", e)

    async def _run_stream_alpaca_ws(self):
        """WebSocket streaming via alpaca-py StockDataStream to get bars in real time."""
        try:
            from alpaca.data.live import StockDataStream
            from alpaca.data.timeframe import TimeFrame
        except Exception as e:
            logger.error("alpaca-py live streaming unavailable: %s", e)
            return
        alpaca_cfg = self.config.get('alpaca', {})
        feed = (self.config.get('live', {}).get('stream', {}).get('feed', 'iex')).lower()
        symbols = [i['symbol'] for i in self._universe_instruments()]
        stream = StockDataStream(alpaca_cfg.get('apiKeyId'), alpaca_cfg.get('apiSecretKey'))
        # subscribe bars callback
        async def on_bar(bar):
            # bar has .symbol and OHLCV fields; append to minute_df
            ts = pd.to_datetime(bar.timestamp, utc=True).floor('T')
            sym = bar.symbol
            row = pd.DataFrame({
                ('open', sym): [bar.open],
                ('high', sym): [bar.high],
                ('low', sym): [bar.low],
                ('close', sym): [bar.close],
                ('volume', sym): [bar.volume]
            }, index=[ts])
            row.columns = pd.MultiIndex.from_tuples(row.columns)
            self.minute_df = pd.concat([self.minute_df, row]).sort_index()
            bt = Backtester()
            allocs = bt.run(self.strategy, self.minute_df)
            logger.info("Alpaca ws rebalancing allocations_count=%d", len(allocs) if allocs else 0)
        stream.subscribe_bars(on_bar, *symbols)
        await stream.run()

    async def run_backtest(self) -> List[Allocation]:
        bt_cfg = self.config.get('backtest', {})
        src = bt_cfg.get('source', 'alpaca').lower()
        ds = self.config.get('data_sources', {})
        output_size = bt_cfg.get('output_size', 'compact')
        rapidapi_key = None
        interval = bt_cfg.get('interval', self.config.get('data', {}).get('timeframe', '1Min'))
        logger.info("Backtest config source=%s interval=%s", src, interval)
        bt = Backtester()
        universe_instruments = self._universe_instruments()
        if src == 'alpaca':
            df = self.alpaca_data.load_minute_bars(universe_instruments, bt_cfg.get('start'), bt_cfg.get('end'))
        else:
            loader = MinuteDataLoader(universe_instruments)
            bt_conf = BacktestConfig(
                start=bt_cfg.get('start'),
                end=bt_cfg.get('end'),
                interval=interval,
                source=src,
                api_key=(ds.get('alphaVantage', {}) or {}).get('apiKey'),
                output_size=output_size,
                rapidapi_key=None
            )
            df = loader.load(bt_conf)
        universe_syms = set(self.config.get('universe', {}).get('models', []))
        if df is None or df.empty:
            logger.warning("Backtest loader returned empty dataframe")
            self.minute_df = pd.DataFrame()
        else:
            df = df.loc[:, [c for c in df.columns if c[1] in universe_syms]]
            logger.info("Backtest dataframe shape=%s columns_count=%d", df.shape, len(df.columns))
            na_pct = df.isna().mean().mean()
            logger.debug("Backtest dataframe NaN overall pct=%.4f", na_pct)
            self.minute_df = df
        logger.info("Running strategy on minute_df")
        allocs = bt.run(self.strategy, self.minute_df)
        logger.info("Strategy finished - allocation windows=%d", len(allocs) if allocs else 0)
        return allocs
