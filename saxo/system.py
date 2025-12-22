# End-to-end system wiring: config -> ws -> engine -> fix execution
import asyncio
import pandas as pd
import logging
from typing import Dict, Any, List
from .config_loader import ConfigLoader
from .instrument_map import InstrumentMap
from .ws_marketdata import SaxoWSClient, MinuteAggregator
from .engine import Strategy, Backtester, Allocation
from .data import MinuteDataLoader, BacktestConfig

logger = logging.getLogger(__name__)

class TradingSystem:
    def __init__(self, config_path: str = "config/saxo_config.json"):
        loader = ConfigLoader(config_path)
        self.config = loader.load()
        loader.generate_fix_cfg()
        self.instrument_map = InstrumentMap(self.config['instruments'])
        self.ws = SaxoWSClient(self.config['websocket']['url'], self.config['websocket']['token'], self.config['websocket']['contextId'])
        self.minute_df = pd.DataFrame()
        self.strategy = Strategy("CS-Momentum-Minute", self.config['strategy'])
        # Defer FIX setup to live mode to avoid dependency during backtests
        self.fix_settings = None
        self.fix_client = None
        self.aggregator = MinuteAggregator()
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
        logger.info("Starting live stream")
        try:
            import quickfix as fix
            from .fix_client import FixClient
            self.fix_settings = fix.SessionSettings("config/saxo_fix.cfg")
            self.fix_client = FixClient(self.fix_settings)
            logger.info("FIX session settings initialized")
        except Exception as e:
            logger.warning("FIX unavailable: %s", e)
            self.fix_settings = None
            self.fix_client = None
        await self.ws.connect()
        base_url = self.config['websocket'].get('restBaseUrl', 'https://gateway.saxobank.com/openapi')
        await self.ws.rest_subscribe_quotes(base_url, self._universe_instruments(), "ref")
        logger.info("Subscribed to quotes via REST")

        def on_message(msg: Dict[str, Any]):
            if isinstance(msg, dict) and 'Data' in msg:
                for item in msg['Data']:
                    self.aggregator.on_quote(item)
            else:
                self.aggregator.on_quote(msg)
            df = self.aggregator.finalize_minute()
            if not df.empty:
                cols = df.columns
                new_cols = []
                for price_type, ref_id in cols:
                    uic = int(ref_id.split('-')[-1])
                    symbol = self.instrument_map.by_uic[uic]['symbol']
                    new_cols.append((price_type, symbol))
                df.columns = pd.MultiIndex.from_tuples(new_cols)
                universe_syms = set(self.config.get('universe', {}).get('models', []))
                df = df.loc[:, [c for c in df.columns if c[1] in universe_syms]]
                self.minute_df = pd.concat([self.minute_df, df]).sort_index()
                logger.debug("Stream minute_df shape=%s", self.minute_df.shape)
                bt = Backtester()
                allocs = bt.run(self.strategy, self.minute_df)
                logger.info("Stream rebalancing allocations_count=%d", len(allocs) if allocs else 0)

        await self.ws.stream(on_message)

    async def run_backtest(self) -> List[Allocation]:
        bt_cfg = self.config.get('backtest', {})
        src = bt_cfg.get('source', 'yahoo').lower()
        ds = self.config.get('data_sources', {})
        output_size = bt_cfg.get('output_size', 'compact')
        rapidapi_key = None
        if src == 'yahoo':
            interval = (ds.get('yahoo', {})).get('interval', '1h')
            api_key = None
        elif src in ('alpha_vantage', 'alphavantage'):
            interval = (ds.get('alphaVantage', {})).get('interval', '1min')
            api_key = (ds.get('alphaVantage', {})).get('apiKey')
        elif src in ('alpha_vantage_rapidapi', 'alphavantage_rapidapi', 'rapidapi'):
            interval = bt_cfg.get('interval', '1min')
            rapidapi_key = bt_cfg.get('rapidapi_key') or bt_cfg.get('apiKey')
            api_key = rapidapi_key  # for logging only
        else:
            interval = bt_cfg.get('interval', '1m')
            api_key = bt_cfg.get('apiKey')
        logger.info("Backtest config source=%s interval=%s api_key_present=%s", src, interval, bool(api_key))
        bt = Backtester()
        universe_instruments = self._universe_instruments()
        loader = MinuteDataLoader(universe_instruments)
        # Build BacktestConfig with RapidAPI-specific fields when needed
        bt_conf = BacktestConfig(
            start=bt_cfg.get('start'),
            end=bt_cfg.get('end'),
            interval=interval,
            source=src,
            api_key=(None if src in ('alpha_vantage_rapidapi', 'alphavantage_rapidapi', 'rapidapi') else api_key),
            output_size=output_size,
            rapidapi_key=rapidapi_key
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

    def start_fix(self):
        try:
            import quickfix as fix
            from .fix_client import FixClient
            self.fix_settings = fix.SessionSettings("config/saxo_fix.cfg")
            self.fix_client = FixClient(self.fix_settings)
            self.fix_client.start()
            logger.info("FIX client started")
        except Exception as e:
            logger.warning("FIX start failed: %s", e)

    def stop_fix(self):
        if self.fix_client:
            self.fix_client.stop()
            logger.info("FIX client stopped")
