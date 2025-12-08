# End-to-end system wiring: config -> ws -> engine -> fix execution
import asyncio
import pandas as pd
from typing import Dict, Any
from .config_loader import ConfigLoader
from .instrument_map import InstrumentMap
from .ws_marketdata import SaxoWSClient, MinuteAggregator
from .engine import Strategy, Backtester
from .fix_client import FixClient
import quickfix as fix

class TradingSystem:
    def __init__(self, config_path: str = "config/saxo_config.json"):
        loader = ConfigLoader(config_path)
        self.config = loader.load()
        loader.generate_fix_cfg()
        self.instrument_map = InstrumentMap(self.config['instruments'])
        self.ws = SaxoWSClient(self.config['websocket']['url'], self.config['websocket']['token'], self.config['websocket']['contextId'])
        self.minute_df = pd.DataFrame()
        self.strategy = Strategy("CS-Momentum-Minute", self.config['strategy'])
        self.fix_settings = fix.SessionSettings("config/saxo_fix.cfg")
        self.fix_client = FixClient(self.fix_settings)
        self.aggregator = MinuteAggregator()

    async def run_stream(self):
        await self.ws.connect()
        # Perform REST subscriptions
        base_url = self.config['websocket'].get('restBaseUrl', 'https://gateway.saxobank.com/openapi')
        await self.ws.rest_subscribe_quotes(base_url, self.config['instruments'], "ref")

        def on_message(msg: Dict[str, Any]):
            # Saxo streaming envelopes contain data array; handle both single and batch
            if isinstance(msg, dict) and 'Data' in msg:
                for item in msg['Data']:
                    self.aggregator.on_quote(item)
            else:
                self.aggregator.on_quote(msg)
            # finalize every minute tick
            df = self.aggregator.finalize_minute()
            if not df.empty:
                # Map ReferenceId to symbols if needed; for now use ReferenceId columns
                # Build multiindex (PriceType, Symbol) as engine expects
                # Assuming ref ids are ref-<uic>, convert to symbol
                cols = df.columns
                new_cols = []
                for price_type, ref_id in cols:
                    uic = int(ref_id.split('-')[-1])
                    symbol = self.instrument_map.by_uic[uic]['symbol']
                    new_cols.append((price_type, symbol))
                df.columns = pd.MultiIndex.from_tuples(new_cols)
                self.minute_df = pd.concat([self.minute_df, df]).sort_index()
                # Optionally run rebalancing
                bt = Backtester()
                allocs = bt.run(self.strategy, self.minute_df)
                # TODO: pass latest allocation to execution router

        await self.ws.stream(on_message)

    def start_fix(self):
        self.fix_client.start()

    def stop_fix(self):
        self.fix_client.stop()
