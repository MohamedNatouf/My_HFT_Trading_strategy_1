# Strategy engine implementing cross-sectional momentum on minute bars with parameter-driven logic
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import logging
import warnings

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    HAS_PYPOPT = True
except Exception:
    HAS_PYPOPT = False

# Suppress specific pct_change FutureWarning noise in logs
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*fill_method.*DataFrame.pct_change.*deprecated.*"
)

logger = logging.getLogger(__name__)

@dataclass
class Position:
    symbol: str
    weight: float

@dataclass
class Allocation:
    date: pd.Timestamp
    positions: List[Position]
    emergency: bool = False

class Strategy:
    def __init__(self, name: str, params: Dict):
        self.name = name
        self.params = params
        self.state = {
            "emergency": False,
            "equity_wait": 0,
            "bond_wait": 0,
            "hold_days": 0
        }
        self.minutes_per_day = int(self.params.get("minutes_per_day", 390))
        self.rebalance_minutes = int(self.params.get("rebalance_minutes", 5))
        self.emergency_cfg: Dict = self.params.get("emergency", {}) or {}
        self.emergency_enabled: bool = bool(self.params.get("Emergency_System_Enabled", True))
        self.universe_models: List[str] = self.params.get("universe_models", []) or []

    @staticmethod
    def _sma_instantaneous_slope(series: pd.Series, period: int) -> float:
        if len(series) < period+1:
            return 1.0
        short = series.iloc[-period:].mean()
        long = series.iloc[-(period+1):].mean()
        if long == 0:
            return 1.0
        val = (short/long) - 1
        return val if val != 0 else 1.0

    def _high_low_diff_ratio(self, series: pd.Series) -> float:
        # Use raw minute bars counts (start,end,step)
        return Strategy._high_low_diff_ratio_static(series, 22, 132, 22)

    @staticmethod
    def _high_low_diff_ratio_static(series: pd.Series, start: int, end: int, step: int) -> float:
        max_signal = -1000.0
        for p in range(start, end+1, step):
            if len(series) < p+1:
                continue
            sub = series.iloc[-p:]
            cur = sub.iloc[-1]
            mx = sub.max()
            mn = sub.min()
            std = sub.std()
            if std == 0:
                std = 1.0
            high_diff = (mx/cur)/std if cur != 0 else 0
            low_diff = (cur/mn)/std if mn != 0 else 0
            value = (low_diff - high_diff) * 100
            max_signal = max(max_signal, value)
        return 1.0 if max_signal == -1000.0 else max_signal

    def _linear_regression_ratio(self, series: pd.Series) -> float:
        return Strategy._linear_regression_ratio_static(series, 11, 132, 11)

    @staticmethod
    def _linear_regression_ratio_static(series: pd.Series, start: int, end: int, step: int) -> float:
        max_signal = -1000.0
        for p in range(start, end+1, step):
            if len(series) < p+1:
                continue
            sub = series.iloc[-p:]
            x = np.arange(1, len(sub)+1)
            n = len(sub)
            sum_x = x.sum(); sum_y = sub.sum(); sum_xy = (x*sub.values).sum(); sum_x2 = (x*x).sum()
            denom = (n*sum_x2 - sum_x**2)
            if denom == 0 or n == 0:
                continue
            b = ((n*sum_xy) - (sum_x*sum_y)) / denom
            a = ((sum_y*sum_x2) - (sum_x*sum_xy)) / denom
            forecast = a + b*n
            if a != 0:
                ratio = (forecast/a) - 1
                max_signal = max(max_signal, ratio)
        return 1.0 if max_signal == -1000.0 else max_signal

    def _stochastic_oscillator(self, series: pd.Series) -> float:
        return Strategy._stochastic_oscillator_static(series, 22, 132, 22)

    @staticmethod
    def _stochastic_oscillator_static(series: pd.Series, start: int, end: int, step: int) -> float:
        max_signal = -1000.0
        for p in range(start, end+1, step):
            if len(series) < p+1:
                continue
            sub = series.iloc[-p:]
            cur = sub.iloc[-1]; mx = sub.max(); mn = sub.min()
            if (mx - mn) != 0:
                k = ((cur - mn) / (mx - mn)) * 100
                if k > 100 or k == 0:
                    k = 1
                max_signal = max(max_signal, k)
        return 1.0 if max_signal == -1000.0 else max_signal

    @staticmethod
    def _price_percent_rank(sub: List[float], value: float) -> float:
        if len(sub) == 0:
            return 1.0
        sorted_vals = sorted(sub)
        try:
            idx = sorted_vals.index(value)
        except ValueError:
            idx = np.searchsorted(sorted_vals, value)
        n = len(sorted_vals)
        if n <= 1:
            return 1.0
        pr = (idx) / (n - 1)
        return pr if pr != 0 else 1.0

    def _price_percent_rank_metric(self, series: pd.Series) -> float:
        return Strategy._price_percent_rank_metric_static(series, 4, 110, 2)

    @staticmethod
    def _price_percent_rank_metric_static(series: pd.Series, start: int, end: int, step: int) -> float:
        max_signal = -1000.0
        for p in range(start, end+1, step):
            if len(series) < p+1:
                continue
            sub = series.iloc[-p:]
            cur = float(sub.iloc[-1])
            pr = Strategy._price_percent_rank([round(v,3) for v in sub.tolist()], round(cur,3))
            max_signal = max(max_signal, pr)
        return 1.0 if max_signal == -1000.0 else max_signal

    def _rsi_index(self, series: pd.Series) -> float:
        return Strategy._rsi_index_static(series, 11, 132, 11)

    @staticmethod
    def _rsi_index_static(series: pd.Series, start: int, end: int, step: int) -> float:
        max_signal = -1000.0
        for p in range(start, end+1, step):
            if len(series) < p+1:
                continue
            sub = series.iloc[-p:]
            diffs = sub.diff().dropna()
            gains = diffs.clip(lower=0)
            losses = (-diffs.clip(upper=0))
            avg_gain = gains.mean(); avg_loss = losses.mean()
            if avg_loss != 0:
                rs = avg_gain/avg_loss
                if rs != -1:
                    rsi = 100 - (100/(1+rs))
                    max_signal = max(max_signal, rsi)
        if max_signal != -1000.0 and max_signal != 1:
            return max_signal/100.0
        return 1.0

    def Momentum_MultiFactor_Maximizer(self, close: pd.DataFrame, lookback: int, method: str) -> pd.Series:
        scores = {}
        lookback_minutes = int(lookback)  # raw minute bars count
        for symbol in close.columns:
            s = close[symbol].dropna()
            if len(s) < lookback_minutes+2:
                scores[symbol] = -np.inf
                continue
            hl = self._high_low_diff_ratio(s)
            lr = self._linear_regression_ratio(s)
            pr = self._price_percent_rank_metric(s)
            rsi = self._rsi_index(s)
            sma = self._sma_instantaneous_slope(s, lookback_minutes)
            stoch = self._stochastic_oscillator(s)
            value = hl * abs(lr) * pr * rsi * abs(sma) * abs(stoch)
            scores[symbol] = value
        ser = pd.Series(scores)
        if method == "Metrics Count":
            arrays = []
            arr_hl = close.apply(lambda col: self._high_low_diff_ratio(col.dropna()))
            arr_lr = close.apply(lambda col: self._linear_regression_ratio(col.dropna()))
            arr_pr = close.apply(lambda col: self._price_percent_rank_metric(col.dropna()))
            arr_rsi = close.apply(lambda col: self._rsi_index(col.dropna()))
            arr_sma = close.apply(lambda col: self._sma_instantaneous_slope(col.dropna(), lookback_minutes))
            arr_stoch = close.apply(lambda col: self._stochastic_oscillator(col.dropna()))
            arrays = [arr_hl, arr_lr, arr_pr, arr_rsi, arr_sma, arr_stoch]
            rank_sum = pd.Series(0.0, index=close.columns)
            for ser_arr in arrays:
                rank_sum += ser_arr.rank(ascending=False)
            return rank_sum.sort_values(ascending=False)
        return ser.sort_values(ascending=False)

    def _apply_special_filter(self, rets: pd.Series, symbols_meta: Optional[Dict[str, Dict]] = None) -> pd.Series:
        if not self.params.get("Special_Filter_Status", False):
            return rets
        return rets

    def _compute_momentum(self, df: pd.DataFrame) -> pd.Series:
        lookback_bars = int(self.params.get("Volatility_Measuring_Lookback_Period", 60))
        # Accept either a MultiIndex with ('Close', symbol) or a single-level frame of closes
        close: pd.DataFrame
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                close = df.xs('Close', axis=1, level=0)
            else:
                # Already a close-only frame with single-level columns
                close = df
        else:
            close = df
        method = self.params.get("Selected_method_of_Momuntum_Metric", "Metrics Combind")
        logger.debug("compute_momentum lookback_bars=%d method=%s cols=%d", lookback_bars, method, len(close.columns))
        return self.Momentum_MultiFactor_Maximizer(close, lookback_bars, method)

    def _weights(self, df: pd.DataFrame, selected_symbols: List[str]) -> pd.Series:
        method = self.params.get("Method_Of_Wieghting", "Assets Inverse Volatility")
        lookback_bars = int(self.params.get("Volatility_Measuring_Lookback_Period", 60))
        vol_strength = float(self.params.get("Volatility_Weight_Strength", 1))
        if isinstance(df.columns, pd.MultiIndex) and 'Close' in df.columns.get_level_values(0):
            prices = df.xs('Close', axis=1, level=0)[selected_symbols]
        else:
            prices = df[selected_symbols]
        prices = prices.ffill().bfill()
        try:
            vol = prices.pct_change(fill_method=None).tail(lookback_bars).std()
        except Exception:
            vol = prices.pct_change().tail(lookback_bars).std()
        logger.debug("weights method=%s lookback_bars=%d vol_strength=%.3f vol_nonnull=%d", method, lookback_bars, vol_strength, vol.notna().sum())
        if method == "Assets Inverse Volatility":
            w = 1/vol.replace(0, np.nan)
            w = w / w.sum()
            logger.debug("weights computed=InverseVol w=%s", {s: float(w.get(s, 0)) for s in selected_symbols})
        elif method == "Assets Volume":
            w = pd.Series(1.0, index=selected_symbols)
            w = w / w.sum()
            logger.debug("weights computed=EqualVol w=%s", {s: float(w.get(s, 0)) for s in selected_symbols})
        elif method == "Volume & Inverse Volume":
            w_vol = 1/vol.replace(0, np.nan)
            w_eq = pd.Series(1.0, index=selected_symbols)
            w = vol_strength*(w_vol/w_vol.sum()) + (1-vol_strength)*(w_eq/w_eq.sum())
            w = w / w.sum()
            logger.debug("weights computed=Blend vol_strength=%.3f w=%s", vol_strength, {s: float(w.get(s, 0)) for s in selected_symbols})
        elif method == "Efficient Frontier Optimization" and HAS_PYPOPT:
            try:
                # pypfopt expects frequency per year; using minutes_per_day*252 for minute data
                freq = self.minutes_per_day * 252
                mu = expected_returns.mean_historical_return(prices, frequency=freq)
                S = risk_models.sample_cov(prices, frequency=freq)
                ef = EfficientFrontier(mu, S)
                w_dict = ef.max_sharpe()
                w = pd.Series(w_dict)
                w = w.reindex(selected_symbols).fillna(0)
                if w.sum() <= 0:
                    raise ValueError("EF returned zero/negative weights")
                w = w / w.sum()
                logger.debug("weights computed=EfficientFrontier w=%s", {s: float(w.get(s, 0)) for s in selected_symbols})
            except Exception as e:
                logger.warning("Efficient Frontier failed; falling back to equal weights. err=%s", e)
                w = pd.Series(1.0, index=selected_symbols)
                w = w / w.sum()
        else:
            if method == "Efficient Frontier Optimization" and not HAS_PYPOPT:
                logger.warning("pypfopt not installed; falling back to equal weights for Efficient Frontier")
            w = pd.Series(1.0, index=selected_symbols)
            w = w / w.sum()
            logger.debug("weights computed=EqualDefault w=%s", {s: float(w.get(s, 0)) for s in selected_symbols})
        return w.fillna(0)

    def _update_emergency(self, df: pd.DataFrame):
        if not self.emergency_enabled:
            self.state["emergency"] = False
            self.state["equity_wait"] = 0
            self.state["bond_wait"] = 0
            self.state["hold_days"] = 0
            return
        eq_lb_bars = int(self.params.get("Equity_Lookback_Period", 55))
        bond_lb_bars = int(self.params.get("Bond_Lookback_Period", 110))
        eq_wait_req = int(self.params.get("Equity_Signal_Wait_Period", 3))
        bond_wait_req = int(self.params.get("Bond_Signal_Wait_Period", 1))
        hold_status = bool(self.params.get("Equity_N_Bond_Hold_Period_Status", True))
        hold_period = int(self.params.get("Equity_N_Bond_Hold_Period", 3))
        close = df["Close"]
        # Fill before pct_change to avoid warnings
        close = close.ffill().bfill()
        eq_syms = self.emergency_cfg.get("equity_signal", [])
        bond_syms = self.emergency_cfg.get("bond_signal", [])
        risk_off = False
        if set(eq_syms).issubset(set(close.columns)) and len(close) > eq_lb_bars:
            eq_ret = close[eq_syms].pct_change(eq_lb_bars, fill_method=None).iloc[-1].mean()
            risk_off = risk_off or (eq_ret < 0)
        if set(bond_syms).issubset(set(close.columns)) and len(close) > bond_lb_bars:
            bond_ret = close[bond_syms].pct_change(bond_lb_bars, fill_method=None).iloc[-1].mean()
            risk_off = risk_off or (bond_ret < 0)
        if risk_off:
            self.state["equity_wait"] += 1
            self.state["bond_wait"] += 1
        else:
            self.state["equity_wait"] = 0
            self.state["bond_wait"] = 0
        if (self.state["bond_wait"] >= bond_wait_req) or (self.state["equity_wait"] >= eq_wait_req):
            self.state["emergency"] = True
            if hold_status:
                self.state["hold_days"] = hold_period
        else:
            if self.state["hold_days"] > 0:
                self.state["hold_days"] -= 1
            else:
                self.state["emergency"] = False
        logger.debug("emergency state=%s eq_wait=%d bond_wait=%d hold_days=%d", self.state["emergency"], self.state["equity_wait"], self.state["bond_wait"], self.state["hold_days"])

    def generate_allocation(self, minute_df: pd.DataFrame) -> Allocation:
        self._update_emergency(minute_df)
        if isinstance(minute_df.columns, pd.MultiIndex) and 'Close' in minute_df.columns.get_level_values(0):
            available_syms = list(minute_df.xs('Close', axis=1, level=0).columns)
        else:
            available_syms = list(minute_df.columns)
        if self.emergency_enabled and self.state["emergency"]:
            emerg_syms = [s for s in self.emergency_cfg.get("active", []) if s in available_syms]
            selected_universe = emerg_syms if emerg_syms else self.universe_models
        else:
            selected_universe = [s for s in self.universe_models if s in available_syms] or available_syms
        if isinstance(minute_df.columns, pd.MultiIndex) and 'Close' in minute_df.columns.get_level_values(0):
            close_frame = minute_df.xs('Close', axis=1, level=0)[selected_universe]
        else:
            close_frame = minute_df[selected_universe]
        scores = self._compute_momentum(close_frame)
        scores = self._apply_special_filter(scores)
        k = int(self.params.get('Number_of_Portfolio_Allocations', 3))
        selected = list(scores.head(k).index)
        weights = self._weights(close_frame, selected)
        positions = [Position(symbol=s, weight=float(weights[s])) for s in selected]
        # Format weights as percentages for tidy logging
        weights_pct = {s: f"{float(weights[s])*100.0:.2f}%" for s in selected}
        logger.info("allocation ts=%s emergency=%s selected=%s weights=%s", minute_df.index[-1], self.emergency_enabled and self.state["emergency"], selected, weights_pct)
        return Allocation(date=minute_df.index[-1], positions=positions, emergency=(self.emergency_enabled and self.state["emergency"]))

class Backtester:
    def run(self, strategy: Strategy, minute_df: pd.DataFrame, rebalancing: Optional[str] = None) -> List[Allocation]:
        allocs: List[Allocation] = []
        method = rebalancing or strategy.params.get("Rebalancing_Method", "active")
        step = int(strategy.params.get('rebalance_minutes', strategy.rebalance_minutes))
        # Minimum bars required equals the volatility lookback (bars), not days
        min_bars = int(strategy.params.get("Volatility_Measuring_Lookback_Period", 60))
        logger.info("backtest start rows=%d cols=%d method=%s step=%d min_bars=%d", len(minute_df), len(minute_df.columns), method, step, min_bars)
        # Initialize starting equity at $100 for logging
        equity = 100.0
        max_equity = 100.0
        for i in range(len(minute_df)):
            window = minute_df.iloc[:i+1]
            if len(window) < max(3, min_bars):
                continue
            ts = window.index[-1]
            do_rebalance = False
            if method == '1':
                do_rebalance = ts.isoweekday() == 5 and ts.hour == 16 and ts.minute == 0
            elif method == '2':
                next_ts = ts + pd.Timedelta(minutes=1)
                do_rebalance = next_ts.month != ts.month and ts.hour == 16 and ts.minute == 0
            elif method == 'active' or method == '3':
                do_rebalance = (i % step) == 0
            if not do_rebalance:
                if i % 1000 == 0:
                    logger.debug("skip rebalance ts=%s i=%d", ts, i)
                continue
            logger.debug("do rebalance ts=%s i=%d", ts, i)
            alloc = strategy.generate_allocation(window)
            # Compute simple portfolio stats on the window for logging
            try:
                if isinstance(window.columns, pd.MultiIndex) and 'Close' in window.columns.get_level_values(0):
                    close = window.xs('Close', axis=1, level=0)
                else:
                    close = window
                sel_syms = [p.symbol for p in alloc.positions]
                w = pd.Series({p.symbol: p.weight for p in alloc.positions})
                px = close[sel_syms].iloc[-1]
                prev_px = close[sel_syms].iloc[-2] if len(close) > 1 else px
                ret_vec = (px / prev_px) - 1
                port_ret = float((w * ret_vec).sum()) if len(w) > 0 else 0.0
                # Update equity in dollars
                equity *= (1.0 + port_ret)
                max_equity = max(max_equity, equity)
                dd = (equity / max_equity) - 1.0
                cum = (equity / 100.0) - 1.0
                # Tidy logging: separate sections with formatted values
                logger.info(
                    "ALLOCATION | ts=%s | emergency=%s\n  positions=%s\nSTATS      | ret=%+.4f%% | cum=%+.2f%% | equity=$%.2f | max_eq=$%.2f | drawdown=%+.4f%%",
                    ts,
                    strategy.emergency_enabled and strategy.state.get("emergency"),
                    [(p.symbol, f"{p.weight*100.0:.2f}%") for p in alloc.positions],
                    port_ret * 100.0,
                    cum * 100.0,
                    equity,
                    max_equity,
                    dd * 100.0,
                )
            except Exception as e:
                logger.debug("stats error ts=%s err=%s", ts, e)
            allocs.append(alloc)
        logger.info("backtest done allocations=%d", len(allocs))
        # Persist ending equity/max equity for visibility
        logger.info("final equity=$%.2f max_equity=$%.2f", equity, max_equity)
        return allocs

# simple dict to carry running stats in logs
logger_extra = {}
