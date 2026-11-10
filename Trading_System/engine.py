# Strategy engine implementing cross-sectional momentum on minute bars with parameter-driven logic
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
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
    # Last observed (non-NaN) close of the symbol at allocation time. Used by
    # the execution router to convert weights into share quantities.
    last_close: Optional[float] = None

@dataclass
class Allocation:
    date: pd.Timestamp
    positions: List[Position]
    emergency: bool = False


def _extract_close_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a close-only DataFrame regardless of column case. If df has MultiIndex
    columns, look for a top-level named 'close' (case-insensitive) and return that slice.
    Otherwise assume df already contains close prices only.
    """
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        # Map lower->original label to keep exact key
        lower_map: Dict[str, str] = {}
        for lbl in lvl0.unique():
            lower_map[str(lbl).lower()] = lbl
        if 'close' in lower_map:
            return df.xs(lower_map['close'], axis=1, level=0)
    return df


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
            if not np.isfinite(cur) or cur == 0:
                continue
            mx = sub.max()
            mn = sub.min()
            # Dispersion of prices normalised by the current price so the
            # score is dimensionless (scale-invariant). Using the raw price
            # std made a 10x higher-priced asset score 10x lower.
            std = (sub / cur).std()
            if not np.isfinite(std) or std == 0:
                # Flat window: mx == mn == cur, so the signal is zero.
                value = 0.0
            else:
                high_diff = (mx/cur)/std
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
        """Return momentum scores ordered best-first (``.head(k)`` = top k).

        Symbols with insufficient history (< lookback+2 non-NaN bars) are
        excluded entirely, for both methods.
        """
        lookback_minutes = int(lookback)  # raw minute bars count
        eligible = [sym for sym in close.columns if close[sym].dropna().shape[0] >= lookback_minutes + 2]
        if method == "Metrics Count":
            if not eligible:
                return pd.Series(dtype=float)
            sub_close = close[eligible]
            arr_hl = sub_close.apply(lambda col: self._high_low_diff_ratio(col.dropna()))
            arr_lr = sub_close.apply(lambda col: self._linear_regression_ratio(col.dropna()))
            arr_pr = sub_close.apply(lambda col: self._price_percent_rank_metric(col.dropna()))
            arr_rsi = sub_close.apply(lambda col: self._rsi_index(col.dropna()))
            arr_sma = sub_close.apply(lambda col: self._sma_instantaneous_slope(col.dropna(), lookback_minutes))
            arr_stoch = sub_close.apply(lambda col: self._stochastic_oscillator(col.dropna()))
            arrays = [arr_hl, arr_lr, arr_pr, arr_rsi, arr_sma, arr_stoch]
            rank_sum = pd.Series(0.0, index=sub_close.columns)
            for ser_arr in arrays:
                # rank 1 = highest metric value (best)
                rank_sum += ser_arr.astype(float).rank(ascending=False)
            # Lowest rank sum = best overall -> ascending sort puts best first
            return rank_sum.sort_values(ascending=True, kind="mergesort")
        scores = {}
        for symbol in eligible:
            s = close[symbol].dropna()
            hl = self._high_low_diff_ratio(s)
            lr = self._linear_regression_ratio(s)
            pr = self._price_percent_rank_metric(s)
            rsi = self._rsi_index(s)
            sma = self._sma_instantaneous_slope(s, lookback_minutes)
            stoch = self._stochastic_oscillator(s)
            value = hl * abs(lr) * pr * rsi * abs(sma) * abs(stoch)
            scores[symbol] = value
        ser = pd.Series(scores, dtype=float)
        ser = ser[np.isfinite(ser)]
        return ser.sort_values(ascending=False)

    def _apply_special_filter(self, rets: pd.Series, symbols_meta: Optional[Dict[str, Dict]] = None) -> pd.Series:
        if not self.params.get("Special_Filter_Status", False):
            return rets
        return rets

    def _compute_momentum(self, df: pd.DataFrame) -> pd.Series:
        lookback_bars = int(self.params.get("Volatility_Measuring_Lookback_Period", 60))
        close = _extract_close_frame(df)
        method = self.params.get("Selected_method_of_Momuntum_Metric", "Metrics Combind")
        logger.debug("compute_momentum lookback_bars=%d method=%s cols=%d", lookback_bars, method, len(close.columns))
        return self.Momentum_MultiFactor_Maximizer(close, lookback_bars, method)

    def _weights(self, df: pd.DataFrame, selected_symbols: List[str]) -> pd.Series:
        method = self.params.get("Method_Of_Wieghting", "Assets Inverse Volatility")
        lookback_bars = int(self.params.get("Volatility_Measuring_Lookback_Period", 60))
        vol_strength = float(self.params.get("Volatility_Weight_Strength", 1))
        prices = _extract_close_frame(df)[selected_symbols]
        # Forward-fill only: back-filling would leak future prices into the past.
        prices = prices.ffill()
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
        close = _extract_close_frame(df)
        # Forward-fill only (no bfill: that would leak future prices backwards)
        close = close.ffill()
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

    def required_window(self) -> int:
        """Number of trailing bars needed to reproduce a full-history allocation.

        max of: indicator lookbacks (132 bars + 2), volatility lookback + 2,
        emergency equity/bond lookbacks + 1. Doubled (+10) to leave head-room
        for symbols with sparse (NaN) bars inside the window.
        """
        vol_lb = int(self.params.get("Volatility_Measuring_Lookback_Period", 60))
        eq_lb = int(self.params.get("Equity_Lookback_Period", 55))
        bond_lb = int(self.params.get("Bond_Lookback_Period", 110))
        need = max(132 + 2, vol_lb + 2, eq_lb + 1, bond_lb + 1)
        return int(self.params.get("max_window", 2 * need + 10))

    def latest_allocation(self, minute_df: pd.DataFrame) -> Optional[Allocation]:
        """Compute ONLY the allocation for the latest bar (live mode).

        Uses just the trailing ``required_window()`` bars, so the cost does not
        grow with the amount of history loaded.
        """
        if minute_df is None or minute_df.empty:
            return None
        window = minute_df.iloc[-self.required_window():]
        return self.generate_allocation(window)

    def generate_allocation(self, minute_df: pd.DataFrame) -> Allocation:
        self._update_emergency(minute_df)
        # Determine available symbols strictly from close prices
        close_all = _extract_close_frame(minute_df)
        available_syms = list(close_all.columns)
        if self.emergency_enabled and self.state["emergency"]:
            emerg_syms = [s for s in self.emergency_cfg.get("active", []) if s in available_syms]
            selected_universe = emerg_syms if emerg_syms else [s for s in self.universe_models if s in available_syms]
        else:
            selected_universe = [s for s in self.universe_models if s in available_syms] or available_syms
        close_frame = close_all[selected_universe]
        scores = self._compute_momentum(close_frame)
        scores = self._apply_special_filter(scores)
        k = int(self.params.get('Number_of_Portfolio_Allocations', 3))
        selected = list(scores.head(k).index)
        if not selected:
            logger.warning("allocation ts=%s: no eligible symbols (insufficient history); returning empty allocation", minute_df.index[-1])
            return Allocation(date=minute_df.index[-1], positions=[], emergency=(self.emergency_enabled and self.state["emergency"]))
        weights = self._weights(close_frame, selected)
        positions = []
        for s in selected:
            last_valid = close_frame[s].dropna()
            last_close = float(last_valid.iloc[-1]) if len(last_valid) else None
            positions.append(Position(symbol=s, weight=float(weights[s]), last_close=last_close))
        # Format weights as percentages for tidy logging
        weights_pct = {s: f"{float(weights[s])*100.0:.2f}%" for s in selected}
        logger.info("allocation ts=%s emergency=%s selected=%s weights=%s", minute_df.index[-1], self.emergency_enabled and self.state["emergency"], selected, weights_pct)
        return Allocation(date=minute_df.index[-1], positions=positions, emergency=(self.emergency_enabled and self.state["emergency"]))

class Backtester:
    """Walk-forward backtest over minute bars.

    P&L is realised WITHOUT look-ahead: at each rebalance bar t the previous
    allocation's weights earn (px_t / px_prev_rebalance - 1); only then are the
    new weights (chosen with data up to and including bar t) put on. An
    optional linear cost ``params['costs']['c2'] * turnover`` is charged at each
    rebalance, where turnover = sum |w_new - w_prev|.
    """

    def __init__(self):
        self.rebalance_returns: List[float] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []

    def run(self, strategy: Strategy, minute_df: pd.DataFrame, rebalancing: Optional[str] = None) -> List[Allocation]:
        allocs: List[Allocation] = []
        self.rebalance_returns = []
        self.equity_curve = []
        if minute_df is None or minute_df.empty:
            logger.warning("backtest: empty minute_df; nothing to do")
            return allocs
        method = rebalancing or strategy.params.get("Rebalancing_Method", "active")
        step = max(1, int(strategy.params.get('rebalance_minutes', strategy.rebalance_minutes)))
        # Minimum bars required equals the volatility lookback (bars), not days
        min_bars = int(strategy.params.get("Volatility_Measuring_Lookback_Period", 60))
        max_window = strategy.required_window()
        c2 = float(((strategy.params.get('costs') or {}).get('c2', 0.0)) or 0.0)
        logger.info("backtest start rows=%d cols=%d method=%s step=%d min_bars=%d max_window=%d c2=%g",
                    len(minute_df), len(minute_df.columns), method, step, min_bars, max_window, c2)
        # Last-known price per symbol at every bar (ffill only; computed once, O(N)).
        close_ff = _extract_close_frame(minute_df).ffill()
        index = minute_df.index
        equity = 100.0
        max_equity = 100.0
        prev_w = pd.Series(dtype=float)
        prev_px = pd.Series(dtype=float)
        for i in range(len(minute_df)):
            if i + 1 < max(3, min_bars):
                continue
            ts = index[i]
            do_rebalance = False
            if method == '1':
                do_rebalance = ts.isoweekday() == 5 and ts.hour == 16 and ts.minute == 0
            elif method == '2':
                next_ts = ts + pd.Timedelta(minutes=1)
                do_rebalance = next_ts.month != ts.month and ts.hour == 16 and ts.minute == 0
            elif method == 'active' or method == '3':
                do_rebalance = (i % step) == 0
            if not do_rebalance:
                continue
            # Only the trailing window is needed by the strategy (avoids O(N^2)).
            window = minute_df.iloc[max(0, i + 1 - max_window):i + 1]
            alloc = strategy.generate_allocation(window)
            try:
                px_now = close_ff.iloc[i]
                # 1) realise P&L of the PREVIOUS weights over (prev rebalance, t]
                if len(prev_w) > 0:
                    rets = (px_now.reindex(prev_w.index) / prev_px.reindex(prev_w.index)) - 1.0
                    rets = rets.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    port_ret = float((prev_w * rets).sum())
                else:
                    port_ret = 0.0
                # 2) switch to the new weights, paying costs on turnover
                new_w = pd.Series({p.symbol: float(p.weight) for p in alloc.positions}, dtype=float)
                all_syms = prev_w.index.union(new_w.index)
                turnover = float((new_w.reindex(all_syms, fill_value=0.0) - prev_w.reindex(all_syms, fill_value=0.0)).abs().sum())
                cost = c2 * turnover
                net_ret = port_ret - cost
                equity *= (1.0 + net_ret)
                max_equity = max(max_equity, equity)
                dd = (equity / max_equity) - 1.0
                cum = (equity / 100.0) - 1.0
                self.rebalance_returns.append(net_ret)
                self.equity_curve.append((ts, equity))
                prev_w = new_w
                prev_px = px_now.reindex(new_w.index)
                logger.info(
                    "ALLOCATION | ts=%s | emergency=%s\n  positions=%s\nSTATS      | ret(prev alloc)=%+.4f%% | cost=%.4f%% | turnover=%.4f | cum=%+.2f%% | equity=$%.2f | max_eq=$%.2f | drawdown=%+.4f%%",
                    ts,
                    strategy.emergency_enabled and strategy.state.get("emergency"),
                    [(p.symbol, f"{p.weight*100.0:.2f}%") for p in alloc.positions],
                    port_ret * 100.0,
                    cost * 100.0,
                    turnover,
                    cum * 100.0,
                    equity,
                    max_equity,
                    dd * 100.0,
                )
            except Exception as e:
                logger.warning("stats error ts=%s err=%s", ts, e)
            allocs.append(alloc)
        logger.info("backtest done allocations=%d", len(allocs))
        logger.info("final equity=$%.2f max_equity=$%.2f", equity, max_equity)
        return allocs

# simple dict to carry running stats in logs
logger_extra = {}
