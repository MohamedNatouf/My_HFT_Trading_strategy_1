# Strategy engine implementing cross-sectional momentum on minute bars with parameter-driven logic
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    HAS_PYPOPT = True
except Exception:
    HAS_PYPOPT = False

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
        # Allow direct minute-based rebalancing override
        self.rebalance_minutes = int(self.params.get("rebalance_minutes", 5))

    # --- Old model metric equivalents on minute data with windows mapped from days to minutes ---
    def _days(self, d: int) -> int:
        return max(1, d * self.minutes_per_day)

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
        return Strategy._high_low_diff_ratio_static(series, self._days(22), self._days(132), self._days(22))

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
        return Strategy._linear_regression_ratio_static(series, self._days(11), self._days(132), self._days(11))

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
        return Strategy._stochastic_oscillator_static(series, self._days(22), self._days(132), self._days(22))

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
        return Strategy._price_percent_rank_metric_static(series, self._days(4), self._days(110), self._days(2))

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
        return Strategy._rsi_index_static(series, self._days(11), self._days(132), self._days(11))

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
        lookback_minutes = self._days(lookback)
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
        # Example: filter by subClass to keep top within each group
        # Integrate full metadata mapping when available
        return rets

    def _compute_momentum(self, df: pd.DataFrame) -> pd.Series:
        lookback_days = int(self.params.get("Volatility_Measuring_Lookback_Period", 60))
        close = df["Close"]
        method = self.params.get("Selected_method_of_Momuntum_Metric", "Metrics Combind")
        return self.Momentum_MultiFactor_Maximizer(close, lookback_days, method)

    def _weights(self, df: pd.DataFrame, selected_symbols: List[str]) -> pd.Series:
        method = self.params.get("Method_Of_Wieghting", "Assets Inverse Volatility")
        lookback_days = int(self.params.get("Volatility_Measuring_Lookback_Period", 60))
        lookback = self._days(lookback_days)
        vol_strength = float(self.params.get("Volatility_Weight_Strength", 1))
        vol = df["Close"][selected_symbols].pct_change().tail(lookback).std()
        if method == "Assets Inverse Volatility":
            w = 1/vol.replace(0, np.nan)
            w = w / w.sum()
        elif method == "Assets Volume":
            w = pd.Series(1.0, index=selected_symbols)
            w = w / w.sum()
        elif method == "Volume & Inverse Volume":
            w_vol = 1/vol.replace(0, np.nan)
            w_eq = pd.Series(1.0, index=selected_symbols)
            w = vol_strength*(w_vol/w_vol.sum()) + (1-vol_strength)*(w_eq/w_eq.sum())
            w = w / w.sum()
        elif method == "Efficient Frontier Optimization" and HAS_PYPOPT:
            prices = df["Close"][selected_symbols]
            mu = expected_returns.mean_historical_return(prices, frequency=self.minutes_per_day)
            S = risk_models.sample_cov(prices, frequency=self.minutes_per_day)
            ef = EfficientFrontier(mu, S)
            w_dict = ef.max_sharpe()
            w = pd.Series(w_dict)
            w = w / w.sum()
        else:
            w = pd.Series(1.0, index=selected_symbols)
            w = w / w.sum()
        return w.fillna(0)

    def _update_emergency(self, df: pd.DataFrame):
        eq_lb_days = int(self.params.get("Equity_Lookback_Period", 55))
        bond_lb_days = int(self.params.get("Bond_Lookback_Period", 110))
        eq_wait_req = int(self.params.get("Equity_Signal_Wait_Period", 3))
        bond_wait_req = int(self.params.get("Bond_Signal_Wait_Period", 1))
        hold_status = bool(self.params.get("Equity_N_Bond_Hold_Period_Status", True))
        hold_period = int(self.params.get("Equity_N_Bond_Hold_Period", 3))
        close = df["Close"]
        model_symbols = list(close.columns)
        eq_lb = self._days(eq_lb_days)
        model_ret = close[model_symbols].pct_change(eq_lb).iloc[-1].mean()
        risk_off = model_ret < 0
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

    def generate_allocation(self, minute_df: pd.DataFrame) -> Allocation:
        self._update_emergency(minute_df)
        scores = self._compute_momentum(minute_df)
        scores = self._apply_special_filter(scores)
        k = int(self.params.get('Number_of_Portfolio_Allocations', 3))
        selected = list(scores.head(k).index)
        weights = self._weights(minute_df, selected)
        positions = [Position(symbol=s, weight=float(weights[s])) for s in selected]
        return Allocation(date=minute_df.index[-1], positions=positions, emergency=self.state["emergency"])

class Backtester:
    def run(self, strategy: Strategy, minute_df: pd.DataFrame, rebalancing: Optional[str] = None) -> List[Allocation]:
        allocs: List[Allocation] = []
        method = rebalancing or strategy.params.get("Rebalancing_Method", "active")
        step = int(strategy.params.get('rebalance_minutes', strategy.rebalance_minutes))
        for i in range(len(minute_df)):
            window = minute_df.iloc[:i+1]
            if len(window) < max(strategy._days(3), strategy._days(int(strategy.params.get("Volatility_Measuring_Lookback_Period",60)))):
                continue
            ts = window.index[-1]
            do_rebalance = False
            if method == '1':
                # weekly at Friday close (16:00)
                do_rebalance = ts.isoweekday() == 5 and ts.hour == 16 and ts.minute == 0
            elif method == '2':
                # monthly at month end close (16:00)
                next_ts = ts + pd.Timedelta(minutes=1)
                do_rebalance = next_ts.month != ts.month and ts.hour == 16 and ts.minute == 0
            elif method == 'active' or method == '3':
                # frequent rebalancing every N minutes
                do_rebalance = (i % step) == 0
            if not do_rebalance:
                continue
            allocs.append(strategy.generate_allocation(window))
        return allocs
