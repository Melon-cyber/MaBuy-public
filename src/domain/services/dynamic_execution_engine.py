# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

@dataclass(frozen=True)
class ExecutionTargetParameters:
    symbol: str
    direction: str
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float
    effective_volatility: float
    garman_klass_vol: float
    parkinson_vol: float
    natr_vol: float
    alpha_confluence_score: float
    initial_sl_distance_usd: float
    initial_tp_distance_usd: float

@dataclass(frozen=True)
class TrailingUpdateResult:
    current_price: float
    peak_price: float
    updated_stop_loss: float
    take_profit_price: float
    progress_phi: float
    momentum_score: float
    efficiency_ratio: float
    lambda_trail_multiplier: float
    delta_trail_usd: float
    is_break_even_active: bool
    is_profit_lock_active: bool
    is_stopped_out: bool
    is_tp_hit: bool
    unrealized_pnl_pct: float

class DynamicParametricExecutionEngine:
    def __init__(
        self,
        min_rr: float = 1.80,
        max_rr: float = 3.20,
        lambda_trail_base: float = 1.20,
        mu_momentum: float = 0.50,
        nu_progress: float = 0.50,
        be_progress_threshold: float = 0.35,
        lock_progress_threshold: float = 0.70,
        fee_slippage_buffer_pct: float = 0.0012,
    ):
        self.min_rr = min_rr
        self.max_rr = max_rr
        self.lambda_trail_base = lambda_trail_base
        self.mu_momentum = mu_momentum
        self.nu_progress = nu_progress
        self.be_progress_threshold = be_progress_threshold
        self.lock_progress_threshold = lock_progress_threshold
        self.fee_slippage_buffer_pct = fee_slippage_buffer_pct

    @staticmethod
    def calculate_parkinson_volatility(highs: np.ndarray, lows: np.ndarray) -> float:
        n = len(highs)
        if n < 2:
            return 0.015
        log_hl = np.log(np.maximum(highs, 1e-9) / np.maximum(lows, 1e-9))
        sigma_p2 = (1.0 / (4.0 * np.log(2.0) * n)) * np.sum(log_hl**2)
        return float(np.sqrt(max(1e-8, sigma_p2)))

    @staticmethod
    def calculate_garman_klass_volatility(
        opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> float:
        n = len(highs)
        if n < 2:
            return 0.015
        log_hl = np.log(np.maximum(highs, 1e-9) / np.maximum(lows, 1e-9))
        log_co = np.log(np.maximum(closes, 1e-9) / np.maximum(opens, 1e-9))
        const_term = 2.0 * np.log(2.0) - 1.0
        term = 0.5 * (log_hl**2) - const_term * (log_co**2)
        sigma_gk2 = (1.0 / n) * np.sum(term)
        return float(np.sqrt(max(1e-8, sigma_gk2)))

    @staticmethod
    def calculate_natr(
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
    ) -> float:
        n = len(highs)
        if n < 2:
            return 0.015
        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        tr = np.maximum.reduce([tr1, tr2, tr3])
        atr = float(np.mean(tr[-period:])) if len(tr) >= period else float(np.mean(tr))
        curr_price = float(closes[-1])
        return float(atr / curr_price) if curr_price > 0 else 0.015

    def calculate_composite_effective_volatility(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        chop_index: float = 38.2,
    ) -> Tuple[float, float, float, float]:
        sig_p = self.calculate_parkinson_volatility(highs, lows)
        sig_gk = self.calculate_garman_klass_volatility(opens, highs, lows, closes)
        sig_natr = self.calculate_natr(highs, lows, closes)

        sig_weighted = 0.50 * sig_gk + 0.20 * sig_p + 0.30 * sig_natr
        sig_max_envelope = max(sig_gk, sig_p, sig_natr)
        sig_base = 0.70 * sig_weighted + 0.30 * sig_max_envelope

        chop_penalty = 1.0 + 0.20 * float(np.clip((chop_index - 38.2) / 23.6, 0.0, 1.0))
        sig_eff = float(np.clip(sig_base * chop_penalty, 0.003, 0.12))
        return sig_eff, sig_gk, sig_p, sig_natr

    def calculate_dynamic_risk_reward(
        self, model_prob_win: float, ofi_aligned: float
    ) -> Tuple[float, float]:
        s_ml = float(np.clip((model_prob_win - 0.50) / 0.45, 0.0, 1.0))
        s_ofi = float(np.clip((ofi_aligned - 0.00) / 0.75, 0.0, 1.0))

        s_alpha = 0.50 * s_ml + 0.30 * s_ofi + 0.20 * (s_ml * s_ofi)
        s_alpha = float(np.clip(s_alpha, 0.0, 1.0))

        g_s = 3.0 * (s_alpha**2) - 2.0 * (s_alpha**3)
        rr = self.min_rr + (self.max_rr - self.min_rr) * g_s
        return float(np.clip(rr, self.min_rr, self.max_rr)), s_alpha

    def calculate_entry_targets(
        self,
        symbol: str,
        direction: str,
        current_price: float,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        model_prob_win: float,
        raw_ofi: float = 0.0,
        market_regime: str = 'NORMAL',
        chop_index: float = 38.2,
        orderbook_wall_price: Optional[float] = None,
    ) -> ExecutionTargetParameters:
        is_long = direction.upper() in ['BULLISH', 'LONG', 'BUY']
        ofi_aligned = raw_ofi if is_long else -raw_ofi

        sig_eff, sig_gk, sig_p, sig_natr = self.calculate_composite_effective_volatility(
            opens, highs, lows, closes, chop_index=chop_index
        )

        if market_regime in ['BULL_TREND', 'BEAR_TREND']:
            k_sl = 1.35
        elif market_regime == 'HIGH_VOLATILITY':
            k_sl = 1.80
        else:
            k_sl = 1.50

        sl_distance = current_price * sig_eff * k_sl
        rr, s_alpha = self.calculate_dynamic_risk_reward(model_prob_win, ofi_aligned)
        tp_distance = rr * sl_distance

        if is_long:
            sl_price = current_price - sl_distance
            tp_price = current_price + tp_distance
            if orderbook_wall_price is not None and orderbook_wall_price > (current_price + 1.2 * sl_distance):
                tp_price = min(tp_price, orderbook_wall_price * 0.9995)
        else:
            sl_price = current_price + sl_distance
            tp_price = current_price - tp_distance
            if orderbook_wall_price is not None and orderbook_wall_price < (current_price - 1.2 * sl_distance):
                tp_price = max(tp_price, orderbook_wall_price * 1.0005)

        return ExecutionTargetParameters(
            symbol=symbol.upper(),
            direction='BULLISH' if is_long else 'BEARISH',
            entry_price=round(current_price, 4),
            stop_loss_price=round(sl_price, 4),
            take_profit_price=round(tp_price, 4),
            risk_reward_ratio=round(rr, 2),
            effective_volatility=round(sig_eff, 5),
            garman_klass_vol=round(sig_gk, 5),
            parkinson_vol=round(sig_p, 5),
            natr_vol=round(sig_natr, 5),
            alpha_confluence_score=round(s_alpha, 4),
            initial_sl_distance_usd=round(sl_distance, 4),
            initial_tp_distance_usd=round(tp_distance, 4),
        )

    @staticmethod
    def calculate_efficiency_ratio(prices: np.ndarray, period: int = 10) -> float:
        if len(prices) < period + 1:
            return 0.50
        net_change = abs(prices[-1] - prices[-period - 1])
        sum_changes = np.sum(np.abs(np.diff(prices[-period - 1 :])))
        if sum_changes <= 1e-9:
            return 0.50
        return float(np.clip(net_change / sum_changes, 0.0, 1.0))

    @staticmethod
    def calculate_normalized_velocity(prices: np.ndarray, sigma_eff: float, k: int = 5) -> float:
        if len(prices) < k + 1 or sigma_eff <= 1e-9:
            return 0.0
        ret = (prices[-1] - prices[-k - 1]) / prices[-k - 1]
        denom = np.sqrt(k) * sigma_eff
        return float(ret / denom)

    def update_trailing_stop(
        self,
        direction: str,
        entry_price: float,
        current_price: float,
        peak_price: float,
        initial_sl_price: float,
        previous_sl_price: float,
        tp_price: float,
        sigma_eff: float,
        recent_price_series: np.ndarray,
    ) -> TrailingUpdateResult:
        is_long = direction.upper() in ['BULLISH', 'LONG', 'BUY']
        dir_sign = 1.0 if is_long else -1.0

        if is_long:
            updated_peak = max(peak_price, current_price)
            peak_progress = max(0.0, updated_peak - entry_price)
        else:
            updated_peak = min(peak_price, current_price)
            peak_progress = max(0.0, entry_price - updated_peak)

        tp_distance = abs(tp_price - entry_price)
        phi = float(np.clip(peak_progress / max(1e-9, tp_distance), 0.0, 5.0)) if tp_distance > 0 else 0.0

        er = self.calculate_efficiency_ratio(recent_price_series, period=10)
        vel = self.calculate_normalized_velocity(recent_price_series, sigma_eff, k=5)
        aligned_vel = vel * dir_sign
        vel_norm = float(np.clip(aligned_vel / 3.0, 0.0, 1.0))
        momentum_score = 0.60 * er + 0.40 * vel_norm

        lambda_trail = (
            self.lambda_trail_base
            * (1.0 + self.mu_momentum * momentum_score)
            * (1.0 - self.nu_progress * min(1.0, phi))
        )
        delta_trail = updated_peak * sigma_eff * lambda_trail

        sl_candidates = [initial_sl_price, previous_sl_price]

        is_be_active = phi >= self.be_progress_threshold
        if is_be_active:
            be_price = entry_price * (1.0 + dir_sign * self.fee_slippage_buffer_pct)
            sl_candidates.append(be_price)
            trail_candidate = (updated_peak - delta_trail) if is_long else (updated_peak + delta_trail)
            sl_candidates.append(trail_candidate)

        is_lock_active = phi >= self.lock_progress_threshold
        if is_lock_active:
            lock_price = entry_price + dir_sign * (0.50 * peak_progress)
            sl_candidates.append(lock_price)

        if is_long:
            updated_sl = max(sl_candidates)
            is_stopped_out = current_price <= updated_sl
            is_tp_hit = current_price >= tp_price
            pnl_pct = (current_price - entry_price) / entry_price * 100.0
        else:
            updated_sl = min(sl_candidates)
            is_stopped_out = current_price >= updated_sl
            is_tp_hit = current_price <= tp_price
            pnl_pct = (entry_price - current_price) / entry_price * 100.0

        return TrailingUpdateResult(
            current_price=round(current_price, 4),
            peak_price=round(updated_peak, 4),
            updated_stop_loss=round(updated_sl, 4),
            take_profit_price=round(tp_price, 4),
            progress_phi=round(phi, 3),
            momentum_score=round(momentum_score, 3),
            efficiency_ratio=round(er, 3),
            lambda_trail_multiplier=round(lambda_trail, 3),
            delta_trail_usd=round(delta_trail, 2),
            is_break_even_active=is_be_active,
            is_profit_lock_active=is_lock_active,
            is_stopped_out=is_stopped_out,
            is_tp_hit=is_tp_hit,
            unrealized_pnl_pct=round(pnl_pct, 2),
        )
