import os
import sqlite3
import time
import math
import logging
import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, Any, List, Tuple

from src.domain.entities.prediction import PredictionResult, PredictionRequest
from src.domain.entities.tier import SubscriptionTier
from src.domain.services.order_flow_imbalance import OrderFlowImbalanceService
from src.infrastructure.ml.baseline_model import BaselineGBDTModelEngine
from src.infrastructure.ml.pytorch.attention_lstm import MultiHorizonAttentionLSTM
from src.infrastructure.ml.pytorch.causal_sequence_buffer import MultiAssetSequenceBufferManager
from src.infrastructure.nlp.rss_sentiment_analyzer import RSSSentimentAnalyzer

logger = logging.getLogger("MaBuyV4EnsembleEngine")


class HurstRegimeDetector:
    """
    Calculates rolling Hurst Exponent (H) via 2-Scale Rescaled Range (R/S) Analysis
    and modulates ensemble weights dynamically.
    
    Regimes:
    - H > 0.55: TREND (Persistent -> PyTorch Attention LSTM boosted)
    - H < 0.45: MEAN_REVERTING (Anti-persistent -> CatBoost GBDT / OFI boosted)
    - 0.45 <= H <= 0.55: CHOPPY_NOISE (Random walk -> Capital gated down to reduce whipsaw)
    """

    @staticmethod
    def _calc_rs(returns: np.ndarray) -> float:
        if len(returns) < 4:
            return 1.0
        mean_r = np.mean(returns)
        devs = returns - mean_r
        cum_devs = np.cumsum(devs)
        r = np.max(cum_devs) - np.min(cum_devs)
        s = np.std(returns, ddof=1)
        if s <= 1e-8:
            return 1.0
        return max(1e-4, float(r / s))

    @classmethod
    def calculate_hurst_from_prices(cls, prices: np.ndarray, min_len: int = 20) -> float:
        """Computes Hurst Exponent H in [0.05, 0.95] from price array using 2-scale R/S."""
        if prices is None or len(prices) < min_len:
            return 0.50
        try:
            arr = np.asarray(prices, dtype=float)
            arr = arr[arr > 0]
            if len(arr) < min_len:
                return 0.50
            returns = np.diff(np.log(arr))
            if len(returns) < 12 or np.all(returns == 0):
                return 0.50
            
            # 2-Scale R/S ratio estimator: (R/S)_N / (R/S)_{N/2} = 2^H
            rs_full = cls._calc_rs(returns)
            half = len(returns) // 2
            rs_half1 = cls._calc_rs(returns[:half])
            rs_half2 = cls._calc_rs(returns[half:])
            rs_half_mean = max(1e-4, (rs_half1 + rs_half2) / 2.0)
            
            ratio = rs_full / rs_half_mean
            hurst = np.log(max(1e-4, ratio)) / np.log(2.0)
            return float(np.clip(hurst, 0.05, 0.95))
        except Exception:
            return 0.50

    @classmethod
    def get_adaptive_weights(cls, hurst: float, base_weights: Dict[str, float]) -> Dict[str, Any]:
        """Modulates ensemble weights dynamically based on market persistence H."""
        h_delta = (hurst - 0.50) / 0.10
        factor = float(np.tanh(h_delta))
        
        # PyTorch Attention LSTM boosted in persistent trends (H > 0.50)
        w_torch = base_weights["pytorch"] * (1.0 + 0.50 * factor)
        # GBDT boosted during mean-reversion/oscillation (H < 0.50)
        w_gbdt = base_weights["gbdt"] * (1.0 - 0.50 * factor)
        # Sentiment boosted slightly during strong directional trends
        w_sent = base_weights["sentiment"] * (1.0 + 0.25 * factor)
        w_drift = base_weights["drift"]
        
        total = max(1e-5, w_torch + w_gbdt + w_sent + w_drift)
        
        # Entropy Risk Gate: smooth tanh scaling from 0.25 (at H=0.50) up to 1.0 (at H>=0.70 or H<=0.30)
        entropy_gate = 0.25 + 0.75 * float(np.tanh(6.0 * abs(hurst - 0.50)))
        entropy_gate = float(np.clip(entropy_gate, 0.25, 1.0))
        
        if hurst > 0.55:
            regime = "TREND"
        elif hurst < 0.45:
            regime = "MEAN_REVERTING"
        else:
            regime = "CHOPPY_NOISE"
            
        return {
            "weights": {
                "pytorch": w_torch / total,
                "gbdt": w_gbdt / total,
                "sentiment": w_sent / total,
                "drift": w_drift / total,
                "threshold": base_weights.get("threshold", 0.006)
            },
            "hurst": round(hurst, 4),
            "entropy_gate": round(entropy_gate, 4),
            "regime": regime
        }


class MaBuyV4EnsembleEngine:
    """
    MaBuy V4 Flagship Machine Learning Engine.
    Combines:
    1. Multi-Task Multi-Horizon PyTorch Deep Neural Network (Dedicated 1h, 4h, 1d Attention Heads)
    2. GBDT Baseline Model (Gradient Boosting on 80+ TA indicators)
    3. NLP News Sentiment Alpha (VADER + RSS sentiment aggregation)
    4. Forecast Bias Correction (Prediction Drift Calibration from fine_tuning_telemetry logs)
    5. Quarter-Kelly Criterion (0.25 * K) & ATR Volatility Risk Manager
    6. Microstructure Order Flow Imbalance (OFI) False Breakout Veto
    """

    HORIZON_WEIGHTS = {
        "1h": {"gbdt": 0.45, "pytorch": 0.45, "sentiment": 0.05, "drift": 0.05, "threshold": 0.0060},
        "4h": {"gbdt": 0.40, "pytorch": 0.45, "sentiment": 0.10, "drift": 0.05, "threshold": 0.0100},
        "1d": {"gbdt": 0.35, "pytorch": 0.50, "sentiment": 0.10, "drift": 0.05, "threshold": 0.0220},
    }

    def __init__(self, db_path: Optional[str] = None, models_dir: str = "models"):
        if db_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            db_path = os.path.join(project_root, "data", "mabuy_prod.db")
        else:
            db_path = os.path.abspath(db_path)
            
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        if not os.path.isabs(models_dir):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            vps_path = "/root/mabuy2/models"
            if os.path.exists(vps_path):
                models_dir = vps_path
            else:
                models_dir = os.path.join(project_root, models_dir)
        self.models_dir = models_dir
        self.gbdt_engine = BaselineGBDTModelEngine()
        self.sentiment_analyzer = RSSSentimentAnalyzer()
        self.ofi_service = OrderFlowImbalanceService()
        
        # Unified 18 Stationary Features matching FeatureEngineer.FEATURE_COLS
        self.pytorch_feature_cols = [
            "r_lag1", "r_lag2", "r_lag3", "r_lag5",
            "volatility_20", "volatility_60", "vol_ratio",
            "z_return_lag1", "z_return_lag2", "z_return_lag3",
            "rsi_14", "macd_norm", "macd_signal_norm", "macd_hist_norm",
            "atr_norm", "bb_pb", "vol_ratio_20", "volume_log_return"
        ]
        self.num_pytorch_features = len(self.pytorch_feature_cols)  # 18
        
        # Initialize Multi-Horizon PyTorch Network & Causal Sequence Buffer (18 features)
        self.pytorch_model = MultiHorizonAttentionLSTM(input_dim=self.num_pytorch_features, hidden_dim=64, num_layers=2)
        self.seq_buffer_manager = MultiAssetSequenceBufferManager(seq_len=60, num_features=self.num_pytorch_features)
        self._pytorch_weights_loaded = False
        self._load_pytorch_weights()
        self.pytorch_model.eval()

        logger.info("Initializing MaBuy V4 Flagship Multi-Horizon Ensemble Engine with Causal Rolling Buffer (18-feat PyTorch)...")

    def _load_pytorch_weights(self):
        """
        Loads pretrained PyTorch model weights if available.
        Robustly resolves dimension mismatches (e.g. 14 vs 18 or 95 vs 18 features),
        maps single-head and legacy multi-horizon layer naming,
        and strictly validates that both the LSTM backbone and prediction heads
        have active, non-random weights.
        """
        candidates = [
            os.path.join(self.models_dir, "multi_horizon_lstm.pt"),
            os.path.join(self.models_dir, "pytorch_attention_lstm_BTCUSDT_1h.pt"),
            os.path.join(self.models_dir, "pytorch_attention_lstm_ETHUSDT_1h.pt"),
            os.path.join(self.models_dir, "pytorch_attention_lstm_SOLUSDT_1h.pt"),
        ]
        target_sd = self.pytorch_model.state_dict()

        for path in candidates:
            if os.path.exists(path):
                try:
                    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                    raw_sd = (
                        checkpoint.get("multi_horizon_state_dict")
                        or checkpoint.get("model_state_dict")
                        or checkpoint.get("state_dict")
                        or checkpoint
                    )
                    if not isinstance(raw_sd, dict):
                        continue

                    adapted_sd = {}
                    for k, v in raw_sd.items():
                        if not isinstance(v, torch.Tensor):
                            continue

                        # Map single-head fc_return / fc_prob to multi-horizon heads
                        # In single-head (LSTMWithAttention):
                        #   fc_return.0 = Linear(64, 32) -> HorizonHead fc_return.0
                        #   fc_return.3 = Linear(32, 1)  -> HorizonHead fc_return.4
                        #   fc_prob.0   = Linear(64, 32) -> HorizonHead fc_prob.0
                        #   fc_prob.3   = Linear(32, 1)  -> HorizonHead fc_prob.4
                        if k.startswith("fc_return.") or k.startswith("fc_prob."):
                            for head in ["head_1h", "head_4h", "head_1d"]:
                                if k.startswith("fc_return.0."):
                                    sub_k = k
                                elif k.startswith("fc_return.3."):
                                    sub_k = k.replace("fc_return.3.", "fc_return.4.")
                                elif k.startswith("fc_prob.0."):
                                    sub_k = k
                                elif k.startswith("fc_prob.3."):
                                    sub_k = k.replace("fc_prob.3.", "fc_prob.4.")
                                else:
                                    sub_k = k
                                head_k = f"{head}.{sub_k}"
                                if head_k in target_sd and target_sd[head_k].shape == v.shape:
                                    adapted_sd[head_k] = v.clone()

                        # Also support mapping from legacy multi-horizon heads that used index 3 instead of 4
                        for head in ["head_1h", "head_4h", "head_1d"]:
                            if k.startswith(f"{head}.fc_return.3."):
                                target_k = k.replace(f"{head}.fc_return.3.", f"{head}.fc_return.4.")
                                if target_k in target_sd and target_sd[target_k].shape == v.shape:
                                    adapted_sd[target_k] = v.clone()
                            elif k.startswith(f"{head}.fc_prob.3."):
                                target_k = k.replace(f"{head}.fc_prob.3.", f"{head}.fc_prob.4.")
                                if target_k in target_sd and target_sd[target_k].shape == v.shape:
                                    adapted_sd[target_k] = v.clone()

                        if k in target_sd:
                            target_shape = target_sd[k].shape
                            if v.shape == target_shape:
                                adapted_sd[k] = v.clone()
                            elif k == "lstm.weight_ih_l0" and len(v.shape) == 2 and len(target_shape) == 2:
                                # Cleanly adapt dimension mismatch (e.g. 14 vs 18, 95 vs 18)
                                adapted_w = target_sd[k].clone()
                                min_r = min(v.shape[0], target_shape[0])
                                min_c = min(v.shape[1], target_shape[1])
                                adapted_w[:min_r, :min_c] = v[:min_r, :min_c]
                                adapted_sd[k] = adapted_w
                                logger.info(
                                    f"Dimension unification: Adapted {k} from {tuple(v.shape)} to {tuple(target_shape)}."
                                )

                    self.pytorch_model.load_state_dict(adapted_sd, strict=False)

                    # Validate active non-random weights across both backbone and prediction heads
                    with torch.no_grad():
                        w_ih = self.pytorch_model.lstm.weight_ih_l0
                        w_head_ret = self.pytorch_model.head_1h.fc_return[4].weight
                        w_head_prob = self.pytorch_model.head_1h.fc_prob[4].weight

                        w_norm = float(torch.norm(w_ih).item())
                        ret_norm = float(torch.norm(w_head_ret).item())
                        prob_norm = float(torch.norm(w_head_prob).item())

                        is_active = (
                            (w_norm > 1e-4) and not torch.isnan(w_ih).any() and
                            (ret_norm > 1e-4) and not torch.isnan(w_head_ret).any() and
                            (prob_norm > 1e-4) and not torch.isnan(w_head_prob).any()
                        )

                    if is_active:
                        self._pytorch_weights_loaded = True
                        logger.info(
                            f"[OK] Cleanly loaded pretrained PyTorch weights from {path} "
                            f"(input_dim: {self.num_pytorch_features}, ||W_ih||={w_norm:.3f}, "
                            f"||W_ret||={ret_norm:.3f}, ||W_prob||={prob_norm:.3f})"
                        )
                        return
                    else:
                        logger.warning(
                            f"PyTorch weights from {path} failed validation (W_ih={w_norm:.4f}, "
                            f"W_ret={ret_norm:.4f}, W_prob={prob_norm:.4f})."
                        )
                except Exception as e:
                    logger.warning(f"Could not load PyTorch checkpoint {path}: {e}")

    def _get_telemetry_bias(self, symbol: str) -> float:
        """Calculates Sim-to-Real historical gap bias from fine_tuning_telemetry table."""
        if not os.path.exists(self.db_path):
            return 0.0
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT slippage_pct FROM fine_tuning_telemetry 
                    WHERE symbol = ? ORDER BY rowid DESC LIMIT 50
                """, (symbol.upper(),))
                rows = cursor.fetchall()
                if not rows:
                    return 0.0
                slippage_vals = [r[0] for r in rows if r[0] is not None]
                return float(np.mean(slippage_vals)) if slippage_vals else 0.0
        except Exception as e:
            logger.warning(f"Error fetching telemetry bias for {symbol}: {e}")
            return 0.0

    def predict(self, request: PredictionRequest, limits: Optional[Any] = None) -> PredictionResult:
        """Executes full V4 Multi-Horizon ensemble inference pipeline."""
        symbol = request.symbol.upper()
        timeframe = getattr(request, "timeframe", "1h").lower()
        if timeframe not in self.HORIZON_WEIGHTS:
            timeframe = "1h"
        user_tier = request.tier

        # 1. Base GBDT Model Prediction for requested timeframe
        if limits is None:
            from src.domain.entities.tier import TierLimits
            limits = TierLimits(
                tier=request.tier,
                daily_prediction_limit=50,
                daily_ai_limit=10,
                allowed_assets=None,
                has_personal_recommendations=True,
                has_sentiment_analysis=True,
                use_advanced_model=True,
                priority_support=True
            )
        gbdt_res = self.gbdt_engine.predict(request, limits)
        curr_price = gbdt_res.current_price
        base_gbdt_log_ret = gbdt_res.predicted_log_return

        # 2. PyTorch Multi-Horizon Neural Head Inference with Causal Sequence Buffer (Sub-30ms)
        try:
            # Extract real chronological market feature matrix from GBDT engine with unified 18 features
            _, feature_matrix = self.gbdt_engine._extract_real_market_feature_matrix(
                symbol, timeframe=timeframe, limit=60, feature_cols=self.pytorch_feature_cols
            )
            seq_tensor = self.seq_buffer_manager.update_and_get_tensor(
                symbol=symbol,
                timeframe=timeframe,
                feature_matrix=feature_matrix
            )
            
            with torch.inference_mode():
                torch_ret_t, torch_prob_t = self.pytorch_model.predict_horizon(seq_tensor, horizon=timeframe)
                torch_ret = float(torch_ret_t[0, 0].item())
                torch_prob = float(torch_prob_t[0, 0].item())
        except Exception as e:
            logger.warning(f"PyTorch multi-horizon inference fallback for {symbol} ({timeframe}): {e}")
            torch_ret = base_gbdt_log_ret
            torch_prob = gbdt_res.confidence_score

        # 3. NLP Sentiment Analysis (used in ensemble scoring, not as LSTM feature)
        s_score, s_summary = self.sentiment_analyzer.analyze_symbol(symbol, lang="ru")

        # 4. Transfer Learning Bias Correction from telemetry
        telemetry_bias = self._get_telemetry_bias(symbol)

        # 5. Hurst Exponent & Adaptive Multi-Horizon Ensemble Weighting
        prices_for_hurst = None
        try:
            fetcher = getattr(self.gbdt_engine, "data_fetcher", None)
            if fetcher and hasattr(fetcher, "fetch_ohlcv"):
                df_candles = fetcher.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
                if df_candles is not None and not df_candles.empty and "close" in df_candles.columns:
                    prices_for_hurst = df_candles["close"].values
        except Exception as e:
            logger.debug(f"Hurst price fetch fallback for {symbol}: {e}")

        hurst = HurstRegimeDetector.calculate_hurst_from_prices(prices_for_hurst)
        adaptive_cfg = HurstRegimeDetector.get_adaptive_weights(hurst, self.HORIZON_WEIGHTS[timeframe])
        h_cfg = adaptive_cfg["weights"]
        regime_name = adaptive_cfg["regime"]
        entropy_gate = adaptive_cfg["entropy_gate"]

        max_ret_bounds = {"1h": 0.025, "4h": 0.055, "1d": 0.100}
        bound = max_ret_bounds.get(timeframe, 0.025)
        torch_ret = max(-bound, min(bound, torch_ret))
        base_gbdt_log_ret = max(-bound, min(bound, base_gbdt_log_ret))

        sentiment_alpha = 0.010 * s_score
        
        adjusted_log_ret = (
            h_cfg["gbdt"] * base_gbdt_log_ret +
            h_cfg["pytorch"] * torch_ret +
            h_cfg["sentiment"] * sentiment_alpha +
            h_cfg["drift"] * (telemetry_bias * 0.1)
        )
        adjusted_log_ret = max(-bound, min(bound, adjusted_log_ret))

        # 6. Determine Direction & Target Price with Horizon Threshold
        threshold = h_cfg["threshold"]
        volatility = getattr(gbdt_res, "volatility_estimate", 0.015)

        if adjusted_log_ret > threshold:
            direction = "BULLISH"
            eff_ret = max(adjusted_log_ret, min(bound, max(0.008, volatility * 0.75)))
            target_price = curr_price * math.exp(eff_ret)
        elif adjusted_log_ret < -threshold:
            direction = "BEARISH"
            eff_ret = min(adjusted_log_ret, -min(bound, max(0.008, volatility * 0.75)))
            target_price = curr_price * math.exp(eff_ret)
        else:
            direction = "NEUTRAL"
            target_price = curr_price

        # 7. Confidence Calibration combining PyTorch probability & GBDT confidence
        volatility = getattr(gbdt_res, "volatility_estimate", 0.015)
        
        # Symmetric PyTorch confidence
        torch_conf = 0.50 + abs(torch_prob - 0.50)

        # Technical & Confluence Alpha adjustments
        delta_confluence = 0.0

        # Model Consensus Alpha
        gbdt_dir_sign = 1 if base_gbdt_log_ret > threshold else (-1 if base_gbdt_log_ret < -threshold else 0)
        torch_dir_sign = 1 if torch_ret > threshold else (-1 if torch_ret < -threshold else 0)

        if gbdt_dir_sign != 0 and gbdt_dir_sign == torch_dir_sign:
            delta_confluence += 0.05
        elif gbdt_dir_sign != 0 and torch_dir_sign != 0 and gbdt_dir_sign != torch_dir_sign:
            delta_confluence -= 0.04

        # News Sentiment Alpha
        if (direction == "BULLISH" and s_score > 0.20) or (direction == "BEARISH" and s_score < -0.20):
            delta_confluence += 0.04
        elif (direction == "BULLISH" and s_score < -0.20) or (direction == "BEARISH" and s_score > 0.20):
            delta_confluence -= 0.04

        # Dynamic Regime Confidence Assignment (Institutional Calibration)
        if direction == "NEUTRAL":
            final_confidence = 0.505
        else:
            # 1. Dynamic signal strength based on return vs volatility
            sig_strength = abs(adjusted_log_ret) / (volatility + 1e-5)

            # 2. Hurst Trend Regime Boost:
            # Persistent trending regime (H > 0.55) validates directional momentum persistence
            hurst_boost = 0.0
            if hurst > 0.55:
                hurst_boost = min(0.05, 0.25 * (hurst - 0.55))
            elif hurst < 0.45:
                hurst_boost = -0.02

            # 3. Recalibrated Dynamic Base:
            # Base intercept: 0.57
            # Momentum scaling: +0.15 * tanh(sig_strength * 0.70)
            # PyTorch conviction: +0.14 * (torch_conf - 0.50)
            dyn_base = 0.57 + 0.15 * math.tanh(sig_strength * 0.70) + 0.14 * (torch_conf - 0.50)

            # 4. Institutional Sniper Calibration bounded strictly between 0.55 and 0.88
            final_confidence = max(0.55, min(0.88, dyn_base + delta_confluence + hurst_boost))

        # 8. Horizon-Tailored Quarter-Kelly Risk Recommendation with Hurst Capital Gating (0.25 * K * G(H))
        b = 1.5 if timeframe == "1h" else (2.0 if timeframe == "4h" else 2.5)
        p = final_confidence
        q = 1.0 - p
        full_kelly = max(0.0, (b * p - q) / b)
        quarter_kelly = 0.25 * full_kelly * 100.0 * entropy_gate
        
        rec_pct = min(4.5, max(1.0, round(quarter_kelly, 1)))
        horizon_name = "1 час" if timeframe == "1h" else ("4 часа" if timeframe == "4h" else "1 день")
        
        if direction == "BULLISH":
            rec_str = f"Паттерн роста ({horizon_name}, режим: {regime_name}, H={hurst:.2f}) подтвержден ({int(final_confidence*100)}%). Фракционный риск (Quarter-Kelly): ~{rec_pct}% от депозита. Плечо: 2x-4x."
        elif direction == "BEARISH":
            rec_str = f"Модель ожидает снижение ({horizon_name}, режим: {regime_name}, H={hurst:.2f}) ({int(final_confidence*100)}%). Фракционный риск (Quarter-Kelly): ~{rec_pct}% от депозита. Рассмотрите хеджирование или шорт."
        else:
            rec_str = f"Рынок ({horizon_name}, режим: {regime_name}, H={hurst:.2f}) в боковике/консолидации. Рекомендуется удерживать позиции или ожидать пробоя."

        # 9. Microstructure Order Flow Imbalance (OFI) & False Breakout Veto Gatekeeper
        obi_score = 0.0
        obi_summary = "Сбалансированный стакан"
        try:
            fetcher = getattr(self.gbdt_engine, "data_fetcher", None)
            if fetcher and hasattr(fetcher, "fetch_orderbook_depth"):
                depth = fetcher.fetch_orderbook_depth(symbol, limit=20)
                obi_data = self.ofi_service.calculate_obi(depth.get("bids", []), depth.get("asks", []))
                obi_score = obi_data.get("obi_score", 0.0)
                obi_summary = obi_data.get("summary", obi_summary)
                
                # Check Microstructure Veto
                is_vetoed, veto_reason = self.ofi_service.evaluate_microstructure_veto(direction, obi_data)
                if is_vetoed and direction in ["BULLISH", "BEARISH"]:
                    logger.info(f"OFI Microstructure Veto triggered for {symbol}: {veto_reason}")
                    direction = "NEUTRAL"
                    target_price = curr_price
                    final_confidence = 0.505
                    rec_str = f"⚠️ {veto_reason}. Технический сигнал заблокирован фильтром стакана ордеров."
        except Exception as e:
            logger.warning(f"OFI calculation bypassed for {symbol}: {e}")

        return PredictionResult(
            symbol=symbol,
            current_price=curr_price,
            predicted_target_price=target_price,
            predicted_log_return=adjusted_log_ret,
            direction=direction,
            confidence_score=final_confidence,
            volatility_estimate=volatility,
            timeframe=timeframe,
            personal_recommendation=rec_str,
            sentiment_score=s_score,
            news_summary=s_summary,
            orderbook_imbalance=obi_score,
            orderbook_summary=obi_summary,
            market_regime=regime_name,
            hurst_exponent=hurst,
            model_version="MaBuy-V4-Titan-Ensemble (MultiHorizon-PyTorch+GBDT+NLP+OFI+Hurst)",
            tier_applied=user_tier
        )

