import ccxt, os, gc, sys, json, time, math, tempfile, logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = '/root/mabuy2' if os.path.exists('/root/mabuy2') else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.infrastructure.ml.feature_engineering.pipeline import CryptoFeaturePipeline
from src.infrastructure.ml.pytorch.dataset import CryptoSequenceDataset
from src.infrastructure.ml.pytorch.attention_lstm import LSTMWithAttention
from src.infrastructure.ml.pytorch.financial_loss import AsymmetricFinancialLoss
from src.infrastructure.mlops.model_registry import ModelRegistry
from src.domain.services.triple_barrier_labeler import (
    make_barriers,
    BarrierSpec,
    label_single_event,
    compute_sample_uniqueness
)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "retrain_pipeline.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger("WeeklyRetrainPipeline")

TARGET_SYMBOLS: List[str] = ["BTC", "ETH", "SOL"]
DEFAULT_TIMEFRAME: str = "1h"
MODELS_DIR: str = os.path.join(BASE_DIR, "models")

# Guard thresholds
MIN_DIRECTIONAL_ACCURACY: float = 0.515
MIN_ANNUALIZED_SHARPE: float = 0.80

class FeatureEngineer:
    FEATURE_COLS: List[str] = [
        "r_lag1", "r_lag2", "r_lag3", "r_lag5",
        "volatility_20", "volatility_60", "vol_ratio",
        "z_return_lag1", "z_return_lag2", "z_return_lag3",
        "rsi_14", "macd_norm", "macd_signal_norm", "macd_hist_norm",
        "atr_norm", "bb_pb", "vol_ratio_20", "volume_log_return"
    ]

    @classmethod
    def compute_features(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        data = df.copy().sort_values("timestamp").reset_index(drop=True)

        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        log_return = np.log(close / close.shift(1))
        data["log_return"] = log_return

        # 1-Bar Momentum Blind Spot Fixed: r_lag1 is the return of the bar that just closed
        data["r_lag1"] = log_return
        data["r_lag2"] = log_return.shift(1)
        data["r_lag3"] = log_return.shift(2)
        data["r_lag5"] = log_return.shift(4)

        vol_20 = log_return.rolling(window=20).std() + 1e-7
        vol_60 = log_return.rolling(window=60).std() + 1e-7
        data["volatility_20"] = vol_20
        data["volatility_60"] = vol_60
        data["vol_ratio"] = vol_20 / vol_60

        z_return = log_return / vol_20
        data["z_return_lag1"] = z_return
        data["z_return_lag2"] = z_return.shift(1)
        data["z_return_lag3"] = z_return.shift(2)

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        data["rsi_14"] = (100.0 - (100.0 / (1.0 + rs))) / 100.0

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal
        data["macd_norm"] = macd / (close + 1e-8)
        data["macd_signal_norm"] = macd_signal / (close + 1e-8)
        data["macd_hist_norm"] = macd_hist / (close + 1e-8)

        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - close.shift(1)),
                np.abs(low - close.shift(1))
            )
        )
        data["atr_norm"] = tr.rolling(window=14).mean() / (close + 1e-8)

        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std() + 1e-8
        upper_band = sma_20 + (2.0 * std_20)
        lower_band = sma_20 - (2.0 * std_20)
        data["bb_pb"] = (close - lower_band) / (upper_band - lower_band + 1e-8)

        vol_sma_20 = volume.rolling(window=20).mean() + 1e-8
        data["vol_ratio_20"] = volume / vol_sma_20
        data["volume_log_return"] = np.log((volume + 1e-8) / (volume.shift(1) + 1e-8))

        data["target"] = np.log(close.shift(-1) / close)

        clean_df = data.dropna(subset=cls.FEATURE_COLS + ["target"]).reset_index(drop=True)
        return clean_df, cls.FEATURE_COLS

class BinanceHistoricalIngestor:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})

    def fetch_paginated_candles(self, symbol: str, timeframe: str = "1h", total_candles: int = 2500) -> pd.DataFrame:
        clean_sym = symbol.upper().replace('USDT', '').strip() + "/USDT"
        logger.info(f"Ingesting {total_candles} {timeframe} candles for {clean_sym}...")
        try:
            timeframe_ms = self.exchange.parse_timeframe(timeframe) * 1000
            now_ms = self.exchange.milliseconds()
            since = now_ms - (total_candles * timeframe_ms)

            all_candles = []
            while len(all_candles) < total_candles:
                remaining = total_candles - len(all_candles)
                limit = min(1000, remaining)
                candles = self.exchange.fetch_ohlcv(clean_sym, timeframe=timeframe, since=since, limit=limit)
                if not candles:
                    break
                all_candles.extend(candles)
                since = candles[-1][0] + 1  # +1ms avoids skipping boundary candles; drop_duplicates handles overlaps
                if len(candles) < limit or since >= now_ms:
                    break
                time.sleep(0.05)

            if all_candles:
                df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                logger.info(f"Successfully loaded {len(df)} candles for {clean_sym} ({df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}).")
                return df
        except Exception as e:
            logger.warning(f"Fetch error for {clean_sym}: {e}")
        return pd.DataFrame()


class WeeklyRetrainingPipeline:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.ingestor = BinanceHistoricalIngestor()
        self.registry = ModelRegistry(registry_dir=self.models_dir)

    def _atomic_save_joblib(self, obj: Any, target_path: str):
        target_dir = os.path.dirname(os.path.abspath(target_path))
        os.makedirs(target_dir, exist_ok=True)
        fd, temp_file_path = tempfile.mkstemp(dir=target_dir, suffix=".tmp")
        os.close(fd)
        try:
            joblib.dump(obj, temp_file_path, compress=3)
            os.replace(temp_file_path, target_path)
            logger.info(f"Atomic write successful: {target_path}")
        except Exception as e:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise e

    def evaluate_out_of_sample(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        fee_roundtrip: float = 0.0004,
        slippage: float = 0.0002
    ) -> Tuple[float, float, float, float]:
        """
        Evaluates Out-of-Sample model predictions with Astra Cost Deduction:
        Gross returns are penalized by round-trip exchange fees (0.02% Maker in + 0.02% Maker out = 0.04%)
        and adverse micro-slippage (0.02%), ensuring reported Sharpe and Expectancy reflect REAL NET ALPHA.
        Formula (Astra Growth Point 1 & Lopez de Prado Triple Barrier):
        E_net = p_win * mean_gross_win - (1 - p_win) * mean_gross_loss - E[fees + slippage]
        """
        y_pred = model.predict(X_test)
        signs_pred = np.sign(y_pred)
        signs_actual = np.sign(y_test)
        da = float(np.mean(signs_pred == signs_actual))

        gross_returns = signs_pred * y_test
        # Deduct execution friction on active trades
        active_mask = (signs_pred != 0)
        net_returns = np.where(active_mask, gross_returns - (fee_roundtrip + slippage), 0.0)

        mean_ret = float(np.mean(net_returns))
        std_ret = float(np.std(net_returns)) + 1e-8
        sharpe_ratio = float((mean_ret / std_ret) * math.sqrt(8760))

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(math.sqrt(mean_squared_error(y_test, y_pred)))
        return da, sharpe_ratio, mae, rmse

    def walk_forward_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        gbdt_params: dict,
        n_folds: int = 5,
        min_train_size: int = 300
    ) -> Tuple[List[dict], float, float, float, float]:
        """
        Expanding-window Walk-Forward Validation with Astra Purging & Embargo (Lopez de Prado Standard).

        Splits the time series into (n_folds + 1) equal chunks. For each fold k (1..n_folds):
          - Train on chunks [0..k] (expanding)
          - Enforce 24-bar (24h) embargo: test_start = train_end + 24
            Guarantees: [feature_time_i, label_end_i] intersect [test_start_j, test_end_j] = empty
          - Test on chunk [test_start..test_end] (strictly unseen future, zero label horizon overlap)

        Returns:
            fold_results: list of per-fold metric dicts
            mean_da, mean_sharpe, mean_mae, mean_rmse: averaged across folds
        """
        N = len(X)
        chunk_size = N // (n_folds + 1)

        if chunk_size < 60:
            logger.warning(f"WFV chunk size {chunk_size} too small for {n_folds} folds. Falling back to 3 folds.")
            n_folds = 3
            chunk_size = N // (n_folds + 1)

        fold_results = []
        all_da, all_sharpe, all_mae, all_rmse = [], [], [], []

        for k in range(n_folds):
            train_end = chunk_size * (k + 1)
            # Astra Purged & Embargo Guard: 24-bar embargo to eliminate label horizon overlap
            test_start = min(train_end + 24, N - 10)
            test_end = train_end + chunk_size

            # Last fold gets all remaining data
            if k == n_folds - 1:
                test_end = N

            if train_end < min_train_size or test_end > N or test_start >= test_end:
                logger.warning(f"WFV fold {k+1}: skipped (train={train_end}, test={test_start}:{test_end})")
                continue

            X_train_fold, y_train_fold = X[:train_end], y[:train_end]
            X_test_fold, y_test_fold = X[test_start:test_end], y[test_start:test_end]

            model_fold = HistGradientBoostingRegressor(**gbdt_params)
            model_fold.fit(X_train_fold, y_train_fold)

            da, sharpe, mae, rmse = self.evaluate_out_of_sample(model_fold, X_test_fold, y_test_fold)

            fold_info = {
                "fold": k + 1,
                "train_size": train_end,
                "test_size": test_end - test_start,
                "da": round(da, 5),
                "sharpe": round(sharpe, 4),
                "mae": round(mae, 6),
                "rmse": round(rmse, 6)
            }
            fold_results.append(fold_info)
            all_da.append(da)
            all_sharpe.append(sharpe)
            all_mae.append(mae)
            all_rmse.append(rmse)

            logger.info(
                f"  WFV Fold {k+1}/{n_folds}: "
                f"train={train_end} test={test_end - test_start} | "
                f"DA={da*100:.2f}% Sharpe={sharpe:.2f} MAE={mae:.5f}"
            )

            del model_fold, X_train_fold, y_train_fold, X_test_fold, y_test_fold

        if not all_da:
            return fold_results, 0.0, 0.0, 0.0, 0.0

        mean_da = float(np.mean(all_da))
        mean_sharpe = float(np.mean(all_sharpe))
        mean_mae = float(np.mean(all_mae))
        mean_rmse = float(np.mean(all_rmse))

        logger.info(
            f"  WFV MEAN ({len(all_da)} folds): "
            f"DA={mean_da*100:.2f}% Sharpe={mean_sharpe:.2f} MAE={mean_mae:.5f}"
        )
        return fold_results, mean_da, mean_sharpe, mean_mae, mean_rmse

    def retrain_symbol(self, symbol: str, total_candles: int = 2500, force_deploy: bool = False) -> dict:
        logger.info(f"--- Weekly Retraining for: {symbol} ---")
        raw_df = self.ingestor.fetch_paginated_candles(symbol, timeframe=DEFAULT_TIMEFRAME, total_candles=total_candles)
        if raw_df is None or len(raw_df) < 300:
            err_msg = f"Insufficient candles ({len(raw_df) if raw_df is not None else 0}). Min: 300."
            logger.error(err_msg)
            return {"symbol": symbol, "guard_passed": False, "error": err_msg}

        processed_df, feature_cols = FeatureEngineer.compute_features(raw_df)

        N = len(processed_df)
        X = processed_df[feature_cols].values
        y = processed_df["target"].values

        # Asset-specific regularization:
        # Altcoins like SOL have higher volatility and noise on 1h candles;
        # shallower trees (depth=2) with stronger L2 penalty (10.0) prevent overfitting on noise.
        clean_sym_key = symbol.upper().replace("USDT", "").strip()
        gbdt_params = {
            "max_iter": 200,
            "learning_rate": 0.01 if clean_sym_key == "SOL" else 0.03,
            "max_depth": 2 if clean_sym_key == "SOL" else 4,
            "min_samples_leaf": 20,
            "l2_regularization": 10.0 if clean_sym_key == "SOL" else 0.5,
            "early_stopping": False,
            "random_state": 42
        }

        # ── Walk-Forward Validation (5 folds) ──
        # Instead of a single 80/10/10 split, evaluate on 5 sequential OOS windows.
        # The deployment guard uses MEAN metrics across all folds.
        fold_results, mean_da, mean_sharpe, mean_mae, mean_rmse = self.walk_forward_evaluate(
            X, y, gbdt_params, n_folds=5, min_train_size=300
        )

        logger.info(f"[{symbol} WFV Summary] Mean DA: {mean_da*100:.2f}% | Mean Sharpe: {mean_sharpe:.2f} | Mean MAE: {mean_mae:.5f}")

        reasons = []
        if mean_da < MIN_DIRECTIONAL_ACCURACY:
            reasons.append(f"Mean DA {mean_da*100:.2f}% < {MIN_DIRECTIONAL_ACCURACY*100:.1f}%")
        if mean_sharpe < MIN_ANNUALIZED_SHARPE:
            reasons.append(f"Mean Sharpe {mean_sharpe:.2f} < {MIN_ANNUALIZED_SHARPE:.2f}")

        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        version_str = time.strftime("v_%Y%m%d_%H%M%S_gbdt", time.gmtime())

        clean_name = symbol.upper().replace("USDT", "").strip()
        artifact_filename = f"gbdt_{clean_name}USDT_1h.joblib"
        target_artifact_path = os.path.join(self.models_dir, artifact_filename)

        # ── Production model: train on all data except last OOS chunk ──
        # Uses maximum information while keeping the final chunk truly unseen.
        production_train_end = N - (N // 6)  # ~83% of data, last chunk reserved
        model = HistGradientBoostingRegressor(**gbdt_params)
        model.fit(X[:production_train_end], y[:production_train_end])

        # Final single-window OOS on the reserved last chunk with 1-bar embargo for reporting
        final_test_start = min(production_train_end + 1, N)
        final_da, final_sharpe, final_mae, final_rmse = self.evaluate_out_of_sample(
            model, X[final_test_start:], y[final_test_start:]
        )
        logger.info(f"[{symbol} Final OOS] Net DA: {final_da*100:.2f}% | Net Sharpe: {final_sharpe:.2f} | MAE: {final_mae:.5f}")

        if not reasons or force_deploy:
            if force_deploy and reasons:
                logger.warning(f"⚠️ Guard bypassed via force_deploy for {symbol}: {', '.join(reasons)}")
            logger.info(f"✅ DEPLOYMENT GUARD PASSED for {symbol}! Saving model...")
            self._atomic_save_joblib(model, target_artifact_path)

            metrics = {
                "mean_directional_accuracy": round(mean_da, 5),
                "sharpe_ratio": round(mean_sharpe, 4),
                "mean_mae": round(mean_mae, 6),
                "mean_rmse": round(mean_rmse, 6),
                "final_oos_da": round(final_da, 5),
                "final_oos_sharpe": round(final_sharpe, 4),
                "wfv_folds": len(fold_results),
                "sample_size_total": N
            }

            self.registry.register_model(
                model_name=f"gbdt_{clean_name}USDT",
                version=version_str,
                artifact_path=os.path.join("models", artifact_filename),
                model_type="GBDT",
                metrics=metrics,
                symbol=f"{clean_name}USDT",
                set_active=True
            )

            res = {
                "symbol": symbol, "timeframe": DEFAULT_TIMEFRAME, "total_samples": N,
                "mean_da": mean_da, "mean_sharpe": mean_sharpe, "mean_mae": mean_mae,
                "final_oos_da": final_da, "final_oos_sharpe": final_sharpe,
                "wfv_folds": fold_results,
                "guard_passed": True, "artifact_path": target_artifact_path, "timestamp": timestamp_str
            }
        else:
            rejection_text = ", ".join(reasons)
            logger.warning(f"⚠️ Guard not met for {symbol} ({rejection_text}). Preserving current production model.")
            res = {
                "symbol": symbol, "timeframe": DEFAULT_TIMEFRAME, "total_samples": N,
                "mean_da": mean_da, "mean_sharpe": mean_sharpe, "mean_mae": mean_mae,
                "final_oos_da": final_da, "final_oos_sharpe": final_sharpe,
                "wfv_folds": fold_results,
                "guard_passed": False, "rejection_reason": rejection_text, "timestamp": timestamp_str
            }

        # 4. Fast 5-epoch warm-start online fine-tuning for PyTorch Attention LSTM (concept drift defense)
        pt_res = self.finetune_pytorch_lstm(symbol, raw_df)
        res["pytorch_finetuning"] = pt_res

        del raw_df, processed_df, X, y, model
        gc.collect()
        return res

    def finetune_pytorch_lstm(self, symbol: str, raw_df: pd.DataFrame, epochs: int = 5, lr: float = 1e-4) -> dict:
        """
        Fast 5-epoch warm-start online fine-tuning loop for pytorch_attention_lstm_{clean_name}USDT_1h.pt
        using learning rate 1e-4 on recent 300-500 candles to eliminate neural network concept drift.
        """
        clean_name = symbol.upper().replace("USDT", "").strip()
        artifact_filename = f"pytorch_attention_lstm_{clean_name}USDT_{DEFAULT_TIMEFRAME}.pt"
        target_path = os.path.join(self.models_dir, artifact_filename)

        logger.info(f"🧠 [Warm-Start PyTorch] Fine-tuning Attention LSTM for {clean_name}USDT ({epochs} epochs, LR={lr})...")
        try:
            candle_count = min(len(raw_df), 500)
            recent_candles = raw_df.tail(candle_count).copy().reset_index(drop=True)
            if len(recent_candles) < 100:
                logger.warning(f"Insufficient candles ({len(recent_candles)}) for PyTorch fine-tuning of {clean_name}.")
                return {"success": False, "error": "insufficient_data"}

            processed_df, feat_names = FeatureEngineer.compute_features(recent_candles)
            if len(processed_df) <= 60:
                logger.warning(f"Insufficient transformed feature rows ({len(processed_df)}) for seq_len=60 on {clean_name}.")
                return {"success": False, "error": "insufficient_seq_length"}

            X_mat = processed_df[feat_names].values.astype(np.float32)
            y_tar = processed_df["target"].values.astype(np.float32)

            dataset = CryptoSequenceDataset(X_mat, y_tar, seq_len=60)
            drop_last = len(dataset) > 32
            loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=drop_last)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = LSTMWithAttention(input_dim=len(feat_names), hidden_dim=64, num_layers=2).to(device)

            # Warm-start from existing checkpoint if available
            warm_started = False
            if os.path.exists(target_path):
                try:
                    ckpt = torch.load(target_path, map_location="cpu", weights_only=False)
                    s_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
                    if isinstance(s_dict, dict):
                        # Support loading from multi_horizon_state_dict if single-head keys missing
                        if "fc_return.3.weight" not in s_dict:
                            if "head_1h.fc_return.4.weight" in s_dict:
                                s_dict["fc_return.0.weight"] = s_dict["head_1h.fc_return.0.weight"]
                                s_dict["fc_return.0.bias"] = s_dict["head_1h.fc_return.0.bias"]
                                s_dict["fc_return.3.weight"] = s_dict["head_1h.fc_return.4.weight"]
                                s_dict["fc_return.3.bias"] = s_dict["head_1h.fc_return.4.bias"]
                                s_dict["fc_prob.0.weight"] = s_dict["head_1h.fc_prob.0.weight"]
                                s_dict["fc_prob.0.bias"] = s_dict["head_1h.fc_prob.0.bias"]
                                s_dict["fc_prob.3.weight"] = s_dict["head_1h.fc_prob.4.weight"]
                                s_dict["fc_prob.3.bias"] = s_dict["head_1h.fc_prob.4.bias"]
                            elif "head_1h.fc_return.3.weight" in s_dict:
                                s_dict["fc_return.0.weight"] = s_dict["head_1h.fc_return.0.weight"]
                                s_dict["fc_return.0.bias"] = s_dict["head_1h.fc_return.0.bias"]
                                s_dict["fc_return.3.weight"] = s_dict["head_1h.fc_return.3.weight"]
                                s_dict["fc_return.3.bias"] = s_dict["head_1h.fc_return.3.bias"]
                                s_dict["fc_prob.0.weight"] = s_dict["head_1h.fc_prob.0.weight"]
                                s_dict["fc_prob.0.bias"] = s_dict["head_1h.fc_prob.0.bias"]
                                s_dict["fc_prob.3.weight"] = s_dict["head_1h.fc_prob.3.weight"]
                                s_dict["fc_prob.3.bias"] = s_dict["head_1h.fc_prob.3.bias"]

                        # Robustly adapt input dimension if shape differs from 18 features
                        curr_w = model.lstm.weight_ih_l0.data
                        if "lstm.weight_ih_l0" in s_dict:
                            old_w = s_dict["lstm.weight_ih_l0"]
                            if old_w.shape != curr_w.shape:
                                adapted_w = curr_w.clone()
                                min_r = min(old_w.shape[0], curr_w.shape[0])
                                min_c = min(old_w.shape[1], curr_w.shape[1])
                                adapted_w[:min_r, :min_c] = old_w[:min_r, :min_c]
                                s_dict["lstm.weight_ih_l0"] = adapted_w

                        model.load_state_dict(s_dict, strict=False)
                        warm_started = True
                        if any(torch.isnan(p).any() for p in model.parameters()):
                            logger.warning(f"Existing checkpoint for {clean_name} contained NaN weights. Initializing fresh.")
                            model = LSTMWithAttention(input_dim=len(feat_names), hidden_dim=64, num_layers=2).to(device)
                            warm_started = False
                        else:
                            logger.info(f"Loaded existing PyTorch weights for warm-start from {target_path}")
                except Exception as load_err:
                    logger.warning(f"Could not load existing checkpoint for warm-start ({load_err}). Initializing fresh weights.")

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = AsymmetricFinancialLoss(lambda_sharpe=1.0, lambda_bce=0.5, lambda_mse=1.0)

            model.train()
            final_loss = 0.0
            for epoch in range(epochs):
                epoch_loss = 0.0
                batch_count = 0
                for X_b, y_b, y_dir_b in loader:
                    if len(X_b) <= 1:
                        continue
                    X_b, y_b, y_dir_b = X_b.to(device), y_b.to(device), y_dir_b.to(device)
                    optimizer.zero_grad()
                    pred_ret, pred_prob = model(X_b)
                    loss = criterion(pred_ret, pred_prob, y_b, y_dir_b)
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += float(loss.item()) * len(X_b)
                    batch_count += len(X_b)
                if batch_count > 0:
                    final_loss = epoch_loss / batch_count
                logger.debug(f"[{clean_name} PyTorch Fine-Tune] Epoch {epoch+1}/{epochs} - Loss: {final_loss:.5f}")

            # Verify no NaN parameters exist before persisting
            if any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters()):
                logger.critical(f"Refusing to save {clean_name} model: NaN or Inf parameters detected after training.")
                return {"success": False, "error": "nan_parameters_detected"}

            # Multi-horizon mapped state dict for v4_titan_ensemble MultiHorizonAttentionLSTM
            sd = model.state_dict()
            mh_sd = dict(sd)
            for head_name in ["head_1h", "head_4h", "head_1d"]:
                for k, v in sd.items():
                    if k.startswith("fc_return.0.") or k.startswith("fc_prob.0."):
                        mh_sd[f"{head_name}.{k}"] = v.clone()
                    elif k.startswith("fc_return.3."):
                        sub_k = k.replace("fc_return.3.", "fc_return.4.")
                        mh_sd[f"{head_name}.{sub_k}"] = v.clone()
                    elif k.startswith("fc_prob.3."):
                        sub_k = k.replace("fc_prob.3.", "fc_prob.4.")
                        mh_sd[f"{head_name}.{sub_k}"] = v.clone()
                # Initialize LayerNorm identity defaults for HorizonHead
                mh_sd[f"{head_name}.fc_return.1.weight"] = torch.ones(32)
                mh_sd[f"{head_name}.fc_return.1.bias"] = torch.zeros(32)
                mh_sd[f"{head_name}.fc_prob.1.weight"] = torch.ones(32)
                mh_sd[f"{head_name}.fc_prob.1.bias"] = torch.zeros(32)

            # Atomic save checkpoint with dual state_dict keys for v4_titan_ensemble compatibility
            save_payload = {
                "model_state_dict": sd,
                "state_dict": sd,
                "multi_horizon_state_dict": mh_sd,
                "feature_names": feat_names,
                "seq_len": 60,
                "symbol": f"{clean_name}USDT",
                "timeframe": DEFAULT_TIMEFRAME,
                "summary_metrics": {"fine_tune_loss": round(final_loss, 5)},
                "last_finetuned_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "warm_started": warm_started
            }

            temp_path = target_path + f".tmp_{os.getpid()}"
            torch.save(save_payload, temp_path)
            os.replace(temp_path, target_path)

            # Also persist multi_horizon_lstm.pt candidate
            mh_path = os.path.join(self.models_dir, "multi_horizon_lstm.pt")
            try:
                torch.save(save_payload, mh_path + f".tmp_{os.getpid()}")
                os.replace(mh_path + f".tmp_{os.getpid()}", mh_path)
            except Exception:
                pass

            # Register PyTorch checkpoint in ModelRegistry to touch manifest and notify daemon
            version_str = time.strftime("pt_%Y%m%d_%H%M%S_ft", time.gmtime())
            try:
                self.registry.register_model(
                    model_name=f"pytorch_attention_lstm_{clean_name}USDT",
                    version=version_str,
                    artifact_path=os.path.join("models", artifact_filename),
                    model_type="PyTorch-Attention-LSTM",
                    metrics={
                        "fine_tune_loss": round(final_loss, 5),
                        "epochs": epochs,
                        "lr": lr,
                        "warm_started": int(warm_started),
                        "samples": len(dataset)
                    },
                    symbol=f"{clean_name}USDT",
                    set_active=True
                )
            except Exception as reg_err:
                logger.warning(f"Could not register PyTorch model in manifest: {reg_err}")

            logger.info(f"✅ [PyTorch Fine-Tune] Successfully saved {clean_name}USDT Attention LSTM (loss={final_loss:.4f}) to {target_path}")
            return {"success": True, "loss": round(final_loss, 5), "warm_started": warm_started, "artifact": target_path}

        except Exception as pt_err:
            logger.error(f"Error fine-tuning PyTorch LSTM for {symbol}: {pt_err}", exc_info=True)
            return {"success": False, "error": str(pt_err)}

    def finetune_symbol_only(self, symbol: str, total_candles: int = 500, epochs: int = 5, lr: float = 1e-4) -> dict:
        """Lightweight online fine-tuning for a single asset without full GBDT recomputation."""
        logger.info(f"--- Online Fine-Tuning for: {symbol} ---")
        raw_df = self.ingestor.fetch_paginated_candles(symbol, timeframe=DEFAULT_TIMEFRAME, total_candles=total_candles)
        if raw_df is None or len(raw_df) < 100:
            err_msg = f"Insufficient candles ({len(raw_df) if raw_df is not None else 0}) for fine-tuning."
            logger.error(err_msg)
            return {"symbol": symbol, "success": False, "error": err_msg}
        res = self.finetune_pytorch_lstm(symbol, raw_df, epochs=epochs, lr=lr)
        del raw_df
        gc.collect()
        return res

    def run_finetune_pipeline(
        self,
        symbols: Optional[List[str]] = None,
        epochs: int = 5,
        lr: float = 1e-4,
        candles: int = 500
    ) -> Dict[str, Any]:
        """Runs fast online warm-start fine-tuning across all target symbols."""
        targets = symbols or TARGET_SYMBOLS
        logger.info(f"=== Starting Daily Online Fine-Tuning for {targets} (epochs={epochs}, lr={lr}) ===")
        start_time = time.time()
        results: Dict[str, Any] = {}

        for symbol in targets:
            try:
                results[symbol] = self.finetune_symbol_only(symbol, total_candles=candles, epochs=epochs, lr=lr)
            except Exception as ex:
                logger.error(f"Error fine-tuning {symbol}: {ex}", exc_info=True)
                results[symbol] = {"symbol": symbol, "success": False, "error": str(ex)}

        elapsed = round(time.time() - start_time, 2)
        summary = {
            "execution_timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "mode": "online_finetune",
            "elapsed_seconds": elapsed,
            "results": results,
            "success_count": sum(1 for r in results.values() if r.get("success") is True)
        }

        summary_path = os.path.join(BASE_DIR, "logs", "latest_finetune_summary.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save fine-tune summary JSON: {e}")

        logger.info(f"Fine-tuning completed in {elapsed}s. Success: {summary['success_count']}/{len(targets)}.")
        return summary

    def run_pipeline(
        self,
        symbols: Optional[List[str]] = None,
        total_candles: int = 2500,
        force_deploy: bool = False
    ) -> Dict[str, Any]:
        targets = symbols or TARGET_SYMBOLS
        logger.info(f"=== Starting Weekly Autonomous Retraining Pipeline for {targets} ===")
        start_time = time.time()
        results: Dict[str, Any] = {}

        for symbol in targets:
            try:
                res = self.retrain_symbol(symbol, total_candles=total_candles, force_deploy=force_deploy)
                results[symbol] = res
            except Exception as ex:
                logger.error(f"Error retraining {symbol}: {ex}", exc_info=True)
                results[symbol] = {"symbol": symbol, "error": str(ex), "guard_passed": False}

        elapsed = round(time.time() - start_time, 2)
        summary = {
            "execution_timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "elapsed_seconds": elapsed,
            "results": results,
            "promoted_count": sum(1 for r in results.values() if r.get("guard_passed") is True)
        }

        summary_path = os.path.join(BASE_DIR, "logs", "latest_retrain_summary.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save summary JSON: {e}")

        logger.info(f"Pipeline completed in {elapsed}s. Promoted: {summary['promoted_count']}/{len(targets)}.")
        return summary

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MaBuy Autonomous MLOps Retraining & Online Fine-Tuning Engine")
    parser.add_argument("--mode", choices=["full", "finetune", "gbdt"], default="full", help="Execution mode: full retrain or daily online fine-tune")
    parser.add_argument("--symbols", type=str, default="BTC,ETH,SOL", help="Comma-separated symbols (e.g. BTC,ETH,SOL)")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for PyTorch fine-tuning (default: 5)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for PyTorch (default: 1e-4)")
    parser.add_argument("--candles", type=int, default=None, help="Candles to fetch (default: 2500 for full, 500 for finetune)")
    parser.add_argument("--force-deploy", action="store_true", help="Bypass deployment guard if set")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    pipeline = WeeklyRetrainingPipeline()

    if args.mode == "finetune":
        candles = args.candles or 500
        summary = pipeline.run_finetune_pipeline(symbols=symbols, epochs=args.epochs, lr=args.lr, candles=candles)
    else:
        candles = args.candles or 2500
        summary = pipeline.run_pipeline(symbols=symbols, total_candles=candles, force_deploy=args.force_deploy)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
