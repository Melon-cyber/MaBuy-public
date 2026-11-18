import pandas as pd
import numpy as np
import os
import joblib
from src.domain.entities.tick import MLSignalRequest, MLSignalResponse

class QuantAnalyzer:
    """
    Класс для квантового анализа потока сырых тиков.
    Ожидает на вход батч последних N сделок и возвращает торговый сигнал.
    Использует ML-модель RandomForestClassifier для предсказания направления.
    """
    
    def __init__(self, model_path: str = "models/tick_model.pkl"):
        self.confidence_threshold = 0.40
        self.model = None
        
        # Загружаем модель, если она существует
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print(f"[QuantAnalyzer] Модель {model_path} успешно загружена.")
            except Exception as e:
                print(f"[QuantAnalyzer] Ошибка при загрузке модели: {e}")
        else:
            print(f"[QuantAnalyzer] ВНИМАНИЕ: Модель {model_path} не найдена. Будет использоваться бейзлайн (эвристика).")
            
    def analyze(self, request: MLSignalRequest) -> MLSignalResponse:
        if not request.ticks:
            return MLSignalResponse(direction="NEUTRAL", confidence_score=0.0)
            
        df = pd.DataFrame([tick.model_dump() for tick in request.ticks])
        df = df.sort_values(by="timestamp").reset_index(drop=True)
        
        # 1. Расчет фичей (должен в точности совпадать с dataset_builder.py)
        # Обрати внимание: в проде лучше считать фичи инкрементально
        df["signed_qty"] = df["size"] * df["side"]
        df["quote_vol"] = df["price"] * df["size"]
        
        total_qty = df["size"].sum()
        total_signed_qty = df["signed_qty"].sum()
        total_quote_vol = df["quote_vol"].sum()
        
        # OFI
        ofi = total_signed_qty / total_qty if total_qty > 0 else 0
        
        # Волатильность (по всему батчу)
        volatility = df["price"].std() if len(df) > 1 else 0
        
        # VWAP Dev
        vwap = total_quote_vol / total_qty if total_qty > 0 else df["price"].iloc[-1]
        current_price = df["price"].iloc[-1]
        vwap_dev = (current_price - vwap) / current_price if current_price > 0 else 0
        
        # Микро-тренд
        micro_trend = current_price - df["price"].iloc[0]
        
        features = pd.DataFrame([{
            'ofi': ofi,
            'volatility': volatility,
            'vwap_dev': vwap_dev,
            'micro_trend': micro_trend
        }])
        
        # 2. Предсказание (ML Inference)
        if self.model is not None:
            # Predict Proba возвращает вероятности для классов: [-1, 0, 1] (если они есть в датасете)
            classes = list(self.model.classes_)
            probas = self.model.predict_proba(features)[0]
            
            # Находим класс с максимальной вероятностью
            max_idx = np.argmax(probas)
            pred_class = classes[max_idx]
            confidence = probas[max_idx]
            
            if pred_class == 1 and confidence > self.confidence_threshold:
                direction = "BULLISH"
            elif pred_class == -1 and confidence > self.confidence_threshold:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"
                
            return MLSignalResponse(
                direction=direction,
                confidence_score=round(confidence, 4)
            )
            
        else:
            # Fallback на эвристику (Baseline)
            score = 0.5 
            score += ofi * 0.3 
            if current_price > vwap: score += 0.1
            elif current_price < vwap: score -= 0.1
            if micro_trend > 0: score += 0.1
            elif micro_trend < 0: score -= 0.1
                
            score = max(0.0, min(1.0, float(score)))
            
            if score > self.confidence_threshold:
                direction = "BULLISH"
            elif score < (1 - self.confidence_threshold):
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"
                
            confidence = abs(score - 0.5) * 2 if direction != "NEUTRAL" else 0.0
            return MLSignalResponse(direction=direction, confidence_score=round(confidence, 4))
