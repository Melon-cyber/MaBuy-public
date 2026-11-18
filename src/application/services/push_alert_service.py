import time
import logging
from typing import List, Dict, Any, Optional
from src.domain.entities.tier import SubscriptionTier
from src.domain.entities.prediction import PredictionRequest
from src.application.use_cases.get_prediction_use_case import GetPredictionUseCase
from src.domain.services.position_size_calculator import PositionSizeCalculatorService

logger = logging.getLogger(__name__)


class PushAlertService:
    """
    Automated Crypto Signal Push Alert Engine (Pro & Titan Tiers Only).
    Periodically scans major assets (BTC, ETH, SOL, BNB, XRP, DOGE, AVAX, ADA, LINK, DOT)
    and identifies high-confidence breakout signals with immediate Take-Profit, Stop-Loss and Safe Leverage.
    """

    def __init__(self, get_prediction_use_case: GetPredictionUseCase, position_calc: Optional[PositionSizeCalculatorService] = None):
        self.get_prediction_use_case = get_prediction_use_case
        self.position_calc = position_calc or PositionSizeCalculatorService()
        self.monitored_symbols = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "ADA", "LINK", "DOT"]
        self._last_alert_time: Dict[str, float] = {}
        self._last_alert_direction: Dict[str, str] = {}
        self.cooldown_seconds = 3600 # 1 hour cooldown per symbol unless trend flips

    def scan_market_for_breakout_signals(self, lang: str = "ru") -> List[Dict[str, Any]]:
        """
        Scans market symbols and returns list of actionable high-confidence breakout alerts
        with pre-calculated TP, SL, R:R and Safe Leverage.
        """
        alerts = []
        now = time.time()

        for symbol in self.monitored_symbols:
            req = PredictionRequest(
                user_id="alert_engine_scanner",
                symbol=symbol,
                tier=SubscriptionTier.PRO,
                timeframe="1h",
                lang=lang
            )

            try:
                result = self.get_prediction_use_case.execute(req)
                
                # Filter for strong high-confidence directional signals (> 58% confidence and abs(return) > 0.12%)
                if result.direction in ("BULLISH", "BEARISH") and result.confidence_score >= 0.58 and abs(result.predicted_log_return) >= 0.0012:
                    last_time = self._last_alert_time.get(symbol, 0)
                    last_dir = self._last_alert_direction.get(symbol, "")

                    # Send alert if cooldown expired OR direction flipped (e.g. LONG -> SHORT)
                    if (now - last_time >= self.cooldown_seconds) or (last_dir and last_dir != result.direction):
                        self._last_alert_time[symbol] = now
                        self._last_alert_direction[symbol] = result.direction

                        # Compute actionable trade setup (TP, SL, Leverage, Kelly)
                        plan = self.position_calc.calculate_position_plan(
                            account_balance=1000.0,
                            symbol=symbol,
                            direction=result.direction,
                            risk_level="balanced",
                            model_probability=result.confidence_score
                        )

                        alerts.append({
                            "symbol": result.symbol,
                            "direction": result.direction,
                            "current_price": result.current_price,
                            "target_price": plan["take_profit_price"],
                            "stop_loss_price": plan["stop_loss_price"],
                            "tp_pct": plan["take_profit_pct"],
                            "sl_pct": plan["stop_loss_pct"],
                            "reward_risk_ratio": plan["reward_risk_ratio"],
                            "max_safe_leverage": plan["max_safe_leverage"],
                            "kelly_risk_pct": plan["risk_pct"],
                            "confidence_pct": int(result.confidence_score * 100),
                            "timeframe": result.timeframe,
                            "personal_recommendation": result.personal_recommendation or "Импульс подтвержден ансамблем моделей."
                        })
            except Exception as e:
                logger.warning(f"Push alert scanner exception for {symbol}: {e}")

        return alerts
