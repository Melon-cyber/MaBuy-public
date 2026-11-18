import math
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from src.domain.services.position_size_calculator import PositionSizeCalculatorService
from src.domain.services.symbol_normalizer import SymbolNormalizer


@dataclass(frozen=True)
class SafeLeveragePlan:
    symbol: str
    direction: str
    account_balance: float
    current_price: float
    stop_loss_price: float
    take_profit_price: float
    stop_loss_pct: float
    take_profit_pct: float
    reward_risk_ratio: float
    risk_pct: float
    risk_amount_usd: float
    position_size_usd: float
    position_size_coins: float
    potential_profit_usd: float
    max_safe_leverage: int
    liquidation_distance_pct: float
    safety_buffer_ratio: float
    formatted_message: str


class SafeLeverageWizardService:
    """
    Safe-Leverage & Position Size Wizard for MaBuy Telegram Bot.
    Provides 1-second actionable sizing calculation and anti-liquidation protection.
    """

    def __init__(self, position_calculator: Optional[PositionSizeCalculatorService] = None):
        self.position_calculator = position_calculator or PositionSizeCalculatorService()

    def generate_wizard_plan(
        self,
        symbol: str,
        account_balance: float,
        direction: str = "BULLISH",
        risk_level: str = "balanced",
        model_probability: float = 0.58,
        lang: str = "ru"
    ) -> SafeLeveragePlan:
        if account_balance <= 0:
            account_balance = 100.0

        raw_plan = self.position_calculator.calculate_position_plan(
            account_balance=account_balance,
            symbol=symbol,
            direction=direction,
            risk_level=risk_level,
            model_probability=model_probability
        )

        sym = raw_plan["symbol"]
        current_p = raw_plan["current_price"]
        sl_p = raw_plan["stop_loss_price"]
        tp_p = raw_plan["take_profit_price"]
        sl_pct = raw_plan["stop_loss_pct"]
        tp_pct = raw_plan["take_profit_pct"]
        rr = raw_plan["reward_risk_ratio"]
        risk_pct = raw_plan["risk_pct"]
        risk_usd = raw_plan["risk_amount_usd"]
        pos_usd = raw_plan["position_size_usd"]
        pos_coins = raw_plan["position_size_coins"]
        prof_usd = raw_plan["potential_profit_usd"]
        safe_lev = raw_plan["max_safe_leverage"]

        liq_dist_pct = round(100.0 / max(1, safe_lev), 2)
        safety_buffer = round(liq_dist_pct / max(0.1, sl_pct), 1)

        # Build clean, localized Telegram message
        dir_emoji = "🟢 LONG" if direction.upper() == "BULLISH" else "🔴 SHORT"
        
        if lang == "ru":
            msg = (
                f"🧮 <b>Калькулятор безопасного входа MaBuy Safe-Leverage</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>Актив:</b> <code>{sym}</code> | <b>Направление:</b> {dir_emoji}\n"
                f"💰 <b>Торговый депозит:</b> <code>${account_balance:,.2f}</code>\n\n"
                f"🛡️ <b>БЕЗОПАСНЫЙ РАСЧЕТ РИСКА (Quarter-Kelly):</b>\n"
                f"• Допустимый риск: <b>{risk_pct:.2f}%</b> (<code>${risk_usd:.2f}</code>)\n"
                f"• Рекомендуемый объем: <b>${pos_usd:,.2f}</b> (<code>{pos_coins:.4f} {sym.replace('USDT', '')}</code>)\n"
                f"• Потолок безопасного плеча: <b>{safe_lev}x Изолированное</b>\n"
                f"• Запас до ликвидации: <b>~{liq_dist_pct}%</b> (в <b>{safety_buffer}x</b> дальше стоп-лосса ✅)\n\n"
                f"🎯 <b>ЦЕЛИ И СТОП-ЛОСС (R:R = 1:{rr:.1f}):</b>\n"
                f"• Вход (Текущая): <code>${current_p:,.2f}</code>\n"
                f"• Тейк-профит (+{tp_pct:.2f}%): <code>${tp_p:,.2f}</code> ➔ <b>+${prof_usd:.2f}</b>\n"
                f"• Стоп-лосс (-{sl_pct:.2f}%): <code>${sl_p:,.2f}</code> ➔ <b>-${risk_usd:.2f}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"<i>⚠️ Защита депозита: Никогда не превышайте плечо {safe_lev}x во избежание риска ликвидации.</i>"
            )
        else:
            msg = (
                f"🧮 <b>MaBuy Safe-Leverage Position Wizard</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>Asset:</b> <code>{sym}</code> | <b>Direction:</b> {dir_emoji}\n"
                f"💰 <b>Account Deposit:</b> <code>${account_balance:,.2f}</code>\n\n"
                f"🛡️ <b>QUANTUM RISK SIZING (Quarter-Kelly):</b>\n"
                f"• Max Risk Limit: <b>{risk_pct:.2f}%</b> (<code>${risk_usd:.2f}</code>)\n"
                f"• Recommended Position: <b>${pos_usd:,.2f}</b> (<code>{pos_coins:.4f} {sym.replace('USDT', '')}</code>)\n"
                f"• Safe Leverage Ceiling: <b>{safe_lev}x Isolated</b>\n"
                f"• Liquidation Distance: <b>~{liq_dist_pct}%</b> (<b>{safety_buffer}x</b> farther than Stop-Loss ✅)\n\n"
                f"🎯 <b>LEVELS & RATIO (R:R = 1:{rr:.1f}):</b>\n"
                f"• Entry: <code>${current_p:,.2f}</code>\n"
                f"• Take-Profit (+{tp_pct:.2f}%): <code>${tp_p:,.2f}</code> ➔ <b>+${prof_usd:.2f}</b>\n"
                f"• Stop-Loss (-{sl_pct:.2f}%): <code>${sl_p:,.2f}</code> ➔ <b>-${risk_usd:.2f}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"<i>⚠️ Capital Defense: Do not exceed {safe_lev}x leverage to avoid liquidation cascades.</i>"
            )

        return SafeLeveragePlan(
            symbol=sym,
            direction=direction.upper(),
            account_balance=account_balance,
            current_price=current_p,
            stop_loss_price=sl_p,
            take_profit_price=tp_p,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            reward_risk_ratio=rr,
            risk_pct=risk_pct,
            risk_amount_usd=risk_usd,
            position_size_usd=pos_usd,
            position_size_coins=pos_coins,
            potential_profit_usd=prof_usd,
            max_safe_leverage=safe_lev,
            liquidation_distance_pct=liq_dist_pct,
            safety_buffer_ratio=safety_buffer,
            formatted_message=msg
        )
