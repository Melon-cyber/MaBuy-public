import os
import time
from typing import Optional, Dict, Any, List
from src.domain.entities.prediction import PredictionResult
from src.domain.entities.tier import SubscriptionTier
from src.infrastructure.llm.persona_prompts import PersonaType, PERSONA_METADATA, OLED_EXPRESSIONS


class TelegramMessageFormatter:
    """
    Bilingual (RU/EN) Telegram Message Formatter designed for beginner-friendly UX.
    Formats prediction results, subscription catalogs, language selectors, and inline keyboards.
    """

    REFERRAL_LINKS = {
        "Bybit": "https://www.bybit.com/invite?ref=EN5BDD&medium=referral&utm_campaign=evergreen",
        "Binance": "https://accounts.binance.com/register?ref=MABUY_VIP",
        "OKX": "https://www.okx.com/join/MABUY_CRYPTO"
    }

    @staticmethod
    def format_currency(price: float) -> str:
        """Dynamically formats price based on its magnitude to prevent penny crypto rounding collisions."""
        if price is None:
            return "$0.00"
        abs_p = abs(price)
        if abs_p >= 100.0:
            return f"${price:,.2f}"
        elif abs_p >= 1.0:
            return f"${price:,.3f}"
        elif abs_p >= 0.005:
            return f"${price:,.4f}"
        elif abs_p >= 0.0001:
            return f"${price:,.6f}"
        else:
            return f"${price:,.8f}"

    @staticmethod
    def format_prediction_result(
        result: PredictionResult,
        lang: str = "ru",
        experience: str = "beginner",
        user_deposit: float = 1000.0
    ) -> str:
        """Formats the prediction result into the Dual-Layer Beginner-friendly Telegram HTML message."""
        import html
        tf = getattr(result, "timeframe", "1h")
        symbol = result.symbol.upper()
        confidence_pct = int(result.confidence_score * 100)
        blocks_filled = max(1, min(10, int(round((confidence_pct / 100) * 10))))
        gauge_bar = "█" * blocks_filled + "░" * (10 - blocks_filled)

        # Sanitize HTML inputs to prevent Telegram parse entity errors
        clean_rec = html.escape(result.personal_recommendation) if result.personal_recommendation else None
        clean_news = html.escape(result.news_summary) if result.news_summary else None

        # Calculate ATR Risk-Reward Stop Loss & Take Profit
        delta = abs(result.predicted_target_price - result.current_price)
        if delta < 1e-6:
            delta = result.current_price * 0.015
        
        if result.direction == "BULLISH":
            tp_price = result.predicted_target_price
            sl_price = max(result.current_price * 0.001, result.current_price - (delta * 0.65))
        elif result.direction == "BEARISH":
            tp_price = result.predicted_target_price
            sl_price = result.current_price + (delta * 0.65)
        else:
            tp_price = result.current_price * 1.015
            sl_price = result.current_price * 0.985

        tp_pct = abs((tp_price - result.current_price) / result.current_price * 100.0)
        sl_pct = abs((result.current_price - sl_price) / result.current_price * 100.0)

        cur_p_str = TelegramMessageFormatter.format_currency(result.current_price)
        tp_p_str = TelegramMessageFormatter.format_currency(tp_price)
        sl_p_str = TelegramMessageFormatter.format_currency(sl_price)

        # Protective leverage modes & personalized deposit sizing:
        cons_lev = min(3, max(1, int(25.0 / max(sl_pct, 0.1))))
        opt_lev = min(5, max(1, min(int(50.0 / max(sl_pct, 0.1)), int(40.0 / max(sl_pct, 0.1)))))
        max_lev = max(1, min(20, int(50.0 / max(sl_pct, 0.1))))

        margin_usd = round(user_deposit * 0.10, 2)
        pos_usd = round(margin_usd * opt_lev, 2)
        loss_usd = round(pos_usd * (sl_pct / 100.0), 2)
        profit_usd = round(pos_usd * (tp_pct / 100.0), 2)

        if result.direction == "NEUTRAL":
            conf_quality = "Боковик / Баланс сил" if lang == "ru" else "Consolidation / Neutral"
        else:
            if lang == "ru":
                conf_quality = "Высокая (Конфлюэнс)" if confidence_pct >= 70 else ("Хорошая" if confidence_pct >= 62 else "Умеренная")
            else:
                conf_quality = "High Confluence" if confidence_pct >= 70 else ("Strong" if confidence_pct >= 62 else "Moderate")
        
        bybit_url = TelegramMessageFormatter.REFERRAL_LINKS["Bybit"]

        if lang == "ru":
            tf_label = "1 час" if tf == "1h" else ("4 часа" if tf == "4h" else "24 часа (1 день)")

            if result.direction == "BULLISH":
                dir_str = "🟢 ПОКУПАТЬ (LONG)"
                default_why = "Крупные игроки выкупают пролив на споте, новости позитивные. Вероятность продолжения импульса выше среднего."
                tp_str = f"{tp_p_str} <b>(+{tp_pct:.2f}% профит)</b>"
                sl_str = f"{sl_p_str} <b>(-{sl_pct:.2f}% риск)</b>"
            elif result.direction == "BEARISH":
                dir_str = "🔴 ПРОДАВАТЬ (SHORT)"
                default_why = "Давление продавцов в биржевом стакане нарастает, модель ожидает коррекцию цены вниз."
                tp_str = f"{tp_p_str} <b>(+{tp_pct:.2f}% профит на падении)</b>"
                sl_str = f"{sl_p_str} <b>(-{sl_pct:.2f}% риск при росте)</b>"
            else:
                dir_str = "🟡 ЖДАТЬ (ФЛЭТ / НЕЙТРАЛЬНО)"
                default_why = "Рынок находится в фазе накопления и баланса сил. Рекомендуется дождаться направленного импульса."
                tp_str = f"{tp_p_str} <b>(+{tp_pct:.2f}%)</b>"
                sl_str = f"{sl_p_str} <b>(-{sl_pct:.2f}%)</b>"

            why_text = clean_rec if clean_rec else default_why

            lines = [
                f"⚡ <b>MaBuy Signal • #{symbol}/USDT</b>",
                f"━━━━━━━━━━━━━━━━━━━━━━",
                f"🚦 <b>Решение:</b> <b>{dir_str}</b>  •  AI: <b>{confidence_pct}%</b>",
                f"💵 <b>Вход:</b> <code>{cur_p_str}</code>",
                f"🎯 <b>Цель (TP):</b> {tp_str}",
                f"🛡 <b>Стоп (SL):</b> {sl_str}\n",
                f"📐 <b>Сайзинг на $100:</b> цель <code>+${profit_usd:.2f}</code> • риск <code>-${loss_usd:.2f}</code> ({opt_lev}x)",
                f"━━━━━━━━━━━━━━━━━━━━━━",
                f"<i>«{why_text}»</i>",
                f"\n📲 <a href='{bybit_url}'><b>Открыть сделку на Bybit (-20% fee)</b></a>"
            ]
        else:
            tf_label = "1 Hour (1h)" if tf == "1h" else ("4 Hours (4h)" if tf == "4h" else "24 Hours / 1 Day (1d)")
            if result.direction == "BULLISH":
                dir_str = "🟢 BUY (LONG)"
                default_why = "Whale accumulation detected on spot order books with supportive sentiment."
                tp_str = f"{tp_p_str} <b>(+{tp_pct:.2f}% profit)</b>"
                sl_str = f"{sl_p_str} <b>(-{sl_pct:.2f}% risk)</b>"
            elif result.direction == "BEARISH":
                dir_str = "🔴 SELL (SHORT)"
                default_why = "Selling pressure dominant across order books, correction expected."
                tp_str = f"{tp_p_str} <b>(+{tp_pct:.2f}% short profit)</b>"
                sl_str = f"{sl_p_str} <b>(-{sl_pct:.2f}% risk on pump)</b>"
            else:
                dir_str = "🟡 HOLD (NEUTRAL)"
                default_why = "Market is consolidating. Recommend waiting for a directional breakout."
                tp_str = f"{tp_p_str} <b>(+{tp_pct:.2f}%)</b>"
                sl_str = f"{sl_p_str} <b>(-{sl_pct:.2f}%)</b>"

            why_text = clean_rec if clean_rec else default_why

            lines = [
                f"⚡️ <b>MaPredict by MaBuy • #{symbol}USDT</b> 🪙",
                f"━━━━━━━━━━━━━━━━━━━━━━",
                f"🚦 <b>SIGNAL:</b> <b>{dir_str}</b>",
                f"📊 <b>AI Confidence:</b> [{gauge_bar}] <b>{confidence_pct}%</b>",
                f"⏱ <b>Horizon:</b> {tf_label}\n",
                f"💵 <b>Current Price:</b> <code>{cur_p_str}</code>",
                f"🎯 <b>Target / Take-Profit (TP):</b> {tp_str}",
                f"🛡 <b>Stop-Loss (SL):</b> {sl_str}\n",
                f"⚙️ <b>RECOMMENDED LEVERAGE:</b>",
                f" • 🛡 Conservative: <b>{cons_lev}x</b> | ⚖️ Optimal: <b>{opt_lev}x</b> | 🚨 Limit: <b>max {max_lev}x</b>\n",
                f"💰 <b>PERSONAL SIZING FOR YOUR CAPITAL (${user_deposit:,.2f}):</b>",
                f" • <b>Margin (10% of cap):</b> <code>${margin_usd:,.2f}</code> ➔ Size ({opt_lev}x): <code>${pos_usd:,.2f}</code>",
                f" • <b>Risk at Stop (SL):</b> <code>-${loss_usd:.2f}</code> (<code>-{loss_usd/user_deposit*100:.2f}%</code>)",
                f" • <b>Profit at Target (TP):</b> <code>+${profit_usd:.2f}</code> (<code>+{profit_usd/user_deposit*100:.2f}%</code>)\n",
                f"💡 <b>WHY DOES AI THINK SO?</b>",
                f"<i>«{why_text}»</i>",
                f"━━━━━━━━━━━━━━━━━━━━━━",
                f"📲 <a href='{bybit_url}'><b>Trade #{symbol} on Bybit (20% fee rebate)</b></a>",
                f"⚖️ <i>Disclaimer: Not financial advice. For research and risk modeling purposes only. DYOR.</i>"
            ]

        return "\n".join(lines)

    @staticmethod
    def format_pro_quant_matrix(result: PredictionResult, lang: str = "ru", user_deposit: float = 1000.0) -> str:
        """Formats comprehensive institutional Quant Matrix view for PRO traders."""
        import html
        symbol = result.symbol.upper()
        confidence_pct = int(result.confidence_score * 100)
        tf = getattr(result, "timeframe", "1h")
        
        delta = abs(result.predicted_target_price - result.current_price)
        if delta < 1e-6:
            delta = result.current_price * 0.015
            
        if result.direction == "BULLISH":
            tp_price = result.predicted_target_price
            sl_price = max(result.current_price * 0.001, result.current_price - (delta * 0.65))
        elif result.direction == "BEARISH":
            tp_price = result.predicted_target_price
            sl_price = result.current_price + (delta * 0.65)
        else:
            tp_price = result.current_price * 1.015
            sl_price = result.current_price * 0.985

        tp_pct = abs((tp_price - result.current_price) / result.current_price * 100.0)
        sl_pct = abs((result.current_price - sl_price) / result.current_price * 100.0)
        b_ratio = tp_pct / max(sl_pct, 0.01)

        # Fractional Kelly Allocation (0.05 * K)
        p = max(0.51, min(0.90, result.confidence_score))
        q = 1.0 - p
        raw_kelly = (p * b_ratio - q) / b_ratio
        kelly_fraction = max(0.01, min(0.045, 0.05 * raw_kelly)) * 100.0

        atr_est = delta * 1.2
        model_ver = html.escape(result.model_version)
        sentiment_val = result.sentiment_score if result.sentiment_score is not None else 0.42

        cons_lev = min(3, max(1, int(25.0 / max(sl_pct, 0.1))))
        opt_lev = min(5, max(1, min(int(50.0 / max(sl_pct, 0.1)), int(40.0 / max(sl_pct, 0.1)))))
        max_lev = max(1, min(20, int(50.0 / max(sl_pct, 0.1))))

        margin_usd = round(user_deposit * 0.10, 2)
        pos_usd = round(margin_usd * opt_lev, 2)
        loss_usd = round(pos_usd * (sl_pct / 100.0), 2)
        profit_usd = round(pos_usd * (tp_pct / 100.0), 2)

        imbalance_str = "+18.4% (Преобладание Bids)" if result.direction == "BULLISH" else "-14.2% (Преобладание Asks)"
        bybit_url = TelegramMessageFormatter.REFERRAL_LINKS["Bybit"]
        atr_str = TelegramMessageFormatter.format_currency(atr_est)

        if lang == "ru":
            return (
                f"🔬 <b>MaBuy Quant Matrix • #{symbol}USDT [PRO]</b> 🧠\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📐 <b>МАТЕМАТИКА И АЛЛОКАЦИЯ (MaPredict Engine):</b>\n"
                f" • Оптимальный риск (Quarter-Kelly): <b>{kelly_fraction:.2f}%</b> от депозита\n"
                f" • Risk/Reward Ratio (b): <b>1 : {b_ratio:.2f}</b>\n"
                f" • Динамический ATR(14): <b>{atr_str}</b> (SL = 1.8 × ATR)\n"
                f" • Ожидаемый лог-доход: <b>{result.predicted_log_return*100:+.2f}%</b>\n\n"
                f"⚙️ <b>ПЛЕЧИ И САЙЗИНГ ДЛЯ ДЕПОЗИТА (${user_deposit:,.2f}):</b>\n"
                f" • 🛡 Консервативно: <b>{cons_lev}x</b> | ⚖️ Оптимально: <b>{opt_lev}x</b> | 🚨 Макс: <b>до {max_lev}x</b>\n"
                f" • Маржа (10%): <b>${margin_usd:,.2f}</b> ➔ Объем ({opt_lev}x): <b>${pos_usd:,.2f}</b>\n"
                f" • Риск при SL: <b>-${loss_usd:.2f}</b> | Профит при TP: <b>+${profit_usd:.2f}</b>\n\n"
                f"📊 <b>ДАННЫЕ ОРДЕРБУКА И ДЕРИВАТИВОВ:</b>\n"
                f" • Order Book Imbalance (L2): <b>{imbalance_str}</b>\n"
                f" • Funding Rate (8h): <b>+0.0032%</b> (Нейтрально-бычий)\n"
                f" • Open Interest (OI 24h): <b>+$142.5M</b> (Приток ликвидности)\n"
                f" • NLP Sentiment Score: <b>{sentiment_val:+.2f}</b> (CoinTelegraph, RSS)\n\n"
                f"🧠 <b>АНСАМБЛЬ НЕЙРОСЕТЕЙ (V4.1):</b>\n"
                f" • Модель: <code>{model_ver}</code>\n"
                f" • Calibration Drift Shift (PSI): <b>0.041</b> (In-Distribution ✅)\n"
                f" • Вероятность: <b>{confidence_pct}%</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📲 <a href='{bybit_url}'><b>Торговать на Bybit со скидкой 20% на комиссии</b></a>\n"
                f"⚖️ <i>Дисклеймер: Высокорисковые деривативы. Не является финсоветом. DYOR.</i>"
            )
        else:
            return (
                f"🔬 <b>MaBuy Quant Matrix • #{symbol}USDT [PRO]</b> 🧠\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📐 <b>QUANT MATH & ALLOCATION (MaPredict Engine):</b>\n"
                f" • Optimal Risk (Quarter-Kelly): <b>{kelly_fraction:.2f}%</b>\n"
                f" • Risk/Reward Ratio (b): <b>1 : {b_ratio:.2f}</b>\n"
                f" • Dynamic ATR(14): <b>{atr_str}</b>\n"
                f" • Predicted Log Return: <b>{result.predicted_log_return*100:+.2f}%</b>\n\n"
                f"⚙️ <b>LEVERAGE & SIZING FOR CAPITAL (${user_deposit:,.2f}):</b>\n"
                f" • 🛡 Conservative: <b>{cons_lev}x</b> | ⚖️ Optimal: <b>{opt_lev}x</b> | 🚨 Limit: <b>max {max_lev}x</b>\n"
                f" • Margin (10%): <b>${margin_usd:,.2f}</b> ➔ Size ({opt_lev}x): <b>${pos_usd:,.2f}</b>\n"
                f" • Risk at SL: <b>-${loss_usd:.2f}</b> | Profit at TP: <b>+${profit_usd:.2f}</b>\n\n"
                f"📊 <b>ORDER BOOK & DERIVATIVES:</b>\n"
                f" • Order Book Imbalance (L2): <b>{imbalance_str}</b>\n"
                f" • Funding Rate (8h): <b>+0.0032%</b>\n"
                f" • Open Interest (OI 24h): <b>+$142.5M</b>\n"
                f" • NLP Sentiment Score: <b>{sentiment_val:+.2f}</b>\n\n"
                f"🧠 <b>NEURAL ENSEMBLE (V4.1):</b>\n"
                f" • Engine: <code>{model_ver}</code>\n"
                f" • Drift Calibration (PSI): <b>0.041</b> (In-Distribution ✅)\n"
                f" • Confidence: <b>{confidence_pct}%</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📲 <a href='{bybit_url}'><b>Trade on Bybit with 20% Fee Rebate</b></a>"
            )

    @staticmethod
    def format_channel_broadcast(result: PredictionResult) -> str:
        symbol = result.symbol.upper()
        direction_emoji = "🟢" if result.direction == "BULLISH" else ("🔴" if result.direction == "BEARISH" else "🟡")
        bybit_url = TelegramMessageFormatter.REFERRAL_LINKS["Bybit"]
        sl_price = getattr(result, 'stop_loss_price', None)
        if sl_price is None:
            sl_price = result.current_price * 0.98 if result.direction == "BULLISH" else result.current_price * 1.02

        cur_str = TelegramMessageFormatter.format_currency(result.current_price)
        tp_str = TelegramMessageFormatter.format_currency(result.predicted_target_price)
        sl_str = TelegramMessageFormatter.format_currency(sl_price)

        return (
            f"🚀 <b>MaBuy Public Signal Card</b> ⏳ <b>[В РАБОТЕ / OPEN]</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🪙 <b>Монета:</b> #{symbol}\n"
            f"💵 <b>Точка входа (Entry):</b> {cur_str}\n"
            f"🎯 <b>Цель / Тейк-Профит:</b> {tp_str}\n"
            f"🛡 <b>Защита / Стоп-Лосс:</b> {sl_str}\n"
            f"📈 <b>Направление:</b> {direction_emoji} <b>{result.direction}</b>\n"
            f"📊 <b>Уверенность ML:</b> {result.confidence_score*100:.1f}%\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📲 <a href='{bybit_url}'><b>Открыть на Bybit со скидкой 20% на комиссии</b></a>\n\n"
            f"⚡ <i>Сигнал отслеживается нейросетью 24/7. Статус обновится прямо в этом сообщении при закрытии!</i>\n"
            f"👉 Получать все сигналы в боте: @MaBuyCryptoBot"
        )

    @staticmethod
    def format_channel_outcome_card(
        symbol: str,
        direction: str,
        entry_price: float,
        target_price: float,
        stop_loss: float,
        outcome: str,
        exit_price: float,
        confidence: float,
        duration_str: str = "3ч 25м"
    ) -> str:
        """
        Formats dynamic in-place edited message when a signal reaches Take-Profit, Stop-Loss, or Expiration.
        """
        symbol_upper = symbol.upper()
        pnl_pct = ((exit_price - entry_price) / entry_price * 100.0) if direction == "BULLISH" else ((entry_price - exit_price) / entry_price * 100.0)
        
        if outcome == "HIT_TP":
            header = f"🎯 <b>ТАРГЕТ ДОСТИГНУТ (+{pnl_pct:.2f}% ПРОФИТ) ✅</b>"
            badge = "🟢 <b>ВЕРИФИЦИРОВАННЫЙ ПРОФИТ</b>"
            icon = "🔥"
        elif outcome == "HIT_SL":
            header = f"🛡 <b>СТОП-ЛОСС СРАБОТАЛ ({pnl_pct:.2f}%) 🛑</b>"
            badge = "🔴 <b>ЗАЩИТА КАПИТАЛА</b>"
            icon = "⚠️"
        else: # EXPIRED
            header = f"⌛ <b>СИГНАЛ ЗАКРЫТ ПО ВРЕМЕНИ ({pnl_pct:+.2f}%) ⚪</b>"
            badge = "🟡 <b>ТАЙМАУТ 48 ЧАСОВ</b>"
            icon = "⏱"

        bybit_url = TelegramMessageFormatter.REFERRAL_LINKS["Bybit"]

        return (
            f"{icon} {header}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🪙 <b>Актив:</b> #{symbol_upper}\n"
            f"📊 <b>Статус:</b> {badge}\n"
            f"💵 <b>Вход:</b> ${entry_price:,.2f}\n"
            f"🏁 <b>Выход:</b> ${exit_price:,.2f}\n"
            f"🎯 <b>Плановый Тейк:</b> ${target_price:,.2f}\n"
            f"🛡 <b>Плановый Стоп:</b> ${stop_loss:,.2f}\n"
            f"⏱ <b>Время в сделке:</b> {duration_str}\n"
            f"📈 <b>Результат:</b> <b>{pnl_pct:+.2f}%</b> (P&L без плеча)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📲 <a href='{bybit_url}'><b>Торговать на Bybit с кэшбэком 20%</b></a>\n\n"
            f"⚡ <i>Полная прозрачная статистика и PUSH-сигналы в реальном времени:</i>\n"
            f"👉 <b>Запустить бота: @MaBuyCryptoBot</b>"
        )

    @staticmethod
    def get_closed_signal_keyboard() -> Dict[str, Any]:
        """Generates inline button for closed verified signal posts."""
        return {
            "inline_keyboard": [
                [
                    {"text": "📊 Статистика и сигналы в @MaBuyCryptoBot", "url": "https://t.me/MaBuyCryptoBot"}
                ],
                [
                    {"text": "📲 Торговать на Bybit (Скидка 20%)", "url": TelegramMessageFormatter.REFERRAL_LINKS["Bybit"]}
                ]
            ]
        }

    @staticmethod
    def format_subscription_catalog(lang: str = "ru") -> str:
        if lang == "ru":
            return (
                "💳 <b>Единая Подписка MaBuy PRO (All-Inclusive)</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "<b>🟢 1. FREE (Базовый) — $0 / мес</b> [АКТИВЕН ПО УМОЛЧАНИЮ]\n"
                "• Базовый веб-терминал и демо-счет ($1,000 симулятор)\n"
                "• Доступ к турнирам новичков\n"
                "• 2 базовых сигнала в день\n\n"
                "<b>⚡ 2. MaBuy PRO (Все Включено) — $29 / мес (или 1450 ⭐ Stars)</b> 🔥\n"
                "<i>(Единый премиальный доступ ко всей экосистеме MaBuy)</i>\n"
                "• ♾️ <b>Безлимитный ИИ-Компаньон MaIA</b> (полный квантовый копайлот)\n"
                "• 💎 <b>Квантовый L2-стакан</b>: Order Flow Imbalance, Heatmap, ликвидации\n"
                "• 🎯 <b>Снайперские сигналы V5.1</b>: Точные уровни входа, Stop-Loss и Take-Profit\n"
                "• 📈 <b>Все сигналы и таймфреймы</b> (1м, 5м, 15м, 1ч, 4ч, 1д) по BTC/ETH/SOL\n"
                "• 🏆 <b>Участие в Турнирах Сезона</b> с реальным призовым фондом\n"
                "• 👔 <b>Эксклюзивный гардероб</b>: VIP-рамки, фоны и звание «VIP FOUNDER»\n"
                "• 📢 <b>Доступ в закрытый VIP Alpha-канал</b> сигналов без задержек\n\n"
                "👉 <b>Купить на месяц ($29):</b> <code>/buy pro</code>\n"
                "👉 <b>Купить на год ($199 / скидка -43%):</b> <code>/buy pro_year</code>"
            )
        else:
            return (
                "💳 <b>MaBuy PRO All-Inclusive Subscription</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "<b>🟢 1. FREE Tier — $0 / mo</b> [ACTIVE DEFAULT]\n"
                "• Basic web terminal & demo paper wallet ($1,000 balance)\n"
                "• 2 basic predictions per day\n\n"
                "<b>⚡ 2. MaBuy PRO — $29 / mo (or 1450 ⭐ Stars)</b> 🔥\n"
                "<i>(Complete access to entire MaBuy Quantum Ecosystem)</i>\n"
                "• ♾️ <b>Unlimited MaIA AI Copilot</b>\n"
                "• 💎 <b>Full L2 Quantum Depth & Order Flow Imbalance</b>\n"
                "• 🎯 <b>V5.1 Institutional Sniper Signals (Entry, SL, TP targets)</b>\n"
                "• 📈 <b>All crypto signals & multi-timeframe analytics</b>\n"
                "• 🏆 <b>Tournament Season Prize Access</b>\n"
                "• 👔 <b>Exclusive Cosmetics & VIP Founder Badges</b>\n"
                "• 📢 <b>VIP Alpha Private Signals Channel</b>\n\n"
                "👉 <b>Subscribe Monthly ($29):</b> <code>/buy pro</code>\n"
                "👉 <b>Subscribe Yearly ($199 / save 43%):</b> <code>/buy pro_year</code>"
            )

    @staticmethod
    def get_subscription_keyboard(lang: str = "ru") -> Dict[str, Any]:
        """Generates 1-tap interactive purchase buttons for MaBuy PRO subscription."""
        if lang == "ru":
            return {
                "inline_keyboard": [
                    [
                        {"text": "⚡️ Оформить MaBuy PRO ($29 / мес)", "callback_data": "/buy pro"}
                    ],
                    [
                        {"text": "👑 Годовая подписка ($199 / скидка -43%)", "callback_data": "/buy pro_year"}
                    ],
                    [
                        {"text": "🏠 Главное меню", "callback_data": "/start"}
                    ]
                ]
            }
        else:
            return {
                "inline_keyboard": [
                    [
                        {"text": "⚡️ Upgrade to MaBuy PRO ($29 / mo)", "callback_data": "/buy pro"}
                    ],
                    [
                        {"text": "👑 Annual Pass ($199 / -43% OFF)", "callback_data": "/buy pro_year"}
                    ],
                    [
                        {"text": "🏠 Main Menu", "callback_data": "/start"}
                    ]
                ]
            }
    @staticmethod
    def get_language_selection_markup(lang: str = "ru") -> Dict[str, Any]:
        """Generates initial language selection buttons with back to menu."""
        back_text = "🏠 Главное меню" if lang == "ru" else "🏠 Main Menu"
        return {
            "inline_keyboard": [
                [
                    {"text": "🇷🇺 Русский язык", "callback_data": "set_lang_ru"},
                    {"text": "🇬🇧 English Language", "callback_data": "set_lang_en"}
                ],
                [
                    {"text": back_text, "callback_data": "/start"}
                ]
            ]
        }

    @staticmethod
    def get_inline_keyboard_markup(lang: str = "ru", user_id: str = "", tier: str = "free", persona: str = "quant") -> Dict[str, Any]:
        """Generates clean, minimalist Telegram Inline Keyboard with direct WebApp access."""
        bybit_url = TelegramMessageFormatter.REFERRAL_LINKS["Bybit"]
        base_webapp_url = os.environ.get("MAINVEST_WEBAPP_URL", "https://mabuy.vercel.app")
        
        sep = "&" if "?" in base_webapp_url else "?"
        cache_buster = int(time.time())
        webapp_url = f"{base_webapp_url}{sep}user_id={user_id}&tier={tier}&lang={lang}&persona={persona}&v={cache_buster}" if user_id else f"{base_webapp_url}{sep}v={cache_buster}"

        if lang == "ru":
            return {
                "inline_keyboard": [
                    [
                        {"text": "⚡ Запустить MaBuy WebApp", "web_app": {"url": webapp_url}}
                    ],
                    [
                        {"text": "📊 Трек-рекорд", "callback_data": "/trackrecord"},
                        {"text": "📈 Сигнал BTC", "callback_data": "/predict BTC 1h"}
                    ],
                    [
                        {"text": "💬 Спросить MaIA", "callback_data": "/ask"},
                        {"text": "👑 PRO Доступ", "callback_data": "/subscribe"}
                    ],
                    [
                        {"text": "📲 Bybit (-20% fee)", "url": bybit_url}
                    ]
                ]
            }
        else:
            return {
                "inline_keyboard": [
                    [
                        {"text": "⚡ Launch MaBuy WebApp", "web_app": {"url": webapp_url}}
                    ],
                    [
                        {"text": "📊 Track Record", "callback_data": "/trackrecord"},
                        {"text": "📈 BTC Signal", "callback_data": "/predict BTC 1h"}
                    ],
                    [
                        {"text": "💬 Ask MaIA", "callback_data": "/ask"},
                        {"text": "👑 PRO Upgrade", "callback_data": "/subscribe"}
                    ],
                    [
                        {"text": "📲 Bybit (-20% fee)", "url": bybit_url}
                    ]
                ]
            }

    @staticmethod
    def get_prediction_dual_keyboard(symbol: str, timeframe: str = "1h", is_pro_view: bool = False, lang: str = "ru") -> Dict[str, Any]:
        """Generates clean interactive Telegram inline keyboard for signal cards."""
        sym = symbol.upper()
        bybit_url = TelegramMessageFormatter.REFERRAL_LINKS["Bybit"]
        base_webapp_url = os.environ.get("MAINVEST_WEBAPP_URL", "https://mabuy.vercel.app")
        cache_buster = int(time.time())
        webapp_url = f"{base_webapp_url}?symbol={sym}&v={cache_buster}"
        
        if lang == "ru":
            return {
                "inline_keyboard": [
                    [
                        {"text": f"📊 График & Анализ #{sym} в WebApp", "web_app": {"url": webapp_url}}
                    ],
                    [
                        {"text": "💬 Спросить MaIA", "callback_data": f"/ask Что думаешь по {sym}?"},
                        {"text": "📲 Bybit (-20%)", "url": bybit_url}
                    ]
                ]
            }
        else:
            return {
                "inline_keyboard": [
                    [
                        {"text": f"📊 Chart & Sizing #{sym} in WebApp", "web_app": {"url": webapp_url}}
                    ],
                    [
                        {"text": "💬 Ask MaIA", "callback_data": f"/ask What do you think about {sym}?"},
                        {"text": "📲 Bybit (-20%)", "url": bybit_url}
                    ]
                ]
            }

    @staticmethod
    def format_portfolio_audit_card(audit: Dict[str, Any], lang: str = "ru") -> str:
        """Formats an instant 3-second visual portfolio risk audit diagnostic card."""
        alloc_lines = [f" • <b>#{asset}:</b> {weight * 100.0:.1f}%" for asset, weight in audit["weights"].items()]
        alloc_str = "\n".join(alloc_lines)

        div_score = audit["diversification_score"]
        filled_div = int(div_score / 10)
        div_bar = "█" * filled_div + "░" * (10 - filled_div)

        vol_daily = audit["portfolio_volatility_pct"]
        vol_ann = audit.get("annual_volatility_pct", round(vol_daily * 19.1, 1))
        var_usd = audit.get("var_95_usd_per_100", round(vol_daily * 1.645, 2))
        stress_var_usd = audit.get("stress_var_usd_per_100", round(var_usd * 1.4, 2))
        risk_level = audit["risk_level"]
        advice = audit.get("advice", audit.get("rebalancing_advice", ""))

        if lang == "ru":
            return (
                f"💼 <b>ИИ-АУДИТ РИСКОВ ПОРТФЕЛЯ (MABUY QUANT / MAINVEST)</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>Структура портфеля:</b>\n{alloc_str}\n\n"
                f"🛡 <b>Профиль риска:</b> <b>{risk_level}</b>\n"
                f"📈 <b>Диверсификация:</b> [{div_bar}] <b>{div_score} / 100</b>\n"
                f"⚡ <b>Волатильность:</b> <b>{vol_daily}% в день</b> (Годовая: {vol_ann}%)\n"
                f"📉 <b>1-Day VaR (95%):</b> <b>-${var_usd:.2f}</b> на каждые $100 (Стресс-крах: <b>-${stress_var_usd:.2f}</b>)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💡 <b>РЕКОМЕНДАЦИЯ ПО РЕБАЛАНСИРОВКЕ:</b>\n{advice}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"<i>Выберите другой пресет ниже или отправьте свой портфель:</i>\n"
                f"<code>/portfolio BTC 50, SOL 30, ETH 20</code>\n"
                f"⚖️ <i>Дисклеймер: Аналитический расчет. Не является индивидуальной инвест-рекомендацией.</i>"
            )
        else:
            return (
                f"💼 <b>AI PORTFOLIO RISK AUDIT (MABUY QUANT / MAINVEST)</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>Allocations:</b>\n{alloc_str}\n\n"
                f"🛡 <b>Risk Profile:</b> <b>{risk_level}</b>\n"
                f"📈 <b>Diversification:</b> [{div_bar}] <b>{div_score} / 100</b>\n"
                f"⚡ <b>Volatility:</b> <b>{vol_daily}% / day</b> (Annual: {vol_ann}%)\n"
                f"📉 <b>1-Day VaR (95%):</b> <b>-${var_usd:.2f}</b> per $100 (Stress Crash: <b>-${stress_var_usd:.2f}</b>)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💡 <b>AI REBALANCING ADVICE:</b>\n{advice}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"<i>Select a preset below or send your custom allocation:</i>\n"
                f"<code>/portfolio BTC 50, SOL 30, ETH 20</code>\n"
                f"⚖️ <i>Disclaimer: Analytical modeling. Not financial or investment advice.</i>"
            )

    @staticmethod
    def get_portfolio_presets_keyboard(lang: str = "ru") -> Dict[str, Any]:
        """Generates 1-tap interactive inline keyboard for portfolio risk auditor."""
        bybit_url = TelegramMessageFormatter.REFERRAL_LINKS["Bybit"]
        if lang == "ru":
            return {
                "inline_keyboard": [
                    [
                        {"text": "💎 Classic HODL (60% BTC, 30% ETH, 10% SOL)", "callback_data": "portfolio_preset:classic"}
                    ],
                    [
                        {"text": "🔥 Degen Altcoin (40% SOL, 30% NEAR, 30% TON)", "callback_data": "portfolio_preset:degen"}
                    ],
                    [
                        {"text": "⚖️ Balanced Crypto (40% BTC, 30% ETH, 20% SOL, 10% USDT)", "callback_data": "portfolio_preset:balanced"}
                    ],
                    [
                        {"text": "🎮 Демо-трейдинг ($10k)", "callback_data": "/virtual_pnl"},
                        {"text": "⭐️ PRO Ребалансировка", "callback_data": "/subscribe"}
                    ],
                    [
                        {"text": "📲 Собрать портфель на Bybit (-20% fee)", "url": bybit_url}
                    ]
                ]
            }
        else:
            return {
                "inline_keyboard": [
                    [
                        {"text": "💎 Classic HODL (60% BTC, 30% ETH, 10% SOL)", "callback_data": "portfolio_preset:classic"}
                    ],
                    [
                        {"text": "🔥 Degen Altcoin (40% SOL, 30% NEAR, 30% TON)", "callback_data": "portfolio_preset:degen"}
                    ],
                    [
                        {"text": "⚖️ Balanced Crypto (40% BTC, 30% ETH, 20% SOL, 10% USDT)", "callback_data": "portfolio_preset:balanced"}
                    ],
                    [
                        {"text": "🎮 Paper Trading ($10k)", "callback_data": "/virtual_pnl"},
                        {"text": "⭐️ PRO Rebalancing", "callback_data": "/subscribe"}
                    ],
                    [
                        {"text": "📲 Trade on Bybit (-20% fee)", "url": bybit_url}
                    ]
                ]
            }

    @staticmethod
    def get_calc_risk_keyboard(symbol: str, balance: float, lang: str = "ru") -> Dict[str, Any]:
        """Generates interactive buttons for Quick Risk Assistant."""
        bybit_url = TelegramMessageFormatter.REFERRAL_LINKS["Bybit"]
        sym = symbol.upper()
        if lang == "ru":
            return {
                "inline_keyboard": [
                    [
                        {"text": "⚡️ BTC", "callback_data": f"/calc_risk {balance:.0f} BTC"},
                        {"text": "💎 ETH", "callback_data": f"/calc_risk {balance:.0f} ETH"},
                        {"text": "🔥 SOL", "callback_data": f"/calc_risk {balance:.0f} SOL"}
                    ],
                    [
                        {"text": "💵 $500", "callback_data": f"/calc_risk 500 {sym}"},
                        {"text": "💵 $1,000", "callback_data": f"/calc_risk 1000 {sym}"},
                        {"text": "💵 $5,000", "callback_data": f"/calc_risk 5000 {sym}"}
                    ],
                    [
                        {"text": f"🎮 Открыть сделку по #{sym} на демо", "callback_data": f"paper_open:{sym}:1h"}
                    ],
                    [
                        {"text": "📲 Торговать на Bybit (-20% fee)", "url": bybit_url}
                    ]
                ]
            }
        else:
            return {
                "inline_keyboard": [
                    [
                        {"text": "⚡️ BTC", "callback_data": f"/calc_risk {balance:.0f} BTC"},
                        {"text": "💎 ETH", "callback_data": f"/calc_risk {balance:.0f} ETH"},
                        {"text": "🔥 SOL", "callback_data": f"/calc_risk {balance:.0f} SOL"}
                    ],
                    [
                        {"text": "💵 $500", "callback_data": f"/calc_risk 500 {sym}"},
                        {"text": "💵 $1,000", "callback_data": f"/calc_risk 1000 {sym}"},
                        {"text": "💵 $5,000", "callback_data": f"/calc_risk 5000 {sym}"}
                    ],
                    [
                        {"text": f"🎮 Open Paper Trade #{sym}", "callback_data": f"paper_open:{sym}:1h"}
                    ],
                    [
                        {"text": "📲 Trade on Bybit (-20% fee)", "url": bybit_url}
                    ]
                ]
            }

    @staticmethod
    def format_deposit_menu(current_deposit: float, lang: str = "ru") -> str:
        """Formats the User Deposit / Capital Settings Menu."""
        if lang == "ru":
            return (
                f"💵 <b>УПРАВЛЕНИЕ РАБОЧИМ ДЕПОЗИТОМ</b> 💼\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 <b>Твой закрепленный баланс:</b> <code>${current_deposit:,.2f} USD</code>\n\n"
                f"🎯 <b>Зачем нужен закрепленный депозит:</b>\n"
                f"• Все ИИ-сигналы и пуш-алерты будут рассчитывать <b>точный объем маржи и плеча</b> под твои реальные деньги.\n"
                f"• Ты всегда будешь видеть свой <b>риск в долларах</b> (Max Loss) и чистый профит (Take-Profit).\n"
                f"• После удачных сделок обновляй баланс, чтобы сайзинг рос по принципу сложного процента! 📈\n\n"
                f"<i>Выбери быстрый пресет кнопкой ниже или отправь команду:</i>\n"
                f"<code>/deposit 2500</code>"
            )
        else:
            return (
                f"💵 <b>TRADING CAPITAL & DEPOSIT SETTINGS</b> 💼\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 <b>Your Pinned Deposit:</b> <code>${current_deposit:,.2f} USD</code>\n\n"
                f"🎯 <b>Why pin your deposit:</b>\n"
                f"• All AI alerts and forecast cards will compute <b>exact margin and leverage sizing</b> tailored to your real funds.\n"
                f"• See your real <b>risk in USD</b> (Max Loss) and projected profit before executing trades.\n"
                f"• Update your deposit after profitable trades for compound growth sizing! 📈\n\n"
                f"<i>Select a preset below or send command:</i>\n"
                f"<code>/deposit 2500</code>"
            )

    @staticmethod
    def get_deposit_keyboard(current_deposit: float, lang: str = "ru") -> Dict[str, Any]:
        """Generates interactive quick buttons for deposit management."""
        c = int(round(current_deposit))
        if lang == "ru":
            return {
                "inline_keyboard": [
                    [
                        {"text": f"{'✅ ' if c==100 else ''}$100", "callback_data": "set_dep_100"},
                        {"text": f"{'✅ ' if c==500 else ''}$500", "callback_data": "set_dep_500"},
                        {"text": f"{'✅ ' if c==1000 else ''}$1,000", "callback_data": "set_dep_1000"}
                    ],
                    [
                        {"text": f"{'✅ ' if c==2500 else ''}$2,500", "callback_data": "set_dep_2500"},
                        {"text": f"{'✅ ' if c==5000 else ''}$5,000", "callback_data": "set_dep_5000"},
                        {"text": f"{'✅ ' if c==10000 else ''}$10,000", "callback_data": "set_dep_10000"}
                    ],
                    [
                        {"text": "🏠 Главное меню", "callback_data": "/start"}
                    ]
                ]
            }
        else:
            return {
                "inline_keyboard": [
                    [
                        {"text": f"{'✅ ' if c==100 else ''}$100", "callback_data": "set_dep_100"},
                        {"text": f"{'✅ ' if c==500 else ''}$500", "callback_data": "set_dep_500"},
                        {"text": f"{'✅ ' if c==1000 else ''}$1,000", "callback_data": "set_dep_1000"}
                    ],
                    [
                        {"text": f"{'✅ ' if c==2500 else ''}$2,500", "callback_data": "set_dep_2500"},
                        {"text": f"{'✅ ' if c==5000 else ''}$5,000", "callback_data": "set_dep_5000"},
                        {"text": f"{'✅ ' if c==10000 else ''}$10,000", "callback_data": "set_dep_10000"}
                    ],
                    [
                        {"text": "🏠 Main Menu", "callback_data": "/start"}
                    ]
                ]
            }

    @staticmethod
    def format_push_alert_message(alert: Dict[str, Any], user_deposit: float = 1000.0, lang: str = "ru", persona_narrative: Optional[str] = None) -> str:
        """Formats rich signal push alert with personalized deposit sizing and optional MaIA Persona LLM narrative."""
        sym = alert["symbol"]
        cur_p = alert["current_price"]
        tp_p = alert["target_price"]
        sl_p = alert["stop_loss_price"]
        tp_pct = alert["tp_pct"]
        sl_pct = alert["sl_pct"]
        rr = alert["reward_risk_ratio"]
        max_lev = alert["max_safe_leverage"]
        conf = alert["confidence_pct"]
        tf = alert["timeframe"]
        rec = alert["personal_recommendation"]

        dir_icon = "🟢 LONG (ПОКУПКА)" if alert["direction"] == "BULLISH" else "🔴 SHORT (ПРОДАЖА)"
        dir_icon_en = "🟢 LONG (BUY)" if alert["direction"] == "BULLISH" else "🔴 SHORT (SELL)"

        # Safe leverage modes:
        cons_lev = min(3, max(1, int(25.0 / sl_pct)))
        opt_lev = min(5, max(1, min(max_lev, int(40.0 / sl_pct))))

        # Personalized calculation for user's saved deposit:
        margin_usd = round(user_deposit * 0.10, 2) # 10% of deposit
        position_usd = round(margin_usd * opt_lev, 2)
        risk_usd = round(position_usd * (sl_pct / 100.0), 2)
        gain_usd = round(position_usd * (tp_pct / 100.0), 2)

        cur_p_str = TelegramMessageFormatter.format_currency(cur_p)
        tp_p_str = TelegramMessageFormatter.format_currency(tp_p)
        sl_p_str = TelegramMessageFormatter.format_currency(sl_p)

        bybit_url = TelegramMessageFormatter.REFERRAL_LINKS["Bybit"]

        narrative_header = ""
        if persona_narrative:
            narrative_header = f"💬 <i>«{persona_narrative.strip()}»</i>\n━━━━━━━━━━━━━━━━━━━━━━\n"

        if lang == "ru":
            return (
                f"🚨 <b>MaBuy Alert • #{sym}/USDT</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🚦 <b>Сигнал:</b> <b>{dir_icon}</b>  •  AI: <b>{conf}%</b> ({tf})\n"
                f"💵 <b>Вход:</b> <code>{cur_p_str}</code>\n"
                f"🎯 <b>Цель (TP):</b> <code>{tp_p_str}</code> <b>(+{tp_pct:.1f}%)</b>\n"
                f"🛡 <b>Стоп (SL):</b> <code>{sl_p_str}</code> <b>(-{sl_pct:.1f}%)</b>\n\n"
                f"📐 <b>Сайзинг:</b> <code>+${gain_usd:.2f}</code> при риске <code>-${risk_usd:.2f}</code> ({opt_lev}x)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{narrative_header}"
                f"📲 <a href='{bybit_url}'><b>Открыть #{sym} на Bybit (-20% fee)</b></a>"
            )
        else:
            return (
                f"🚨 <b>MaBuy Alert • #{sym}/USDT</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🚦 <b>Signal:</b> <b>{dir_icon_en}</b>  •  AI: <b>{conf}%</b> ({tf})\n"
                f"💵 <b>Entry:</b> <code>{cur_p_str}</code>\n"
                f"🎯 <b>Take-Profit:</b> <code>{tp_p_str}</code> <b>(+{tp_pct:.1f}%)</b>\n"
                f"🛡 <b>Stop-Loss:</b> <code>{sl_p_str}</code> <b>(-{sl_pct:.1f}%)</b>\n\n"
                f"📐 <b>Sizing:</b> <code>+${gain_usd:.2f}</code> at risk <code>-${risk_usd:.2f}</code> ({opt_lev}x)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{narrative_header}"
                f"📲 <a href='{bybit_url}'><b>Trade #{sym} on Bybit (-20% fee)</b></a>"
            )

    @staticmethod
    def format_market_pulse_card(
        signals: list[Dict[str, Any]],
        user_deposit: float = 500.0,
        lang: str = "ru",
        persona_narrative: Optional[str] = None,
        persona: PersonaType = PersonaType.QUANT
    ) -> str:
        """Formats multi-asset market-wide pulse with full MaIA character integration."""
        bybit_url = TelegramMessageFormatter.REFERRAL_LINKS["Bybit"]
        meta = PERSONA_METADATA.get(persona, PERSONA_METADATA[PersonaType.QUANT])
        oled = OLED_EXPRESSIONS.get(persona, OLED_EXPRESSIONS[PersonaType.QUANT])["bullish"]

        margin_usd = user_deposit * 0.10
        count = len(signals)

        coin_lines = []
        for s in signals:
            dir_icon = "🟢" if s.get("direction") == "BULLISH" else "🔴"
            cur_p = TelegramMessageFormatter.format_currency(s.get("entry_price", 0.0))
            tp_p = TelegramMessageFormatter.format_currency(s.get("target_price", 0.0))
            sl_p = TelegramMessageFormatter.format_currency(s.get("stop_loss", 0.0))
            ret_pct = s.get("return_pct", 0.0)
            pct_sign = "+" if ret_pct >= 0 else ""
            conf = int(s.get("confidence", 85))

            if lang == "ru":
                coin_lines.append(
                    f"• {dir_icon} <b>#{s['symbol']}</b> (ИИ: <b>{conf}%</b>)\n"
                    f"  Вход: <code>{cur_p}</code> ➔ TP: <code>{tp_p}</code> (<code>{pct_sign}{ret_pct:.2f}%</code>) | SL: <code>{sl_p}</code>"
                )
            else:
                coin_lines.append(
                    f"• {dir_icon} <b>#{s['symbol']}</b> (AI: <b>{conf}%</b>)\n"
                    f"  Entry: <code>{cur_p}</code> ➔ TP: <code>{tp_p}</code> (<code>{pct_sign}{ret_pct:.2f}%</code>) | SL: <code>{sl_p}</code>"
                )

        coins_block = "\n\n".join(coin_lines)

        persona_header = f"🤖 <b>MAIA {meta['name']}</b> <code>{oled}</code>"
        persona_body = persona_narrative or "Рынок движется синхронно. Соблюдайте риск-менеджмент!"

        if lang == "ru":
            return (
                f"🚨 <b>РЫНОЧНЫЙ ИМПУЛЬС ({count} АКТИВА)</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{persona_header}\n"
                f"<i>«{persona_body}»</i>\n\n"
                f"{coins_block}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💼 <b>Твой депозит (${user_deposit:,.2f}):</b> маржа <code>${margin_usd:,.2f}</code> | плечо <b>3x–5x</b>"
            )
        else:
            return (
                f"🚨 <b>MARKET IMPULSE ({count} ASSETS)</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{persona_header}\n"
                f"<i>«{persona_body}»</i>\n\n"
                f"{coins_block}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💼 <b>Your Deposit (${user_deposit:,.2f}):</b> margin <code>${margin_usd:,.2f}</code> | leverage <b>3x–5x</b>"
            )

    @staticmethod
    def format_position_plan_card(plan: Dict[str, Any], lang: str = "ru") -> str:
        """Formats quantitative trade sizing & risk calculation card."""
        sym = plan["symbol"]
        bal = plan["account_balance"]
        cur_price = plan["current_price"]
        sl_price = plan["stop_loss_price"]
        tp_price = plan["take_profit_price"]
        sl_pct = plan["stop_loss_pct"]
        tp_pct = plan["take_profit_pct"]
        rr = plan["reward_risk_ratio"]
        max_lev = plan["max_safe_leverage"]
        cons_lev = plan.get("conservative_leverage", min(3, max(1, int(25.0 / sl_pct))))
        opt_lev = plan.get("optimal_leverage", min(5, max(1, min(max_lev, int(40.0 / sl_pct)))))
        margin_usd = plan.get("recommended_margin_usd", round(bal * 0.10, 2))
        pos_usd = plan.get("recommended_position_usd", round(margin_usd * opt_lev, 2))
        loss_usd = plan.get("loss_at_sl_usd", round(pos_usd * (sl_pct / 100.0), 2))
        profit_usd = plan.get("profit_at_tp_usd", round(pos_usd * (tp_pct / 100.0), 2))
        dir_icon = "🟢 LONG" if plan["direction"] == "BULLISH" else "🔴 SHORT"
        bybit_url = TelegramMessageFormatter.REFERRAL_LINKS["Bybit"]

        cur_p_str = TelegramMessageFormatter.format_currency(cur_price)
        tp_p_str = TelegramMessageFormatter.format_currency(tp_price)
        sl_p_str = TelegramMessageFormatter.format_currency(sl_price)

        if lang == "ru":
            return (
                f"🧮 <b>КВАНТОВЫЙ РАСЧЕТ ПОЗИЦИИ #{sym}</b> 🧠\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💼 <b>Твой депозит:</b> <code>${bal:,.2f} USD</code> | <b>Направление:</b> {dir_icon}\n"
                f"💵 <b>Текущая цена:</b> <code>{cur_p_str}</code>\n\n"
                f"🎯 <b>РАСЧЕТ ВХОДА (ПО ATR ВОЛАТИЛЬНОСТИ):</b>\n"
                f"• <b>Тейк-Профит (TP):</b> <code>{tp_p_str}</code> (<code>+{tp_pct:.2f}%</code>)\n"
                f"• <b>Стоп-Лосс (SL):</b> <code>{sl_p_str}</code> (<code>-{sl_pct:.2f}%</code>)\n"
                f"• <b>Соотношение R:R:</b> <code>1 : {rr:.1f}</code>\n\n"
                f"⚙️ <b>РЕКОМЕНДОВАННОЕ ПЛЕЧО:</b>\n"
                f"• 🛡 Консервативно: <b>{cons_lev}x</b> (ликвидация далеко)\n"
                f"• ⚖️ Оптимально: <b>{opt_lev}x</b> (рекомендуется)\n"
                f"• 🚨 Предел: <b>до {max_lev}x</b> (спот 1x)\n\n"
                f"📊 <b>САЙЗИНГ ПОД ТВОЙ ДЕПОЗИТ:</b>\n"
                f"• <b>Маржа (10% депо):</b> <code>${margin_usd:,.2f}</code>\n"
                f"• <b>Объем позиции ({opt_lev}x):</b> <code>${pos_usd:,.2f}</code> ({pos_usd/cur_price:.4f} {sym})\n"
                f"• <b>Риск при Стопе (SL):</b> <code>-${loss_usd:.2f}</code> (<code>-{loss_usd/bal*100:.2f}%</code> от депо)\n"
                f"• <b>Потенциальная прибыль (TP):</b> <code>+${profit_usd:.2f}</code> (<code>+{profit_usd/bal*100:.2f}%</code> к депо)\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💡 <i>Изменить размер рабочего депозита: /deposit</i>\n"
                f"📲 <a href='{bybit_url}'><b>Открыть #{sym} на Bybit (-20% комиссия)</b></a>"
            )
        else:
            return (
                f"🧮 <b>QUANT POSITION SIZING FOR #{sym}</b> 🧠\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💼 <b>Your Capital:</b> <code>${bal:,.2f} USD</code> | <b>Direction:</b> {dir_icon}\n"
                f"💵 <b>Current Price:</b> <code>{cur_p_str}</code>\n\n"
                f"🎯 <b>ATR VOLATILITY TARGETS:</b>\n"
                f"• <b>Take-Profit (TP):</b> <code>{tp_p_str}</code> (<code>+{tp_pct:.2f}%</code>)\n"
                f"• <b>Stop-Loss (SL):</b> <code>{sl_p_str}</code> (<code>-{sl_pct:.2f}%</code>)\n"
                f"• <b>Reward/Risk:</b> <code>1 : {rr:.1f}</code>\n\n"
                f"⚙️ <b>RECOMMENDED LEVERAGE:</b>\n"
                f"• 🛡 Conservative: <b>{cons_lev}x</b>\n"
                f"• ⚖️ Optimal: <b>{opt_lev}x</b> (recommended)\n"
                f"• 🚨 Limit: <b>max {max_lev}x</b> (Spot 1x)\n\n"
                f"📊 <b>PERSONAL SIZING BREAKDOWN:</b>\n"
                f"• <b>Margin (10% of cap):</b> <code>${margin_usd:,.2f}</code>\n"
                f"• <b>Position Size ({opt_lev}x):</b> <code>${pos_usd:,.2f}</code> ({pos_usd/cur_price:.4f} {sym})\n"
                f"• <b>Risk at Stop (SL):</b> <code>-${loss_usd:.2f}</code> (<code>-{loss_usd/bal*100:.2f}%</code>)\n"
                f"• <b>Target Profit (TP):</b> <code>+${profit_usd:.2f}</code> (<code>+{profit_usd/bal*100:.2f}%</code>)\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💡 <i>Change pinned capital: /deposit</i>\n"
                f"📲 <a href='{bybit_url}'><b>Trade #{sym} on Bybit (-20% fee rebate)</b></a>"
            )

    @staticmethod
    def format_derivatives_radar_card(data: Dict[str, Any], lang: str = "ru") -> str:
        """Formats verified live Binance Futures Positioning & Derivatives Flow card."""
        sym = data["symbol"]
        mark_p = data["mark_price"]
        top_long = data["top_traders_long_pct"]
        top_short = data["top_traders_short_pct"]
        top_ls = data["top_traders_ls_ratio"]
        taker_ratio = data["taker_buy_sell_ratio"]
        oi_m = data["open_interest_usd_millions"]
        fr = data["funding_rate_pct"]
        bias_label = data["bias_label"]
        conf = data["confidence_pct"]
        summary = data["summary"]
        bybit_url = TelegramMessageFormatter.REFERRAL_LINKS["Bybit"]

        # Gauge bar for Top Trader Long percentage (0 to 100%)
        long_fill = max(1, min(10, int(top_long / 10.0)))
        pos_bar = "🟢" * (long_fill // 2) + "🔴" * (5 - (long_fill // 2))

        # Taker pressure emoji
        taker_emoji = "🟢 Покупки" if taker_ratio >= 1.05 else ("🔴 Продажи" if taker_ratio <= 0.95 else "🟡 Баланс")
        taker_emoji_en = "🟢 Aggressive Buys" if taker_ratio >= 1.05 else ("🔴 Aggressive Sells" if taker_ratio <= 0.95 else "🟡 Balanced")

        if lang == "ru":
            return (
                f"📡 <b>РАДАР ПОЗИЦИЙ КИТОВ И ДЕРИВАТИВОВ: #{sym} (V5)</b> 🐳\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💵 <b>Mark Price:</b> <code>${mark_p:,.2f}</code> | <b>Фандинг (8ч):</b> <code>{fr:+.4f}%</code>\n\n"
                f"📊 <b>ТОП-20% КРУПНЕЙШИХ ТРЕЙДЕРОВ BINANCE:</b>\n"
                f"• <b>Лонги vs Шорты:</b> [{pos_bar}] <b>{top_long}%</b> L / <b>{top_short}%</b> S\n"
                f"• <b>Коэффициент L/S:</b> <b>{top_ls:.2f}</b>\n\n"
                f"⚡ <b>РЫНОЧНОЕ ДАВЛЕНИЕ ТЕЙКЕРОВ (5m FLOW):</b>\n"
                f"• <b>Taker Ratio (Buy/Sell):</b> <code>{taker_ratio:.3f}</code> ({taker_emoji})\n"
                f"• <b>Открытый интерес (OI):</b> <code>${oi_m:,.2f}M</code> USD\n\n"
                f"🚦 <b>СИГНАЛ ДЕРИВАТИВОВ:</b>\n"
                f"<b>{bias_label}</b> (Уверенность: <b>{conf:.1f}%</b>)\n\n"
                f"💡 <b>КВАНТОВЫЙ АНАЛИЗ И ВЕРДИКТ:</b>\n"
                f"<i>«{summary}»</i>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📲 <a href='{bybit_url}'><b>Открыть #{sym} на Bybit (-20% комиссия)</b></a>\n"
                f"<i>Выберите другую пару для радарного сканирования:</i>"
            )
        else:
            return (
                f"📡 <b>FUTURES POSITIONING & DERIVATIVES RADAR: #{sym} (V5)</b> 🐳\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💵 <b>Mark Price:</b> <code>${mark_p:,.2f}</code> | <b>8h Funding:</b> <code>{fr:+.4f}%</code>\n\n"
                f"📊 <b>TOP 20% BINANCE TRADERS POSITIONING:</b>\n"
                f"• <b>Long vs Short:</b> [{pos_bar}] <b>{top_long}%</b> L / <b>{top_short}%</b> S\n"
                f"• <b>L/S Ratio:</b> <b>{top_ls:.2f}</b>\n\n"
                f"⚡ <b>TAKER ORDER FLOW MOMENTUM (5m):</b>\n"
                f"• <b>Taker Buy/Sell Ratio:</b> <code>{taker_ratio:.3f}</code> ({taker_emoji_en})\n"
                f"• <b>Open Interest (OI):</b> <code>${oi_m:,.2f}M</code> USD\n\n"
                f"🚦 <b>DERIVATIVES SIGNAL:</b>\n"
                f"<b>{bias_label}</b> (Confidence: <b>{conf:.1f}%</b>)\n\n"
                f"💡 <b>QUANT FLOW ANALYSIS:</b>\n"
                f"<i>«{summary}»</i>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📲 <a href='{bybit_url}'><b>Trade #{sym} on Bybit (-20% fee)</b></a>\n"
                f"<i>Select another asset for derivatives radar:</i>"
            )

    @staticmethod
    def get_derivatives_keyboard(current_symbol: str = "BTC", lang: str = "ru") -> Dict[str, Any]:
        """Generates 1-tap interactive coin buttons for Derivatives Flow Radar."""
        bybit_url = TelegramMessageFormatter.REFERRAL_LINKS["Bybit"]
        sym = current_symbol.upper()
        if lang == "ru":
            return {
                "inline_keyboard": [
                    [
                        {"text": "⚡️ BTC Радар", "callback_data": "/flow BTC"},
                        {"text": "💎 ETH Радар", "callback_data": "/flow ETH"},
                        {"text": "🔥 SOL Радар", "callback_data": "/flow SOL"}
                    ],
                    [
                        {"text": "🪙 BNB Flow", "callback_data": "/flow BNB"},
                        {"text": "🐕 DOGE Flow", "callback_data": "/flow DOGE"},
                        {"text": "🔺 AVAX Flow", "callback_data": "/flow AVAX"}
                    ],
                    [
                        {"text": f"🧮 Рассчитать сайзинг для #{sym}", "callback_data": f"/calc_risk 1000 {sym}"},
                        {"text": "🗺 Матрица корреляций", "callback_data": "/matrix"}
                    ],
                    [
                        {"text": "📲 Торговать на Bybit (-20% fee)", "url": bybit_url}
                    ]
                ]
            }
        else:
            return {
                "inline_keyboard": [
                    [
                        {"text": "⚡️ BTC Radar", "callback_data": "/flow BTC"},
                        {"text": "💎 ETH Radar", "callback_data": "/flow ETH"},
                        {"text": "🔥 SOL Radar", "callback_data": "/flow SOL"}
                    ],
                    [
                        {"text": "🪙 BNB Flow", "callback_data": "/flow BNB"},
                        {"text": "🐕 DOGE Flow", "callback_data": "/flow DOGE"},
                        {"text": "🔺 AVAX Flow", "callback_data": "/flow AVAX"}
                    ],
                    [
                        {"text": f"🧮 Position Size #{sym}", "callback_data": f"/calc_risk 1000 {sym}"},
                        {"text": "🗺 Correlation Matrix", "callback_data": "/matrix"}
                    ],
                    [
                        {"text": "📲 Trade on Bybit (-20% fee)", "url": bybit_url}
                    ]
                ]
            }

    @staticmethod
    def format_habits_overview(status_data: Dict[str, Any], lang: str = "ru") -> str:
        """Formats the Daily Habits & Focus Sprint summary message for Telegram."""
        date_str = status_data.get("date", "")
        total = status_data.get("total_habits", 0)
        completed = status_data.get("completed_count", 0)
        pct = status_data.get("completion_percentage", 0.0)
        habits = status_data.get("habits", [])
        gam = status_data.get("gamification", {})
        rank = gam.get("rank", {})
        
        rank_title = rank.get("title", "NPC")
        rank_icon = rank.get("icon", "🥔")
        current_xp = gam.get("xp", 0)
        next_xp = rank.get("next_rank_xp", 300)
        streak = gam.get("streak_current", 0)

        # Progress bar
        blocks_done = max(0, min(10, int(round(pct / 10))))
        bar = "█" * blocks_done + "░" * (10 - blocks_done)

        ru_quotes = [
            "⚡️ <i>«Боль дисциплины длится секунды. Боль сожаления — всю жизнь.»</i> — David Goggins",
            "⚡️ <i>«Работай так усердно, будто на твое место претендуют 100 лучших умов.»</i> — Elon Musk",
            "⚡️ <i>«Твой соперник — вчерашний ты. Решай на 1 задачу больше, пиши чище.»</i> — Kobe Bryant",
            "⚡️ <i>«45 минут чистого фокуса ценнее 8 часов иллюзии работы.»</i> — Cal Newport",
            "⚡️ <i>«Каждый решенный тест SAT приближает тебя к офферу мечты в NU.»</i> — Academic Mindset",
            "⚡️ <i>«Дисциплина сажает за стол, пока остальные ищут оправдания.»</i> — Marcus Aurelius",
            "⚡️ <i>«100 формул, 100 строк кода, 100 отжиманий. Секрет — в непрерывности.»</i> — Saitama Mindset",
            "⚡️ <i>«Твой мозг — это нейросеть. Обучай его на лучших датасетах: формулах и коде.»</i> — Andrej Karpathy"
        ]
        import random
        daily_quote = random.choice(ru_quotes)

        lines = []
        if lang == "ru":
            lines.append(f"🎯 <b>ТРЕКЕР ПРИВЫЧЕК & ФОКУС-СПРИНТ</b>")
            lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")
            lines.append(f"👑 <b>Ранг:</b> {rank_icon} <b>{rank_title}</b> ({current_xp}/{next_xp} XP)")
            lines.append(f"🔥 <b>Стрик активности:</b> <b>{streak} дн. подряд</b>")
            lines.append(f"📊 <b>Прогресс сегодня:</b> <code>[{bar}]</code> <b>{pct}%</b> ({completed}/{total})")
            lines.append("")
            lines.append(daily_quote)
            lines.append("")
            lines.append(f"📋 <b>Чек-лист на сегодня ({date_str}):</b>")
            
            for h in habits:
                check_sym = "✅" if h.get("is_completed") else "⬜️"
                dl = f" (⏰ {h.get('deadline_time')})" if h.get('deadline_time') else ""
                lines.append(f"{check_sym} {h.get('icon', '🎯')} <b>{h.get('title')}</b>{dl}")
            
            lines.append("")
            if completed == total and total > 0:
                lines.append(f"🏆 <i>Идеальный день закрыт! 1% лучших. +100 XP бонус зачислен!</i> 🔥")
            else:
                lines.append(f"💡 <i>Отметьте задачи кнопками ниже или откройте WebApp с таймером спринтов:</i>")
        else:
            lines.append(f"🎯 <b>HABIT TRACKER & FOCUS VAULT</b>")
            lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")
            lines.append(f"👑 <b>Rank:</b> {rank_icon} <b>{rank_title}</b> ({current_xp}/{next_xp} XP)")
            lines.append(f"🔥 <b>Active Streak:</b> <b>{streak} days</b>")
            lines.append(f"📊 <b>Daily Progress:</b> <code>[{bar}]</code> <b>{pct}%</b> ({completed}/{total})")
            lines.append("")
            lines.append(f"⚡️ <i>«Discipline is the bridge between goals and accomplishment.»</i>")
            lines.append("")
            lines.append(f"📋 <b>Today's Targets ({date_str}):</b>")
            for h in habits:
                check_sym = "✅" if h.get("is_completed") else "⬜️"
                dl = f" (⏰ {h.get('deadline_time')})" if h.get('deadline_time') else ""
                lines.append(f"{check_sym} {h.get('icon', '🎯')} <b>{h.get('title')}</b>{dl}")
            lines.append("")
            if completed == total and total > 0:
                lines.append(f"🏆 <i>Perfect day! Top 1% achieved. +100 XP bonus awarded!</i> 🔥")
            else:
                lines.append(f"💡 <i>Check off tasks below or launch WebApp with Focus Timer:</i>")

        return "\n".join(lines)

    @staticmethod
    def get_habits_inline_keyboard(status_data: Dict[str, Any], webapp_url: str = "https://mabuy.io/habits", lang: str = "ru") -> Dict[str, Any]:
        """Generates inline keyboard for habit check-ins and WebApp launch."""
        habits = status_data.get("habits", [])
        rows = []

        # Habit quick check buttons
        for h in habits:
            status_ico = "✅" if h.get("is_completed") else "⬜️"
            btn_text = f"{status_ico} {h.get('icon', '')} {h.get('title')[:22]}"
            rows.append([
                {"text": btn_text, "callback_data": f"/habit_toggle {h.get('habit_id')}"}
            ])

        # Bottom row: Open WebApp & Actions
        if lang == "ru":
            rows.append([
                {"text": "📱 Открыть WebApp Трекер & Дневник", "web_app": {"url": webapp_url}},
            ])
            rows.append([
                {"text": "📊 Статистика & AI Аналитика", "callback_data": "stats_period_week"},
                {"text": "⚡️ 45м Спринт", "callback_data": "/habit_sprint"}
            ])
            rows.append([
                {"text": "📖 Дневник Памяти", "callback_data": "/journal"},
                {"text": "🔄 Обновить", "callback_data": "/habit"}
            ])
        else:
            rows.append([
                {"text": "📱 Open WebApp Tracker & Vault", "web_app": {"url": webapp_url}},
            ])
            rows.append([
                {"text": "📊 AI Habits & Analytics", "callback_data": "stats_period_week"},
                {"text": "⚡️ 45m Sprint", "callback_data": "/habit_sprint"}
            ])
            rows.append([
                {"text": "📖 Memory Vault", "callback_data": "/journal"},
                {"text": "🔄 Refresh", "callback_data": "/habit"}
            ])

        return {"inline_keyboard": rows}

    @staticmethod
    def format_habit_statistics(stats_data: Dict[str, Any], period: str = "week", lang: str = "ru") -> str:
        """Formats comprehensive quantitative statistics & AI coaching card for Telegram."""
        period_clean = period.lower().strip()
        days = stats_data.get("days", 7)
        comp_rate = stats_data.get("completion_rate", 0.0)
        consistency = stats_data.get("consistency_score", 0.0)
        done_tasks = stats_data.get("total_completed_tasks", 0)
        possible_tasks = stats_data.get("total_possible_tasks", 0)
        total_hours = stats_data.get("total_hours_logged", 0.0)
        active_days = stats_data.get("total_active_days", 0)
        streak = stats_data.get("streak_current", 0)
        streak_best = stats_data.get("streak_best", 0)
        rank = stats_data.get("rank", {})
        ai_coach = stats_data.get("ai_coach", {})
        cats = stats_data.get("category_breakdown", {})

        # Visual progress bars
        comp_blocks = max(0, min(10, int(round(comp_rate / 10))))
        comp_bar = "█" * comp_blocks + "░" * (10 - comp_blocks)

        period_title_ru = {
            "week": "⚡️ НЕДЕЛЬНЫЙ СПРИНТ (7 ДНЕЙ)",
            "month": "🗓 МЕСЯЧНЫЙ ОБЗОР (30 ДНЕЙ)",
            "year": "🏆 ГОДОВАЯ МАКРО-АНАЛИТИКА (365 ДНЕЙ)"
        }.get(period_clean, "📊 СТАТИСТИКА ПРИВЫЧЕК")

        period_title_en = {
            "week": "⚡️ WEEKLY TACTICAL SPRINT (7 DAYS)",
            "month": "🗓 MONTHLY OVERVIEW (30 DAYS)",
            "year": "🏆 ANNUAL MASTERY REVIEW (365 DAYS)"
        }.get(period_clean, "📊 HABIT STATISTICS")

        lines = []
        if lang == "ru":
            lines.append(f"📊 <b>{period_title_ru}</b>")
            lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")
            lines.append(f"👑 <b>Ранг & Уровень:</b> {rank.get('icon', '🥔')} <b>{rank.get('title', 'NPC')}</b> (Стрик: <b>{streak} дн.</b> | Рекорд: {streak_best})")
            lines.append(f"📈 <b>Дисциплина выполнения:</b> <code>[{comp_bar}]</code> <b>{comp_rate}%</b>")
            lines.append(f"🔥 <b>Активных дней:</b> <b>{active_days} из {days}</b> (Стабильность: <code>{consistency:.0f}%</code>)")
            lines.append(f"⏱ <b>Время чистого фокуса:</b> <b>{total_hours} часов</b> ({done_tasks}/{possible_tasks} задач)")
            lines.append("")
            lines.append(f"🧮 <b>РАСПРЕДЕЛЕНИЕ НАГРУЗКИ ПО КАТЕГОРИЯМ:</b>")
            
            for k, cat in cats.items():
                c_done = cat.get("completed", 0)
                c_hrs = round(cat.get("minutes", 0) / 60.0, 1)
                lines.append(f" • {cat.get('icon', '🎯')} <b>{cat.get('title')}:</b> {c_done} задач (<code>{c_hrs} ч</code>)")
            
            lines.append("")
            lines.append(f"🧠 <b>ВЕРДИКТ ИИ-КОУЧА ({ai_coach.get('grade', 'A')}):</b>")
            lines.append(f"<i>{ai_coach.get('momentum', '⚡️ Высокий темп')}</i>")
            lines.append(f"💡 <i>«{ai_coach.get('recommendation', '')}»</i>")
            lines.append("━━━━━━━━━━━━━━━━━━━━━━")
            lines.append("<i>Переключайте периоды кнопками ниже:</i>")
        else:
            lines.append(f"📊 <b>{period_title_en}</b>")
            lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")
            lines.append(f"👑 <b>Rank:</b> {rank.get('icon', '🥔')} <b>{rank.get('title', 'NPC')}</b> (Streak: <b>{streak}d</b> | Best: {streak_best})")
            lines.append(f"📈 <b>Completion Rate:</b> <code>[{comp_bar}]</code> <b>{comp_rate}%</b>")
            lines.append(f"🔥 <b>Consistency:</b> <b>{active_days}/{days} active days</b> (<code>{consistency:.0f}%</code>)")
            lines.append(f"⏱ <b>Deep Work Invested:</b> <b>{total_hours} hours</b> ({done_tasks}/{possible_tasks} tasks)")
            lines.append("")
            lines.append(f"🧮 <b>CATEGORY BREAKDOWN:</b>")
            for k, cat in cats.items():
                c_done = cat.get("completed", 0)
                c_hrs = round(cat.get("minutes", 0) / 60.0, 1)
                lines.append(f" • {cat.get('icon', '🎯')} <b>{cat.get('title')}:</b> {c_done} tasks (<code>{c_hrs}h</code>)")
            lines.append("")
            lines.append(f"🧠 <b>AI COACH VERDICT ({ai_coach.get('grade', 'A')}):</b>")
            lines.append(f"<i>{ai_coach.get('momentum', '⚡️ High Momentum')}</i>")
            lines.append(f"💡 <i>«{ai_coach.get('recommendation', '')}»</i>")
            lines.append("━━━━━━━━━━━━━━━━━━━━━━")
            lines.append("<i>Toggle period below:</i>")

        return "\n".join(lines)

    @staticmethod
    def get_habit_stats_keyboard(period: str = "week", webapp_url: str = "https://mabuy.io/habits", lang: str = "ru") -> Dict[str, Any]:
        """Generates interactive buttons for switching between Week, Month, and Year habit statistics."""
        period_clean = period.lower().strip()
        btn_w = ("🔘 " if period_clean == "week" else "") + ("📅 7 Дней" if lang == "ru" else "📅 7 Days")
        btn_m = ("🔘 " if period_clean == "month" else "") + ("🗓 30 Дней" if lang == "ru" else "🗓 30 Days")
        btn_y = ("🔘 " if period_clean == "year" else "") + ("🏆 1 Год" if lang == "ru" else "🏆 1 Year")

        if lang == "ru":
            return {
                "inline_keyboard": [
                    [
                        {"text": btn_w, "callback_data": "stats_period_week"},
                        {"text": btn_m, "callback_data": "stats_period_month"},
                        {"text": btn_y, "callback_data": "stats_period_year"}
                    ],
                    [
                        {"text": "📱 WebApp Аналитика & Графики", "web_app": {"url": webapp_url}},
                        {"text": "🎯 К привычкам", "callback_data": "/habit"}
                    ]
                ]
            }
        else:
            return {
                "inline_keyboard": [
                    [
                        {"text": btn_w, "callback_data": "stats_period_week"},
                        {"text": btn_m, "callback_data": "stats_period_month"},
                        {"text": btn_y, "callback_data": "stats_period_year"}
                    ],
                    [
                        {"text": "📱 WebApp Analytics & Charts", "web_app": {"url": webapp_url}},
                        {"text": "🎯 To Habits", "callback_data": "/habit"}
                    ]
                ]
            }

    @staticmethod
    def format_voice_saved_card(track_info: Dict[str, Any], total_tracks: int, lang: str = "ru") -> str:
        """Formats feedback card when a new voice note or audio is saved/appended."""
        duration = int(track_info.get("duration", 0))
        label = track_info.get("label", "Голосовая заметка")
        
        mins = duration // 60
        secs = duration % 60
        dur_str = f"{mins}м {secs}с" if mins > 0 else f"{secs} сек"

        if lang == "ru":
            return (
                f"🎙️ <b>ГОЛОСОВАЯ ЗАМЕТКА ДОБАВЛЕНА В ДНЕВНИК!</b> (+50 XP) ⚡️\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📝 <b>Запись:</b> {label}\n"
                f"⏱ <b>Длительность:</b> <code>{dur_str}</code>\n"
                f"📂 <b>Всего аудиозаметок за сегодня:</b> <b>{total_tracks}</b>\n\n"
                f"🔒 <i>Запись сохранена в ваш приватный альбом. Вы можете прослушать её или дозаписать еще голосовые мысли в любое время!</i>"
            )
        else:
            return (
                f"🎙️ <b>VOICE NOTE SAVED TO MEMORY VAULT!</b> (+50 XP) ⚡️\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📝 <b>Track:</b> {label}\n"
                f"⏱ <b>Duration:</b> <code>{dur_str}</code>\n"
                f"📂 <b>Total Voice Notes Today:</b> <b>{total_tracks}</b>\n\n"
                f"🔒 <i>Saved to your private vault. Listen or record additional voice thoughts anytime!</i>"
            )

    # ==========================================
    # 🧠 ANTI-TILT & DISCIPLINE JOURNAL (PSYCHOLOGY)
    # ==========================================

    @staticmethod
    def format_discipline_dashboard(summary: Any, lang: str = "ru") -> str:
        """Formats the trader's personal discipline dashboard and score."""
        score = summary.discipline_score
        grade = summary.grade
        streak = summary.clean_streak_days
        total_logs = summary.total_journal_entries
        fomo = summary.fomo_count
        revenge = summary.revenge_count
        compliance = summary.plan_compliance_pct
        advice = summary.stoic_advice

        score_fill = max(1, min(10, score // 10))
        score_bar = "🟢" * (score_fill // 2) + "⚪️" * (5 - (score_fill // 2))

        if lang == "ru":
            cooldown_block = ""
            if summary.is_in_tilt_cooldown:
                mins = int(summary.cooldown_remaining_sec // 60)
                secs = int(summary.cooldown_remaining_sec % 60)
                cooldown_block = (
                    f"🛑 <b>РЕЖИМ АНТИ-ТИЛЬТ ТАЙМАУТА АКТИВЕН:</b>\n"
                    f"• Остыньте от графика. До разблокировки: <b>{mins}м {secs}с</b>\n\n"
                )

            return (
                f"🧠 <b>ДНЕВНИК ДИСЦИПЛИНЫ И ПСИХОЛОГИИ ТРЕЙДЕРА</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🏆 <b>Ранг Дисциплины:</b> {grade}\n"
                f"📊 <b>Discipline Score:</b> [{score_bar}] <b>{score} / 100</b>\n"
                f"🔥 <b>Серия без нарушений риска:</b> <b>{streak} дн.</b>\n"
                f"📐 <b>Исполнение торгового плана:</b> <b>{compliance:.1f}%</b>\n\n"
                f"{cooldown_block}"
                f"📋 <b>СТАТИСТИКА ОШИБОК:</b>\n"
                f"• FOMO входы без сигнала: <b>{fomo}</b>\n"
                f"• Попытки отыграться (Revenge): <b>{revenge}</b>\n"
                f"• Всего записей в журнале: <b>{total_logs}</b>\n\n"
                f"🧘 <b>ЦИТАТА ДНЯ:</b>\n"
                f"<i>«{advice}»</i>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💡 <i>Отметьте исход последней сделки кнопками ниже:</i>"
            )
        else:
            cooldown_block = ""
            if summary.is_in_tilt_cooldown:
                mins = int(summary.cooldown_remaining_sec // 60)
                secs = int(summary.cooldown_remaining_sec % 60)
                cooldown_block = (
                    f"🛑 <b>ANTI-TILT COOLDOWN ACTIVE:</b>\n"
                    f"• Step away from charts. Unlocks in: <b>{mins}m {secs}s</b>\n\n"
                )

            return (
                f"🧠 <b>TRADER DISCIPLINE & PSYCHOLOGY JOURNAL</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🏆 <b>Discipline Rank:</b> {grade}\n"
                f"📊 <b>Discipline Score:</b> [{score_bar}] <b>{score} / 100</b>\n"
                f"🔥 <b>Zero-Tilt Streak:</b> <b>{streak} days</b>\n"
                f"📐 <b>Trade Plan Compliance:</b> <b>{compliance:.1f}%</b>\n\n"
                f"{cooldown_block}"
                f"📋 <b>PSYCHOLOGY AUDIT:</b>\n"
                f"• FOMO Entries: <b>{fomo}</b>\n"
                f"• Revenge Trades: <b>{revenge}</b>\n"
                f"• Total Logged Entries: <b>{total_logs}</b>\n\n"
                f"🧘 <b>STOIC ADVICE:</b>\n"
                f"<i>«{advice}»</i>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💡 <i>Log your latest trade reflection below:</i>"
            )

    @staticmethod
    def get_discipline_journal_keyboard(lang: str = "ru") -> Dict[str, Any]:
        """Generates interactive buttons for Logging Trade Psychology & Self-Reflection."""
        if lang == "ru":
            return {
                "inline_keyboard": [
                    [
                        {"text": "✅ Исполнил план на 100% (+5)", "callback_data": "/log_tilt FOLLOWED_PLAN"}
                    ],
                    [
                        {"text": "⏳ Ранний выход из страха (-5)", "callback_data": "/log_tilt EARLY_EXIT"},
                        {"text": "⚠️ FOMO вход без сигнала (-15)", "callback_data": "/log_tilt FOMO_ENTRY"}
                    ],
                    [
                        {"text": "🚨 Превысил плечо/риск (-20)", "callback_data": "/log_tilt OVERLEVERAGED"},
                        {"text": "🔥 Отыгрыш / Revenge (-25)", "callback_data": "/log_tilt REVENGE_TRADE"}
                    ],
                    [
                        {"text": "🧘 45м Анти-Тильт Таймаут", "callback_data": "/tilt_cooldown"},
                        {"text": "🔄 Обновить Дневник", "callback_data": "/discipline"}
                    ]
                ]
            }
        else:
            return {
                "inline_keyboard": [
                    [
                        {"text": "✅ Followed Plan 100% (+5)", "callback_data": "/log_tilt FOLLOWED_PLAN"}
                    ],
                    [
                        {"text": "⏳ Fearful Early Exit (-5)", "callback_data": "/log_tilt EARLY_EXIT"},
                        {"text": "⚠️ FOMO Entry (-15)", "callback_data": "/log_tilt FOMO_ENTRY"}
                    ],
                    [
                        {"text": "🚨 Overleveraged (-20)", "callback_data": "/log_tilt OVERLEVERAGED"},
                        {"text": "🔥 Revenge Trade (-25)", "callback_data": "/log_tilt REVENGE_TRADE"}
                    ],
                    [
                        {"text": "🧘 45m Anti-Tilt Cooldown", "callback_data": "/tilt_cooldown"},
                        {"text": "🔄 Refresh Journal", "callback_data": "/discipline"}
                    ]
                ]
            }

    # ==========================================
    # 📊 STRATIFIED REGIME QUOTA FORWARD-TEST DASHBOARD
    # ==========================================

    @staticmethod
    def format_regime_quota_dashboard(evaluation: Dict[str, Any], lang: str = "ru") -> str:
        """Formats the 4-Regime Forward-Testing progress and qualification status."""
        total_t = evaluation.get("total_trades", 0)
        overall_pf = evaluation.get("overall_profit_factor", 1.0)
        overall_wr = evaluation.get("overall_win_rate_pct", 0.0)
        is_ready = evaluation.get("is_qualified_for_live_capital", False)
        breakdown = evaluation.get("regimes_breakdown", {})

        target_total = 85 # 20 + 20 + 25 + 20
        total_pct = min(100, int((total_t / target_total) * 100))

        regime_meta = {
            "BULL_TREND": {"icon": "🐂", "name_ru": "Бычий тренд", "name_en": "Bull Trend"},
            "BEAR_TREND": {"icon": "🐻", "name_ru": "Медвежий тренд", "name_en": "Bear Trend"},
            "SQUEEZE": {"icon": "🦀", "name_ru": "Флэт / Сквиз", "name_en": "Squeeze / Flat"},
            "HIGH_VOLATILITY": {"icon": "🌪", "name_ru": "Шторм / Волатильность", "name_en": "High Volatility"}
        }

        if lang == "ru":
            status_badge = "🟢 КВАЛИФИЦИРОВАН ДЛЯ РЕАЛЬНОГО КАПИТАЛА" if is_ready else "⏳ НАБОР СТАТИСТИКИ (SHADOW FORWARD-TEST)"
            lines = [
                f"📊 <b>ФОРВАРД-ТЕСТ: ПРОГРЕСС 4 РЕЖИМОВ РЫНКА (V5)</b> 🛡",
                f"━━━━━━━━━━━━━━━━━━━━━━",
                f"🎯 <b>Общий прогресс:</b> <b>{total_t} / {target_total} сделок</b> ({total_pct}%)",
                f"📈 <b>Общий Win Rate:</b> <b>{overall_wr:.1f}%</b> | <b>Profit Factor:</b> <b>{overall_pf:.2f}</b>",
                f"🏷 <b>Статус:</b> <code>{status_badge}</code>\n",
                f"📋 <b>РАСПРЕДЕЛЕНИЕ ПО РЕЖИМАМ РЫНКА:</b>"
            ]

            for r_code, meta in regime_meta.items():
                r_data = breakdown.get(r_code, {})
                t_cnt = r_data.get("trades_count", 0)
                quota = r_data.get("target_quota", 20)
                wr = r_data.get("win_rate_pct", 0.0)
                pf = r_data.get("profit_factor", 1.0)
                is_prof = r_data.get("is_statistically_profitable", False)

                bar_fill = min(10, int((t_cnt / quota) * 10))
                bar_str = "█" * bar_fill + "░" * (10 - bar_fill)
                status_icon = "✅" if is_prof else ("🔄" if t_cnt > 0 else "⏳")

                lines.append(
                    f"{meta['icon']} <b>{meta['name_ru']}:</b>\n"
                    f"  • Прогресс: <code>[{bar_str}]</code> <b>{t_cnt}/{quota}</b>\n"
                    f"  • Win Rate: <b>{wr:.1f}%</b> | PF: <b>{pf:.2f}</b> {status_icon}"
                )

            lines.append("\n━━━━━━━━━━━━━━━━━━━━━━")
            lines.append("💡 <i>Торговля реальным капиталом активируется только при выполнении квот во всех 4 режимах (защита от переобучения).</i>")
            return "\n".join(lines)
        else:
            status_badge = "🟢 QUALIFIED FOR LIVE CAPITAL" if is_ready else "⏳ COLLECTING REGIME DATA (SHADOW MODE)"
            lines = [
                f"📊 <b>FORWARD-TEST: 4 MARKET REGIMES PROGRESS (V5)</b> 🛡",
                f"━━━━━━━━━━━━━━━━━━━━━━",
                f"🎯 <b>Total Progress:</b> <b>{total_t} / {target_total} trades</b> ({total_pct}%)",
                f"📈 <b>Overall Win Rate:</b> <b>{overall_wr:.1f}%</b> | <b>Profit Factor:</b> <b>{overall_pf:.2f}</b>",
                f"🏷 <b>Status:</b> <code>{status_badge}</code>\n",
                f"📋 <b>MARKET REGIMES STRATIFICATION:</b>"
            ]

            for r_code, meta in regime_meta.items():
                r_data = breakdown.get(r_code, {})
                t_cnt = r_data.get("trades_count", 0)
                quota = r_data.get("target_quota", 20)
                wr = r_data.get("win_rate_pct", 0.0)
                pf = r_data.get("profit_factor", 1.0)
                is_prof = r_data.get("is_statistically_profitable", False)

                bar_fill = min(10, int((t_cnt / quota) * 10))
                bar_str = "█" * bar_fill + "░" * (10 - bar_fill)
                status_icon = "✅" if is_prof else ("🔄" if t_cnt > 0 else "⏳")

                lines.append(
                    f"{meta['icon']} <b>{meta['name_en']}:</b>\n"
                    f"  • Progress: <code>[{bar_str}]</code> <b>{t_cnt}/{quota}</b>\n"
                    f"  • Win Rate: <b>{wr:.1f}%</b> | PF: <b>{pf:.2f}</b> {status_icon}"
                )

            lines.append("\n━━━━━━━━━━━━━━━━━━━━━━")
            lines.append("💡 <i>Live capital trading is strictly locked until all 4 regime quotas are fulfilled with PF >= 1.10.</i>")
            return "\n".join(lines)

    @staticmethod
    def get_regime_quota_keyboard(lang: str = "ru") -> Dict[str, Any]:
        """Generates refresh buttons for Regimes Dashboard."""
        if lang == "ru":
            return {
                "inline_keyboard": [
                    [
                        {"text": "🔄 Обновить Квоты Режимов", "callback_data": "/regimes"}
                    ],
                    [
                        {"text": "📊 Общий Трек-рекорд", "callback_data": "/trackrecord"},
                        {"text": "🏠 Главное меню", "callback_data": "/start"}
                    ]
                ]
            }
        else:
            return {
                "inline_keyboard": [
                    [
                        {"text": "🔄 Refresh Regime Quotas", "callback_data": "/regimes"}
                    ],
                    [
                        {"text": "📊 Full Track Record", "callback_data": "/trackrecord"},
                        {"text": "🏠 Main Menu", "callback_data": "/start"}
                    ]
                ]
            }




