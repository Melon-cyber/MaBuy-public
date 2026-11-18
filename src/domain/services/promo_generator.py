import random
from typing import Dict, List

class PromoPostGeneratorService:
    """
    Generates high-converting, viral promotional posts for Telegram Channels, VK, and Social Media
    to drive massive organic user traffic into @MaBuyCryptoBot.
    """

    PROMO_TEMPLATES_RU = [
        # Template 1: High-Energy Tech & Proof (Focus on Veto & Zero Emotions)
        (
            "🛡 <b>ПОЧЕМУ 90% ТРЕЙДЕРОВ СЛИВАЮТ, А ИИ ФИЛЬТРУЕТ ЭМОЦИИ?</b> 🤖📈\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "Большинство теряет депозит из-за двух факторов: <b>ФОМО и торговли без риск-менеджмента</b>.\n\n"
            "Модель <b>MaBuy V4</b> работает строго на прикладной математике:\n"
            "• 📊 Анализирует 80+ рыночных факторов 24/7\n"
            "• 🛑 Накладывает Veto на ложные пробои по Order Flow\n"
            "• 🧮 Рассчитывает безопасный риск и размер позиции по формуле Келли\n"
            "• 🔒 Неизменяемый SQLite-журнал форвард-тестов\n\n"
            "👉 <b>Протестируйте вероятностный анализ бесплатно:</b>\n"
            "📲 @MaBuyCryptoBot"
        ),
        # Template 2: Analytical & Case Study (Safe-Leverage & Math)
        (
            "📊 <b>МАТЕМАТИЧЕСКИЙ АНАЛИЗ РЫНКА НА PYTHON & ML</b> ⚡\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "Мы переложили рутину анализа рисков на строгий Data Science.\n\n"
            "<b>Что умеет MaBuy AI:</b>\n"
            "1. Safe-Leverage Wizard: расчет безопасного плеча и размера позиции (ATR & Kelly)\n"
            "2. Оценка рисков: расчет 95% Daily VaR и CVaR для криптопортфеля\n"
            "3. Veto-фильтрация: отсев сделок при аномальных стенах ликвидности\n"
            "4. Адаптивный интерфейс: режим 🐣 <b>Новичок</b> и 🔬 <b>Data Science Pro</b>\n\n"
            "🔗 Получите доступ к открытой бета-версии прямо сейчас:\n"
            "📲 @MaBuyCryptoBot"
        ),
        # Template 3: Urgent Signal / Market Insight Style
        (
            "🚨 <b>ВЕРОЯТНОСТНЫЙ ИИ-АНАЛИЗ РЫНКА #BTC / #ETH</b> 📈\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "Алгоритм зафиксировал изменение микроструктуры рынка и проверил сентимент новостей.\n\n"
            "💡 <i>Вместо хаотичных решений получайте проверенные вероятностные сценарии с точным расчетом риска и точек выхода!</i>\n\n"
            "👇 <b>Запустить бота в 1 клик:</b>\n"
            "📲 @MaBuyCryptoBot"
        )
    ]

    PROMO_TEMPLATES_EN = [
        (
            "🛡 <b>CAN DATA SCIENCE BEAT EMOTIONAL CRYPTO TRADING?</b> 🤖📈\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "Most traders lose due to FOMO and poor risk management.\n\n"
            "<b>MaBuy AI V4</b> operates with strict applied mathematics:\n"
            "• 📊 Scans 80+ indicators and news sentiment 24/7\n"
            "• 🛑 Enforces Microstructure Veto against false breakouts\n"
            "• 🧮 Calculates Safe Leverage and dynamic TP/SL (Kelly & ATR)\n"
            "• 🔒 Immutable SQLite Forward-Test Journal\n\n"
            "👉 <b>Try free probability analytics right now:</b>\n"
            "📲 @MaBuyCryptoBot"
        )
    ]

    @staticmethod
    def generate_post(lang: str = "ru") -> str:
        if lang == "ru":
            return random.choice(PromoPostGeneratorService.PROMO_TEMPLATES_RU)
        else:
            return random.choice(PromoPostGeneratorService.PROMO_TEMPLATES_EN)
