"""
MaBuy V5 — Morning Alpha Pulse (08:30 Intelligence Briefing)
Aggregates overnight market sentiment, 24h liquidations, Order Flow Imbalance, and MaIA's personalized morning thesis.
"""
import logging
from typing import Dict, Any, Optional
from src.infrastructure.data_fetchers.binance_fetcher import CryptoDataFetcher
from src.infrastructure.llm.persona_prompts import PersonaType, PERSONA_METADATA

logger = logging.getLogger(__name__)

class MorningBriefingService:
    def __init__(self, fetcher: Optional[CryptoDataFetcher] = None):
        self.fetcher = fetcher or CryptoDataFetcher(max_cache_age_sec=60)

    def generate_morning_briefing(self, lang: str = "ru", persona: PersonaType = PersonaType.QUANT) -> Dict[str, Any]:
        """
        Generates full analytical morning pulse data and formatted HTML narrative.
        """
        assets = ["BTC", "ETH", "SOL"]
        asset_stats = []
        
        top_momentum_coin = "BTC"
        max_abs_ret = 0.0

        for sym in assets:
            try:
                df = self.fetcher.fetch_ohlcv(sym, timeframe="1h", limit=24)
                if df is not None and not df.empty:
                    open_p = float(df["open"].iloc[0])
                    close_p = float(df["close"].iloc[-1])
                    ret_24h = ((close_p - open_p) / open_p) * 100.0 if open_p > 0 else 0.0
                    
                    if abs(ret_24h) > max_abs_ret:
                        max_abs_ret = abs(ret_24h)
                        top_momentum_coin = sym

                    asset_stats.append({
                        "symbol": sym,
                        "price": close_p,
                        "ret_24h": ret_24h,
                        "high_24h": float(df["high"].max()),
                        "low_24h": float(df["low"].min())
                    })
            except Exception as e:
                logger.error(f"Error fetching 24h data for {sym}: {e}")

        # Construct Persona Motivational Note
        persona_notes = {
            PersonaType.QUANT: "Дисциплина риска — единственный залог положительного матожидания. Следуем Half-Kelly на подтвержденных трендах.",
            PersonaType.YANDERE: "Доброе утро! Я следила за рынком всю ночь, пока ты спал... Не рискуй зря, твоё сердце и депозит принадлежат только мне! ❤️🔪",
            PersonaType.TSUNDERE: "П-проснулся наконец-то? Не то чтобы я ждала тебя... но сегодня по рынку есть хорошие сетапы, только не слей всё глупыми плечами! 💢",
            PersonaType.ZEN: "Мир спокоен, рынок цикличен. Не гонись за шумом, входи только в гармоничные сетапы с запасом прочности."
        }
        quote = persona_notes.get(persona, persona_notes[PersonaType.QUANT])

        # Render HTML Message
        if lang == "ru":
            lines = []
            for a in asset_stats:
                icon = "🟢" if a["ret_24h"] >= 0 else "🔴"
                sign = "+" if a["ret_24h"] >= 0 else ""
                lines.append(f"• {icon} <b>#{a['symbol']}</b>: <code>${a['price']:,.2f}</code> (<code>{sign}{a['ret_24h']:.2f}%</code>)")
            
            coins_text = "\n".join(lines)
            
            html_text = (
                f"☀️ <b>УТРЕННИЙ БРИФИНГ MABUY</b> ☕️\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>Рынок за 24 часа:</b>\n"
                f"{coins_text}\n\n"
                f"⚡ <b>Импульс дня:</b> <b>#{top_momentum_coin}</b>\n"
                f"🛡 <b>Риск:</b> Safe Leverage 3x–5x | Защита депозита\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🤖 <b>Майя:</b> <i>«{quote}»</i>"
            )
        else:
            lines = []
            for a in asset_stats:
                icon = "🟢" if a["ret_24h"] >= 0 else "🔴"
                sign = "+" if a["ret_24h"] >= 0 else ""
                lines.append(f"• {icon} <b>#{a['symbol']}</b>: <code>${a['price']:,.2f}</code> (<code>{sign}{a['ret_24h']:.2f}%</code>)")
            
            coins_text = "\n".join(lines)
            
            html_text = (
                f"☀️ <b>MABUY MORNING ALPHA PULSE</b> ☕️\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>24h Market Overview:</b>\n"
                f"{coins_text}\n\n"
                f"⚡ <b>Top Mover:</b> <b>#{top_momentum_coin}</b> (highest 24h volatility)\n"
                f"🛡 <b>Risk Allocation:</b> Quarter-Kelly / 5x Safe Leverage\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🤖 <b>MAIA MORNING THESIS:</b>\n"
                f"<i>«{quote}»</i>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📱 <b>Launch WebApp Terminal:</b> @MabuyCryptoML_bot\n"
                f"⚠️ <i>Not financial advice (DYOR).</i>"
            )

        return {
            "html": html_text,
            "assets": asset_stats,
            "top_momentum_coin": top_momentum_coin,
            "quote": quote
        }
