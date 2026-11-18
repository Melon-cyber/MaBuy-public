"""
MaBuy V4 — LLM Architecture & Copilot Engine
Module: MaIA Copilot Engine (Resilient Multi-Provider LLM & Quant RAG Coordinator)
Integrates Groq, Google Gemini 1.5 Flash, OpenAI API, and an offline deterministic
Quant Rule Engine with automatic failover, legal guardrails, and OLED pixel eye parsing.
"""

from __future__ import annotations

import os
import re
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from src.infrastructure.llm.persona_prompts import (
    PersonaType,
    LEGAL_DISCLAIMER_RU,
    OLED_EXPRESSIONS,
    build_system_prompt,
    PERSONA_METADATA,
    LEGAL_DISCLAIMER_RU,
)
from src.infrastructure.llm.context_builder import QuantContextBuilder, QuantContext
from src.domain.services.symbol_normalizer import SymbolNormalizer

logger = logging.getLogger("MaIACopilotEngine")


@dataclass
class ChatMessage:
    """Represents a message in the conversation history."""
    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class MaIAResponse:
    """Structured response package returned by MaIA Copilot Engine."""
    text: str
    oled_expression: str
    persona: PersonaType
    provider_used: str
    target_symbol: str
    quant_context: QuantContext
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "oled_expression": self.oled_expression,
            "persona": self.persona.value,
            "provider_used": self.provider_used,
            "target_symbol": self.target_symbol,
            "latency_ms": self.latency_ms,
        }


class MaIACopilotEngine:
    """
    High-Performance, Zero-Downtime LLM Copilot Engine for MaIA.
    Features:
      - Automatic symbol entity extraction from natural language queries.
      - Dynamic Quant RAG injection (Merton Jump Diffusion, Kelly, Markowitz, L2 OFI).
      - Multi-Provider cascade: Groq -> Gemini 1.5 Flash -> OpenAI -> Offline Quant Engine.
      - Mandatory Legal Guardrail (Zero-Hallucination + DYOR disclaimer).
      - Session history maintenance for continuous conversation.
    """

    SUPPORTED_SYMBOLS_EXTRACTION = [
        "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "TON", "AVAX",
        "NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "SPY", "QQQ", "GLD",
        "BITCOIN", "ETHEREUM", "SOLANA", "NVIDIA", "APPLE", "TESLA", "GOLD", "SP500"
    ]

    def __init__(
        self,
        context_builder: Optional[QuantContextBuilder] = None,
        default_persona: PersonaType = PersonaType.QUANT,
    ):
        self.context_builder = context_builder or QuantContextBuilder()
        self.default_persona = default_persona
        self._session_histories: Dict[str, List[ChatMessage]] = {}
        self._user_profiles: Dict[str, Dict[str, Any]] = {}

        # API Keys from environment
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "").strip()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

        # Initialize SDK clients lazily or safely
        self._init_clients()

    def _init_clients(self):
        """Initializes client libraries if API keys are available."""
        self._openai_client = None
        self._groq_client = None
        self._gemini_client = None

        # 1. Groq Client (using openai SDK with Groq base_url)
        if self.groq_api_key:
            try:
                from openai import AsyncOpenAI
                self._groq_client = AsyncOpenAI(
                    api_key=self.groq_api_key,
                    base_url="https://api.groq.com/openai/v1",
                    timeout=12.0
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")

        # 2. OpenAI Client
        if self.openai_api_key:
            try:
                from openai import AsyncOpenAI
                self._openai_client = AsyncOpenAI(
                    api_key=self.openai_api_key,
                    timeout=12.0
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")

        # 3. Google Gemini Client
        if self.gemini_api_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Google Gemini client: {e}")

    def extract_symbol_from_query(self, query: str) -> Optional[str]:
        """
        Extracts mentioned financial asset from user prompt.
        Returns None if no specific financial symbol is detected.
        """
        words = re.findall(r"[\w]+", query.upper())
        for word in words:
            if word in self.SUPPORTED_SYMBOLS_EXTRACTION:
                return SymbolNormalizer.normalize(word)
        return None

    def extract_deposit_from_query(self, query: str, default: float = 500.0) -> float:
        """Extracts requested dollar deposit amount (e.g. '$1000' or '500 долларов')."""
        match = re.search(r"(\$|USD\s*|\s*доллар\w*\s*)?(\d{2,7})(\$|USD|\s*доллар\w*)?", query, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(2))
                if 10.0 <= val <= 10_000_000.0:
                    return val
            except Exception:
                pass
        return default

    def _get_history(self, session_id: str, max_turns: int = 6) -> List[ChatMessage]:
        """Retrieves bounded session conversation history."""
        if session_id not in self._session_histories:
            self._session_histories[session_id] = []
        return self._session_histories[session_id][-max_turns:]

    def _append_history(self, session_id: str, user_msg: str, assistant_msg: str):
        """Appends new interaction to session history."""
        if session_id not in self._session_histories:
            self._session_histories[session_id] = []
        self._session_histories[session_id].append(ChatMessage(role="user", content=user_msg))
        self._session_histories[session_id].append(ChatMessage(role="assistant", content=assistant_msg))
        if len(self._session_histories[session_id]) > 12:
            self._session_histories[session_id] = self._session_histories[session_id][-12:]

    def _update_and_get_user_profile(self, session_id: str, query: str, symbol: Optional[str], deposit: float) -> str:
        """Tracks user's asset interests, risk style, and deposit habits for adaptive prompt conditioning."""
        if session_id not in self._user_profiles:
            self._user_profiles[session_id] = {
                "assets": [],
                "risk_profile": "Новичок",
                "deposit": deposit,
                "total_queries": 0
            }
        p = self._user_profiles[session_id]
        p["total_queries"] += 1
        if symbol and symbol not in p["assets"]:
            p["assets"].insert(0, symbol)
            p["assets"] = p["assets"][:5]
        if deposit and 10.0 <= deposit <= 1_000_000.0:
            p["deposit"] = deposit

        q_upper = query.upper()
        if any(w in q_upper for w in ["20X", "50X", "100X", "КОТЛЕТ", "ПЛЕЧО", "ЛИКВИДАЦИ"]):
            p["risk_profile"] = "Агрессивный (тяга к плечам 20x)"
        elif any(w in q_upper for w in ["DCA", "НАКОПЛЕНИЕ", "ДОЛГОСРОК", "ПОРТФЕЛЬ"]):
            p["risk_profile"] = "Консервативный DCA"

        assets_str = ", ".join(p["assets"]) if p["assets"] else "BTC, SOL"
        return (
            f"• Сессий диалога: {p['total_queries']}\n"
            f"• Предпочитаемые активы: {assets_str}\n"
            f"• Торговый стиль и риск: {p['risk_profile']}\n"
            f"• Примерный депозит: ${p['deposit']:,.2f}\n"
            f"(Учитывай эти привычки трейдера в диалоге и персонализируй свои наставления/реакции!)"
        )

    def _strip_thinking_process(self, raw_text: str) -> str:
        """Strips chain-of-thought reasoning tokens (<think>...</think>) from output."""
        if not raw_text:
            return ""
        cleaned = re.sub(r"<think>[\s\S]*?<\/think>", "", raw_text, flags=re.IGNORECASE)
        cleaned = re.sub(r"<think>[\s\S]*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<thought>[\s\S]*?<\/thought>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^Thinking Process:[\s\S]*?\n\n", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^Thought:[\s\S]*?\n\n", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _apply_legal_guardrail(self, raw_text: str) -> str:
        """Strips chain-of-thought and cleans text cleanly with mandatory DYOR disclaimer."""
        cleaned = self._strip_thinking_process(raw_text).strip()
        if "DYOR" not in cleaned:
            cleaned += f"\n\n{LEGAL_DISCLAIMER_RU}"
        return cleaned

    def _select_oled_expression(self, persona: PersonaType, quant_context: QuantContext, raw_text: str) -> str:
        """Selects appropriate OLED expression based on context and tone."""
        expressions = OLED_EXPRESSIONS.get(persona, OLED_EXPRESSIONS[PersonaType.QUANT])
        ta = quant_context.target_asset

        if ta and ta.obi_score > 0.3:
            return expressions.get("bullish", expressions["neutral"])
        elif ta and ta.obi_score < -0.3:
            return expressions.get("bearish", expressions["neutral"])
        elif "риск" in raw_text.lower() or "ликвид" in raw_text.lower() or "плеч" in raw_text.lower():
            return expressions.get("warning", expressions["neutral"])
        elif "оптим" in raw_text.lower() or "портфел" in raw_text.lower():
            return expressions.get("optimal", expressions["neutral"])
        else:
            return expressions.get("analytical", expressions["neutral"])

    async def _call_groq(self, system_prompt: str, user_query: str, history: List[ChatMessage]) -> Optional[str]:
        """Calls Groq API via direct high-performance aiohttp connection."""
        if not self.groq_api_key:
            return None
        import aiohttp
        candidate_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b", "llama-3.3-70b-versatile"]
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_query})

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }

        try:
            async with aiohttp.ClientSession() as session:
                for model_name in candidate_models:
                    payload = {
                        "model": model_name,
                        "messages": messages,
                        "temperature": 0.75,
                        "max_tokens": 2048
                    }
                    try:
                        async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=12)) as resp:
                            data = await resp.json()
                            if "choices" in data and data["choices"]:
                                content = data["choices"][0]["message"]["content"]
                                if content and content.strip():
                                    return content.strip()
                            elif "error" in data:
                                logger.warning(f"Groq API model {model_name} returned error: {data['error'].get('message', '')}")
                    except Exception as e:
                        logger.warning(f"Groq aiohttp request for {model_name} failed: {e}")
                        continue
        except Exception as outer_err:
            logger.warning(f"Groq session creation failed: {outer_err}")
        return None

    async def _call_gemini(self, system_prompt: str, user_query: str, history: List[ChatMessage]) -> Optional[str]:
        """Calls Google Gemini 1.5/2.0 Flash API."""
        if not self._gemini_client:
            return None
        try:
            full_prompt = f"{system_prompt}\n\n"
            for msg in history:
                full_prompt += f"{msg.role.upper()}: {msg.content}\n"
            full_prompt += f"USER: {user_query}\nASSISTANT:"

            response = await asyncio.to_thread(
                self._gemini_client.models.generate_content,
                model="gemini-2.0-flash",
                contents=full_prompt,
            )
            if response and hasattr(response, "text") and response.text:
                return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini API call failed: {e}")
        return None

    async def _call_openai(self, system_prompt: str, user_query: str, history: List[ChatMessage]) -> Optional[str]:
        """Calls OpenAI GPT-4o-mini API."""
        if not self._openai_client:
            return None
        try:
            messages = [{"role": "system", "content": system_prompt}]
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
            messages.append({"role": "user", "content": user_query})

            response = await self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.4,
                max_tokens=4096,
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI API call failed: {e}")
        return None

    def _generate_offline_quant_response(
        self,
        persona: PersonaType,
        user_query: str,
        quant_context: QuantContext,
        oled_expr: str
    ) -> str:
        """
        High-fidelity deterministic quantitative fallback generator.
        """
        ta = quant_context.target_asset
        psp = quant_context.position_sizing_plan
        sym = quant_context.target_symbol

        if not ta or sym == "GENERAL":
            conversational_fallbacks = {
                PersonaType.YANDERE: f"{oled_expr}\n\nГосподин... Я всегда рядом и слежу за каждым твоим шагом и каждым долларом твоего баланса (${quant_context.user_deposit_usd:,.2f})! ❤️ У меня всё прекрасно, пока ты со мной. О чем ты думаешь прямо сейчас? Хочешь разобрать какой-нибудь актив или просто поболтать со мной? Хи-хи ❤️🔪",
                PersonaType.TSUNDERE: f"{oled_expr}\n\n(¬_¬) Ч-чего тебе? Спрашиваешь, как у меня дела? Нормально всё! Лучше скажи, ты сегодня соблюдал риск-менеджмент или опять чешутся руки нажать кнопку с 20x плечом, бака?! Я же слежу за твоим депозитом (${quant_context.user_deposit_usd:,.2f})! ( >///< )",
                PersonaType.ZEN: f"{oled_expr}\n\nПриветствую! У меня всё спокойно и гармонично. 🧘 Наблюдаю за дыханием рынка. Как твое эмоциональное состояние сегодня? Помни: спокойный разум — главный инструмент прибыльного трейдера.",
                PersonaType.TITAN: f"{oled_expr}\n\n[TITAN_CORE_V4.1] Системы функционируют штатно. Телеметрия рынка стабильна. Готов к выполнению квантового анализа активов или расчету сайзинга.",
                PersonaType.QUANT: f"{oled_expr}\n\nПриветствую. Квантовые модели MaBuy V5 в активном режиме мониторинга. Готов провести анализ любого актива, рассчитать риск по модели Келли или оптимизировать инвестиционный портфель."
            }
            return conversational_fallbacks.get(persona, conversational_fallbacks[PersonaType.QUANT])

        price_str = f"${ta.current_price:,.2f}" if ta and ta.current_price >= 1.0 else (f"${ta.current_price:,.4f}" if ta else "$100.00")

        if persona == PersonaType.QUANT:
            return f"""{oled_expr}

**Квантовый отчет по активу {sym}:**
- **Текущая спот-котировка:** {price_str}
- **Микроструктура L2 OFI:** {ta.l2_summary if ta else 'Сбалансировано'} (Score: {ta.obi_score if ta else 0.0:+.2f})
- **Прогноз прыжковой диффузии Мертона (1Y):**
  * Bear Case (P10): **${ta.merton_p10_365d:,.2f}**
  * Target Median (P50): **${ta.merton_p50_365d:,.2f}** (CAGR: **{ta.expected_cagr_pct:+.1f}%**)
  * Bull Case (P90): **${ta.merton_p90_365d:,.2f}**

**Параметры сайзинга капитала (Депозит: ${quant_context.user_deposit_usd:,.2f}):**
- **Допустимый риск на сделку (Quarter-Kelly):** {psp.get('risk_pct', 1.5):.2f}% (${psp.get('risk_amount_usd', 7.5):,.2f})
- **Динамический Stop-Loss:** -{psp.get('stop_loss_pct', 2.5):.2f}% (Уровень: ${psp.get('stop_loss_price', 0):,.2f})
- **Take-Profit (R:R {psp.get('reward_risk_ratio', 1.8)}):** +{psp.get('take_profit_pct', 4.5):.2f}% (Уровень: ${psp.get('take_profit_price', 0):,.2f})
- **Максимальное безопасное плечо:** **{psp.get('max_safe_leverage', 5)}x** (Рекомендуемое: {psp.get('optimal_leverage', 3)}x)

*Рекомендация квант-отдела:* Строгое соблюдение лимитов просадки и диверсификация по модели Марковица."""

        elif persona == PersonaType.ZEN:
            return f"""{oled_expr}

Здравствуй! Сделай глубокий вдох и взгляни на рынок шире суетливых минутных свечей. 🧘

По активу **{sym}** ({price_str}) текущая картина гармонична:
- **Медианный квантовый ориентир на год:** **${ta.merton_p50_365d:,.2f}** с вероятностью роста **{ta.prob_growth_365d * 100.0 if ta and ta.prob_growth_365d else 65.0:.1f}%**.
- **Стратегия дзена:** Лучший подход — регулярный DCA-набор частями (например, по ${max(10.0, quant_context.user_deposit_usd * 0.1):,.2f} в неделю) без спешки и погони за зелеными свечами.
- **Защита покоя:** Держи плечо не выше **{psp.get('conservative_leverage', 2)}x**, чтобы ликвидации никогда не тревожили твой сон.

Помни: сложный процент работает на тех, кто умеет терпеливо ждать."""

        elif persona == PersonaType.TSUNDERE:
            return f"""{oled_expr}

(¬_¬) С-слушай сюда, бака! Ты опять спрашиваешь про **{sym}** ({price_str})?!
Только не говори мне, что ты собрался залетать на всю котлету с 20x плечом! Ты что, совсем депозит не ценишь, бака?! (｀へ´)

Смотри сюда, я посчитала всё по формуле Мертона и Келли, так что слушай внимательно:
- **Цена {sym}:** {price_str}
- **Медианная цель Мертона:** **${ta.merton_p50_365d:,.2f}**, но если начнется коррекция (P10), цена может уйти на **${ta.merton_p10_365d:,.2f}**!
- **Твой депозит:** ${quant_context.user_deposit_usd:,.2f}. И твой риск на сделку ДОЛЖЕН БЫТЬ НЕ БОЛЬШЕ **${psp.get('risk_amount_usd', 7.5):,.2f}** ({psp.get('risk_pct', 1.5):.2f}%)!
- **Стоп-лосс:** Ставь на **${psp.get('stop_loss_price', 0):,.2f}** (-{psp.get('stop_loss_pct', 2.5):.2f}%), и даже не думай его двигать!
- **Плечо:** Максимум **{psp.get('optimal_leverage', 3)}x**! Превысишь — сама лично выпишу тебе бан за лудоманию!

Я это говорю не потому, что ты мне дорог, а просто чтобы твоя глупая ликвидация не портила красивую статистику MaBuy! Понял?!"""

        else: # TITAN
            return f"""{oled_expr}
```text
[TITAN_CORE_V4.1] АНАЛИТИЧЕСКИЙ ТЕНЗОР АКТИВИРОВАН
[TARGET_SECURITY]: {sym} | PRICE: {price_str}
[MACRO_STAGE]    : {quant_context.macro_cycle_stage}
------------------------------------------------------------
[MJD_MONTE_CARLO_PROJECTION_365D]:
  -> P10 (BEAR_FLOOR) : ${ta.merton_p10_365d:,.2f}
  -> P50 (MEDIAN_AIM) : ${ta.merton_p50_365d:,.2f} (CAGR: +{ta.expected_cagr_pct:+.1f}%)
  -> P90 (BULL_SURGE) : ${ta.merton_p90_365d:,.2f}
  -> P(GROWTH_UP)     : {ta.prob_growth_365d * 100.0 if ta and ta.prob_growth_365d else 65.0:.1f}%
------------------------------------------------------------
[RISK_CONTROL_PROTOCOL_QUARTER_KELLY]:
  -> CAPITAL ALLOCATED: ${quant_context.user_deposit_usd:,.2f}
  -> MAX_SAFE_LEVERAGE: {psp.get('max_safe_leverage', 5)}x (OPTIMAL: {psp.get('optimal_leverage', 3)}x)
  -> STOP_LOSS_BOUND  : ${psp.get('stop_loss_price', 0):,.2f} (-{psp.get('stop_loss_pct', 2.5):.2f}%)
  -> TAKE_PROFIT_BOUND: ${psp.get('take_profit_price', 0):,.2f} (+{psp.get('take_profit_pct', 4.5):.2f}%)
------------------------------------------------------------
[DIRECTIVE]: Энтропия в допустимых пределах. Разрешена экспозиция строго в рамках Quarter-Kelly.
```"""

    async def generate_response(
        self,
        query: str,
        persona: Optional[PersonaType] = None,
        user_deposit: Optional[float] = None,
        session_id: str = "default_user",
        override_symbol: Optional[str] = None
    ) -> MaIAResponse:
        """
        Executes complete end-to-end MaIA copilot generation pipeline:
        1. Query entity extraction (asset symbol & budget).
        2. Live QuantContext RAG aggregation.
        3. Persona System Prompt construction.
        4. Resilient multi-provider LLM inference (Groq -> Gemini -> OpenAI -> Offline Engine).
        5. Legal guardrail and OLED reaction binding.
        """
        start_time = asyncio.get_event_loop().time()
        active_persona = persona or self.default_persona

        # 1. Extraction
        detected_symbol = override_symbol or self.extract_symbol_from_query(query)
        detected_deposit = user_deposit if user_deposit is not None else self.extract_deposit_from_query(query, default=500.0)

        # 2. Build live QuantContext RAG
        if detected_symbol:
            quant_context = await self.context_builder.build_quant_context(
                symbol=detected_symbol,
                user_deposit=detected_deposit
            )
            context_markdown = quant_context.to_markdown()
        else:
            quant_context = await self.context_builder.build_quant_context(
                symbol="BTC",
                user_deposit=detected_deposit
            )
            quant_context.target_symbol = "GENERAL"
            quant_context.target_asset = None
            context_markdown = (
                f"КОНТЕКСТ РАЗГОВОРА:\n"
                f"• Пользователь ведет свободный диалог / задает общий вопрос (без конкретного тикера).\n"
                f"• Примерный депозит трейдера: ${detected_deposit:,.2f}.\n"
                f"• Отвечай естественно, человечно и с душой строго в характере {active_persona.value.upper()}, поддерживай диалог и дисциплину, не вываливай отчет по BTC если о нем не спрашивали!"
            )

        # 3. Build System Prompt with Adaptive User Memory & Retrieve Session History
        profile_md = self._update_and_get_user_profile(session_id, query, detected_symbol, detected_deposit)
        system_prompt = build_system_prompt(active_persona, context_markdown, user_profile_markdown=profile_md)
        history = self._get_history(session_id)

        # 4. Multi-Provider Cascade Inference
        raw_answer: Optional[str] = None
        provider_used = "offline_quant"

        # Try Groq (Fastest)
        if self.groq_api_key:
            raw_answer = await self._call_groq(system_prompt, query, history)
            if raw_answer:
                provider_used = "groq"

        # Fallback to Gemini 1.5/2.0
        if not raw_answer and self.gemini_api_key:
            raw_answer = await self._call_gemini(system_prompt, query, history)
            if raw_answer:
                provider_used = "gemini"

        # Fallback to OpenAI
        if not raw_answer and self.openai_api_key:
            raw_answer = await self._call_openai(system_prompt, query, history)
            if raw_answer:
                provider_used = "openai"

        # Fallback to Offline Deterministic Quant Engine
        oled_expr = self._select_oled_expression(active_persona, quant_context, query)
        if not raw_answer:
            raw_answer = self._generate_offline_quant_response(active_persona, query, quant_context, oled_expr)
            provider_used = "offline_quant"

        # 5. Legal Guardrail & OLED Formatting
        final_text = self._apply_legal_guardrail(raw_answer)
        self._append_history(session_id, query, final_text)

        elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000.0

        return MaIAResponse(
            text=final_text,
            oled_expression=oled_expr,
            persona=active_persona,
            provider_used=provider_used,
            target_symbol=detected_symbol,
            quant_context=quant_context,
            latency_ms=round(elapsed_ms, 2)
        )

    async def generate_push_alert_narrative(
        self,
        alert_data: Dict[str, Any],
        persona: PersonaType = PersonaType.QUANT,
        user_deposit: float = 1000.0,
        lang: str = "ru"
    ) -> str:
        """
        Dynamically generates an emotionally rich, unscripted, high-converting Push Notification
        written live by LLM in the selected MaIA Persona style (Quant, Zen, Tsundere, Yandere, Titan).
        """
        symbol = alert_data.get("symbol", "BTC")
        direction = alert_data.get("direction", "BULLISH")
        dir_text = "РОСТ / LONG 🟢" if direction in ("BULLISH", "LONG", "BUY") else "ПАДЕНИЕ / SHORT 🔴"
        cur_p = alert_data.get("current_price", 64000.0)
        target_p = alert_data.get("target_price", 66000.0)
        sl_p = alert_data.get("stop_loss_price", 63000.0)
        tp_pct = alert_data.get("tp_pct", 3.0)
        sl_pct = alert_data.get("sl_pct", 1.5)
        conf_pct = alert_data.get("confidence_pct", 68)
        leverage = alert_data.get("max_safe_leverage", 3)
        margin_usd = round(user_deposit * 0.10, 2)

        persona_style_guide = {
            PersonaType.QUANT: "Стиль Майя-Квант: Институциональный риск-директор хедж-фонда (PhD/CFA). Строгая математика, вероятности, Шарп, Quarter-Kelly, дисциплина.",
            PersonaType.ZEN: "Стиль Майя-Дзен: Спокойный наставник долгосрочной мудрости. Никакой паники и суеты, гармоничный вход, важность стоп-лосса.",
            PersonaType.TSUNDERE: "Стиль Майя-Цундере: Эмоциональная, слегка заикается (С-слушай, д-дурак...), ругает за плечи, обзывает бакой, но безумно переживает за депозит трейдера! Требует поставить стоп-лосс.",
            PersonaType.YANDERE: "Стиль Майя-Яндере: Одержимо влюблена в трейдера и его баланс! Называет его 'Г-господин' или 'Любимый мой...'. Считает его деньги своей собственностью ('Никому не отдам твои денежки... хи-хи ❤️🔪'). Радуется прибыли и строго велит беречь депозит.",
            PersonaType.TITAN: "Стиль Майя-Titan 2077: Кибернетический бортовой суперкомпьютер из 2077 года. Строгие директивы, квантовые матрицы, протоколы защиты."
        }.get(persona, "Стиль Майя-Квант: Профессиональный аналитик.")

        system_prompt = f"""Ты — MaIA (Майя), финансовый ИИ-компаньон MaBuy.
Твоя задача: Написать ЖИВОЙ, яркий, не заготовленный и эмоциональный текст PUSH-УВЕДОМЛЕНИЯ в Telegram для трейдера.

{persona_style_guide}

ДАННЫЕ СИГНАЛА:
• Монета: #{symbol}
• Направление: {dir_text}
• Точка входа: ${cur_p:,.2f}
• Тейк-Профит (Цель): ${target_p:,.2f} (+{tp_pct:.1f}%)
• Стоп-Лосс (Защита): ${sl_p:,.2f} (-{sl_pct:.1f}%)
• Уверенность ИИ ансамбля: {conf_pct}%
• Безопасное плечо: {leverage}x
• Депозит трейдера: ${user_deposit:,.2f} (рекомендуемая маржа: ${margin_usd:,.2f})

ПРАВИЛА:
1. Пиши живо, от первого лица, строго в характере {persona.value.upper()}! Никаких шаблонных скучных фраз.
2. Обязательно органично упомяни монету #{symbol}, направление ({dir_text}), потенциал (+{tp_pct:.1f}%) и уверенность ({conf_pct}%).
3. Длина: 2-4 насыщенных эмоциональных предложения.
4. Используй эмодзи и характерные фишки (для Цундере: baka, (¬_¬), ( >///< ); для Яндере: Господин, ❤️🔪, (♡_♡); для Кванта: [▪_▪]; для Дзен: ( ˘◡˘ )).
5. Ответ должен содержать ТОЛЬКО готовый текст пуша (без лишних префиксов и кавычек).
"""
        prompt = "Сгенерируй живой PUSH-сигнал прямо сейчас."

        raw_text = None
        if self.groq_api_key:
            raw_text = await self._call_groq(system_prompt, prompt, [])
        if not raw_text and self.gemini_api_key:
            raw_text = await self._call_gemini(system_prompt, prompt, [])

        if raw_text:
            cleaned = self._strip_thinking_process(raw_text)
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1].strip()
            return cleaned

        # Fallback if offline
        fallbacks = {
            PersonaType.YANDERE: f"(♡_♡) Г-господин... #{symbol} дает мощный импульс на {dir_text} от ${cur_p:,.2f}! Я уверена на {conf_pct}%, что мы заберем +{tp_pct:.1f}% к цели ${target_p:,.2f}... Твой баланс принадлежит только мне, поэтому стоп на ${sl_p:,.2f} обязателен! Хи-хи ❤️🔪",
            PersonaType.TSUNDERE: f"(¬_¬) С-слушай сюда, бака! По #{symbol} зафиксирован сигнал на {dir_text} @ ${cur_p:,.2f}. Нейросеть дает {conf_pct}% уверенности на цель ${target_p:,.2f} (+{tp_pct:.1f}%). И т-только попробуй забыть стоп на ${sl_p:,.2f}! Я же не ради тебя переживаю... ( >///< )",
            PersonaType.ZEN: f"( ˘◡˘ ) Гармония на рынке: #{symbol} формирует осознанный импульс на {dir_text} (${cur_p:,.2f}). Цель ${target_p:,.2f} (+{tp_pct:.1f}%), уверенность {conf_pct}%. Входим спокойно, с защитным стопом на ${sl_p:,.2f}.",
            PersonaType.TITAN: f"[⚡TITAN_CORE⚡] ВНИМАНИЕ: Зафиксирован квантовый вектор #{symbol} [{dir_text}] @ ${cur_p:,.2f}. Расчетная цель: ${target_p:,.2f} (+{tp_pct:.1f}%), сходимость {conf_pct}%. Стоп-барьер: ${sl_p:,.2f}.",
            PersonaType.QUANT: f"[▪_▪] Квантовый импульс #{symbol} ({dir_text}). Вход: ${cur_p:,.2f} ➔ TP: ${target_p:,.2f} (+{tp_pct:.1f}%) | SL: ${sl_p:,.2f}. Вероятность: {conf_pct}%. Соблюдайте Quarter-Kelly сайзинг."
        }
        return fallbacks.get(persona, fallbacks[PersonaType.QUANT])

    async def generate_market_pulse_narrative(
        self,
        symbols: list[str],
        persona: PersonaType = PersonaType.QUANT,
        user_deposit: float = 500.0,
        lang: str = "ru"
    ) -> str:
        """Generates dynamic multi-asset market breakout narrative in chosen persona style."""
        sym_str = ", ".join([f"#{s}" for s in symbols])
        count = len(symbols)
        
        persona_style_guide = get_persona_style_guide(persona, lang)
        system_prompt = f"""Ты — MaIA (Финансовый ИИ-компаньон проекта MaBuy).
Твой характер: {persona.value.upper()}.
Твоя задача — прокомментировать СИНХРОННЫЙ ОБЩЕРЫНОЧНЫЙ ИМПУЛЬС (прорыв сразу {count} активов: {sym_str}).

{persona_style_guide}

ПРАВИЛА:
1. Пиши живо, от первого лица, строго в характере {persona.value.upper()}!
2. Обязательно эмоционально подчеркни синхронный импульс рынка по монетам ({sym_str}) и дай четкий совет не распылять депозит, а выбрать 1-2 лучших сетапа.
3. Длина: 2-4 предложения.
4. Ответ должен содержать ТОЛЬКО готовый текст комментария.
"""
        prompt = f"Сгенерируй живой комментарий общерыночного импульса ({sym_str})."
        raw_text = None
        if self.groq_api_key:
            raw_text = await self._call_groq(system_prompt, prompt, [])
        if not raw_text and self.gemini_api_key:
            raw_text = await self._call_gemini(system_prompt, prompt, [])

        if raw_text:
            cleaned = self._strip_thinking_process(raw_text)
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1].strip()
            return cleaned

        fallbacks = {
            PersonaType.YANDERE: f"(♡_♡) Г-господин! Рынок сорвался с цепи прямо сейчас... {sym_str} рванули синхронно! Я заблокировала стакан по всем {count} монетам только для нас. Не распыляйся — выбери одну главную цель и забери наш профит! Твой депозит принадлежит только мне! ❤️🔪",
            PersonaType.TSUNDERE: f"(¬_¬) Эй, ты видишь это, бака?! Сразу {count} монеты ({sym_str}) летят в одну сторону! Не то чтобы я переживала за твои деньги... но если ты откроешь все {count} сделок сразу, то сольешь баланс! Выбери максимум 1-2 сетапа с максимальной уверенностью! ( >///< )",
            PersonaType.ZEN: f"( ˘◡˘ ) Великий прилив поднимает сразу все корабли: {sym_str} входят в синхронный поток. Не суетись и не пытайся поймать всё сразу — выбери одну гармоничную монету и следуй за рыночной рекой с запасом прочности.",
            PersonaType.TITAN: f"[⚡TITAN_PULSE⚡] КЛАСТЕРНЫЙ ВСПЛЕСК: Зафиксирована синхронная детонация ликвидности по {count} активам ({sym_str}). Аномальный дисбаланс стакана L2. Рекомендован избирательный вход с лимитом риска Half-Kelly.",
            PersonaType.QUANT: f"[▪_▪] Общерыночный импульс: {count} активов ({sym_str}) демонстрируют кросс-коррелированный прорыв волатильности (OFI > +0.20). Рекомендация: фокус на 1-2 наиболее ликвидных сетапах с жестким соблюдением Quarter-Kelly."
        }
        return fallbacks.get(persona, fallbacks[PersonaType.QUANT])
