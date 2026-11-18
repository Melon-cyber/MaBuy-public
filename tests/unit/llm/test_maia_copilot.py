"""
Unit & Integration Tests for MaIA LLM Architecture and Copilot Engine.
Verifies Persona Prompts, Quant RAG Context Builder, Legal Guardrails, and Copilot Engine.
"""

import unittest
import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.infrastructure.llm.persona_prompts import (
    PersonaType,
    LEGAL_DISCLAIMER_RU,
    OLED_EXPRESSIONS,
    PERSONA_METADATA,
    build_system_prompt,
)
from src.infrastructure.llm.context_builder import (
    QuantContextBuilder,
    QuantContext,
    AssetQuantSnapshot,
)
from src.infrastructure.llm.copilot_engine import (
    MaIACopilotEngine,
    MaIAResponse,
)


class TestMaIACopilotModule(unittest.TestCase):
    """Test suite covering MaIA LLM Architecture."""

    def test_persona_prompts_and_metadata(self):
        """Verifies that all 4 personas are properly registered with unique metadata and OLED expressions."""
        for p in PersonaType:
            self.assertIn(p, PERSONA_METADATA)
            meta = PERSONA_METADATA[p]
            self.assertIn("name", meta)
            self.assertIn("badge", meta)
            self.assertIn("description", meta)

            self.assertIn(p, OLED_EXPRESSIONS)
            oleds = OLED_EXPRESSIONS[p]
            self.assertIn("neutral", oleds)
            self.assertIn("analytical", oleds)

            # Test prompt construction
            prompt = build_system_prompt(p, quant_context_markdown="TEST_QUANT_DATA")
            self.assertIn("TEST_QUANT_DATA", prompt)
            self.assertIn("ZERO-HALLUCINATION PROTOCOL", prompt)

    def test_quant_context_builder(self):
        """Verifies that QuantContextBuilder aggregates live prices, Merton forecasts, Kelly sizing, and Markowitz weights."""
        async def _run():
            builder = QuantContextBuilder()
            ctx = await builder.build_quant_context(symbol="BTC", user_deposit=1000.0)
            return ctx

        ctx = asyncio.run(_run())

        self.assertIsInstance(ctx, QuantContext)
        self.assertEqual(ctx.target_symbol, "BTC")
        self.assertEqual(ctx.user_deposit_usd, 1000.0)
        self.assertIn("BTC", ctx.universe_prices)
        self.assertGreater(ctx.universe_prices["BTC"], 0)

        # Check target asset snapshot
        ta = ctx.target_asset
        self.assertIsNotNone(ta)
        self.assertEqual(ta.symbol, "BTC")
        self.assertGreater(ta.current_price, 0)
        self.assertIsNotNone(ta.merton_p50_365d)
        self.assertIsNotNone(ta.merton_p10_365d)
        self.assertIsNotNone(ta.merton_p90_365d)
        self.assertGreater(ta.merton_p90_365d, ta.merton_p10_365d)

        # Check position sizing plan
        psp = ctx.position_sizing_plan
        self.assertIn("risk_pct", psp)
        self.assertIn("risk_amount_usd", psp)
        self.assertIn("stop_loss_pct", psp)
        self.assertIn("take_profit_pct", psp)
        self.assertGreaterEqual(psp["reward_risk_ratio"], 1.5)

        # Check portfolio optimization
        self.assertGreater(len(ctx.optimal_portfolio_max_sharpe), 0)
        self.assertGreater(len(ctx.optimal_portfolio_min_variance), 0)

        # Check markdown serialization
        md = ctx.to_markdown()
        self.assertIn("СПОТ-ЦЕНЫ 13 ТОП-АКТИВОВ", md)
        self.assertIn("ГЛУБОКИЙ КВАНТОВЫЙ АНАЛИЗ", md)
        self.assertIn("РАСЧЕТ РИСКА И САЙЗИНГА", md)
        self.assertIn("ОПТИМАЛЬНЫЕ ДОЛИ ПОРТФЕЛЯ", md)

    def test_copilot_engine_entity_extraction(self):
        """Verifies extraction of asset symbols and user budget amounts from natural queries."""
        engine = MaIACopilotEngine()

        self.assertEqual(engine.extract_symbol_from_query("Что думаешь по SOL на споте?"), "SOL")
        self.assertEqual(engine.extract_symbol_from_query("Стоит ли сейчас покупать акции NVDA?"), "NVDA")
        self.assertIsNone(engine.extract_symbol_from_query("Помоги распределить баланс"))

        self.assertEqual(engine.extract_deposit_from_query("У меня есть $1500, как войти?"), 1500.0)
        self.assertEqual(engine.extract_deposit_from_query("Баланс 2500 долларов для торговли"), 2500.0)
        self.assertEqual(engine.extract_deposit_from_query("Просто дай совет"), 500.0)

    def test_copilot_engine_all_personas_generation(self):
        """Verifies end-to-end response generation, legal guardrails, and OLED states across all 4 personas."""
        engine = MaIACopilotEngine()

        queries_and_personas = [
            ("Рассчитай математическое ожидание и стоп-лосс для BTC", PersonaType.QUANT),
            ("Как мне спокойно накапливать SOL без паники и с DCA?", PersonaType.ZEN),
            ("Хочу взять ETH на 20x плечо на всю котлету!", PersonaType.TSUNDERE),
            ("Активируй квантовое сканирование NVDA", PersonaType.TITAN),
        ]

        async def _run_all():
            results = []
            for query, persona in queries_and_personas:
                resp = await engine.generate_response(
                    query=query,
                    persona=persona,
                    user_deposit=1000.0,
                    session_id="test_session_1"
                )
                results.append((persona, resp))
            return results

        results = asyncio.run(_run_all())

        for persona, resp in results:
            self.assertIsInstance(resp, MaIAResponse)
            self.assertEqual(resp.persona, persona)
            self.assertGreater(len(resp.text), 50)
            self.assertGreater(len(resp.oled_expression), 0)

            # Mandatory Legal Guardrail validation
            self.assertIn("Не является индивидуальной инвестиционной рекомендацией", resp.text)

            # Persona-specific stylistic cues
            text_low = resp.text.lower()
            if persona == PersonaType.QUANT:
                self.assertTrue(
                    "квант" in text_low or "мертон" in text_low or "kelly" in text_low or "шарп" in text_low or "cagr" in text_low
                )
            elif persona == PersonaType.ZEN:
                self.assertTrue(
                    "dca" in text_low or "гармон" in text_low or "терпелив" in text_low or "покой" in text_low or "дзен" in text_low or "сложн" in text_low
                )
            elif persona == PersonaType.TSUNDERE:
                self.assertTrue(
                    "бака" in text_low or "котлет" in text_low or "ликвидаци" in text_low or "пф" in text_low or "плеч" in text_low
                )
            elif persona == PersonaType.TITAN:
                self.assertTrue(
                    "titan" in text_low or "тензор" in text_low or "protocol" in text_low or "sys" in text_low or "процесс" in text_low
                )

    def test_copilot_session_history(self):
        """Verifies multi-turn conversation memory within a session."""
        async def _run():
            engine = MaIACopilotEngine()
            session_id = "user_42_vip"

            resp1 = await engine.generate_response("Привет, я планирую войти в SOL на $700", persona=PersonaType.QUANT, session_id=session_id)
            history1 = engine._get_history(session_id)

            resp2 = await engine.generate_response("А какой стоп-лосс мне поставить?", persona=PersonaType.QUANT, session_id=session_id)
            history2 = engine._get_history(session_id)
            return resp1, history1, resp2, history2

        resp1, history1, resp2, history2 = asyncio.run(_run())

        self.assertIsNotNone(resp1)
        self.assertEqual(len(history1), 2)
        self.assertEqual(history1[0].role, "user")
        self.assertEqual(history1[1].role, "assistant")

        self.assertIsNotNone(resp2)
        self.assertEqual(len(history2), 4)


if __name__ == "__main__":
    unittest.main()
