"""
MaBuy Institutional Quantitative Advisory Service (GPT-6 Astra Titan Engine)
Provides deep-reasoning algorithmic audits, regime diagnosis, and risk orchestration.
Configured with maximum reasoning depth and up to 12,000 completion tokens.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("AstraAdvisor")

CANDIDATE_MODELS = [
    "gpt-6-astra",
    "gpt-5.6-sol",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
]

class AstraReasoningTier:
    """
    USAGEMAX Reasoning Effort Tiers:
    - LOW_MED: Quick audits, writing, research, daily market summaries (4k tokens)
    - HIGH: Everyday monster, features, full UI/UX drafts, TMA widgets (8k tokens)
    - EXTRA_HIGH: Deep debugging, quant audits, churn diagnosis, DB migrations (12k tokens)
    - ULTRA: Cybersecurity monster, Black Swan stress tests, full architecture (16k+ tokens)
    """
    LOW_MED = "LOW_MED"
    HIGH = "HIGH"
    EXTRA_HIGH = "EXTRA_HIGH"
    ULTRA = "ULTRA"

TIER_CONFIGS = {
    AstraReasoningTier.LOW_MED: {"reasoning_effort": "low", "max_tokens": 4096},
    AstraReasoningTier.HIGH: {"reasoning_effort": "medium", "max_tokens": 8192},
    AstraReasoningTier.EXTRA_HIGH: {"reasoning_effort": "high", "max_tokens": 12000},
    AstraReasoningTier.ULTRA: {"reasoning_effort": "high", "max_tokens": 16384},
}

class AstraAdvisor:
    """
    Direct interface to GPT-6 Astra reasoning models for quantitative risk defense,
    portfolio optimization, and deep post-loss / micro-structure diagnosis.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self.api_key = api_key or os.environ.get("ASTRA_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("ASTRA_BASE_URL") or "https://ttmapi.site/v1"
        self._client = None
        if self.api_key:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=300.0)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client for AstraAdvisor: {e}")

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def consult(
        self,
        system_prompt: str,
        user_payload: Dict[str, Any],
        reasoning_effort: str = "high",
        max_tokens: int = 12000,
        response_json: bool = True,
        model: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Executes a deep-reasoning consultation with Astra.
        Employs reasoning_effort='high' and up to 16,000 tokens for thorough mathematical proof.
        """
        if not self._client:
            logger.error("AstraAdvisor called but OpenAI client is not initialized (missing API key).")
            return None

        user_content = json.dumps(user_payload, ensure_ascii=False, indent=2)
        effective_system = system_prompt
        if response_json and "json" not in effective_system.lower():
            effective_system += "\nYou MUST return a valid JSON object adhering strictly to RFC 8259."

        models_to_try = [model] if model else CANDIDATE_MODELS
        for target_model in models_to_try:
            try:
                logger.info(f"🔮 Consulting {target_model} (reasoning_effort={reasoning_effort}, max_tokens={max_tokens})...")
                role_name = "developer" if any(m in target_model for m in ["o1", "o3", "astra", "gpt-6"]) else "system"
                kwargs = {
                    "model": target_model,
                    "messages": [
                        {"role": role_name, "content": effective_system},
                        {"role": "user", "content": user_content}
                    ]
                }
                
                # Reasoning models support reasoning_effort
                if any(m in target_model for m in ["o3", "o1", "astra", "sol", "gpt-5", "gpt-6"]):
                    kwargs["reasoning_effort"] = reasoning_effort
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = min(max_tokens, 4096)

                if response_json:
                    kwargs["response_format"] = {"type": "json_object"}

                kwargs["stream"] = True
                response_stream = self._client.chat.completions.create(**kwargs)
                chunks = []
                for chunk in response_stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        content = chunk.choices[0].delta.content
                        if content:
                            chunks.append(content)
                raw_text = "".join(chunks) or "{}"
                logger.info(f"✅ Received response from {target_model} ({len(raw_text)} chars).")

                if response_json:
                    # Clean markdown codeblocks if model wraps output
                    cleaned = raw_text.strip()
                    if cleaned.startswith("```json"):
                        cleaned = cleaned[7:]
                    if cleaned.startswith("```"):
                        cleaned = cleaned[3:]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                    return json.loads(cleaned.strip())
                return {"text": raw_text}

            except Exception as err:
                logger.warning(f"Model {target_model} failed or unavailable: {err}. Attempting fallback...")

        logger.error("All candidate Astra models failed.")
        return None

    def consult_with_tier(
        self,
        system_prompt: str,
        user_payload: Dict[str, Any],
        tier: str = AstraReasoningTier.HIGH,
        response_json: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Executes consultation using standardized USAGEMAX reasoning tier."""
        cfg = TIER_CONFIGS.get(tier, TIER_CONFIGS[AstraReasoningTier.HIGH])
        return self.consult(
            system_prompt=system_prompt,
            user_payload=user_payload,
            reasoning_effort=cfg["reasoning_effort"],
            max_tokens=cfg["max_tokens"],
            response_json=response_json
        )

    def audit_portfolio_health(
        self,
        equity: float,
        active_positions: List[Dict[str, Any]],
        recent_trades: List[Dict[str, Any]],
        rolling_24h_pnl: float
    ) -> Optional[Dict[str, Any]]:
        """Runs an institutional risk audit on the active portfolio."""
        system_prompt = (
            "Role: Chief Quantitative Risk Officer & Lead Portfolio Manager (Citadel / Renaissance caliber).\n"
            "Objective: Conduct a rigorous quantitative audit of active crypto positions, portfolio drawdown, "
            "margin utilization, and serial correlation. Return valid JSON with keys: risk_grade, action_items, "
            "allow_new_entries (bool), maximum_safe_leverage, mathematical_analysis."
        )
        payload = {
            "account_equity_usd": equity,
            "rolling_24h_pnl_usd": rolling_24h_pnl,
            "active_positions": active_positions,
            "recent_trades": recent_trades
        }
        return self.consult(system_prompt, payload, reasoning_effort="high", max_tokens=8192)
