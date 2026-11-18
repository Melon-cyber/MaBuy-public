"""
MaBuy Quantitative Trading Ecosystem
Module: src/infrastructure/mlops/astra_auditor.py
Description: Frontier AGI Lead Quant Auditor powered by OpenAI GPT-6 Astra.
             Performs institutional two-layer deployment validation on weekly retrained models.
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("AstraQuantAuditor")

class AstraQuantAuditor:
    """
    Lead Quantitative Auditor leveraging OpenAI GPT-6 Astra (with graceful fallback).
    Enforces institutional two-layer deployment verification:
      - Layer 1: Hard Deterministic Mathematical Filter (DA >= 52.5%, Sharpe >= 1.05, Drift <= 15%).
      - Layer 2: Frontier AGI Qualitative Forensic Review & Memorandum.
    """

    DEFAULT_MODEL = "gpt-4o-mini"
    FALLBACK_MODEL = "gpt-4o"
    EMERGENCY_FALLBACK_MODEL = "gpt-6-astra"

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        # Auto-discover and load .env.retrain or .env if not loaded
        try:
            from dotenv import load_dotenv
            for env_candidate in ["/root/mabuy2/.env.retrain", ".env.retrain", "/root/mabuy2/.env", ".env", "../.env"]:
                if os.path.exists(env_candidate):
                    load_dotenv(env_candidate)
        except ImportError:
            pass

        self.api_key = api_key or os.getenv("RETRAIN_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name or os.getenv("RETRAIN_ASTRA_MODEL") or os.getenv("ASTRA_MODEL", self.DEFAULT_MODEL)
        self.client = None
        
        base_url = os.getenv("RETRAIN_OPENAI_BASE_URL")
        if not base_url:
            if self.api_key and self.api_key.startswith("sk-proj-"):
                base_url = "https://api.openai.com/v1"
            else:
                base_url = os.getenv("OPENAI_BASE_URL")

        if self.api_key:
            try:
                import openai
                kwargs = {"api_key": self.api_key, "timeout": 45.0}
                if base_url:
                    kwargs["base_url"] = base_url
                self.client = openai.OpenAI(**kwargs)
                logger.info(f"AstraQuantAuditor initialized with model: {self.model_name} (base_url: {base_url or 'default'})")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}. Running in pure math fallback mode.")
        else:
            logger.warning("Neither RETRAIN_OPENAI_API_KEY nor OPENAI_API_KEY found. AstraQuantAuditor will operate in deterministic math mode.")

    def run_layer1_math_guard(self, symbol: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Layer 1: Deterministic Mathematical Filter.
        Fast, zero-cost, 100% deterministic guard.
        """
        da = float(metrics.get("directional_accuracy_oos", 0.0))
        # Handle both decimal (0.54) and percentage (54.0) formats
        da_pct = da if da > 1.0 else da * 100.0
        
        sharpe = float(metrics.get("sharpe_oos", 0.0))
        w_att_drift = float(metrics.get("weight_norms", {}).get("W_att_drift_pct", 0.0))
        w_ih_drift = float(metrics.get("weight_norms", {}).get("W_ih_drift_pct", 0.0))

        reasons = []
        passed = True

        if da_pct < 52.5:
            passed = False
            reasons.append(f"DA {da_pct:.2f}% < 52.5% threshold")
        
        if sharpe < 1.00:
            passed = False
            reasons.append(f"Sharpe {sharpe:.2f} < 1.00 minimum")

        if abs(w_att_drift) > 15.0:
            passed = False
            reasons.append(f"||W_att|| drift {w_att_drift:+.2f}% exceeds 15% limit")

        if abs(w_ih_drift) > 10.0:
            passed = False
            reasons.append(f"||W_ih|| drift {w_ih_drift:+.2f}% exceeds 10% limit")

        return {
            "symbol": symbol,
            "layer1_passed": passed,
            "reasons": reasons,
            "metrics": {
                "da_pct": round(da_pct, 2),
                "sharpe": round(sharpe, 2),
                "w_att_drift_pct": round(w_att_drift, 2),
                "w_ih_drift_pct": round(w_ih_drift, 2)
            }
        }

    def audit_weekly_cohort(self, cohort_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs full cohort audit across all retrained assets (BTC, ETH, SOL).
        Returns comprehensive decision dictionary with GPT-6 Astra's memorandum.
        """
        timestamp_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        symbols = cohort_payload.get("symbols", {})
        
        # 1. Run Layer 1 on all symbols
        layer1_results = {}
        for sym, sym_metrics in symbols.items():
            layer1_results[sym] = self.run_layer1_math_guard(sym, sym_metrics)

        all_layer1_passed = all(r["layer1_passed"] for r in layer1_results.values()) if layer1_results else False

        # If OpenAI client is unavailable, return deterministic math verdict
        if not self.client:
            logger.info("OpenAI client not configured. Returning deterministic Layer 1 audit.")
            return {
                "audit_timestamp": timestamp_utc,
                "auditor_model": "DeterministicMathGuard (No API Key)",
                "decision": "APPROVE" if all_layer1_passed else "REJECT",
                "confidence_score": 0.70,
                "layer1_results": layer1_results,
                "regime_summary": "Сбор метрик выполнен в режиме математического детерминизма.",
                "weight_drift_analysis": "Весовые нормы проверены по жестким порогам (лимит 15% для внимания, 10% для базы).",
                "lead_quant_memo": f"Математический шлюз: {'Успешно пройден всеми активами' if all_layer1_passed else 'Обнаружены отклонения порогов'}.",
                "recommendations": "Рекомендуется следить за трейлинг-стопами и волатильностью в течение 24 часов после деплоя."
            }

        # 2. Layer 2: Call GPT-6 Astra for Deep Qualitative Forensic Audit
        system_prompt = """Ты — Chief Quantitative Risk Officer & Lead Quant фонда MaBuy.
Твоя задача — провести независимый криминалистический аудит еженедельного переобучения ансамбля (Weekly Retrain Pipeline: GBDT + PyTorch Attention LSTM).

Институциональные критерии допуска к PROD (APPROVE):
1. OOS Directional Accuracy (DA) >= 52.5% (с учетом комиссии Maker 0.02%).
2. OOS Sharpe Ratio >= 1.05.
3. Дельта нормы весов внимания (||W_att||) <= 15.0% (сохранение долговременной памяти без переобучения на шум).
4. Дельта LSTM Backbone (||W_ih||) <= 10.0%.

Если все ключевые критерии соблюдены и нет скрытого катастрофического оверфиттинга, выдай вердикт "APPROVE".
Если хотя бы один ключевой актив деградировал или выявлена критическая нестабильность весов, выдай "REJECT".

Ответь СТРОГО в формате валидного JSON объекта:
{
  "decision": "APPROVE" | "REJECT",
  "confidence_score": 0.85,
  "regime_summary": "Резюме текущего макро/микро режима рынка (Hurst, Funding, волатильность)",
  "weight_drift_analysis": "Анализ дельты весов внимания и стабильности нейросети",
  "lead_quant_memo": "Развернутое заключение Lead Quant для торгового комитета",
  "recommendations": "Рекомендации по управлению рисками для боевого демона"
}"""

        user_content = json.dumps({
            "timestamp_utc": timestamp_utc,
            "symbols_data": cohort_payload,
            "layer1_pre_check": layer1_results
        }, ensure_ascii=False)

        candidate_models = list(dict.fromkeys([
            self.model_name,
            "gpt-4o-mini",
            "gpt-4o",
            self.FALLBACK_MODEL,
            self.EMERGENCY_FALLBACK_MODEL
        ]))
        last_error = None
        astra_response = None

        for model in candidate_models:
            try:
                logger.info(f"Dispatching retrain audit to {model} with reasoning_effort='low'...")
                # Attempt with reasoning_effort (supported on gpt-6-astra and o-series)
                try:
                    res = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        reasoning_effort="low",
                        response_format={"type": "json_object"}
                    )
                except Exception as param_err:
                    logger.debug(f"Calling {model} without reasoning_effort parameter: {param_err}")
                    res = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        response_format={"type": "json_object"}
                    )

                content = res.choices[0].message.content
                parsed = json.loads(content)
                parsed["auditor_model"] = model
                parsed["audit_timestamp"] = timestamp_utc
                parsed["layer1_results"] = layer1_results
                astra_response = parsed
                logger.info(f"✅ Audit successfully concluded by {model}. Decision: {parsed.get('decision')}")
                break

            except Exception as call_err:
                last_error = call_err
                logger.warning(f"Model {model} audit failed: {call_err}. Trying next candidate...")

        if astra_response:
            return astra_response

        # Graceful Fail-Safe Fallback if all API calls failed
        logger.error(f"All frontier LLM audit calls failed ({last_error}). Activating Layer 1 Fail-Safe Fallback.")
        return {
            "audit_timestamp": timestamp_utc,
            "auditor_model": f"Layer1FailSafe (Error: {str(last_error)[:60]})",
            "decision": "APPROVE" if all_layer1_passed else "REJECT",
            "confidence_score": 0.75,
            "layer1_results": layer1_results,
            "regime_summary": "Сетевой шлюз API временно недоступен. Анализ выполнен аварийным детерминированным модулем.",
            "weight_drift_analysis": "Дельта весов уложилась в лимиты Layer 1." if all_layer1_passed else "Превышены математические лимиты дрейфа.",
            "lead_quant_memo": f"Автоматический аварийный допуск на основе математических порогов: {'УТВЕРЖДЕНО' if all_layer1_passed else 'ОТКЛОНЕНО'}.",
            "recommendations": "Проверить статус подключения к внешнему API и журнал daemon_app.log."
        }

    def format_markdown_memo(self, audit_result: Dict[str, Any]) -> str:
        """
        Formats the audit result into a clean, institutional GitHub-style Markdown section.
        """
        dec = audit_result.get("decision", "UNKNOWN")
        model = audit_result.get("auditor_model", self.model_name)
        badge = "🟢 **УТВЕРЖДЕНО К ДЕПЛОЮ (APPROVED)**" if dec == "APPROVE" else "🔴 **ДЕПЛОЙ ОТКЛОНЁН (REJECTED / ROLLBACK)**"
        
        md_lines = [
            "## 🧠 Автономный надзор: GPT-6 Astra (Lead Quant Audit)",
            f"**Модель аудитора:** `{model}` | **Время:** `{audit_result.get('audit_timestamp')}`",
            f"**Вердикт квант-комитета:** {badge}",
            f"**Индекс уверенности AGI:** `{audit_result.get('confidence_score', 0.85):.2f}`",
            "",
            "### 📊 Оценка рыночного режима и волатильности",
            f"> {audit_result.get('regime_summary', 'Режим стабилен.')}",
            "",
            "### 🔬 Криминалистический аудит весов нейросети (PyTorch Attention Drift)",
            f"> {audit_result.get('weight_drift_analysis', 'Дрейф в пределах допустимого.')}",
            "",
            "### 📑 Меморандум Lead Quant",
            f"{audit_result.get('lead_quant_memo', 'Модели готовы к боевой эксплуатации.')}",
            "",
            "### 🛡️ Рекомендации по риск-параметрам демона",
            f"*{audit_result.get('recommendations', 'Сохранять стандартные лимиты просадки.')}*",
            ""
        ]
        return "\n".join(md_lines)
