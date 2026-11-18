#!/usr/bin/env python3
"""
Astra Institutional Whitepaper & Investment Memorandum Generator.

Generates a Tier-1 Venture Capital (a16z crypto / Paradigm / Pantera caliber)
Whitepaper and Quantitative Technical Architecture Document for MaBuy.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

from src.domain.services.astra_advisor import AstraAdvisor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AstraWhitepaperGen")


def generate_whitepaper():
    print("================================================================================")
    print("   MABUY INSTITUTIONAL WHITEPAPER GENERATION (TIER-1 VENTURE CAPITAL DECK)      ")
    print("================================================================================")

    advisor = AstraAdvisor()
    if not advisor.is_available:
        raise ValueError("AstraAdvisor is unavailable (missing API key).")

    system_prompt = """You are Astra, Principal Quantitative Architect, Venture Partner at a Tier-1 Crypto Quantitative Fund (Paradigm / Pantera / a16z crypto caliber), and former Head of Quantitative Research at Renaissance Technologies.

Write a definitive, institutional-grade Quantitative Whitepaper & Technical Architecture Document for:
**MaBuy V5.2: The Renaissance Titan Autonomous Crypto Quantitative Trading Ecosystem**

The document must read with absolute institutional authority, mathematical precision, and commercial clarity.
Cover the following 8 comprehensive sections:

# 1. Executive Summary & Investment Thesis
- The macroeconomic crypto futures landscape: retail churn vs. algorithmic predatory microstructure.
- MaBuy's core thesis: Combining Causal Machine Learning, L2 Order Flow Imbalance, and Marcos López de Prado non-ruin risk guards into an autonomous, 24/7 high-payoff execution engine.
- High-level track record metrics: Payoff Ratio R = 3.09 (BTC), R = 2.37 (SOL), Max DD 7.33%, 100% Black Swan survival.

# 2. Mathematical Alpha & Causal Machine Learning
- The Dual-Horizon Hybrid Architecture:
  * HistGradientBoostingRegressor (18 stationary technical, lag, and volatility features).
  * PyTorch Multi-Horizon Attention LSTM (recurrent temporal dependency extraction without future leakage).
  * Causal Sequence Rolling Buffer mechanics.
- Dynamic Confidence Calibration formula: how signal strength, attention confidence, model consensus, and regime indicators scale confidence into the 75-88% institutional sniper zone.

# 3. Market Microstructure & High-Frequency Order Flow (OFI)
- Order Flow Imbalance (OFI) formula: Top-of-book delta, queue depletion, and cancel/fill ratios.
- Notional thresholding: $150,000 USD institutional wall detection vs. retail noise.
- Live microprice skew, Cumulative Volume Delta (CVD), and pre-execution spread friction filters.

# 4. The 5-Tier Non-Ruin Risk Guard Architecture
- Detailed mathematical formulations and rationale for each gate:
  1. Funding Rate Squeeze & Crowding Bias Engine (8-hour rate modulation & carry sizing).
  2. Multi-Timeframe (MTF) Trend Confluence Service (structural alignment).
  3. Rescaled-Range (R/S) Hurst Exponent Dynamical Classifier (persistence H >= 0.58, anti-persistence H <= 0.42, and strict Brownian noise deadband [0.42, 0.58]).
  4. Real-Time Microstructure OFI Veto Gate.
  5. Post-Loss Churn Defense & Minimum Price Displacement Lockout (1.5x ATR displacement required to re-enter).
- Dynamic Adaptive Volatility Circuit Breaker with consecutive-loss tightening and 24h cooling off.
- Cross-Asset Correlation Guard with 40% margin haircut under rho = 0.98.

# 5. Execution Microstructure & Fee Capture
- VIP-0 Maker-First Post-Only execution: 0.02% maker fee vs. 0.05% taker fee (+0.03% edge per side, saving 30% of margin profit per trade).
- Atomic Stop-Loss Replacement Pattern: Proof of zero naked position exposure during trailing stop updates.
- NTP microsecond clock synchronization with Binance matching engines.

# 6. Empirical Performance, DSR, & Black Swan Stress-Testing
- Marcos López de Prado Deflated Sharpe Ratio (DSR), Probabilistic Sharpe Ratio (PSR), and Minimum Track Record Length (MinTRL).
- 81-run Hyperparameter Grid Optimization & Pareto Frontier synthesis.
- 5-Scenario Black Swan Crisis Simulation (FTX -25% crash, Yen Carry -14% wick, 180 bps spread vacuum, rho=0.98 contagion, 72h anti-persistent chop) with mathematical proof of capital non-ruin.

# 7. Unit Economics, Capital Scalability, & Deposit Protection
- Micro-Capital Economics: The $39.54 live account, 100% Cash Flash-Shock protection ($59.00 threshold), and the mathematical rationale for BTC quarantine below $100.
- Scaling trajectory: $39.54 -> $150.00 -> $500.00 -> $50,000+ with Volatility-Targeted Fractional Kelly sizing.
- SaaS & B2B Institutional monetization models (Telegram VIP Bot, API licensing, Proprietary Trading Desk).

# 8. Roadmap to 10.0: The Quantitative Horizon
- Low-latency Rust/C++ execution sidecar.
- Cross-exchange statistical arbitrage (Binance / OKX / Bybit funding basis).
- Online automated Bayesian self-tuning hyperparameters.

Format the whitepaper in exquisite GitHub-Flavored Markdown with LaTeX mathematical formulas, clear diagrams, tables, and executive conclusions."""

    user_payload = {
        "title": "MaBuy V5.2 Renaissance Titan Institutional Whitepaper",
        "request_type": "Institutional Whitepaper & VC Investment Memorandum",
        "target_audience": "Top-Tier Venture Capital & Sovereign Wealth Quantitative Funds",
        "generation_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    logger.info("Submitting Whitepaper generation request to Astra LLM Cluster...")
    res = advisor.consult(
        system_prompt=system_prompt,
        user_payload=user_payload,
        reasoning_effort="high",
        max_tokens=12000,
        response_json=False,
    )

    if not res or not res.get("text"):
        raise RuntimeError("AstraAdvisor returned an empty response for the whitepaper.")

    whitepaper_text = res["text"]
    docs_dir = ROOT_DIR / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out_path = docs_dir / "MABUY_INSTITUTIONAL_WHITEPAPER.md"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# MABUY V5.2: THE RENAISSANCE TITAN QUANTITATIVE ARCHITECTURE\n\n")
        f.write(f"**Document Version**: 5.2.0-INSTITUTIONAL\n")
        f.write(f"**Publication Date**: {dt.datetime.now(dt.timezone.utc).strftime('%B %d, %Y')}\n")
        f.write(f"**Principal Author**: Astra Quantitative Research Group & Mathematical Systems Architecture\n")
        f.write(f"**Classification**: Institutional Investment Memorandum & Technical Whitepaper\n\n")
        f.write("---\n\n")
        f.write(whitepaper_text)
        f.write("\n")

    print(f"\nInstitutional Whitepaper written to: {out_path}")
    print("================================================================================")
    return out_path


if __name__ == "__main__":
    generate_whitepaper()
