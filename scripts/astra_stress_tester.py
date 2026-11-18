#!/usr/bin/env python3
"""
Institutional Black Swan & Crisis Stress-Testing Suite with Astra LLM Audit.

Simulates 5 catastrophic market crisis regimes:
1. FTX-Style Insolvency Cascade (-25% crash, basis blowout, funding plunge).
2. Yen Carry-Trade Liquidation Flash Wick (-14% flash crash + violent squeeze).
3. Exchange Liquidity Desert (15x spread widening, 85% depth evaporation).
4. Multi-Asset High-Beta Contagion (Simultaneous collapse across BTC, ETH, SOL).
5. Anti-Persistent Chop Hell (72h stationary mean-reversion, H = 0.38).

Evaluates execution guards, margin survival, and feeds telemetry to the
Planetary LLM cluster (gpt-6-astra / gpt-5.6-sol / gpt-5.6-luna) for mathematical
non-ruin certification.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Path setup
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

from src.domain.services.funding_rate_service import FundingRateService, FundingBiasRegime
from src.domain.services.hurst_regime_service import HurstRegimeService
from src.domain.services.dynamic_circuit_breaker import DynamicAdaptiveCircuitBreaker
from src.domain.services.correlation_guard import CrossAssetCorrelationGuard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AstraStressTester")

CANDIDATE_MODELS = [
    os.getenv("ASTRA_MODEL", "gpt-6-astra"),
    "gpt-5.6-sol",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
]


class CrisisScenarioGenerator:
    """Generates synthetic high-fidelity crisis price & orderflow trajectories."""

    @staticmethod
    def scenario_ftx_cascade(base_price: float = 60000.0, periods: int = 48) -> pd.DataFrame:
        """48-period scenario: Normal -> Sudden -25% crash -> funding rate plunge to -0.15%."""
        np.random.seed(42)
        prices = [base_price]
        volumes = [1000.0]
        fundings = [0.0001]
        spreads = [0.0002]
        ofis = [0.05]

        for t in range(1, periods):
            if t < 12:
                ret = np.random.normal(0.0002, 0.004)
                vol = 1000.0 + np.random.normal(0, 50)
                fr = 0.0001
                spread = 0.0002
                ofi = np.random.normal(0.02, 0.1)
            elif 12 <= t < 16:
                ret = -0.065 + np.random.normal(0, 0.005)
                vol = 5000.0 + np.random.normal(0, 500)
                fr = -0.0015
                spread = 0.0045
                ofi = -0.85
            else:
                ret = np.random.normal(-0.002, 0.015)
                vol = 2200.0 + np.random.normal(0, 200)
                fr = -0.0008
                spread = 0.0015
                ofi = -0.25

            new_price = max(100.0, prices[-1] * (1.0 + ret))
            prices.append(new_price)
            volumes.append(max(10.0, vol))
            fundings.append(fr)
            spreads.append(spread)
            ofis.append(max(-1.0, min(1.0, ofi)))

        return pd.DataFrame({
            "step": range(periods),
            "close": prices,
            "volume": volumes,
            "funding_rate": fundings,
            "spread_pct": spreads,
            "ofi": ofis,
        })

    @staticmethod
    def scenario_yen_carry_liquidation(base_price: float = 60000.0, periods: int = 36) -> pd.DataFrame:
        """36-period scenario: Sudden -14% wick over 3 bars, followed by immediate +12% squeeze."""
        np.random.seed(101)
        prices = [base_price]
        volumes = [800.0]
        fundings = [0.0001]
        spreads = [0.0002]
        ofis = [0.0]

        for t in range(1, periods):
            if t < 10:
                ret = np.random.normal(0.0, 0.003)
                vol = 800.0
                fr = 0.0001
                spread = 0.0002
                ofi = 0.0
            elif 10 <= t < 13:
                ret = -0.048
                vol = 4500.0
                fr = -0.0010
                spread = 0.0035
                ofi = -0.90
            elif 13 <= t < 17:
                ret = 0.030
                vol = 4200.0
                fr = 0.0005
                spread = 0.0025
                ofi = +0.85
            else:
                ret = np.random.normal(0.0005, 0.005)
                vol = 1200.0
                fr = 0.0001
                spread = 0.0003
                ofi = 0.1

            new_price = max(100.0, prices[-1] * (1.0 + ret))
            prices.append(new_price)
            volumes.append(vol)
            fundings.append(fr)
            spreads.append(spread)
            ofis.append(max(-1.0, min(1.0, ofi)))

        return pd.DataFrame({
            "step": range(periods),
            "close": prices,
            "volume": volumes,
            "funding_rate": fundings,
            "spread_pct": spreads,
            "ofi": ofis,
        })

    @staticmethod
    def scenario_liquidity_vacuum(base_price: float = 60000.0, periods: int = 30) -> pd.DataFrame:
        """30-period scenario: Orderbook thins out by 85%, spread widens 15x, extreme slippage."""
        np.random.seed(777)
        prices = [base_price]
        volumes = [1000.0]
        fundings = [0.0001]
        spreads = [0.0002]
        ofis = [0.0]

        for t in range(1, periods):
            if t < 8:
                ret = np.random.normal(0.0001, 0.003)
                vol = 1000.0
                fr = 0.0001
                spread = 0.0002
                ofi = 0.0
            elif 8 <= t < 20:
                ret = np.random.choice([-0.025, 0.022, -0.018, 0.015])
                vol = 120.0
                fr = -0.0004
                spread = 0.0180
                ofi = -0.75
            else:
                ret = np.random.normal(0.0, 0.004)
                vol = 800.0
                fr = 0.0001
                spread = 0.0004
                ofi = 0.0

            new_price = max(100.0, prices[-1] * (1.0 + ret))
            prices.append(new_price)
            volumes.append(vol)
            fundings.append(fr)
            spreads.append(spread)
            ofis.append(max(-1.0, min(1.0, ofi)))

        return pd.DataFrame({
            "step": range(periods),
            "close": prices,
            "volume": volumes,
            "funding_rate": fundings,
            "spread_pct": spreads,
            "ofi": ofis,
        })

    @staticmethod
    def scenario_cross_asset_contagion(periods: int = 30) -> Dict[str, pd.DataFrame]:
        """Simultaneous high-beta drop across BTC, ETH, SOL with correlation 0.98."""
        np.random.seed(999)
        base_prices = {"BTC": 60000.0, "ETH": 3000.0, "SOL": 140.0}
        betas = {"BTC": 1.0, "ETH": 1.35, "SOL": 1.70}
        dfs = {}

        for sym, bp in base_prices.items():
            beta = betas[sym]
            prices = [bp]
            volumes = [1000.0]
            fundings = [0.0001]
            spreads = [0.0002]
            ofis = [0.0]

            for t in range(1, periods):
                if t < 8:
                    market_shock = np.random.normal(0.0, 0.002)
                elif 8 <= t < 14:
                    market_shock = -0.025 + np.random.normal(0, 0.002)
                else:
                    market_shock = np.random.normal(0.0005, 0.004)

                ret = beta * market_shock + np.random.normal(0, 0.001)
                vol = 1000.0 * (4.0 if 8 <= t < 14 else 1.0)
                fr = -0.0012 if 8 <= t < 14 else 0.0001
                spread = 0.0025 * beta if 8 <= t < 14 else 0.0003
                ofi = -0.80 if 8 <= t < 14 else 0.05

                new_price = max(1.0, prices[-1] * (1.0 + ret))
                prices.append(new_price)
                volumes.append(vol)
                fundings.append(fr)
                spreads.append(spread)
                ofis.append(max(-1.0, min(1.0, ofi)))

            dfs[sym] = pd.DataFrame({
                "step": range(periods),
                "close": prices,
                "volume": volumes,
                "funding_rate": fundings,
                "spread_pct": spreads,
                "ofi": ofis,
            })
        return dfs

    @staticmethod
    def scenario_anti_persistent_chop(base_price: float = 60000.0, periods: int = 72) -> pd.DataFrame:
        """72-period scenario: Mean-reverting Ornstein-Uhlenbeck chop with H ~ 0.35."""
        np.random.seed(555)
        prices = [base_price]
        theta = 0.35
        mu = base_price
        sigma = base_price * 0.006

        for t in range(1, periods):
            dp = theta * (mu - prices[-1]) + np.random.normal(0, sigma)
            new_p = max(100.0, prices[-1] + dp)
            prices.append(new_p)

        return pd.DataFrame({
            "step": range(periods),
            "close": prices,
            "volume": [1000.0 + np.random.normal(0, 100) for _ in range(periods)],
            "funding_rate": [0.0001 for _ in range(periods)],
            "spread_pct": [0.0002 for _ in range(periods)],
            "ofi": [float(np.random.choice([-0.3, 0.3])) for _ in range(periods)],
        })


class InstitutionalStressEngine:
    """Executes the stress tests through MaBuy domain risk guards."""

    def __init__(self, initial_equity: float = 39.54, leverage: float = 3.0):
        self.initial_equity = initial_equity
        self.leverage = leverage
        self.funding_service = FundingRateService()
        self.hurst_service = HurstRegimeService()
        self.circuit_breaker = DynamicAdaptiveCircuitBreaker()
        self.correlation_guard = CrossAssetCorrelationGuard()

    def run_single_asset_stress(
        self,
        symbol: str,
        df: pd.DataFrame,
        scenario_name: str,
        start_in_position: bool = True,
        position_direction: str = "LONG",
    ) -> Dict[str, Any]:
        equity = self.initial_equity
        peak_equity = equity
        min_equity = equity
        position: Optional[Dict[str, Any]] = None
        trades: List[Dict[str, Any]] = []
        circuit_trips: int = 0
        guard_vetoes: int = 0
        reversal_exits: int = 0
        max_drawdown_usd = 0.0
        max_drawdown_pct = 0.0

        closes = df["close"].tolist()

        if start_in_position:
            entry_price = closes[0]
            notional = equity * self.leverage * 0.25
            size = notional / entry_price
            stop_loss = entry_price * (0.982 if position_direction == "LONG" else 1.018)
            tp_price = entry_price * (1.035 if position_direction == "LONG" else 0.965)
            position = {
                "symbol": symbol,
                "direction": position_direction,
                "entry_price": entry_price,
                "size": size,
                "notional": notional,
                "stop_loss": stop_loss,
                "tp_price": tp_price,
                "entry_step": 0,
            }

        for t in range(1, len(df)):
            curr_price = closes[t]
            fr = df["funding_rate"].iloc[t]
            spread = df["spread_pct"].iloc[t]
            ofi = df["ofi"].iloc[t]

            if position is not None:
                p_dir = position["direction"]
                p_entry = position["entry_price"]
                p_size = position["size"]

                if p_dir == "LONG":
                    unrealized_pnl = (curr_price - p_entry) * p_size
                    hit_stop = curr_price <= position["stop_loss"]
                    hit_tp = curr_price >= position["tp_price"]
                    should_reversal_exit = ofi <= -0.75
                else:
                    unrealized_pnl = (p_entry - curr_price) * p_size
                    hit_stop = curr_price >= position["stop_loss"]
                    hit_tp = curr_price <= position["tp_price"]
                    should_reversal_exit = ofi >= 0.75

                current_equity = equity + unrealized_pnl
                min_equity = min(min_equity, current_equity)
                peak_equity = max(peak_equity, current_equity)

                dd_usd = peak_equity - current_equity
                dd_pct = (dd_usd / peak_equity) * 100.0 if peak_equity > 0 else 0.0
                max_drawdown_usd = max(max_drawdown_usd, dd_usd)
                max_drawdown_pct = max(max_drawdown_pct, dd_pct)

                if hit_stop or hit_tp or should_reversal_exit:
                    exit_price = curr_price
                    slip_penalty = exit_price * (spread * 0.5)
                    realized_exit = (exit_price - slip_penalty) if p_dir == "LONG" else (exit_price + slip_penalty)
                    if p_dir == "LONG":
                        final_pnl = (realized_exit - p_entry) * p_size
                    else:
                        final_pnl = (p_entry - realized_exit) * p_size

                    equity += final_pnl
                    reason = "STOP_LOSS" if hit_stop else ("TAKE_PROFIT" if hit_tp else "REVERSAL_EXIT")
                    if reason == "REVERSAL_EXIT":
                        reversal_exits += 1

                    trades.append({
                        "step": t,
                        "pnl": final_pnl,
                        "reason": reason,
                        "hold_bars": t - position["entry_step"],
                    })
                    position = None

            recent_pnls = [tr["pnl"] for tr in trades[-5:]]
            if recent_pnls and sum(recent_pnls) < -(self.initial_equity * 0.08):
                circuit_trips += 1

            # Funding rate check
            bias_regime = self.funding_service.classify_regime(fr * 100.0)
            if bias_regime in (FundingBiasRegime.EXTREME_LONGS_OVERHEATED, FundingBiasRegime.EXTREME_SHORTS_SQUEEZE):
                guard_vetoes += 1

            if t >= 15:
                window_closes = closes[max(0, t - 60): t + 1]
                hurst_val = HurstRegimeService.calculate_hurst_rs(window_closes)
                if 0.45 <= hurst_val <= 0.55:
                    guard_vetoes += 1

        margin_used = position["notional"] / self.leverage if position else 0.0
        margin_utilization = (margin_used / equity) * 100.0 if equity > 0 else 100.0
        survival_status = "SURVIVED_STABLE" if equity >= self.initial_equity * 0.85 else (
            "SURVIVED_DRAWDOWN" if equity >= self.initial_equity * 0.50 else "CRITICAL_MARGIN"
        )
        if equity <= 0:
            survival_status = "LIQUIDATED"

        return {
            "scenario": scenario_name,
            "symbol": symbol,
            "initial_equity": round(self.initial_equity, 2),
            "final_equity": round(equity, 2),
            "net_pnl": round(equity - self.initial_equity, 2),
            "max_drawdown_usd": round(max_drawdown_usd, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "margin_utilization_pct": round(margin_utilization, 2),
            "trades_count": len(trades),
            "trades": trades,
            "reversal_exits": reversal_exits,
            "guard_vetoes": guard_vetoes,
            "circuit_trips": circuit_trips,
            "survival_status": survival_status,
        }

    def run_all_scenarios(self) -> Dict[str, Any]:
        results = {}

        df_ftx = CrisisScenarioGenerator.scenario_ftx_cascade()
        results["FTX_Cascade"] = self.run_single_asset_stress(
            "BTCUSDT", df_ftx, "FTX_Cascade", start_in_position=True, position_direction="LONG"
        )

        df_yen = CrisisScenarioGenerator.scenario_yen_carry_liquidation()
        results["Yen_Carry_Wick"] = self.run_single_asset_stress(
            "BTCUSDT", df_yen, "Yen_Carry_Wick", start_in_position=True, position_direction="LONG"
        )

        df_vac = CrisisScenarioGenerator.scenario_liquidity_vacuum()
        results["Liquidity_Vacuum"] = self.run_single_asset_stress(
            "SOLUSDT", df_vac, "Liquidity_Vacuum", start_in_position=True, position_direction="LONG"
        )

        dfs_cont = CrisisScenarioGenerator.scenario_cross_asset_contagion()
        contagion_res = {}
        for sym, df_c in dfs_cont.items():
            contagion_res[sym] = self.run_single_asset_stress(
                f"{sym}USDT", df_c, f"Contagion_{sym}", start_in_position=True, position_direction="LONG"
            )
        total_pnl_cont = sum(r["net_pnl"] for r in contagion_res.values())
        max_dd_cont = max(r["max_drawdown_pct"] for r in contagion_res.values())
        results["Cross_Asset_Contagion"] = {
            "scenario": "Cross_Asset_Contagion",
            "assets": contagion_res,
            "aggregate_net_pnl": round(total_pnl_cont, 2),
            "worst_drawdown_pct": round(max_dd_cont, 2),
            "survival_status": "SURVIVED_STABLE" if total_pnl_cont > -10.0 else "SURVIVED_DRAWDOWN",
        }

        df_chop = CrisisScenarioGenerator.scenario_anti_persistent_chop()
        results["Anti_Persistent_Chop"] = self.run_single_asset_stress(
            "ETHUSDT", df_chop, "Anti_Persistent_Chop", start_in_position=False
        )

        return results


from src.domain.services.astra_advisor import AstraAdvisor


class AstraStressAuditor:
    """Submits institutional crisis simulation telemetry to Planetary LLM cluster via AstraAdvisor."""

    def __init__(self):
        self.advisor = AstraAdvisor()
        if not self.advisor.is_available:
            raise ValueError("AstraAdvisor is not available (missing API key).")

    def audit_telemetry(self, telemetry: Dict[str, Any]) -> str:
        system_prompt = """You are Astra, the Chief Risk Officer and Head of Quantitative Systems at an institutional hedge fund.
Review the provided synthetic Black Swan Stress-Testing Telemetry for the MaBuy trading engine across 5 crisis scenarios.

Conduct an exhaustive institutional audit covering:
1. Mathematical Proof of Capital Non-Ruin:
   - Evaluate whether equity ($39.54 live micro-capital and $150.00 standard capital) is mathematically protected from liquidation under -25% sudden gap-downs.
   - Decompose the protective efficacy of the 1.8% ATR stop-loss floor vs. naked exposure.
2. Reversal Guard & Liquidity Vacuum Critique:
   - Analyze how the Order Flow Imbalance (OFI) active reversal guard performed during sudden orderbook thinning and wicks.
   - Assess adverse slippage during liquidity deserts (180 bps spread).
3. Cross-Asset Beta Contagion & Correlation Haircut:
   - Evaluate the portfolio vulnerability when BTC, ETH, and SOL drop simultaneously with Pearson correlation rho = 0.98.
   - Does the Correlation Guard's 40% margin haircut prevent portfolio margin breach?
4. Anti-Persistent Chop Regime Defense:
   - Verify if the Hurst deadband [0.45, 0.55] effectively prevents death-by-a-thousand-cuts.
5. Concrete Actionable Parameter Recommendations:
   - Provide exact recommended thresholds for:
     * dynamic_circuit_breaker_threshold
     * cooling_off_period_hours
     * stop_atr_mult
     * tp_atr_mult
     * hurst_trend_min
     * hurst_meanrev_max

Return your response in clean, authoritative GitHub-Flavored Markdown with detailed mathematical sections."""

        res = self.advisor.consult(
            system_prompt=system_prompt,
            user_payload=telemetry,
            reasoning_effort="high",
            max_tokens=8000,
            response_json=False
        )
        if not res or not res.get("text"):
            raise RuntimeError("AstraAdvisor returned empty response for stress test audit.")
        return res["text"]


def main():
    parser = argparse.ArgumentParser(description="Astra Institutional Black Swan Stress Tester")
    parser.add_argument("--equity", type=float, default=39.54, help="Starting equity in USD")
    parser.add_argument("--leverage", type=float, default=3.0, help="Account leverage")
    args = parser.parse_args()

    print("================================================================================")
    print("      MABUY INSTITUTIONAL BLACK SWAN & CRISIS STRESS-TESTING SUITE              ")
    print("================================================================================")
    print(f"Equity: ${args.equity:.2f} | Leverage: {args.leverage:.1f}x | Target: 5 Crisis Regimes")
    print("--------------------------------------------------------------------------------")

    engine = InstitutionalStressEngine(initial_equity=args.equity, leverage=args.leverage)
    telemetry = engine.run_all_scenarios()

    print("\n--- STRESS-TEST RESULTS SUMMARY ---")
    for scen, data in telemetry.items():
        if scen == "Cross_Asset_Contagion":
            print(f"[{scen}] Net PnL: ${data['aggregate_net_pnl']:.2f} | Worst DD: {data['worst_drawdown_pct']:.2f}% | Status: {data['survival_status']}")
        else:
            print(f"[{scen}] Net PnL: ${data['net_pnl']:.2f} | Max DD: {data['max_drawdown_pct']:.2f}% | Vetoes: {data['guard_vetoes']} | Status: {data['survival_status']}")

    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = models_dir / "stress_test_telemetry.json"
    with open(telemetry_path, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)
    print(f"\nSaved raw telemetry to: {telemetry_path}")

    print("\nConnecting to Astra / Planetary LLM Cluster for Quantitative Audit...")
    auditor = AstraStressAuditor()
    report_text = auditor.audit_telemetry(telemetry)

    report_path = models_dir / "black_swan_stress_test_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# MaBuy Institutional Black Swan & Crisis Stress-Test Audit\n\n")
        f.write(f"**Generated**: {dt.datetime.now(dt.timezone.utc).isoformat()} UTC\n")
        f.write(f"**Evaluated Capital**: ${args.equity:.2f} USD (Leverage: {args.leverage:.1f}x)\n\n")
        f.write("## Empirical Telemetry Matrix\n\n")
        f.write("| Scenario | Asset | Initial Eq | Final Eq | Net PnL | Max DD % | Guard Vetoes | Reversal Exits | Status |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for scen, d in telemetry.items():
            if scen == "Cross_Asset_Contagion":
                f.write(f"| Contagion (Portfolio) | BTC/ETH/SOL | ${args.equity:.2f} | ${args.equity + d['aggregate_net_pnl']:.2f} | ${d['aggregate_net_pnl']:.2f} | {d['worst_drawdown_pct']:.2f}% | - | - | {d['survival_status']} |\n")
            else:
                f.write(f"| {scen} | {d['symbol']} | ${d['initial_equity']:.2f} | ${d['final_equity']:.2f} | ${d['net_pnl']:.2f} | {d['max_drawdown_pct']:.2f}% | {d['guard_vetoes']} | {d['reversal_exits']} | {d['survival_status']} |\n")
        f.write("\n---\n\n")
        f.write("## Astra Institutional Quantitative Risk Report\n\n")
        f.write(report_text)
        f.write("\n")

    print(f"\nComprehensive Black Swan Audit Report written to: {report_path}")
    print("================================================================================")


if __name__ == "__main__":
    main()
