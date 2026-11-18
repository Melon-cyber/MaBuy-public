#!/usr/bin/env python3
"""
Astra Institutional Hyperparameter Grid Optimizer with Deflated Sharpe Ratio (DSR).

Executes an exhaustive parameter sweep across real Binance 1h historical data for BTC, ETH, and SOL:
- stop_atr_mult: [1.5, 1.8, 2.2]
- tp_atr_mult: [2.5, 3.2, 4.0]
- allocation_fraction: [0.20, 0.25, 0.30]

Evaluates Marcos López de Prado DSR, PSR, Payoff Ratio R, and Win Rate under realistic Maker fees (0.02%).
Submits the Pareto frontier to Astra (gpt-6-astra / gpt-5.6-sol) to synthesize optimal production settings.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

from src.infrastructure.data_fetchers.binance_fetcher import CryptoDataFetcher
from src.infrastructure.mlops.weekly_retrain_pipeline import FeatureEngineer
from src.domain.services.institutional_backtester import InstitutionalBacktester, BacktestConfig
from src.domain.services.astra_advisor import AstraAdvisor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AstraGridOptimizer")


def generate_backtest_signals(df: pd.DataFrame, symbol: str) -> pd.Series:
    """Generate directional trading signals aligned with feature engineering index."""
    features_df, _ = FeatureEngineer.compute_features(df)
    models_dir = ROOT_DIR / "models"
    candidates = [
        f"gbdt_{symbol}USDT_1h.joblib",
        f"gbdt_{symbol}_1h.joblib",
        f"baseline_gbdt_{symbol}USDT_1h.joblib",
    ]
    model = None
    for cand in candidates:
        cand_path = models_dir / cand
        if cand_path.exists():
            try:
                model = joblib.load(cand_path)
                break
            except Exception:
                continue

    signals = pd.Series(0, index=features_df["timestamp"])
    if model is not None:
        try:
            feat_cols = [c for c in FeatureEngineer.FEATURE_COLS if c in features_df.columns]
            X = features_df[feat_cols].fillna(0.0)
            preds = model.predict(X)
            thresh = 0.0025
            signals = pd.Series(
                np.where(preds > thresh, 1, np.where(preds < -thresh, -1, 0)),
                index=features_df["timestamp"],
            )
        except Exception:
            model = None

    if model is None:
        close = features_df["close"]
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=36, adjust=False).mean()
        rsi = features_df["rsi_14"].fillna(0.5)
        long_cond = (ema_fast > ema_slow) & (rsi > 0.52)
        short_cond = (ema_fast < ema_slow) & (rsi < 0.48)
        sig_arr = np.where(long_cond, 1, np.where(short_cond, -1, 0))
        signals = pd.Series(sig_arr, index=features_df["timestamp"])

    return signals


def run_grid_for_symbol(
    symbol: str,
    df: pd.DataFrame,
    signals: pd.Series,
    stop_mults: List[float],
    tp_mults: List[float],
    allocations: List[float],
) -> List[Dict[str, Any]]:
    """Run parameter sweep for a single symbol."""
    results = []
    total_runs = len(stop_mults) * len(tp_mults) * len(allocations)
    logger.info(f"Running {total_runs} parameter combinations for {symbol}...")

    common_idx = df.index.intersection(signals.index)
    df_aligned = df.loc[common_idx]
    sig_aligned = signals.loc[common_idx]

    for stop_m in stop_mults:
        for tp_m in tp_mults:
            for alloc in allocations:
                config = BacktestConfig(
                    initial_capital=1000.0,
                    leverage=3.0,
                    allocation_fraction=alloc,
                    maintenance_margin_rate=0.005,
                    maker_fee_rate=0.0002,
                    taker_fee_rate=0.0005,
                    base_slippage_pct=0.0001,
                    maker_fill_kappa=1.0,
                    random_seed=42,
                )
                engine = InstitutionalBacktester(config)
                audit = engine.run_backtest(
                    df=df_aligned,
                    signals=sig_aligned,
                    stop_atr_mult=stop_m,
                    tp_atr_mult=tp_m,
                    maker_execution=True,
                )
                m = audit.metrics
                results.append({
                    "symbol": symbol,
                    "stop_atr_mult": stop_m,
                    "tp_atr_mult": tp_m,
                    "allocation": alloc,
                    "target_rr": round(tp_m / stop_m, 2),
                    "trades": m.get("trade_count", 0),
                    "win_rate": round(m.get("win_rate", 0.0), 2),
                    "payoff_ratio": round(m.get("payoff_ratio", 0.0), 2),
                    "net_pnl": round(m.get("net_realized_pnl", 0.0), 2),
                    "return_pct": round(m.get("total_return_pct", 0.0), 2),
                    "max_drawdown_pct": round(m.get("max_drawdown_pct", 0.0), 2),
                    "sharpe": round(m.get("annualized_net_sharpe", 0.0), 2),
                    "psr": round(m.get("psr", 0.0), 4),
                    "dsr": round(m.get("dsr", 0.0), 4),
                })

    # Sort by Sharpe and Payoff Ratio
    results.sort(key=lambda r: (r["sharpe"], r["payoff_ratio"]), reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Astra Hyperparameter Grid Optimizer")
    parser.add_argument("--limit", type=int, default=1000, help="Candle history limit")
    args = parser.parse_args()

    print("================================================================================")
    print("      MABUY INSTITUTIONAL HYPERPARAMETER GRID OPTIMIZER (ASTRA & DSR)           ")
    print("================================================================================")

    fetcher = CryptoDataFetcher()
    symbols = ["BTC", "ETH", "SOL"]
    stop_grid = [1.5, 1.8, 2.2]
    tp_grid = [2.5, 3.2, 4.0]
    alloc_grid = [0.20, 0.25, 0.30]

    all_grid_results: Dict[str, List[Dict[str, Any]]] = {}
    top_configurations: Dict[str, Dict[str, Any]] = {}

    for sym in symbols:
        print(f"\n📥 Fetching {args.limit} 1h candles for {sym}USDT from Binance...")
        df = fetcher.fetch_ohlcv(sym, timeframe="1h", limit=args.limit)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        signals = generate_backtest_signals(df.reset_index(), sym)
        signals.index = pd.to_datetime(signals.index, utc=True)

        res_list = run_grid_for_symbol(
            sym, df, signals, stop_grid, tp_grid, alloc_grid
        )
        all_grid_results[sym] = res_list
        top_configurations[sym] = res_list[0] if res_list else {}

        print(f"\nTop 3 Configurations for {sym}:")
        for i, r in enumerate(res_list[:3], 1):
            print(f"  #{i}: Stop={r['stop_atr_mult']}x ATR, TP={r['tp_atr_mult']}x ATR, Alloc={r['allocation']*100:.0f}% | "
                  f"PnL: ${r['net_pnl']:.2f}, WR: {r['win_rate']}%, R: {r['payoff_ratio']}, DD: {r['max_drawdown_pct']}%, "
                  f"Sharpe: {r['sharpe']}, PSR: {r['psr']:.2f}")

    # Compile synthesis payload for Astra
    synthesis_payload = {
        "evaluation_period": f"{args.limit} hourly bars (real Binance USD-M Futures)",
        "fee_schedule": "Maker 0.02%, Taker 0.05% (VIP-0)",
        "leverage": 3.0,
        "grid_results_summary": {
            sym: {
                "top_1": all_grid_results[sym][0] if all_grid_results[sym] else {},
                "top_2": all_grid_results[sym][1] if len(all_grid_results[sym]) > 1 else {},
                "top_3": all_grid_results[sym][2] if len(all_grid_results[sym]) > 2 else {},
                "worst": all_grid_results[sym][-1] if all_grid_results[sym] else {},
            }
            for sym in symbols
        },
        "top_configurations": top_configurations,
    }

    # Save raw grid results
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    grid_json_path = models_dir / "grid_optimization_results.json"
    with open(grid_json_path, "w", encoding="utf-8") as f:
        json.dump(synthesis_payload, f, indent=2)
    print(f"\nSaved grid optimization telemetry to: {grid_json_path}")

    # Consult Astra
    print("\n🔮 Consulting Astra Quantitative Reasoning Advisor on Multi-Objective Pareto Frontier...")
    advisor = AstraAdvisor()
    if advisor.is_available:
        system_prompt = """You are Astra, Principal Quantitative Systems Architect and Mathematical Optimizer at an institutional hedge fund.
Review the provided multi-parameter grid search telemetry across real Binance 1h data for BTC, ETH, and SOL.

Your tasks:
1. Identify the Pareto-Optimal configuration that maximizes Payoff Ratio R while strictly capping Max Drawdown below 10-12%.
2. Explain the structural difference between BTC (high payoff ratio R > 2.0) and ETH/SOL under Maker-first execution.
3. Formulate the exact mathematical parameters to set in production:
   - recommended_stop_atr_mult
   - recommended_tp_atr_mult
   - recommended_allocation_fraction (per asset: BTC, ETH, SOL)
   - dynamic_trailing_trigger_atr
   - dynamic_trailing_distance_atr
4. Provide risk boundaries for micro-capital ($39.54) scaling to target capital ($150.00 - $500.00).

Respond in clean, authoritative GitHub-Flavored Markdown with LaTeX mathematical formulas where appropriate."""

        res = advisor.consult(
            system_prompt=system_prompt,
            user_payload=synthesis_payload,
            reasoning_effort="high",
            max_tokens=8000,
            response_json=False,
        )

        if res and res.get("text"):
            report_text = res["text"]
            report_path = models_dir / "hyperparameter_optimization_report.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("# MaBuy Institutional Hyperparameter Grid Optimization & Pareto Frontier\n\n")
                f.write(f"**Generated**: {dt.datetime.now(dt.timezone.utc).isoformat()} UTC\n")
                f.write(f"**Data Horizon**: {args.limit} 1h Binance Futures Candles | **Leverage**: 3.0x\n\n")
                f.write("## Top Empirical Grid Configurations\n\n")
                f.write("| Asset | Stop ATR | TP ATR | Target R:R | Alloc % | Win Rate % | Payoff R | Net PnL ($) | Max DD % | Sharpe | PSR |\n")
                f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
                for sym in symbols:
                    top = top_configurations.get(sym, {})
                    f.write(
                        f"| **{sym}** | {top.get('stop_atr_mult')}x | {top.get('tp_atr_mult')}x | {top.get('target_rr')} | "
                        f"{top.get('allocation', 0)*100:.0f}% | {top.get('win_rate')}% | {top.get('payoff_ratio')} | "
                        f"${top.get('net_pnl')} | {top.get('max_drawdown_pct')}% | {top.get('sharpe')} | {top.get('psr')} |\n"
                    )
                f.write("\n---\n\n")
                f.write("## Astra Institutional Optimization Report & Pareto Synthesis\n\n")
                f.write(report_text)
                f.write("\n")

            print(f"\nComprehensive Hyperparameter Report written to: {report_path}")

    print("================================================================================")


if __name__ == "__main__":
    main()
