#!/usr/bin/env python3
"""
Institutional Backtest Pipeline Runner
Executes the Marcos López de Prado Institutional Backtesting Engine across
real historical Binance 1h OHLCV data for BTC, ETH, and SOL.
Generates DSR, PSR, MinTRL, and invokes AstraAdvisor for deep forensic evaluation.
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

import numpy as np
import pandas as pd
import joblib

from src.infrastructure.data_fetchers.binance_fetcher import CryptoDataFetcher
from src.infrastructure.mlops.weekly_retrain_pipeline import FeatureEngineer
from src.domain.services.institutional_backtester import InstitutionalBacktester, BacktestConfig
from src.domain.services.astra_advisor import AstraAdvisor


def generate_backtest_signals(df: pd.DataFrame, symbol: str) -> pd.Series:
    """Generate directional trading signals using available models or trend-momentum fallback."""
    features_df, _ = FeatureEngineer.compute_features(df)
    
    models_dir = os.path.join(BASE_DIR, "models")
    candidates = [
        f"gbdt_{symbol}USDT_1h.joblib",
        f"gbdt_{symbol}_1h.joblib",
        f"baseline_gbdt_{symbol}USDT_1h.joblib",
        "gbdt_model.pkl"
    ]
    model = None
    for cand in candidates:
        cand_path = os.path.join(models_dir, cand)
        if os.path.exists(cand_path):
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
                index=features_df["timestamp"]
            )
        except Exception as e:
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


def run_single_asset_backtest(
    symbol: str,
    limit: int = 1000,
    initial_capital: float = 1000.0,
    leverage: float = 3.0
) -> Dict[str, Any]:
    """Execute complete institutional event-driven backtest for a single symbol."""
    print(f"🔄 Fetching real Binance 1h candles for {symbol} (limit={limit})...")
    fetcher = CryptoDataFetcher()
    df = fetcher.fetch_ohlcv(symbol, timeframe="1h", limit=limit)
    
    if df.empty or len(df) < 50:
        raise ValueError(f"Insufficient candle history returned for {symbol}")
        
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    
    signals = generate_backtest_signals(df.reset_index(), symbol)
    signals.index = pd.to_datetime(signals.index, utc=True)
    
    common_idx = df.index.intersection(signals.index)
    df = df.loc[common_idx]
    signals = signals.loc[common_idx]
    
    config = BacktestConfig(
        initial_capital=initial_capital,
        leverage=leverage,
        allocation_fraction=0.25,
        maintenance_margin_rate=0.005,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
        base_slippage_pct=0.0001,
        maker_fill_kappa=1.0,
        random_seed=42
    )
    
    engine = InstitutionalBacktester(config)
    audit = engine.run_backtest(
        df=df,
        signals=signals,
        stop_atr_mult=2.0,
        tp_atr_mult=3.2,
        maker_execution=True
    )
    
    report_text = engine.generate_report()
    return {
        "symbol": symbol,
        "candles": len(df),
        "audit": audit,
        "report_text": report_text,
        "start_date": str(df.index[0]),
        "end_date": str(df.index[-1])
    }


def main():
    print("=" * 70)
    print("🏛️ MA BUY INSTITUTIONAL EVENT-DRIVEN BACKTEST SUITE (DEFLATED SHARPE)")
    print("=" * 70)
    
    symbols = ["BTC", "ETH", "SOL"]
    results = {}
    
    for sym in symbols:
        try:
            res = run_single_asset_backtest(sym, limit=1000, initial_capital=1000.0, leverage=3.0)
            results[sym] = res
            print(f"\n{res['report_text']}\n")
        except Exception as e:
            print(f"❌ Backtest failed for {sym}: {e}")

    summary_telemetry = {}
    for sym, res in results.items():
        audit = res["audit"]
        m = audit.metrics
        summary_telemetry[sym] = {
            "candle_count": res["candles"],
            "period": f"{res['start_date']} to {res['end_date']}",
            "net_pnl_usd": m.get("net_realized_pnl", 0.0),
            "total_return_pct": m.get("total_return_pct", 0.0),
            "profit_factor": m.get("profit_factor", 0.0),
            "payoff_ratio": m.get("payoff_ratio", 0.0),
            "win_rate": m.get("win_rate", 0.0),
            "trade_count": m.get("trade_count", 0),
            "annualized_net_sharpe": m.get("annualized_net_sharpe", 0.0),
            "annualized_sortino": m.get("annualized_sortino", 0.0),
            "max_drawdown_pct": m.get("max_drawdown_pct", 0.0),
            "calmar_ratio": m.get("calmar_ratio", 0.0),
            "psr": m.get("psr", 0.0),
            "dsr": m.get("dsr", 0.0)
        }

    print("\n🔮 Consulting Astra Quantitative Reasoning Advisor on Real Backtest Metrics...")
    advisor = AstraAdvisor()
    advisory = None
    if advisor.is_available:
        system_prompt = """You are Astra, Principal Quantitative Systems Auditor at an institutional hedge fund.
Review the provided real Binance 1h event-driven backtest results across BTC, ETH, SOL.
Focus on:
1. Marcos Lopez de Prado Deflated Sharpe Ratio (DSR) and Probabilistic Sharpe Ratio (PSR) significance.
2. Payoff Ratio R vs Win Rate tradeoff under 0.02% Maker / 0.05% Taker friction.
3. Micro-capital survivability ($39.54 scaling to $200-$500).
4. Concrete hyperparameter recommendations for:
   - stop_atr_mult
   - tp_atr_mult
   - Hurst regime bounds
   - Dynamic Trailing Stop parameters.
Return a structured JSON object adhering strictly to RFC 8259 with keys:
- executive_summary (str)
- dsr_significance_assessment (str)
- micro_capital_recommendations (list of str)
- recommended_hyperparameters (dict)
- institutional_score (float, 0-100)"""

        advisory = advisor.consult(
            system_prompt=system_prompt,
            user_payload=summary_telemetry,
            max_tokens=4000,
            response_json=True
        )
        print("\n✅ Astra Quantitative Consultation Complete!")
        print(json.dumps(advisory, indent=2, ensure_ascii=False))

    report_file = os.path.join(BASE_DIR, "models", "institutional_backtest_report.md")
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Institutional Backtest & Quantitative Performance Audit\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        f.write(f"**Data Provider:** Binance USD-M Futures (1h OHLCV)\n")
        f.write(f"**Execution Model:** Maker-First (0.02% Maker, 0.05% Taker, Dynamic Slippage)\n\n")
        
        f.write("## 1. Asset Performance Tables\n\n")
        for sym, res in results.items():
            f.write(f"### {sym} Performance Audit\n\n```text\n{res['report_text']}\n```\n\n")

        if advisory:
            f.write("## 2. Astra Quantitative Advisor Evaluation\n\n")
            f.write(f"**Institutional Score:** {advisory.get('institutional_score', 'N/A')}/100\n\n")
            f.write(f"### Executive Summary\n\n{advisory.get('executive_summary', '')}\n\n")
            f.write(f"### Deflated Sharpe Ratio (DSR) Assessment\n\n{advisory.get('dsr_significance_assessment', '')}\n\n")
            f.write("### Micro-Capital Directives\n\n")
            for rec in advisory.get("micro_capital_recommendations", []):
                f.write(f"- {rec}\n")
            f.write("\n### Recommended Parameter Adjustments\n\n```json\n")
            f.write(json.dumps(advisory.get("recommended_hyperparameters", {}), indent=2))
            f.write("\n```\n")

    print(f"\n📄 Complete Institutional Report saved to: {report_file}")


if __name__ == "__main__":
    main()
