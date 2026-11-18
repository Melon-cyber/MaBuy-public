#!/usr/bin/env python3
"""
CLI Tool: Consult GPT-6 Astra Quantitative Advisor
Runs a deep institutional quantitative analysis using maximum reasoning effort.
"""

import os
import sys
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

from src.domain.services.astra_advisor import AstraAdvisor
from src.infrastructure.persistence.sqlite_tracker import SQLiteTracker
from src.infrastructure.execution.binance_trader import BinanceAutoTrader

def main():
    print("🧠 Initializing GPT-6 Astra Quantitative Audit (High Reasoning Mode)...")
    advisor = AstraAdvisor()
    if not advisor.is_available:
        print("❌ Error: OPENAI_API_KEY is not set or OpenAI client failed to initialize.")
        sys.exit(1)

    db_path = os.path.join(BASE_DIR, "data", "mabuy_prod.db")
    db = SQLiteTracker(db_path=db_path)

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    equity = 40.0
    active_positions = []
    if api_key and api_secret:
        try:
            trader = BinanceAutoTrader(api_key=api_key, api_secret=api_secret)
            session = trader._create_isolated_session()
            bal = session.fetch_balance()
            equity = float(bal.get("total", {}).get("USDT", 40.0))
            raw_pos = session.fetch_positions()
            for p in raw_pos:
                amt = float(p.get("info", {}).get("positionAmt", 0) or p.get("contracts", 0) * (1 if p.get("side") == "long" else -1))
                if abs(amt) > 0:
                    active_positions.append({
                        "symbol": p.get("symbol"),
                        "side": "LONG" if amt > 0 else "SHORT",
                        "contracts": amt,
                        "entryPrice": p.get("entryPrice"),
                        "markPrice": p.get("markPrice"),
                        "unrealizedProfit": p.get("unrealizedProfit")
                    })
        except Exception as e:
            print(f"Note on balance fetch: {e}")

    # Fetch recent paper trades
    recent_trades = []
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT symbol, direction, entry_price, exit_price, pnl_usd, status, opened_at, closed_at FROM paper_trades ORDER BY opened_at DESC LIMIT 6")
        for r in cur.fetchall():
            recent_trades.append({
                "symbol": r[0], "direction": r[1], "entry": r[2], "exit": r[3],
                "pnl_usd": r[4], "status": r[5], "opened_at": r[6], "closed_at": r[7]
            })
        conn.close()
    except Exception as e:
        print(f"Note on db trades: {e}")

    print(f"📊 Sending Portfolio State to Astra: Equity=${equity:.2f}, Active Positions={len(active_positions)}, Recent Trades={len(recent_trades)}")
    
    result = advisor.audit_portfolio_health(
        equity=equity,
        active_positions=active_positions,
        recent_trades=recent_trades,
        rolling_24h_pnl=0.0
    )

    if result:
        print("\n" + "="*70)
        print("🏛 GPT-6 ASTRA QUANTITATIVE AUDIT REPORT")
        print("="*70)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        report_path = os.path.join(BASE_DIR, "reports", "astra_latest_audit.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Saved audit report to: {report_path}")
    else:
        print("❌ Consultation failed.")

if __name__ == "__main__":
    main()
