#!/usr/bin/env python3
"""
Institutional Statistical Readiness Auditor for MaBuy V5.2.
Audits trade logs from SQLite to determine if sample size is sufficient for Astra's full re-audit.
Standards: Renaissance Technologies, Citadel, Two Sigma.
"""

import os
import sys
import math
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Any, List

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DB_PATH = "/root/mabuy2/data/mabuy_prod.db"
if not os.path.exists(DB_PATH):
    # Local fallback for development
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DB_PATH = os.path.join(proj_root, "data", "mabuy_prod.db")


def run_statistical_audit(db_path: str = DB_PATH) -> Dict[str, Any]:
    if not os.path.exists(db_path):
        return {"error": f"Database not found at {db_path}"}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Query closed trades from signal_log (new architecture)
    cur.execute("""
        SELECT symbol, direction, entry_price, exit_price, pnl_usd, close_reason, signal_time, status
        FROM signal_log
        WHERE status = 'CLOSED' AND pnl_usd IS NOT NULL
        ORDER BY id ASC
    """)
    sig_rows = cur.fetchall()

    # Query closed trades from paper_trades (virtual execution)
    cur.execute("""
        SELECT symbol, direction, entry_price, exit_price, pnl_usd, opened_at, status
        FROM paper_trades
        WHERE status = 'CLOSED' AND pnl_usd IS NOT NULL
        ORDER BY trade_id ASC
    """)
    paper_rows = cur.fetchall()

    conn.close()

    def analyze_dataset(rows: List[sqlite3.Row], name: str) -> Dict[str, Any]:
        n = len(rows)
        if n == 0:
            return {
                "name": name,
                "count": 0,
                "win_count": 0,
                "loss_count": 0,
                "win_rate": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "net_pnl": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "payoff_ratio": 0.0,
                "max_dd_usd": 0.0,
                "readiness_label": "ОЖИДАНИЕ ПЕРВЫХ СДЕЛОК (0/50)",
                "readiness_color": "⚪",
                "verdict": "Новая архитектура v5.2 запущена. Идет набор первых сделок.",
            }

        pnls = [float(r["pnl_usd"] or 0.0) for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / n) * 100.0 if n > 0 else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        net_pnl = sum(pnls)

        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        avg_win = (gross_profit / win_count) if win_count > 0 else 0.0
        avg_loss = (gross_loss / loss_count) if loss_count > 0 else 0.0
        payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

        # Maximum Drawdown calculation on cumulative equity
        cum_equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cum_equity += p
            if cum_equity > peak:
                peak = cum_equity
            dd = peak - cum_equity
            if dd > max_dd:
                max_dd = dd

        # Sample Readiness Classification (Astra Benchmark)
        if n < 25:
            readiness_label = f"НАКОПЛЕНИЕ ВЫБОРКИ ({n}/50 сделок — {n*2}%)"
            readiness_color = "🟡"
            verdict = "Продолжать автономную работу. Рано делать выводы."
        elif 25 <= n < 50:
            readiness_label = f"ПЕРВИЧНЫЙ КАНАРЕЕЧНЫЙ СРЕЗ ({n}/50 сделок — {n*2}%)"
            readiness_color = "🟠"
            verdict = "Выборка достаточна для промежуточной оценки тренда."
        else:
            readiness_label = f"ПОЛНЫЙ ИНСТИТУЦИОНАЛЬНЫЙ АУДИТ ГОТОВ ({n} сделок >= 50)"
            readiness_color = "🟢"
            verdict = "СТАТИСТИЧЕСКИ ЗНАЧИМО! Пора активировать ключ Astra для ре-аудита."

        return {
            "name": name,
            "count": n,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": win_rate,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "net_pnl": net_pnl,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "payoff_ratio": payoff_ratio,
            "max_dd_usd": max_dd,
            "readiness_label": readiness_label,
            "readiness_color": readiness_color,
            "verdict": verdict,
        }

    return {
        "signal_log": analyze_dataset(sig_rows, "Live Signal Logger (Архитектура v5.2)"),
        "paper_trades": analyze_dataset(paper_rows, "Paper Trades Execution Engine"),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def print_audit_report():
    report = run_statistical_audit()
    if "error" in report:
        print(f"❌ Ошибка аудита: {report['error']}")
        return

    print("=" * 75)
    print("🏛 ИНСТИТУЦИОНАЛЬНЫЙ СТАТИСТИЧЕСКИЙ АУДИТОР MaBuy V5.2 (СТАНДАРТ ASTRA)")
    print(f"⏱ Время среза: {report['timestamp_utc']}")
    print("=" * 75)

    for key in ["signal_log", "paper_trades"]:
        ds = report[key]
        print(f"\n📊 КОНТУР: {ds['name']}")
        print(f"   • Статус выборки: {ds.get('readiness_color', '')} {ds.get('readiness_label', '')}")
        print(f"   • Закрытых сделок: {ds['count']} (Побед: {ds.get('win_count', 0)}, Поражений: {ds.get('loss_count', 0)})")
        print(f"   • Win Rate: {ds['win_rate']:.1f}%")
        print(f"   • Profit Factor: {ds['profit_factor']:.2f}")
        print(f"   • Чистый PnL: ${ds['net_pnl']:+.2f} (Прибыль: +${ds.get('gross_profit', 0):.2f}, Убыток: -${ds.get('gross_loss', 0):.2f})")
        print(f"   • Коэффициент выигрыша (Payoff Ratio): {ds.get('payoff_ratio', 0):.2f} (Ср. победа: ${ds.get('avg_win', 0):.2f} / Ср. убыток: ${ds.get('avg_loss', 0):.2f})")
        print(f"   • Максимальная просадка: ${ds.get('max_dd_usd', 0):.2f}")
        print(f"   📢 ВЕРДИКТ ASTRA: {ds.get('verdict', '')}")
        print("-" * 75)

    print("\n💡 ПРАВИЛО СОЗДАТЕЛЯ:")
    print("   1. До 50 сделок — НЕ трогать руками код, пороги и активные позиции.")
    print("   2. При достижении 50 сделок — купить API-ключ (1000 тенге) и запросить у Astra полный ре-аудит.")
    print("=" * 75)


if __name__ == "__main__":
    print_audit_report()
