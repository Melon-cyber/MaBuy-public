# -*- coding: utf-8 -*-
import os
import sys
import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 65)
print("📊 MABUY V5 LIVE ACCOUNT & TRADING AUDIT")
print("=" * 65)

db_path = "data/mabuy_prod.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# 1. Inspect User Account (100000001)
user_id = "100000001"
print(f"\n👤 [YOUR ACCOUNT: {user_id}]")

# Subscription tier
sub = cur.execute("SELECT tier, expires_at FROM user_subscriptions WHERE user_id = ?", (user_id,)).fetchone()
tier_val = sub[0] if sub else "FREE"
exp_val = sub[1] if sub else "N/A"
print(f"   • Subscription Tier: {tier_val} (Expires: {exp_val})")

# Trading deposit
dep = cur.execute("SELECT deposit_usd, updated_at FROM user_trading_deposits WHERE user_id = ?", (user_id,)).fetchone()
if dep and dep[0] is not None:
    print(f"   • Real Configured Balance: ${dep[0]:.2f} (Updated: {dep[1]})")
else:
    print("   • Real Configured Balance: [НЕ ЗАДАН] (Используйте /deposit <сумма> в Telegram для закрепления)")

# Active open paper positions
open_pos = cur.execute("""
    SELECT trade_id, symbol, direction, entry_price, amount_usd, opened_at, max_pnl_usd 
    FROM paper_trades 
    WHERE user_id = ? AND status = 'OPEN'
""", (user_id,)).fetchall()
print(f"   • Active Open Positions: {len(open_pos)}")
if open_pos:
    for p in open_pos:
        tid, sym, side, entry, amt, opened, max_pnl = p
        max_pnl_str = f"{max_pnl:+.2f}" if max_pnl is not None else "+0.00"
        print(f"     -> #{sym} {side} | Entry: ${entry:,.2f} | Size: ${amt:.2f} | Opened: {opened} | Max PnL: ${max_pnl_str}")
else:
    print("     (0 active positions right now — scanning Binance for high-probability setups >= 70%)")

# Recent closed trades
closed_pos = cur.execute("""
    SELECT trade_id, symbol, direction, entry_price, exit_price, amount_usd, pnl_usd, opened_at, closed_at 
    FROM paper_trades 
    WHERE user_id = ? AND status = 'CLOSED'
    ORDER BY opened_at DESC LIMIT 5
""", (user_id,)).fetchall()
print(f"\n   • Recent Closed Trades (Your Account):")
if closed_pos:
    for cp in closed_pos:
        tid, sym, side, entry, exit_p, amt, pnl, op_dt, cl_dt = cp
        exit_val = exit_p if exit_p is not None else 0.0
        pnl_val = pnl if pnl is not None else 0.0
        print(f"     ✓ #{sym} {side} | In: ${entry:,.2f} -> Out: ${exit_val:,.2f} | Size: ${amt:.2f} | PnL: ${pnl_val:+.2f} | {cl_dt or op_dt}")
else:
    print("     (No closed trades for this specific ID yet)")

# 2. System-wide V5 Signals & Trades
print("\n" + "-" * 65)
print("🌐 SYSTEM-WIDE V5 DYNAMIC ENGINE & SIGNALS STATUS:")
print("-" * 65)

# Total monitored users
trader_count = cur.execute("SELECT count(*) FROM user_trading_deposits").fetchone()[0]
print(f"   • Total Active Traders in System: {trader_count}")

# Open channel signals
open_sigs = cur.execute("""
    SELECT id, symbol, direction, entry_price, target_price, stop_loss, confidence_score, created_at 
    FROM verified_signals_log 
    WHERE status = 'OPEN' 
    ORDER BY id DESC LIMIT 5
""").fetchall()
print(f"   • Active Verified Signals Monitored: {len(open_sigs)}")
for s in open_sigs:
    sig_id, sym, side, entry, tp, sl, conf, cr_at = s
    print(f"     -> #{sym} {side} (Signal #{sig_id}) | Entry: ${entry:,.2f} | TP: ${tp:,.2f} | SL: ${sl:,.2f} | Conf: {conf:.1f}%")

# Recent closed signals with exit price
closed_sigs = cur.execute("""
    SELECT id, symbol, direction, entry_price, exit_price, confidence_score, closed_at 
    FROM verified_signals_log 
    WHERE status = 'CLOSED' 
    ORDER BY id DESC LIMIT 5
""").fetchall()
print(f"\n   • Recent Closed Verified Signals:")
if closed_sigs:
    for cs in closed_sigs:
        sig_id, sym, side, entry, exit_p, conf, cl_at = cs
        exit_val = exit_p if exit_p is not None else 0.0
        print(f"     ✓ #{sym} {side} (Signal #{sig_id}) | Entry: ${entry:,.2f} -> Exit: ${exit_val:,.2f} | Conf: {conf:.1f}% | Closed: {cl_at}")
else:
    print("     (No closed signals in log)")

conn.close()

print("\n" + "=" * 65)
print("✅ AUDIT COMPLETE")
print("=" * 65)
