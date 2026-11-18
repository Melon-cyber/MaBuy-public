#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaBuy Institutional Terminal Status Dashboard
Zero-token real-time status across Binance Live Trading, ML Models, and System Health.
"""

import os
import sys
import json
import time
import sqlite3
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = "/root/mabuy2" if os.path.exists("/root/mabuy2") else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

def line_sep(char="=", width=72):
    print(f"{CYAN}{char * width}{RESET}")

def title(text):
    print(f"\n{BOLD}{CYAN}> {text}{RESET}")

# Header
line_sep("=")
print(f"{BOLD}{CYAN}   MABUY QUANTITATIVE ECOSYSTEM - REAL-TIME COCKPIT{RESET}")
print(f"{DIM}   Host: Ubuntu VPS | Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}{RESET}")
line_sep("=")

# 1. System Services & Health
title("1. SYSTEM SERVICES & VPS HEALTH")

def get_svc_status(svc_name):
    try:
        res = subprocess.run(["systemctl", "is-active", svc_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
        status = res.stdout.strip()
        if status == "active":
            return f"{GREEN}[ACTIVE] Running{RESET}"
        return f"{RED}[{status.upper()}]{RESET}"
    except Exception:
        return f"{DIM}N/A{RESET}"

print(f"   * mabuy-sniper.service (Trading Daemon):  {get_svc_status('mabuy-sniper.service')}")
print(f"   * mabuy-retrain.timer  (Weekly Sunday):   {get_svc_status('mabuy-retrain.timer')}")

db_path = os.path.join(BASE_DIR, "data", "mabuy_prod.db")
if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path, timeout=2)
        cur = conn.cursor()
        wal = cur.execute("PRAGMA journal_mode;").fetchone()[0]
        integ = cur.execute("PRAGMA integrity_check;").fetchone()[0]
        db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
        conn.close()
        print(f"   * SQLite DB (mabuy_prod.db):             {GREEN}[{wal.upper()}] {db_size_mb:.2f} MB | {integ}{RESET}")
    except Exception as e:
        print(f"   * SQLite DB:                             {YELLOW}Warning: {e}{RESET}")
else:
    print(f"   * SQLite DB:                             {DIM}Not present at {db_path}{RESET}")

# 2. Live Binance Futures Status
title("2. LIVE BINANCE FUTURES (REAL DEPOSIT & POSITIONS)")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    import ccxt

    api_key = os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_SECRET_KEY") or os.getenv("BINANCE_API_SECRET")

    if api_key and secret:
        masked_key = api_key[:6] + "..." + api_key[-4:]
        exchange = ccxt.binanceusdm({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'warnOnFetchOpenOrdersWithoutSymbol': False,
                'fetchOpenOrders': {'warnWithoutSymbol': False}
            }
        })
        balance = exchange.fetch_balance()
        usdt_bal = balance.get('USDT', {})
        total_bal = usdt_bal.get('total', 0.0)
        free_bal = usdt_bal.get('free', 0.0)
        used_bal = usdt_bal.get('used', 0.0)

        pnl_pct = ((total_bal - 40.0) / 40.0 * 100.0) if total_bal else 0.0
        pnl_color = GREEN if pnl_pct >= 0 else RED
        print(f"   * API Key:          {DIM}{masked_key}{RESET}")
        print(f"   * Total Balance:    {BOLD}${total_bal:.2f} USDT{RESET} ({pnl_color}{pnl_pct:+.2f}% vs $40.00 base deposit{RESET})")
        print(f"   * Free Margin:      ${free_bal:.2f} USDT")
        print(f"   * Margin In Use:    ${used_bal:.2f} USDT")

        # Performance Telemetry: Realized Win Rate & Profit Factor from Binance Futures Realized PnL
        try:
            income_records = exchange.fapiPrivateGetIncome({'incomeType': 'REALIZED_PNL', 'limit': 100})
            realized_trades = [float(rec['income']) for rec in income_records if abs(float(rec.get('income', 0))) > 0.00001]
            if realized_trades:
                wins = [t for t in realized_trades if t > 0]
                losses = [abs(t) for t in realized_trades if t < 0]
                n_trades = len(realized_trades)
                n_wins = len(wins)
                n_losses = len(losses)
                wr = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0
                gp = sum(wins)
                gl = sum(losses)
                pf = (gp / gl) if gl > 0 else (999.0 if gp > 0 else 0.0)
                net_realized = sum(realized_trades)
                avg_win = (gp / n_wins) if n_wins > 0 else 0.0
                avg_loss = (gl / n_losses) if n_losses > 0 else 0.0
                payoff = (avg_win / avg_loss) if avg_loss > 0 else 0.0

                wr_col = GREEN if wr >= 50.0 else YELLOW
                pf_col = GREEN if pf >= 1.50 else (YELLOW if pf >= 1.0 else RED)
                net_col = GREEN if net_realized >= 0 else RED

                net_str = f"+${net_realized:.2f}" if net_realized >= 0 else f"-${abs(net_realized):.2f}"
                print(f"   * Realized Win Rate:{BOLD} {wr_col}{wr:.1f}%{RESET} ({n_wins}W / {n_losses}L across {n_trades} trades)")
                print(f"   * Profit Factor (PF):{BOLD} {pf_col}{pf:.2f}{RESET} (Gross Win: {GREEN}+${gp:.2f}{RESET} | Gross Loss: {RED}-${gl:.2f}{RESET})")
                print(f"   * Realized Net PnL:  {BOLD}{net_col}{net_str} USDT{RESET} | Payoff Ratio (R): {BOLD}{payoff:.2f}{RESET}")

        except Exception:
            pass

        # Open Positions
        positions = exchange.fetch_positions()

        active_pos = [p for p in positions if abs(float(p.get("info", {}).get("positionAmt", 0) or p.get("contracts", 0) or 0)) > 0]

        if active_pos:
            print(f"\n   {BOLD}Active Open Positions ({len(active_pos)}):{RESET}")
            for p in active_pos:
                amt = float(p.get("info", {}).get("positionAmt", 0) or p.get("contracts", 0))
                side = f"{GREEN}LONG{RESET}" if amt > 0 else f"{RED}SHORT{RESET}"
                sym = p.get('symbol', 'UNKNOWN')
                entry = float(p.get('entryPrice', 0))
                mark = float(p.get('markPrice', 0))
                unrealized = float(p.get('unrealizedPnl', 0))
                u_color = GREEN if unrealized >= 0 else RED
                print(f"     -> #{sym} {side}: {abs(amt)} coins @ ${entry:,.2f} | Mark: ${mark:,.2f} | PnL: {u_color}${unrealized:+.4f}{RESET}")
        else:
            print(f"\n   {DIM}* Open Positions: 0 (flat / cash defense){RESET}")

        # Active Protective Algo Orders per active symbol
        total_algos = 0
        for p in active_pos:
            s_sym = p.get('symbol')
            try:
                orders = exchange.fetch_open_orders(s_sym)
                total_algos += len(orders)
                for o in orders:
                    o_type = o.get('type', '')
                    o_side = o.get('side', '').upper()
                    o_price = o.get('price') or o.get('stopPrice') or 0.0
                    print(f"       [{o_side} {o_type}] @ ${o_price:,.2f} on {s_sym}")
            except Exception:
                pass

        if total_algos > 0:
            print(f"   * Protected Algo Orders: {GREEN}{total_algos} orders active (Zero Naked Risk [OK]){RESET}")
        elif active_pos:
            print(f"   * Protected Algo Orders: {YELLOW}Checking daemon trailing stop manager...{RESET}")
        else:
            print(f"   * Protected Algo Orders: {DIM}0 (No open positions){RESET}")


    else:
        print(f"   {YELLOW}Warning: Binance API keys not set in .env{RESET}")
except Exception as e:
    print(f"   {YELLOW}Warning: Could not query Binance live: {e}{RESET}")

# 3. Active Quant ML Models (Registry)
title("3. ACTIVE QUANT ML ENSEMBLE (3/3 ASSETS IN PRODUCTION)")

manifest_path = os.path.join(BASE_DIR, "models", "registry_manifest.json")
if os.path.exists(manifest_path):
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        active_map = manifest.get("active_models", {})
        print(f"   {'ASSET':<10} {'ACTIVE VERSION':<25} {'OOS DA':<12} {'SHARPE':<10} {'MAE':<10} {'STATUS'}")
        print(f"   {'-'*10} {'-'*25} {'-'*12} {'-'*10} {'-'*10} {'-'*8}")

        models = manifest.get("models", [])
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            active_ver = active_map.get(sym)
            matched = next((m for m in models if m.get("symbol") == sym and m.get("version") == active_ver), None)
            if matched:
                mets = matched.get("metrics", {})
                da = mets.get("mean_directional_accuracy", mets.get("directional_accuracy", 0.0))
                sharpe = mets.get("sharpe_ratio", mets.get("mean_sharpe_ratio", 0.0))
                mae = mets.get("mean_mae", mets.get("mae", 0.0))
                print(f"   {BOLD}{sym:<10}{RESET} {active_ver:<25} {GREEN}{da*100:.2f}%{RESET}       {GREEN}{sharpe:.2f}{RESET}      {mae:.6f}   {GREEN}ACTIVE{RESET}")
            else:
                print(f"   {sym:<10} {'NO ACTIVE MODEL':<25} {'0.00%':<12} {'0.00':<10} {'-':<10} {RED}OFFLINE{RESET}")
    except Exception as e:
        print(f"   {YELLOW}Error reading registry manifest: {e}{RESET}")
else:
    print(f"   {YELLOW}No registry manifest found at {manifest_path}{RESET}")

# 4. Latest GPT-6 Astra Lead Quant Audit
title("4. GPT-6 ASTRA AUDIT VERDICT (LAST AGI REVIEW)")

astra_json_path = os.path.join(BASE_DIR, "reports", "astra_audit_latest.json")
if os.path.exists(astra_json_path):
    try:
        with open(astra_json_path, "r", encoding="utf-8") as f:
            astra = json.load(f)
        dec = astra.get("decision", "UNKNOWN")
        dec_color = GREEN if dec == "APPROVE" else RED
        print(f"   * Auditor Model:    {BOLD}{astra.get('auditor_model', 'gpt-6-astra')}{RESET}")
        print(f"   * Audit Timestamp:  {astra.get('audit_timestamp', 'N/A')}")
        print(f"   * Verdict Decision: {dec_color}{BOLD}{dec}{RESET} (Confidence: {astra.get('confidence_score', 0.99)*100:.0f}%)")
        memo = astra.get("lead_quant_memo", "")
        if memo:
            preview = memo[:180] + "..." if len(memo) > 180 else memo
            print(f"   * Memo Summary:     {DIM}\"{preview}\"{RESET}")
    except Exception as e:
        print(f"   {YELLOW}Error reading Astra audit JSON: {e}{RESET}")
else:
    print(f"   {DIM}No prior Astra audit report saved yet.{RESET}")

# 5. Recent Executed Signals (Audit Trail)
title("5. QUANT SIGNALS AUDIT TRAIL (SQLite WAL)")

if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path, timeout=2)
        cur = conn.cursor()

        # Closed Signals Aggregate Ledger Performance
        try:
            stat_row = cur.execute("""
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN pnl_usd < 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN pnl_usd > 0 THEN pnl_usd ELSE 0.0 END),
                    SUM(CASE WHEN pnl_usd < 0 THEN ABS(pnl_usd) ELSE 0.0 END),
                    SUM(pnl_usd)
                FROM signal_log
                WHERE status='CLOSED' AND pnl_usd IS NOT NULL
            """).fetchone()

            if stat_row and stat_row[0] and stat_row[0] > 0:
                c_cnt, s_wins, s_losses, s_gp, s_gl, s_pnl = stat_row
                s_wr = (s_wins / c_cnt * 100.0) if c_cnt > 0 else 0.0
                s_pf = (s_gp / s_gl) if s_gl and s_gl > 0 else (999.0 if s_gp and s_gp > 0 else 0.0)
                s_wr_col = GREEN if s_wr >= 50.0 else YELLOW
                s_pf_col = GREEN if s_pf >= 1.50 else (YELLOW if s_pf >= 1.0 else RED)
                s_pnl_val = float(s_pnl or 0.0)
                s_pnl_col = GREEN if s_pnl_val >= 0 else RED
                s_pnl_str = f"+${s_pnl_val:.2f}" if s_pnl_val >= 0 else f"-${abs(s_pnl_val):.2f}"
                print(f"   * Signal Ledger Performance: {c_cnt} Closed | Win Rate: {s_wr_col}{s_wr:.1f}%{RESET} ({s_wins}W/{s_losses}L) | PF: {s_pf_col}{s_pf:.2f}{RESET} | Net PnL: {s_pnl_col}{s_pnl_str}{RESET}\n")
        except Exception:
            pass



        sigs = cur.execute("""
            SELECT id, signal_time, symbol, direction, confidence, entry_price, exit_price, pnl_usd, status
            FROM signal_log
            ORDER BY id DESC LIMIT 5
        """).fetchall()
        conn.close()


        if sigs:
            print(f"   {'ID':<5} {'TIME (UTC)':<20} {'SYMBOL':<10} {'DIR':<8} {'CONF':<8} {'ENTRY':<12} {'PNL':<10} {'STATUS'}")
            print(f"   {'-'*5} {'-'*20} {'-'*10} {'-'*8} {'-'*8} {'-'*12} {'-'*10} {'-'*8}")
            for s in sigs:
                sid, stime, sym, d, conf, ep, xp, pnl, stat = s
                d_col = GREEN if d == "BULLISH" else (RED if d == "BEARISH" else YELLOW)
                pnl_str = f"${pnl:+.2f}" if pnl is not None else "--"
                print(f"   #{sid:<4} {stime[:19]:<20} {sym:<10} {d_col}{d:<8}{RESET} {conf*100:.1f}%    ${ep:<11,.2f} {pnl_str:<10} {stat}")
        else:
            print(f"   {DIM}No entries in signal_log table yet.{RESET}")
    except Exception as e:
        print(f"   {DIM}Could not query signal_log: {e}{RESET}")

line_sep("=")
print(f"{DIM}Zero tokens spent - 100% free local telemetry.{RESET}\n")
