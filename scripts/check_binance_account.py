#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaBuy V5 — Direct Live Binance Futures Position & Balance Auditor.
Automatically loads keys from:
1. Project .env file
2. SQLite production database (user_api_keys table for 100000001)
3. Current system environment
"""
import os
import sys
import sqlite3
import ccxt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def load_env():
    potential_paths = [
        BASE_DIR / ".env",
        Path("/root/mabuy2/.env"),
        Path("/root/mabuy/.env"),
        Path("/root/.env"),
        Path.home() / ".env"
    ]
    for p in potential_paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                    for target in ["BINANCE_API_KEY=", "BINANCE_API_SECRET=", "BINANCE_KEY=", "BINANCE_SECRET="]:
                        raw_text = raw_text.replace(target, "\n" + target)
                    for line in raw_text.splitlines():
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            clean_v = v.strip().strip("'").strip('"')
                            os.environ[k.strip()] = clean_v
            except Exception:
                pass

def get_keys():
    load_env()
    
    # 1. Try environment
    api_key = os.environ.get("BINANCE_API_KEY") or os.environ.get("BINANCE_KEY") or os.environ.get("BINANCE_APIKEY")
    api_secret = os.environ.get("BINANCE_API_SECRET") or os.environ.get("BINANCE_SECRET") or os.environ.get("BINANCE_SECRET_KEY")

    # 2. Try checking systemd service file or running process environ on Linux
    if not api_key or not api_secret:
        try:
            svc_path = Path("/etc/systemd/system/mabuy-daemon.service")
            if svc_path.exists():
                with open(svc_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if "EnvironmentFile=" in line:
                            ef = line.split("=", 1)[1].strip()
                            if os.path.exists(ef):
                                with open(ef, "r", encoding="utf-8") as eff:
                                    for eline in eff:
                                        if "=" in eline and not eline.startswith("#"):
                                            ek, ev = eline.split("=", 1)
                                            os.environ[ek.strip()] = ev.strip().strip("'").strip('"')
            api_key = os.environ.get("BINANCE_API_KEY") or os.environ.get("BINANCE_KEY")
            api_secret = os.environ.get("BINANCE_API_SECRET") or os.environ.get("BINANCE_SECRET")
        except Exception:
            pass

    # 3. Try reading /proc of running python daemon on Linux
    if not api_key or not api_secret:
        try:
            import glob
            for env_proc in glob.glob("/proc/[0-9]*/environ"):
                try:
                    with open(env_proc, "rb") as f:
                        raw = f.read().split(b"\x00")
                        for item in raw:
                            s = item.decode("utf-8", errors="ignore")
                            if s.startswith("BINANCE_API_KEY=") or s.startswith("BINANCE_KEY="):
                                api_key = s.split("=", 1)[1].strip().strip("'").strip('"')
                            elif s.startswith("BINANCE_API_SECRET=") or s.startswith("BINANCE_SECRET="):
                                api_secret = s.split("=", 1)[1].strip().strip("'").strip('"')
                    if api_key and api_secret:
                        break
                except Exception:
                    continue
        except Exception:
            pass

    # 4. Try SQLite DB via SQLiteTracker (with decrypt)
    if not api_key or not api_secret:
        try:
            from src.infrastructure.persistence.sqlite_tracker import SQLiteTracker
            tracker = SQLiteTracker()
            k = tracker.get_api_keys("100000001")
            if k and k[0] and k[1]:
                api_key, api_secret = k[0], k[1]
            if not api_key:
                all_vips = tracker.get_all_vip_api_keys()
                if all_vips:
                    first_uid = list(all_vips.keys())[0]
                    api_key, api_secret = all_vips[first_uid]
        except Exception:
            pass

    return api_key, api_secret

def audit_binance():
    print("=" * 65)
    print("⚡ MABUY LIVE BINANCE FUTURES REAL-TIME AUDIT")
    print("=" * 65)

    api_key, api_secret = get_keys()

    if not api_key or not api_secret:
        print("[FAIL] Binance API credentials not found in .env or database!")
        print("Please check /root/mabuy2/.env or enter keys via Telegram bot.")
        return

    masked_key = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
    print(f"🔑 Using Binance API Key: {masked_key}")

    try:
        exchange = ccxt.binanceusdm({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future", "adjustForTimeDifference": True}
        })

        # 1. Fetch Balances
        bal = exchange.fetch_balance()
        usdt_free = bal.get("USDT", {}).get("free", 0.0)
        usdt_total = bal.get("USDT", {}).get("total", 0.0)
        usdt_used = bal.get("USDT", {}).get("used", 0.0)

        print(f"\n💰 REAL BINANCE BALANCES:")
        print(f"  • Total Balance:    ${usdt_total:,.2f} USDT")
        print(f"  • Available Free:   ${usdt_free:,.2f} USDT")
        print(f"  • In Active Margin: ${usdt_used:,.2f} USDT")

        # 2. Fetch Open Positions
        positions = exchange.fetch_positions()
        active_pos = [p for p in positions if abs(float(p.get("contracts", 0) or p.get("info", {}).get("positionAmt", 0))) > 0]

        print(f"\n🎯 ACTIVE OPEN POSITIONS ({len(active_pos)}):")
        if active_pos:
            for p in active_pos:
                sym = p.get("symbol")
                side = p.get("side", "").upper()
                amt = float(p.get("contracts", 0) or p.get("info", {}).get("positionAmt", 0))
                entry_p = float(p.get("entryPrice", 0) or p.get("info", {}).get("entryPrice", 0))
                mark_p = float(p.get("markPrice", 0) or p.get("info", {}).get("markPrice", 0))
                pnl = float(p.get("unrealizedPnl", 0) or p.get("info", {}).get("unRealizedProfit", 0))
                lev = p.get("leverage", 1)
                
                sign = "+" if pnl >= 0 else ""
                print(f"  • #{sym} {side}: {abs(amt)} coins @ ${entry_p:,.4f} | Mark: ${mark_p:,.4f} | PnL: {sign}${pnl:,.2f} | Lev: {lev}x")
        else:
            print("  (0 active positions right now — scanning for high-probability setups)")

        # 3. Fetch Open Orders per active position (Direct Binance Futures API + Algo Engine)
        print(f"\n🛡 ACTIVE SERVER CONDITIONAL ORDERS:")
        found_orders = 0
        for p in active_pos:
            s = p.get("symbol")
            clean_sym = s.replace('/', '').replace(':USDT', '').replace('USDC', '')
            
            # Check 1: Standard Open Orders
            try:
                raw_orders = exchange.fapiPrivateGetOpenOrders({'symbol': clean_sym})
                for o in raw_orders:
                    found_orders += 1
                    o_type = o.get("type", "").upper()
                    o_side = o.get("side", "").upper()
                    stop_p = float(o.get("stopPrice") or o.get("price") or 0.0)
                    oid = o.get("orderId")
                    print(f"  • {o_type} {o_side} on #{s} @ ${stop_p:,.4f} (ID: {oid})")
            except Exception:
                pass
                
            # Check 2: Algo / Conditional Orders (Stop-Loss & Take-Profit)
            try:
                algo_orders = exchange.fapiPrivateGetOpenAlgoOrders({'symbol': clean_sym})
                for o in algo_orders:
                    found_orders += 1
                    o_type = o.get("algoType") or o.get("orderType") or "CONDITIONAL"
                    o_side = o.get("side", "").upper()
                    stop_p = float(o.get("triggerPrice") or o.get("stopPrice") or 0.0)
                    oid = o.get("algoId") or o.get("orderId")
                    print(f"  • {o_type} {o_side} on #{s} @ ${stop_p:,.4f} (Algo ID: {oid})")
            except Exception:
                pass

        if found_orders == 0:
            print("  (0 conditional orders returned by API)")

    except Exception as e:
        print(f"\n[FAIL] Binance API call failed: {e}")

    print("\n" + "=" * 65)

if __name__ == "__main__":
    audit_binance()
