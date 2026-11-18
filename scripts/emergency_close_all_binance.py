#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    import ccxt
except ImportError:
    print('ccxt required: pip install ccxt')
    sys.exit(1)

def get_keys():
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

    api_key = os.environ.get("BINANCE_API_KEY") or os.environ.get("BINANCE_KEY") or os.environ.get("BINANCE_APIKEY")
    api_secret = os.environ.get("BINANCE_API_SECRET") or os.environ.get("BINANCE_SECRET") or os.environ.get("BINANCE_SECRET_KEY")

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

    return api_key, api_secret

def main():
    api_key, api_secret = get_keys()
    if not api_key:
        print('[FAIL] No Binance keys found in .env')
        return
    exchange = ccxt.binanceusdm({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
    })
    positions = exchange.fetch_positions()
    active = [p for p in positions if abs(float(p.get('contracts', 0) or p.get('info', {}).get('positionAmt', 0))) > 0]
    print('Active positions on Binance:', len(active))
    for p in active:
        sym = p.get('symbol')
        amt = float(p.get('contracts', 0) or p.get('info', {}).get('positionAmt', 0))
        clean_sym = sym.replace('/', '').replace(':USDT', '').replace('USDC', '')
        try: exchange.cancel_all_orders(sym)
        except: pass
        try: exchange.fapiPrivateDeleteAllOpenOrders({'symbol': clean_sym})
        except: pass
        side = 'sell' if amt > 0 else 'buy'
        print('Closing', sym, amt, '->', side)
        try:
            o = exchange.create_order(symbol=sym, type='market', side=side, amount=abs(amt), params={'reduceOnly': True})
            print('CLOSED', sym, o.get('id'))
        except Exception as e:
            print('Error closing', sym, e)
    bal = exchange.fetch_balance()
    print('Final USDT Total:', bal.get('USDT', {}).get('total', 0))
    print('Final USDT Free:', bal.get('USDT', {}).get('free', 0))

if __name__ == '__main__':
    main()
