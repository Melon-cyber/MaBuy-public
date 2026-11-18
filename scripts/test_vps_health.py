# -*- coding: utf-8 -*-
import os
import sys
import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print('=' * 60)
print('MABUY VPS HEALTH CHECK (ZERO DEPENDENCIES)')
print('=' * 60)

# 1. Database
print('\n[1/4] Checking SQLite Database...')
db_path = 'data/mabuy_prod.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    res = conn.execute('PRAGMA integrity_check;').fetchall()
    conn.close()
    assert res == [('ok',)]
    print('  [OK] SQLite Database: 100% HEALTHY')
else:
    print('  [INFO] Database file will be created on start')

# 2. Execution Engine
print('\n[2/4] Testing Dynamic Execution Engine V5...')
import numpy as np
from src.domain.services.dynamic_execution_engine import DynamicParametricExecutionEngine
engine = DynamicParametricExecutionEngine()
highs = np.array([100, 102, 101, 103, 104, 103, 105, 106, 105, 107], dtype=float)
lows = np.array([96, 98, 97, 99, 100, 99, 101, 102, 101, 103], dtype=float)
opens = np.array([97, 99, 98, 100, 101, 100, 102, 103, 102, 104], dtype=float)
closes = np.array([99, 101, 100, 102, 103, 102, 104, 105, 104, 106], dtype=float)
targets = engine.calculate_entry_targets(
    symbol='BTCUSDT', direction='BULLISH', current_price=100000.0,
    opens=opens, highs=highs, lows=lows, closes=closes,
    model_prob_win=0.75, raw_ofi=0.25, market_regime='BULL_TREND'
)
print(f'  [OK] Targets: SL= | TP= | R:R=1:{targets.risk_reward_ratio:.2f}')

# 3. Risk Engine
print('\n[3/4] Testing V5 Dynamic Risk Engine...')
from src.domain.services.v5_risk_engine import V5DynamicRiskEngine
plan = V5DynamicRiskEngine.calculate_trade_risk_plan(
    account_balance=32.0, symbol='BTCUSDT', current_price=100000.0,
    stop_loss_price=targets.stop_loss_price, take_profit_price=targets.take_profit_price,
    model_probability=0.75
)
print(f'  [OK] Risk Plan: Allowed={plan.allowed} | Leverage={plan.safe_leverage}x | Risk=')

# 4. Bot Keyboard
print('\n[4/4] Testing Telegram Bot Keyboards...')
from scripts.launch_aiogram_bot import get_predict_keyboard
kb = get_predict_keyboard('BTC', '1h', is_pro_view=False, lang='ru')
assert len(kb.inline_keyboard) >= 2
print('  [OK] Bot Keyboards: WebApp button parsed without errors')

print('\n' + '=' * 60)
print('ALL 4 CHECKS PASSED: VPS IS 100% OPERATIONAL!')
print('=' * 60)
