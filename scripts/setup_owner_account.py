# -*- coding: utf-8 -*-
import os
import sys
import sqlite3
import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 65)
print("⚙️ CONFIGURING OWNER ACCOUNT (100000001) & UNBLOCKING V5 DAEMON")
print("=" * 65)

from src.infrastructure.persistence.sqlite_tracker import SQLiteTracker
from src.domain.entities.tier import SubscriptionTier

db = SQLiteTracker("data/mabuy_prod.db")
owner_id = "100000001"

# 1. Set Real Balance: $32.00 (Micro-Deposit Tier < $100)
db.set_user_deposit(owner_id, 32.0)
print(f"  ✅ Configured Trading Balance for {owner_id}: $32.00")

# 2. Set Lifetime VIP PRO Tier
lifetime = (datetime.datetime.now() + datetime.timedelta(days=3650)).isoformat()
db.set_user_tier(owner_id, SubscriptionTier.PRO, lifetime)
print(f"  ✅ Configured Subscription Tier for {owner_id}: PRO (VIP Founder)")

# 3. Clean up stale historical signals in verified_signals_log
conn = sqlite3.connect("data/mabuy_prod.db")
cur = conn.cursor()
stale_closed = cur.execute("""
    UPDATE verified_signals_log 
    SET status = 'CLOSED', exit_price = entry_price, closed_at = ? 
    WHERE status = 'OPEN' AND created_at < datetime('now', '-6 hours')
""", (datetime.datetime.now().isoformat(),)).rowcount
conn.commit()
print(f"  ✅ Cleaned up {stale_closed} stale signals (>6h old) in verified_signals_log.")

# Check current active open signals
open_count = cur.execute("SELECT count(*) FROM verified_signals_log WHERE status = 'OPEN'").fetchone()[0]
print(f"  ✅ Currently Active Signals in Queue: {open_count} (Cap is max 2)")

conn.close()
print("\n" + "=" * 65)
print("🎉 OWNER ACCOUNT CONFIGURED: Balance $32 | PRO Tier | Scanner Ready!")
print("=" * 65)
