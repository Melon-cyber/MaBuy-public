import sqlite3
import time
import datetime
import json
import uuid
from threading import Lock
from typing import Optional, List, Dict, Any, Tuple

from src.domain.entities.tier import SubscriptionTier
from src.application.interfaces.usage_tracker import DailyUsageTrackerInterface
from src.infrastructure.security.encryption import EncryptionService

class SQLiteTracker(DailyUsageTrackerInterface):
    """
    Production-ready SQLite implementation for tracking daily usage,
    user subscription tiers, and user language preferences.
    Ensures data persists across server reboots.
    """
    def __init__(self, db_path: Optional[str] = None):
        import os
        if db_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            db_path = os.path.join(project_root, "data", "mabuy_prod.db")
        elif db_path != ":memory:":
            db_path = os.path.abspath(db_path)
            
        dirname = os.path.dirname(db_path) if db_path != ":memory:" else None
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self.db_path = db_path
        self._lock = Lock()
        self.encryption_service = EncryptionService()
        self._init_db()

    def _init_db(self):
        from contextlib import closing
        import os, shutil, time, logging
        log = logging.getLogger("SQLiteTracker")
        with self._lock:
            # Check for corrupted database image and auto-heal
            if self.db_path != ":memory:" and os.path.exists(self.db_path):
                try:
                    with sqlite3.connect(self.db_path, timeout=5.0) as check_conn:
                        res = check_conn.execute('PRAGMA integrity_check;').fetchall()
                        if not res or res[0][0] != "ok":
                            raise sqlite3.DatabaseError(f"Integrity check failed: {res}")
                except Exception as db_err:
                    if "malformed" in str(db_err).lower() or "corrupt" in str(db_err).lower() or "integrity" in str(db_err).lower():
                        log.critical(f"🚨 Auto-Healing: Corrupted SQLite detected ({db_err}). Rebuilding pristine database...")
                        backup_p = f"{self.db_path}.corrupted.{int(time.time())}"
                        try:
                            shutil.copy2(self.db_path, backup_p)
                        except Exception:
                            pass
                        for ext in ["", "-wal", "-shm"]:
                            p = f"{self.db_path}{ext}"
                            if os.path.exists(p):
                                try:
                                    os.remove(p)
                                except Exception:
                                    pass
                        log.info(f"✅ Pristine SQLite database recreated. Backup saved to {backup_p}")

            with closing(sqlite3.connect(self.db_path, timeout=30.0)) as conn:
                cursor = conn.cursor()
                # ENFORCE BUSY TIMEOUT AND WAL MODE SAFELY FOR HIGH CONCURRENCY
                cursor.execute('PRAGMA busy_timeout=30000;')
                try:
                    cursor.execute('PRAGMA journal_mode=WAL;')
                except sqlite3.OperationalError:
                    pass  # Already in WAL mode or busy, safe to continue
                cursor.execute('PRAGMA synchronous=NORMAL;')
                cursor.execute('PRAGMA wal_autocheckpoint=1000;')
                # Create table for daily usage
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS daily_usage (
                        user_id TEXT,
                        date_str TEXT,
                        usage_count INTEGER DEFAULT 0,
                        PRIMARY KEY (user_id, date_str)
                    )
                ''')
                # Create table for daily AI copilot usage (Free: 5, Pro/Invest: 10, All-Access: unlimited)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS daily_ai_usage (
                        user_id TEXT,
                        date_str TEXT,
                        usage_count INTEGER DEFAULT 0,
                        PRIMARY KEY (user_id, date_str)
                    )
                ''')
                # Create table for user subscriptions
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_subscriptions (
                        user_id TEXT PRIMARY KEY,
                        tier TEXT NOT NULL,
                        expires_at REAL NOT NULL
                    )
                ''')
                # Create table for user languages
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_languages (
                        user_id TEXT PRIMARY KEY,
                        language TEXT NOT NULL
                    )
                ''')
                # Create table for user MaIA personas (quant, zen, tsundere, yandere, titan)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_personas (
                        user_id TEXT PRIMARY KEY,
                        persona TEXT NOT NULL
                    )
                ''')
                # Create table for user experience level (beginner vs pro)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_experience (
                        user_id TEXT PRIMARY KEY,
                        experience_level TEXT NOT NULL
                    )
                ''')
                # Create table for user profiles (nicknames)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id TEXT PRIMARY KEY,
                        nickname TEXT NOT NULL
                    )
                ''')
                # Create table for encrypted API keys
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_api_keys (
                        user_id TEXT PRIMARY KEY,
                        enc_api_key TEXT NOT NULL,
                        enc_api_secret TEXT NOT NULL
                    )
                ''')
                # Create table for user risk profiles
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_risk_profiles (
                        user_id TEXT PRIMARY KEY,
                        risk_level TEXT NOT NULL
                    )
                ''')
                # Create table for user alerts
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_alerts (
                        alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(user_id, symbol)
                    )
                ''')
                # Create table for user custom trading deposits
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_trading_deposits (
                        user_id TEXT PRIMARY KEY,
                        deposit_usd REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                ''')
                # --- PAPER TRADING TABLES ---
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS virtual_balances (
                        user_id TEXT PRIMARY KEY,
                        usd_balance REAL NOT NULL
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS paper_trades (
                        trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        amount_usd REAL NOT NULL,
                        pnl_usd REAL,
                        status TEXT NOT NULL,
                        opened_at TEXT NOT NULL,
                        closed_at TEXT
                    )
                ''')
                # --- ML SIGNALS HISTORY ---
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ml_signals_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        confidence_score REAL NOT NULL,
                        created_at TEXT NOT NULL
                    )
                ''')
                # --- VERIFIED SIGNALS LOG ---
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS verified_signals_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        target_price REAL NOT NULL,
                        stop_loss REAL NOT NULL,
                        confidence_score REAL NOT NULL,
                        status TEXT NOT NULL,
                        exit_price REAL,
                        channel_message_id INTEGER,
                        channel_id TEXT,
                        created_at TEXT NOT NULL,
                        closed_at TEXT
                    )
                ''')
                # Idempotent schema migration for existing databases
                cursor.execute("PRAGMA table_info(verified_signals_log)")
                existing_cols = {row[1] for row in cursor.fetchall()}
                if "channel_message_id" not in existing_cols:
                    cursor.execute("ALTER TABLE verified_signals_log ADD COLUMN channel_message_id INTEGER")
                if "channel_id" not in existing_cols:
                    cursor.execute("ALTER TABLE verified_signals_log ADD COLUMN channel_id TEXT")
                # --- FINE-TUNING TELEMETRY (SIM-TO-REAL GAP DATASET) ---
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS fine_tuning_telemetry (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        confidence_score REAL NOT NULL,
                        predicted_target_price REAL NOT NULL,
                        actual_fill_price REAL,
                        slippage_pct REAL,
                        execution_latency_ms REAL,
                        realized_pnl_pct REAL,
                        feature_vector_json TEXT,
                        created_at TEXT NOT NULL
                    )
                ''')
                
                # Auto-migrate for Trailing Stop-Loss
                try:
                    cursor.execute('ALTER TABLE paper_trades ADD COLUMN max_pnl_usd REAL DEFAULT 0.0')
                except sqlite3.OperationalError:
                    pass # Column already exists
                
                # --- HABIT TRACKER & GAMIFICATION TABLES ---
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS habits (
                        habit_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        category TEXT NOT NULL,
                        target_type TEXT NOT NULL,
                        target_value INTEGER DEFAULT 1,
                        unit TEXT DEFAULT 'min',
                        icon TEXT DEFAULT '🎯',
                        color TEXT DEFAULT '#3b82f6',
                        deadline_time TEXT DEFAULT '',
                        order_idx INTEGER DEFAULT 0,
                        is_archived INTEGER DEFAULT 0,
                        created_at REAL NOT NULL
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS habit_logs (
                        user_id TEXT NOT NULL,
                        habit_id TEXT NOT NULL,
                        date_str TEXT NOT NULL,
                        value REAL DEFAULT 0,
                        is_completed INTEGER DEFAULT 0,
                        logged_at REAL NOT NULL,
                        PRIMARY KEY (user_id, habit_id, date_str)
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_gamification (
                        user_id TEXT PRIMARY KEY,
                        xp INTEGER DEFAULT 0,
                        streak_current INTEGER DEFAULT 0,
                        streak_best INTEGER DEFAULT 0,
                        last_active_date TEXT DEFAULT '',
                        rank_title TEXT DEFAULT 'NPC'
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS journal_entries (
                        entry_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        date_str TEXT NOT NULL,
                        mood TEXT DEFAULT '🔥',
                        title TEXT DEFAULT '',
                        content_text TEXT DEFAULT '',
                        audio_path TEXT DEFAULT '',
                        audio_tracks_json TEXT DEFAULT '[]',
                        media_paths_json TEXT DEFAULT '[]',
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                ''')
                try:
                    cursor.execute('ALTER TABLE journal_entries ADD COLUMN audio_tracks_json TEXT DEFAULT "[]"')
                except sqlite3.OperationalError:
                    pass
                
                # --- ANTI-TILT & DISCIPLINE JOURNAL ---
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS anti_tilt_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        tag TEXT NOT NULL,
                        pnl_usd REAL DEFAULT 0.0,
                        followed_plan INTEGER NOT NULL,
                        notes TEXT,
                        created_at REAL NOT NULL
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_anti_tilt_state (
                        user_id TEXT PRIMARY KEY,
                        last_tilt_timestamp REAL DEFAULT 0.0
                    )
                ''')
                
                # --- STRATIFIED REGIME QUOTA TRACKER (FORWARD-TEST) ---
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS regime_quota_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        regime TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        is_win INTEGER NOT NULL,
                        pnl_usd REAL NOT NULL,
                        created_at REAL NOT NULL
                    )
                ''')
                
                # Indexes for optimization
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_ml_signals_symbol_time ON ml_signals_history(symbol, created_at DESC)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_paper_trades_user_status ON paper_trades(user_id, status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_alerts_symbol ON user_alerts(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_verified_signals_status ON verified_signals_log(status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_verified_signals_symbol_status ON verified_signals_log(symbol, status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_fine_tuning_telemetry_symbol ON fine_tuning_telemetry(symbol, id DESC)')
                
                # Indexes for habits and journal
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_habits_user ON habits(user_id, is_archived, order_idx)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_habit_logs_user_date ON habit_logs(user_id, date_str)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_journal_user_date ON journal_entries(user_id, date_str DESC)')
                
                # ENFORCE IMMUTABILITY AT DATABASE LEVEL
                cursor.execute('''
                    CREATE TRIGGER IF NOT EXISTS prevent_signal_delete
                    BEFORE DELETE ON verified_signals_log
                    BEGIN
                        SELECT RAISE(ABORT, 'Deleting verified signals is strictly prohibited to guarantee track record immutability.');
                    END;
                ''')
                cursor.execute('''
                    CREATE TRIGGER IF NOT EXISTS prevent_signal_tampering
                    BEFORE UPDATE OF symbol, direction, entry_price, target_price, stop_loss, confidence_score, created_at
                    ON verified_signals_log
                    BEGIN
                        SELECT RAISE(ABORT, 'Tampering with core signal parameters is strictly prohibited.');
                    END;
                ''')

                conn.commit()

    def _get_today_str(self) -> str:
        return datetime.date.today().isoformat()

    # --- Daily Usage Tracking ---
    
    def get_daily_usage(self, user_id: str) -> int:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT usage_count FROM daily_usage WHERE user_id = ? AND date_str = ?', 
                               (str(user_id), self._get_today_str()))
                row = cursor.fetchone()
                return row[0] if row else 0

    def increment_usage(self, user_id: str) -> int:
        with self._lock:
            today = self._get_today_str()
            user_str = str(user_id)
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT usage_count FROM daily_usage WHERE user_id = ? AND date_str = ?', 
                               (user_str, today))
                row = cursor.fetchone()
                if row:
                    new_count = row[0] + 1
                    cursor.execute('UPDATE daily_usage SET usage_count = ? WHERE user_id = ? AND date_str = ?',
                                   (new_count, user_str, today))
                else:
                    new_count = 1
                    cursor.execute('INSERT INTO daily_usage (user_id, date_str, usage_count) VALUES (?, ?, ?)',
                                   (user_str, today, new_count))
                conn.commit()
                return new_count

    def increment_daily_usage(self, user_id: str) -> int:
        return self.increment_usage(user_id)

    def reset_usage(self, user_id: str) -> None:
        """Reset usage for today."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM daily_usage WHERE user_id = ? AND date_str = ?', 
                               (str(user_id), self._get_today_str()))
                conn.commit()

    # --- Daily AI Copilot (MaIA) Usage Tracking ---

    def get_daily_ai_usage(self, user_id: str) -> int:
        """Returns the number of AI questions asked by user today."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT usage_count FROM daily_ai_usage WHERE user_id = ? AND date_str = ?', 
                               (str(user_id), self._get_today_str()))
                row = cursor.fetchone()
                return row[0] if row else 0

    def increment_ai_usage(self, user_id: str) -> int:
        """Increments and returns today's AI usage count for user."""
        with self._lock:
            today = self._get_today_str()
            user_str = str(user_id)
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT usage_count FROM daily_ai_usage WHERE user_id = ? AND date_str = ?', 
                               (user_str, today))
                row = cursor.fetchone()
                if row:
                    new_count = row[0] + 1
                    cursor.execute('UPDATE daily_ai_usage SET usage_count = ? WHERE user_id = ? AND date_str = ?',
                                   (new_count, user_str, today))
                else:
                    new_count = 1
                    cursor.execute('INSERT INTO daily_ai_usage (user_id, date_str, usage_count) VALUES (?, ?, ?)',
                                   (user_str, today, new_count))
                conn.commit()
                return new_count

    def get_user_ai_limit(self, user_id: str) -> int:
        """Returns daily AI limit (Free: 5, MaPredict: 10, MaInvest: 10, All-Access: unlimited)."""
        from src.domain.entities.tier import SubscriptionTier, TIER_CONFIGURATIONS
        user_str = str(user_id)
        if user_str in {"100000001", "owner_admin"}:
            return 999999
        tier_data = self.get_user_tier_data(user_str)
        if not tier_data:
            return TIER_CONFIGURATIONS[SubscriptionTier.FREE].daily_ai_limit
        tier, expires_at = tier_data
        if time.time() > expires_at:
            return TIER_CONFIGURATIONS[SubscriptionTier.FREE].daily_ai_limit
        return TIER_CONFIGURATIONS.get(tier, TIER_CONFIGURATIONS[SubscriptionTier.FREE]).daily_ai_limit

    def can_user_query_ai(self, user_id: str) -> Tuple[bool, int, int]:
        """Checks if user is allowed to make another AI query. Returns (allowed, used, limit)."""
        limit = self.get_user_ai_limit(user_id)
        used = self.get_daily_ai_usage(user_id)
        allowed = used < limit
        return (allowed, used, limit)

    # --- Subscription Tracking ---
    
    def get_user_tier_data(self, user_id: str) -> Optional[tuple[SubscriptionTier, float]]:
        """Returns (Tier, expires_at) if exists, else None."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT tier, expires_at FROM user_subscriptions WHERE user_id = ?', (str(user_id),))
                row = cursor.fetchone()
                if row:
                    try:
                        return SubscriptionTier(row[0]), row[1]
                    except ValueError:
                        if str(row[0]).lower() in ['titan', 'enterprise', 'pro']:
                            return SubscriptionTier.PRO, row[1]
                        return SubscriptionTier.FREE, row[1]
                return None

    def set_user_tier(self, user_id: str, tier: SubscriptionTier, expires_at: float) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_subscriptions (user_id, tier, expires_at) 
                    VALUES (?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET tier=excluded.tier, expires_at=excluded.expires_at
                ''', (str(user_id), tier.value, expires_at))
                conn.commit()

    def delete_user_subscription(self, user_id: str) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM user_subscriptions WHERE user_id = ?', (str(user_id),))
                conn.commit()

    # --- Language Preferences ---
    
    def get_user_language(self, user_id: str, default: str = "ru") -> str:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT language FROM user_languages WHERE user_id = ?', (str(user_id),))
                row = cursor.fetchone()
                return row[0] if row else default

    def set_user_language(self, user_id: str, language: str) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_languages (user_id, language) 
                    VALUES (?, ?) 
                    ON CONFLICT(user_id) DO UPDATE SET language=excluded.language
                ''', (str(user_id), language))
                conn.commit()

    # --- Experience Level Preferences (Beginner vs Pro) ---
    
    def get_user_experience(self, user_id: str, default: str = "beginner") -> str:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT experience_level FROM user_experience WHERE user_id = ?', (str(user_id),))
                row = cursor.fetchone()
                return row[0] if row else default

    def set_user_experience(self, user_id: str, level: str) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_experience (user_id, experience_level) 
                    VALUES (?, ?) 
                    ON CONFLICT(user_id) DO UPDATE SET experience_level=excluded.experience_level
                ''', (str(user_id), level.lower()))
                conn.commit()

    # --- User MaIA Personas (Quant, Zen, Tsundere, Yandere, Titan) ---

    def get_user_persona(self, user_id: str, default: str = "quant") -> str:
        """Returns the user's active MaIA persona (default: 'quant')."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT persona FROM user_personas WHERE user_id = ?', (str(user_id),))
                row = cursor.fetchone()
                return row[0] if row else default

    def set_user_persona(self, user_id: str, persona: str) -> None:
        """Saves the user's selected MaIA persona."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_personas (user_id, persona) 
                    VALUES (?, ?) 
                    ON CONFLICT(user_id) DO UPDATE SET persona = excluded.persona
                ''', (str(user_id), str(persona).lower()))
                conn.commit()

    # --- User Profiles (Nicknames) ---
    
    def get_nickname(self, user_id: str) -> Optional[str]:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT nickname FROM user_profiles WHERE user_id = ?', (str(user_id),))
                row = cursor.fetchone()
                return row[0] if row else None

    def set_nickname(self, user_id: str, nickname: str) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_profiles (user_id, nickname) 
                    VALUES (?, ?) 
                    ON CONFLICT(user_id) DO UPDATE SET nickname=excluded.nickname
                ''', (str(user_id), nickname))
                conn.commit()

    # --- User Risk Profiles ---
    
    def get_user_risk_profile(self, user_id: str) -> str:
        """Returns risk level: 'conservative', 'balanced' (default), or 'aggressive'"""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT risk_level FROM user_risk_profiles WHERE user_id = ?', (str(user_id),))
                row = cursor.fetchone()
                return row[0] if row else "balanced"
                
    def set_user_risk_profile(self, user_id: str, risk_level: str) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_risk_profiles (user_id, risk_level)
                    VALUES (?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET risk_level=excluded.risk_level
                ''', (str(user_id), risk_level))
                conn.commit()

    # --- User Custom Trading Capital / Deposits ---

    def get_user_deposit(self, user_id: str, default: Optional[float] = None) -> Optional[float]:
        """Returns saved user trading deposit in USD. Returns None if not configured by user."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT deposit_usd FROM user_trading_deposits WHERE user_id = ?', (str(user_id),))
                row = cursor.fetchone()
                if row and row[0] is not None:
                    return float(row[0])
                return default

    def set_user_deposit(self, user_id: str, deposit_usd: float) -> float:
        """Saves or updates user custom trading deposit in USD."""
        val = max(10.0, float(deposit_usd))
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_trading_deposits (user_id, deposit_usd, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        deposit_usd=excluded.deposit_usd,
                        updated_at=excluded.updated_at
                ''', (str(user_id), val, time.time()))
                conn.commit()
        return val
                
    # --- Encrypted API Keys Tracking ---
    
    def set_api_keys(self, user_id: str, api_key: str, api_secret: str) -> None:
        enc_api_key = self.encryption_service.encrypt(api_key)
        enc_api_secret = self.encryption_service.encrypt(api_secret)
        
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_api_keys (user_id, enc_api_key, enc_api_secret)
                    VALUES (?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET 
                        enc_api_key=excluded.enc_api_key,
                        enc_api_secret=excluded.enc_api_secret
                ''', (str(user_id), enc_api_key, enc_api_secret))
                conn.commit()

    def get_api_keys(self, user_id: str) -> Optional[tuple[str, str]]:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT enc_api_key, enc_api_secret FROM user_api_keys WHERE user_id = ?', (str(user_id),))
                row = cursor.fetchone()
                if row:
                    dec_api_key = self.encryption_service.decrypt(row[0])
                    dec_api_secret = self.encryption_service.decrypt(row[1])
                    return (dec_api_key, dec_api_secret)
                return None
                
    def get_all_vip_api_keys(self) -> dict[str, tuple[str, str]]:
        """Returns dict mapping user_id -> (api_key, api_secret) for all connected users."""
        users = {}
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT user_id, enc_api_key, enc_api_secret FROM user_api_keys')
                for row in cursor.fetchall():
                    uid = row[0]
                    try:
                        dec_key = self.encryption_service.decrypt(row[1])
                        dec_secret = self.encryption_service.decrypt(row[2])
                        users[uid] = (dec_key, dec_secret)
                    except Exception:
                        pass # Skip invalid/corrupt keys
        return users

    def get_admin_stats(self) -> dict:
        """Calculates total users, tier breakdown, and MRR accurately from active real Telegram users."""
        stats = {
            "total_users": 0,
            "pro_users": 0,
            "titan_users": 0,
            "enterprise_users": 0,
            "mrr_usd": 0.0,
            "api_keys_linked": 0
        }
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(DISTINCT user_id) FROM (
                        SELECT user_id FROM user_languages
                        UNION
                        SELECT user_id FROM user_subscriptions
                        UNION
                        SELECT user_id FROM daily_usage
                    )
                    WHERE user_id GLOB '[0-9]*' AND user_id NOT IN ('broadcast', 'test_user', 'v4_test_user_2')
                """)
                row = cursor.fetchone()
                stats["total_users"] = row[0] if row and row[0] is not None else 0

                now_ts = int(time.time())
                cursor.execute("""
                    SELECT tier, COUNT(*) FROM user_subscriptions 
                    WHERE expires_at > ? 
                      AND user_id GLOB '[0-9]*' 
                      AND user_id NOT IN ('broadcast', 'test_user', 'v4_test_user_2')
                    GROUP BY tier
                """, (now_ts,))
                for tier, count in cursor.fetchall():
                    tier_clean = str(tier).lower().strip()
                    if tier_clean == SubscriptionTier.PRO.value or tier_clean == "pro":
                        stats["pro_users"] += count
                        stats["mrr_usd"] += count * 19.99
                    elif tier_clean == "titan":
                        stats["titan_users"] += count
                        stats["mrr_usd"] += count * 49.99
                    elif tier_clean == "enterprise":
                        stats["enterprise_users"] += count
                        stats["mrr_usd"] += count * 199.99

                cursor.execute("""
                    SELECT COUNT(*) FROM user_api_keys 
                    WHERE user_id GLOB '[0-9]*' AND user_id NOT IN ('broadcast', 'test_user', 'v4_test_user_2')
                """)
                row_api = cursor.fetchone()
                stats["api_keys_linked"] = row_api[0] if row_api and row_api[0] is not None else 0
                
        return stats

    # --- PAPER TRADING METHODS ---

    def get_virtual_balance(self, user_id: str) -> float:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT usd_balance FROM virtual_balances WHERE user_id = ?', (str(user_id),))
                row = cursor.fetchone()
                if row:
                    return row[0]
                # Default start balance
                cursor.execute('INSERT INTO virtual_balances (user_id, usd_balance) VALUES (?, ?)', (str(user_id), 10000.0))
                conn.commit()
                return 10000.0

    def update_virtual_balance(self, user_id: str, pnl_usd: float) -> float:
        current = self.get_virtual_balance(user_id)
        new_balance = current + pnl_usd
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE virtual_balances SET usd_balance = ? WHERE user_id = ?', (new_balance, str(user_id)))
                conn.commit()
        return new_balance

    def get_leaderboard(self, limit: int = 10) -> list[tuple[str, float]]:
        """Returns list of (nickname_or_id, usd_balance) ordered by highest balance."""
        leaders = []
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT v.user_id, v.usd_balance, p.nickname
                    FROM virtual_balances v
                    LEFT JOIN user_profiles p ON v.user_id = p.user_id
                    ORDER BY v.usd_balance DESC
                    LIMIT ?
                ''', (limit,))
                for row in cursor.fetchall():
                    uid = row[0]
                    balance = row[1]
                    nickname = row[2]
                    
                    if nickname:
                        display_name = nickname
                    else:
                        # Mask ID for privacy: 123456789 -> ID 123***789
                        if len(uid) > 6:
                            display_name = f"ID {uid[:3]}***{uid[-3:]}"
                        else:
                            display_name = f"ID {uid}"
                            
                    leaders.append((display_name, balance))
        return leaders

    def open_paper_trade(self, user_id: str, symbol: str, direction: str, entry_price: float, amount_usd: float) -> int:
        opened_at = datetime.datetime.utcnow().isoformat()
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO paper_trades (user_id, symbol, direction, entry_price, amount_usd, status, opened_at, max_pnl_usd)
                    VALUES (?, ?, ?, ?, ?, 'OPEN', ?, 0.0)
                ''', (str(user_id), symbol, direction, entry_price, amount_usd, opened_at))
                conn.commit()
                return cursor.lastrowid

    def close_paper_trade(self, trade_id: int, exit_price: float, pnl_usd: float) -> None:
        closed_at = datetime.datetime.utcnow().isoformat()
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE paper_trades 
                    SET status = 'CLOSED', exit_price = ?, pnl_usd = ?, closed_at = ?
                    WHERE trade_id = ?
                ''', (exit_price, pnl_usd, closed_at, trade_id))
                conn.commit()
                
    def update_max_pnl(self, trade_id: int, max_pnl: float) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE paper_trades SET max_pnl_usd = ? WHERE trade_id = ?
                ''', (max_pnl, trade_id))
                conn.commit()

    def get_open_paper_trades(self, user_id: str) -> list[dict]:
        trades = []
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT trade_id, symbol, direction, entry_price, amount_usd, opened_at, max_pnl_usd
                    FROM paper_trades WHERE user_id = ? AND status = 'OPEN'
                ''', (str(user_id),))
                for row in cursor.fetchall():
                    trades.append({
                        "trade_id": row[0],
                        "symbol": row[1],
                        "direction": row[2],
                        "entry_price": row[3],
                        "amount_usd": row[4],
                        "opened_at": row[5],
                        "max_pnl_usd": row[6] if row[6] is not None else 0.0
                    })
        return trades

    def get_paper_trading_stats(self, user_id: str) -> dict:
        balance = self.get_virtual_balance(user_id)
        stats = {"total_trades": 0, "winrate": 0.0, "total_pnl": 0.0, "balance": balance}
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*), SUM(pnl_usd) 
                    FROM paper_trades WHERE user_id = ? AND status = 'CLOSED'
                ''', (str(user_id),))
                row = cursor.fetchone()
                if row and row[0] > 0:
                    stats["total_trades"] = row[0]
                    stats["total_pnl"] = row[1] if row[1] else 0.0
                    
                    cursor.execute('''
                        SELECT COUNT(*) FROM paper_trades 
                        WHERE user_id = ? AND status = 'CLOSED' AND pnl_usd > 0
                    ''', (str(user_id),))
                    wins = cursor.fetchone()[0]
                    stats["winrate"] = round((wins / stats["total_trades"]) * 100, 2)
        return stats

    def get_user_paper_stats(self, user_id: str) -> dict:
        """Alias for get_paper_trading_stats."""
        return self.get_paper_trading_stats(user_id)

    # --- PUSH ALERTS METHODS ---

    def add_user_alert(self, user_id: str, symbol: str) -> bool:
        """Add a push alert for a symbol. Returns True if added, False if already exists."""
        created_at = datetime.datetime.utcnow().isoformat()
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute('''
                        INSERT INTO user_alerts (user_id, symbol, created_at)
                        VALUES (?, ?, ?)
                    ''', (str(user_id), symbol, created_at))
                    conn.commit()
                    return True
                except sqlite3.IntegrityError:
                    return False # Already exists due to UNIQUE constraint

    def remove_user_alert(self, user_id: str, symbol: str) -> bool:
        """Remove a push alert for a symbol. Returns True if removed."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM user_alerts WHERE user_id = ? AND symbol = ?', (str(user_id), symbol))
                deleted = cursor.rowcount > 0
                conn.commit()
                return deleted

    def get_user_alerts(self, user_id: str) -> list[str]:
        """Get list of symbols the user has alerts for."""
        alerts = []
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT symbol FROM user_alerts WHERE user_id = ?', (str(user_id),))
                for row in cursor.fetchall():
                    alerts.append(row[0])
        return alerts

    def get_users_with_alert(self, symbol: str) -> list[str]:
        """Get all user IDs monitoring a specific symbol."""
        users = []
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT user_id FROM user_alerts WHERE symbol = ?', (symbol,))
                for row in cursor.fetchall():
                    users.append(row[0])
        return users

    # --- ML SIGNALS METHODS ---

    def save_ml_signal(self, symbol: str, direction: str, confidence_score: float) -> None:
        created_at = datetime.datetime.utcnow().isoformat()
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO ml_signals_history (symbol, direction, confidence_score, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (symbol, direction, float(confidence_score), created_at))
                conn.commit()

    def get_recent_ml_signals(self, symbol: str, limit: int = 100) -> list[dict]:
        signals = []
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT direction, confidence_score, created_at
                    FROM ml_signals_history
                    WHERE symbol = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (symbol, limit))
                for row in cursor.fetchall():
                    signals.append({
                        "direction": row[0],
                        "confidence_score": row[1],
                        "created_at": row[2]
                    })
        return signals

    # --- VERIFIED SIGNALS METHODS ---

    def has_open_signal(self, symbol: str) -> bool:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT 1 FROM verified_signals_log WHERE symbol = ? AND status = "OPEN" LIMIT 1', (symbol,))
                return cursor.fetchone() is not None

    def record_verified_signal(
        self,
        symbol: str,
        direction: str,
        entry: float,
        target: float,
        sl: float,
        conf: float,
        channel_message_id: Optional[int] = None,
        channel_id: Optional[str] = None
    ) -> int:
        created_at = datetime.datetime.utcnow().isoformat()
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO verified_signals_log 
                    (symbol, direction, entry_price, target_price, stop_loss, confidence_score, status, channel_message_id, channel_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?)
                ''', (symbol, direction, entry, target, sl, float(conf), channel_message_id, channel_id, created_at))
                conn.commit()
                return cursor.lastrowid

    def update_signal_channel_message_id(self, signal_id: int, channel_message_id: int, channel_id: str) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE verified_signals_log
                    SET channel_message_id = ?, channel_id = ?
                    WHERE id = ?
                ''', (channel_message_id, channel_id, signal_id))
                conn.commit()

    def get_open_verified_signals(self) -> list[dict]:
        signals = []
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, symbol, direction, entry_price, target_price, stop_loss, confidence_score, created_at, channel_message_id, channel_id
                    FROM verified_signals_log
                    WHERE status = 'OPEN'
                ''')
                for row in cursor.fetchall():
                    signals.append({
                        "id": row[0],
                        "symbol": row[1],
                        "direction": row[2],
                        "entry_price": row[3],
                        "target_price": row[4],
                        "stop_loss": row[5],
                        "confidence_score": row[6],
                        "created_at": row[7],
                        "channel_message_id": row[8],
                        "channel_id": row[9]
                    })
        return signals

    def update_verified_signal_status(self, signal_id: int, status: str, exit_price: float) -> None:
        closed_at = datetime.datetime.utcnow().isoformat()
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE verified_signals_log
                    SET status = ?, exit_price = ?, closed_at = ?
                    WHERE id = ?
                ''', (status, exit_price, closed_at, signal_id))
                conn.commit()

    # --- FINE-TUNING TELEMETRY METHODS ---

    def record_fine_tuning_telemetry(
        self,
        symbol: str,
        timeframe: str,
        direction: str,
        confidence_score: float,
        predicted_target: float,
        actual_fill: Optional[float] = None,
        slippage_pct: Optional[float] = None,
        latency_ms: Optional[float] = None,
        feature_vector_json: Optional[str] = None
    ) -> int:
        created_at = datetime.datetime.utcnow().isoformat()
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO fine_tuning_telemetry
                    (symbol, timeframe, direction, confidence_score, predicted_target_price, actual_fill_price, slippage_pct, execution_latency_ms, feature_vector_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, timeframe, direction, float(confidence_score), float(predicted_target), actual_fill, slippage_pct, latency_ms, feature_vector_json, created_at))
                conn.commit()
                return cursor.lastrowid

    def get_all_users(self) -> list[str]:
        """Get a list of all known unique user IDs across all user tables in the database."""
        users = []
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT DISTINCT user_id FROM (
                        SELECT user_id FROM user_languages
                        UNION
                        SELECT user_id FROM daily_usage
                        UNION
                        SELECT user_id FROM user_subscriptions
                        UNION
                        SELECT user_id FROM user_alerts
                        UNION
                        SELECT user_id FROM virtual_balances
                        UNION
                        SELECT user_id FROM habits
                    ) WHERE user_id IS NOT NULL AND user_id != ''
                ''')
                for row in cursor.fetchall():
                    users.append(row[0])
        return users

    def get_active_traders(self) -> list[str]:
        """Returns list of active trader user IDs for paper/live trading automation."""
        users = self.get_all_users()
        if not users:
            # Fallback to default admin/system user if no users in DB yet
            return ["100000001"]
        return users

    # --- HOT ASYNC BACKUP & WAL CHECKPOINT METHODS ---

    def perform_wal_checkpoint(self) -> bool:
        """Performs a non-blocking PASSIVE WAL checkpoint to keep log file size optimal."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                    conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
                return True
            except Exception as e:
                import logging
                logger = logging.getLogger("SQLiteTracker")
                logger.error(f"WAL checkpoint failed: {e}")
                if "malformed" in str(e).lower():
                    logger.warning("Database disk image is malformed. Triggering auto-recovery...")
                    try:
                        import os
                        corrupt_path = f"{self.db_path}.corrupt_{int(time.time())}"
                        os.rename(self.db_path, corrupt_path)
                        logger.warning(f"Moved corrupted database to {corrupt_path}. Re-initializing clean DB...")
                        self._init_db()
                    except Exception as rec_err:
                        logger.error(f"Auto-recovery failed: {rec_err}")
                return False

    def backup_database(self, backup_dir: str = "data/backups") -> Optional[str]:
        """
        Executes a zero-lock online hot backup using SQLite native backup API.
        Creates a compressed snapshot with timestamp.
        """
        import os
        import gzip
        import shutil
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"mabuy_prod_backup_{timestamp}.db"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        with self._lock:
            try:
                # 1. Native Online Hot Backup via sqlite3.Connection.backup
                with sqlite3.connect(self.db_path, timeout=15.0) as src_conn:
                    with sqlite3.connect(backup_path) as dst_conn:
                        src_conn.backup(dst_conn)
                
                # 2. Compress backup to save disk space on VPS
                compressed_path = f"{backup_path}.gz"
                with open(backup_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove uncompressed temp file
                if os.path.exists(backup_path):
                    os.remove(backup_path)

                # 3. Retention policy: Keep last 24 backups
                all_backups = sorted(
                    [os.path.join(backup_dir, f) for f in os.listdir(backup_dir) if f.endswith(".gz")]
                )
                if len(all_backups) > 24:
                    for old_b in all_backups[:-24]:
                        try:
                            os.remove(old_b)
                        except OSError:
                            pass
                            
                return compressed_path
            except Exception as e:
                import logging
                logging.getLogger("SQLiteTracker").error(f"Online hot backup failed: {e}")
                return None

    # =========================================================================
    # --- HABIT TRACKER, FOCUS POMODORO, GAMIFICATION & MULTIMEDIA JOURNAL ---
    # =========================================================================

    @staticmethod
    def calculate_rank(xp: int) -> Dict[str, Any]:
        """
        Calculates user rank based on total XP (10-Level Legendary 1-Year Journey).
        1: NPC (0-999)
        2: Сайтама (1000-2999)
        3: Сигма (3000-7499)
        4: Гигачад (7500-14999)
        5: Грандмастер (15000-29999)
        6: Титан (30000-49999)
        7: Берсерк (50000-74999)
        8: Йохан (75000-99999)
        9: Истинный Воин (100000-149999)
        10: Фаундер (150000+ XP)
        """
        xp = max(0, int(xp))
        if xp < 1000:
            return {
                "lvl": 1,
                "title": "NPC",
                "icon": "🥔",
                "current_xp": xp,
                "min_xp": 0,
                "next_rank_xp": 1000,
                "next_title": "Сайтама",
                "progress_pct": min(100.0, round((xp / 1000.0) * 100, 1))
            }
        elif xp < 3000:
            return {
                "lvl": 2,
                "title": "Сайтама",
                "icon": "🥋",
                "current_xp": xp,
                "min_xp": 1000,
                "next_rank_xp": 3000,
                "next_title": "Сигма",
                "progress_pct": min(100.0, round(((xp - 1000) / 2000.0) * 100, 1))
            }
        elif xp < 7500:
            return {
                "lvl": 3,
                "title": "Сигма",
                "icon": "🗿",
                "current_xp": xp,
                "min_xp": 3000,
                "next_rank_xp": 7500,
                "next_title": "Гигачад",
                "progress_pct": min(100.0, round(((xp - 3000) / 4500.0) * 100, 1))
            }
        elif xp < 15000:
            return {
                "lvl": 4,
                "title": "Гигачад",
                "icon": "🗿",
                "current_xp": xp,
                "min_xp": 7500,
                "next_rank_xp": 15000,
                "next_title": "Грандмастер",
                "progress_pct": min(100.0, round(((xp - 7500) / 7500.0) * 100, 1))
            }
        elif xp < 30000:
            return {
                "lvl": 5,
                "title": "Грандмастер",
                "icon": "🧙‍♂️",
                "current_xp": xp,
                "min_xp": 15000,
                "next_rank_xp": 30000,
                "next_title": "Титан",
                "progress_pct": min(100.0, round(((xp - 15000) / 15000.0) * 100, 1))
            }
        elif xp < 50000:
            return {
                "lvl": 6,
                "title": "Титан",
                "icon": "🦾",
                "current_xp": xp,
                "min_xp": 30000,
                "next_rank_xp": 50000,
                "next_title": "Берсерк",
                "progress_pct": min(100.0, round(((xp - 30000) / 20000.0) * 100, 1))
            }
        elif xp < 75000:
            return {
                "lvl": 7,
                "title": "Берсерк",
                "icon": "🗡️",
                "current_xp": xp,
                "min_xp": 50000,
                "next_rank_xp": 75000,
                "next_title": "Йохан",
                "progress_pct": min(100.0, round(((xp - 50000) / 25000.0) * 100, 1))
            }
        elif xp < 100000:
            return {
                "lvl": 8,
                "title": "Йохан",
                "icon": "♟️",
                "current_xp": xp,
                "min_xp": 75000,
                "next_rank_xp": 100000,
                "next_title": "Истинный Воин",
                "progress_pct": min(100.0, round(((xp - 75000) / 25000.0) * 100, 1))
            }
        elif xp < 150000:
            return {
                "lvl": 9,
                "title": "Истинный Воин",
                "icon": "🛡️",
                "current_xp": xp,
                "min_xp": 100000,
                "next_rank_xp": 150000,
                "next_title": "Фаундер",
                "progress_pct": min(100.0, round(((xp - 100000) / 50000.0) * 100, 1))
            }
        else:
            return {
                "lvl": 10,
                "title": "Фаундер",
                "icon": "💼",
                "current_xp": xp,
                "min_xp": 150000,
                "next_rank_xp": 250000,
                "next_title": "Легендарный Фаундер",
                "progress_pct": 100.0
            }

    def init_default_academic_habits(self, user_id: str, force_reset: bool = False) -> List[Dict[str, Any]]:
        """Initializes preset academic & focus habits for user if none exist or if force_reset is True."""
        user_str = str(user_id)
        default_habits = [
            {
                "habit_id": "sat_math",
                "user_id": user_str,
                "title": "Digital SAT Math (Khan Academy & Desmos)",
                "category": "academic",
                "target_type": "minutes",
                "target_value": 30,
                "unit": "мин",
                "icon": "🧮",
                "color": "#38bdf8",
                "deadline_time": "16:15",
                "order_idx": 1,
                "is_archived": 0,
                "created_at": time.time()
            },
            {
                "habit_id": "ielts_prep",
                "user_id": user_str,
                "title": "IELTS Writing & Speaking",
                "category": "academic",
                "target_type": "minutes",
                "target_value": 30,
                "unit": "мин",
                "icon": "📚",
                "color": "#a855f7",
                "deadline_time": "17:00",
                "order_idx": 2,
                "is_archived": 0,
                "created_at": time.time()
            },
            {
                "habit_id": "ml_code",
                "user_id": user_str,
                "title": "Financial ML & AI Systems (Code/Project)",
                "category": "code",
                "target_type": "minutes",
                "target_value": 45,
                "unit": "мин",
                "icon": "💻",
                "color": "#22c55e",
                "deadline_time": "18:30",
                "order_idx": 3,
                "is_archived": 0,
                "created_at": time.time()
            },
            {
                "habit_id": "sport_steps",
                "user_id": user_str,
                "title": "Спорт & Активность (8000 шагов / тренировка)",
                "category": "health",
                "target_type": "checkbox",
                "target_value": 1,
                "unit": "раз",
                "icon": "🏃",
                "color": "#f97316",
                "deadline_time": "20:00",
                "order_idx": 4,
                "is_archived": 0,
                "created_at": time.time()
            },
            {
                "habit_id": "healthy_sleep",
                "user_id": user_str,
                "title": "Здоровый сон (8 часов)",
                "category": "health",
                "target_type": "checkbox",
                "target_value": 1,
                "unit": "раз",
                "icon": "😴",
                "color": "#6366f1",
                "deadline_time": "23:00",
                "order_idx": 5,
                "is_archived": 0,
                "created_at": time.time()
            }
        ]

        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM habits WHERE user_id = ? AND is_archived = 0', (user_str,))
                count = cursor.fetchone()[0]
                if count == 0 or force_reset:
                    if force_reset:
                        cursor.execute('DELETE FROM habits WHERE user_id = ?', (user_str,))
                    for h in default_habits:
                        cursor.execute('''
                            INSERT OR REPLACE INTO habits (
                                habit_id, user_id, title, category, target_type, target_value,
                                unit, icon, color, deadline_time, order_idx, is_archived, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            h["habit_id"], h["user_id"], h["title"], h["category"], h["target_type"],
                            h["target_value"], h["unit"], h["icon"], h["color"], h["deadline_time"],
                            h["order_idx"], h["is_archived"], h["created_at"]
                        ))
                    # Ensure user gamification record exists
                    cursor.execute('''
                        INSERT OR IGNORE INTO user_gamification (user_id, xp, streak_current, streak_best, last_active_date, rank_title)
                        VALUES (?, 0, 0, 0, '', 'NPC')
                    ''', (user_str,))
                    conn.commit()
        return self.get_user_habits(user_str)

    def get_user_habits(self, user_id: str, include_archived: bool = False) -> List[Dict[str, Any]]:
        """Retrieves all habits for a user."""
        user_str = str(user_id)
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                if include_archived:
                    cursor.execute('SELECT * FROM habits WHERE user_id = ? ORDER BY order_idx ASC, created_at ASC', (user_str,))
                else:
                    cursor.execute('SELECT * FROM habits WHERE user_id = ? AND is_archived = 0 ORDER BY order_idx ASC, created_at ASC', (user_str,))
                rows = cursor.fetchall()
                if rows or include_archived:
                    return [dict(r) for r in rows]
        
        # If no habits exist, init defaults outside lock and return
        return self.init_default_academic_habits(user_str)

    def save_habit(self, user_id: str, habit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Creates or updates a custom habit for user."""
        user_str = str(user_id)
        habit_id = str(habit_data.get("habit_id") or f"h_{uuid.uuid4().hex[:8]}")
        title = str(habit_data.get("title", "Новая привычка")).strip()
        category = str(habit_data.get("category", "custom")).strip()
        target_type = str(habit_data.get("target_type", "checkbox")).strip()
        target_value = int(habit_data.get("target_value", 1))
        unit = str(habit_data.get("unit", "раз" if target_type == "checkbox" else "мин")).strip()
        icon = str(habit_data.get("icon", "🎯")).strip()
        color = str(habit_data.get("color", "#38bdf8")).strip()
        deadline_time = str(habit_data.get("deadline_time", "")).strip()
        order_idx = int(habit_data.get("order_idx", 99))
        is_archived = 1 if habit_data.get("is_archived") else 0
        now_ts = time.time()

        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO habits (
                        habit_id, user_id, title, category, target_type, target_value,
                        unit, icon, color, deadline_time, order_idx, is_archived, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(habit_id) DO UPDATE SET
                        title=excluded.title,
                        category=excluded.category,
                        target_type=excluded.target_type,
                        target_value=excluded.target_value,
                        unit=excluded.unit,
                        icon=excluded.icon,
                        color=excluded.color,
                        deadline_time=excluded.deadline_time,
                        order_idx=excluded.order_idx,
                        is_archived=excluded.is_archived
                ''', (
                    habit_id, user_str, title, category, target_type, target_value,
                    unit, icon, color, deadline_time, order_idx, is_archived, now_ts
                ))
                conn.commit()
        return {
            "habit_id": habit_id,
            "user_id": user_str,
            "title": title,
            "category": category,
            "target_type": target_type,
            "target_value": target_value,
            "unit": unit,
            "icon": icon,
            "color": color,
            "deadline_time": deadline_time,
            "order_idx": order_idx,
            "is_archived": is_archived,
            "created_at": now_ts
        }

    def delete_habit(self, user_id: str, habit_id: str, soft_delete: bool = True) -> bool:
        """Deletes or archives a habit."""
        user_str = str(user_id)
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                if soft_delete:
                    cursor.execute('UPDATE habits SET is_archived = 1 WHERE user_id = ? AND habit_id = ?', (user_str, habit_id))
                else:
                    cursor.execute('DELETE FROM habits WHERE user_id = ? AND habit_id = ?', (user_str, habit_id))
                    cursor.execute('DELETE FROM habit_logs WHERE user_id = ? AND habit_id = ?', (user_str, habit_id))
                conn.commit()
                return cursor.rowcount > 0

    def log_habit_action(self, user_id: str, habit_id: str, date_str: Optional[str] = None,
                         value: float = 1.0, is_completed: Optional[bool] = None) -> Dict[str, Any]:
        """
        Logs habit completion or focus duration, rewards XP, and recalculates daily streaks.
        """
        user_str = str(user_id)
        date_key = date_str or self._get_today_str()
        now_ts = time.time()
        
        # Ensure default habits exist for user
        self.get_user_habits(user_str)

        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get habit definition
                cursor.execute('SELECT * FROM habits WHERE user_id = ? AND habit_id = ?', (user_str, habit_id))
                habit_row = cursor.fetchone()
                if not habit_row:
                    raise ValueError(f"Habit [{habit_id}] not found for user [{user_str}].")
                
                target_type = habit_row["target_type"]
                target_value = habit_row["target_value"]

                # Get previous log for today
                cursor.execute('SELECT * FROM habit_logs WHERE user_id = ? AND habit_id = ? AND date_str = ?',
                               (user_str, habit_id, date_key))
                prev_log = cursor.fetchone()
                prev_completed = bool(prev_log["is_completed"]) if prev_log else False
                prev_value = float(prev_log["value"]) if prev_log else 0.0

                # Determine new state
                if is_completed is not None:
                    new_completed = bool(is_completed)
                    new_value = float(value) if value is not None else (target_value if new_completed else 0.0)
                else:
                    # Toggle or increment
                    if target_type == "checkbox":
                        new_completed = not prev_completed
                        new_value = 1.0 if new_completed else 0.0
                    else:
                        # Minutes timer log
                        new_value = prev_value + float(value)
                        new_completed = new_value >= float(target_value)

                # Save log
                cursor.execute('''
                    INSERT INTO habit_logs (user_id, habit_id, date_str, value, is_completed, logged_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, habit_id, date_str) DO UPDATE SET
                        value=excluded.value,
                        is_completed=excluded.is_completed,
                        logged_at=excluded.logged_at
                ''', (user_str, habit_id, date_key, new_value, 1 if new_completed else 0, now_ts))

                # XP Calculation:
                # Base reward: +50 XP if just transitioned to completed
                # Focus timer minutes: +1 XP per minute tracked
                xp_gained = 0
                if new_completed and not prev_completed:
                    xp_gained += 50
                elif not new_completed and prev_completed:
                    xp_gained -= 50 # Uncheck reverses XP

                if target_type == "minutes" and value > 0 and (is_completed is None or not prev_completed):
                    xp_gained += int(value)

                # Fetch user gamification
                cursor.execute('SELECT * FROM user_gamification WHERE user_id = ?', (user_str,))
                gam_row = cursor.fetchone()
                current_xp = gam_row["xp"] if gam_row else 0
                streak_current = gam_row["streak_current"] if gam_row else 0
                streak_best = gam_row["streak_best"] if gam_row else 0
                last_active_date = gam_row["last_active_date"] if gam_row else ""

                new_xp = max(0, current_xp + xp_gained)

                # Check if all active habits completed for date
                cursor.execute('SELECT COUNT(*) FROM habits WHERE user_id = ? AND is_archived = 0', (user_str,))
                total_active = cursor.fetchone()[0]
                cursor.execute('''
                    SELECT COUNT(*) FROM habit_logs hl
                    JOIN habits h ON hl.habit_id = h.habit_id
                    WHERE hl.user_id = ? AND hl.date_str = ? AND hl.is_completed = 1 AND h.is_archived = 0
                ''', (user_str, date_key))
                completed_count = cursor.fetchone()[0]

                # Full day bonus
                if total_active > 0 and completed_count == total_active:
                    if not prev_completed:
                        new_xp += 100 # +100 XP Full Day Master Bonus

                # Streak management
                if completed_count > 0:
                    try:
                        cur_d = datetime.date.fromisoformat(date_key)
                        yesterday_str = (cur_d - datetime.timedelta(days=1)).isoformat()
                        if last_active_date == yesterday_str:
                            streak_current += 1
                            last_active_date = date_key
                        elif last_active_date != date_key:
                            streak_current = 1
                            last_active_date = date_key
                    except Exception:
                        streak_current = max(1, streak_current)
                        last_active_date = date_key
                    
                    streak_best = max(streak_best, streak_current)

                rank_info = self.calculate_rank(new_xp)
                cursor.execute('''
                    INSERT INTO user_gamification (user_id, xp, streak_current, streak_best, last_active_date, rank_title)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        xp=excluded.xp,
                        streak_current=excluded.streak_current,
                        streak_best=excluded.streak_best,
                        last_active_date=excluded.last_active_date,
                        rank_title=excluded.rank_title
                ''', (user_str, new_xp, streak_current, streak_best, last_active_date, rank_info["title"]))

                conn.commit()

        return self.get_today_habits_status(user_str, date_key)

    def get_today_habits_status(self, user_id: str, date_str: Optional[str] = None) -> Dict[str, Any]:
        """Returns full status of habits and gamification for a given date."""
        user_str = str(user_id)
        date_key = date_str or self._get_today_str()

        # Ensure habits exist
        habits = self.get_user_habits(user_str)

        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get logs for date
                cursor.execute('SELECT * FROM habit_logs WHERE user_id = ? AND date_str = ?', (user_str, date_key))
                logs = {r["habit_id"]: dict(r) for r in cursor.fetchall()}

                # Get gamification
                cursor.execute('SELECT * FROM user_gamification WHERE user_id = ?', (user_str,))
                gam_row = cursor.fetchone()
                if not gam_row:
                    cursor.execute('''
                        INSERT INTO user_gamification (user_id, xp, streak_current, streak_best, last_active_date, rank_title)
                        VALUES (?, 0, 0, 0, '', 'NPC')
                    ''', (user_str,))
                    conn.commit()
                    cursor.execute('SELECT * FROM user_gamification WHERE user_id = ?', (user_str,))
                    gam_row = cursor.fetchone()

        items = []
        completed_count = 0
        for h in habits:
            log = logs.get(h["habit_id"], {})
            val = float(log.get("value", 0.0))
            is_comp = bool(log.get("is_completed", 0))
            if is_comp:
                completed_count += 1
            items.append({
                **h,
                "logged_value": val,
                "is_completed": is_comp,
                "progress_pct": min(100.0, round((val / float(h["target_value"])) * 100, 1)) if h["target_value"] > 0 else (100.0 if is_comp else 0.0)
            })

        total = len(habits)
        day_pct = round((completed_count / total) * 100, 1) if total > 0 else 0.0
        gam_dict = dict(gam_row)
        rank_data = self.calculate_rank(gam_dict["xp"])

        return {
            "date": date_key,
            "total_habits": total,
            "completed_count": completed_count,
            "completion_percentage": day_pct,
            "all_completed": completed_count == total and total > 0,
            "habits": items,
            "gamification": {
                **gam_dict,
                "rank": rank_data
            }
        }

    def get_user_gamification(self, user_id: str) -> Dict[str, Any]:
        """Returns XP, Ranks, Streaks for user."""
        user_str = str(user_id)
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM user_gamification WHERE user_id = ?', (user_str,))
                row = cursor.fetchone()
                if not row:
                    self.init_default_academic_habits(user_str)
                    cursor.execute('SELECT * FROM user_gamification WHERE user_id = ?', (user_str,))
                    row = cursor.fetchone()
                gam_dict = dict(row)
                rank_info = self.calculate_rank(gam_dict["xp"])
                return {
                    **gam_dict,
                    "rank": rank_info
                }

    def add_user_xp(self, user_id: str, xp_delta: int) -> Dict[str, Any]:
        """Manually or programmatically adds XP to user."""
        user_str = str(user_id)
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM user_gamification WHERE user_id = ?', (user_str,))
                row = cursor.fetchone()
                current_xp = row["xp"] if row else 0
                new_xp = max(0, current_xp + xp_delta)
                rank_info = self.calculate_rank(new_xp)
                cursor.execute('''
                    INSERT INTO user_gamification (user_id, xp, streak_current, streak_best, last_active_date, rank_title)
                    VALUES (?, ?, 0, 0, '', ?)
                    ON CONFLICT(user_id) DO UPDATE SET xp = ?, rank_title = ?
                ''', (user_str, new_xp, rank_info["title"], new_xp, rank_info["title"]))
                conn.commit()
        return self.get_user_gamification(user_str)

    def get_activity_heatmap(self, user_id: str, days: int = 60) -> List[Dict[str, Any]]:
        """Returns GitHub-style activity matrix for past N days."""
        user_str = str(user_id)
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=days - 1)
        start_str = start_date.isoformat()

        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT date_str, COUNT(*) as completed_count
                    FROM habit_logs
                    WHERE user_id = ? AND date_str >= ? AND is_completed = 1
                    GROUP BY date_str
                ''', (user_str, start_str))
                completed_by_date = {r["date_str"]: r["completed_count"] for r in cursor.fetchall()}

                cursor.execute('SELECT COUNT(*) FROM habits WHERE user_id = ? AND is_archived = 0', (user_str,))
                total_active = cursor.fetchone()[0] or 5

        matrix = []
        for i in range(days):
            d = start_date + datetime.timedelta(days=i)
            d_str = d.isoformat()
            done = completed_by_date.get(d_str, 0)
            pct = round((done / total_active) * 100, 1) if total_active > 0 else 0
            
            # Level 0 to 4
            if done == 0:
                level = 0
            elif pct < 40:
                level = 1
            elif pct < 75:
                level = 2
            elif pct < 100:
                level = 3
            else:
                level = 4

            matrix.append({
                "date": d_str,
                "day_of_week": d.weekday(), # 0=Mon, 6=Sun
                "completed": done,
                "total": total_active,
                "pct": pct,
                "level": level
            })
        return matrix

    # --- MULTIMEDIA JOURNAL & MEMORY ALBUM METHODS ---

    def save_journal_entry(self, user_id: str, date_str: str, mood: str = '🔥',
                           title: str = '', content_text: str = '',
                           audio_path: str = '', audio_tracks: Optional[List[Dict[str, Any]]] = None,
                           media_paths: Optional[List[str]] = None,
                           entry_id: Optional[str] = None) -> Dict[str, Any]:
        """Saves or updates a daily journal entry with voice note, audio tracks playlist and media."""
        user_str = str(user_id)
        e_id = entry_id or f"j_{date_str}_{uuid.uuid4().hex[:6]}"
        now_ts = time.time()
        media_json = json.dumps(media_paths or [])

        # Fetch existing entry to preserve/merge audio tracks if needed
        existing = self.get_journal_entry_by_date(user_str, date_str)
        is_new = existing is None
        
        current_tracks = []
        if audio_tracks is not None:
            current_tracks = audio_tracks
        elif existing and existing.get("audio_tracks"):
            current_tracks = existing["audio_tracks"]
        elif audio_path:
            current_tracks = [{"id": f"trk_{uuid.uuid4().hex[:6]}", "url": audio_path, "duration": 0, "created_at": now_ts, "label": "Запись 1"}]

        # If audio_path provided and not in current_tracks, append it
        if audio_path and not any(t.get("url") == audio_path for t in current_tracks):
            current_tracks.append({
                "id": f"trk_{uuid.uuid4().hex[:6]}",
                "url": audio_path,
                "duration": 0,
                "created_at": now_ts,
                "label": f"Запись {len(current_tracks) + 1}"
            })

        audio_tracks_json = json.dumps(current_tracks)
        primary_audio = current_tracks[0]["url"] if current_tracks else audio_path

        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT entry_id FROM journal_entries WHERE user_id = ? AND (date_str = ? OR entry_id = ?)', 
                               (user_str, date_str, e_id))
                row = cursor.fetchone()
                
                if row:
                    e_id = row[0]
                    cursor.execute('''
                        UPDATE journal_entries SET
                            mood = ?, title = ?, content_text = ?,
                            audio_path = ?,
                            audio_tracks_json = ?,
                            media_paths_json = ?, updated_at = ?
                        WHERE entry_id = ? AND user_id = ?
                    ''', (mood, title, content_text, primary_audio, audio_tracks_json, media_json, now_ts, e_id, user_str))
                else:
                    cursor.execute('''
                        INSERT INTO journal_entries (
                            entry_id, user_id, date_str, mood, title, content_text,
                            audio_path, audio_tracks_json, media_paths_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (e_id, user_str, date_str, mood, title, content_text, primary_audio, audio_tracks_json, media_json, now_ts, now_ts))
                conn.commit()

        # Award +50 XP for writing in journal if new entry
        if is_new:
            self.add_user_xp(user_str, 50)

        return self.get_journal_entry_by_date(user_str, date_str) or {
            "entry_id": e_id,
            "user_id": user_str,
            "date_str": date_str,
            "mood": mood,
            "title": title,
            "content_text": content_text,
            "audio_path": primary_audio,
            "audio_tracks": current_tracks,
            "media_paths": media_paths or [],
            "created_at": now_ts,
            "updated_at": now_ts
        }

    def append_journal_audio(self, user_id: str, date_str: str, audio_path: str, duration: int = 0, label: str = "") -> Dict[str, Any]:
        """Appends a new voice note or audio file to today's playlist without overwriting existing tracks."""
        user_str = str(user_id)
        now_ts = time.time()
        entry = self.get_journal_entry_by_date(user_str, date_str)
        
        tracks = entry.get("audio_tracks", []) if entry else []
        track_number = len(tracks) + 1
        time_label = datetime.datetime.fromtimestamp(now_ts).strftime("%H:%M")
        track_label = label or f"Голосовая заметка {track_number} ({time_label})"

        new_track = {
            "id": f"trk_{uuid.uuid4().hex[:6]}",
            "url": audio_path,
            "duration": int(duration),
            "created_at": now_ts,
            "label": track_label
        }
        tracks.append(new_track)

        mood = entry.get("mood", "🔥") if entry else "🔥"
        title = entry.get("title", "") if entry else ""
        content_text = entry.get("content_text", "") if entry else ""
        media_paths = entry.get("media_paths", []) if entry else []

        saved = self.save_journal_entry(
            user_id=user_str,
            date_str=date_str,
            mood=mood,
            title=title,
            content_text=content_text,
            audio_path=tracks[0]["url"] if tracks else audio_path,
            audio_tracks=tracks,
            media_paths=media_paths,
            entry_id=entry.get("entry_id") if entry else None
        )
        
        # Award +50 XP for audio logging
        self.add_user_xp(user_str, 50)
        return saved

    def delete_journal_audio_track(self, user_id: str, date_str: str, track_id: str) -> Optional[Dict[str, Any]]:
        """Deletes a specific audio recording from a daily journal entry."""
        user_str = str(user_id)
        entry = self.get_journal_entry_by_date(user_str, date_str)
        if not entry:
            return None
        
        tracks = [t for t in entry.get("audio_tracks", []) if t.get("id") != track_id]
        primary_audio = tracks[0]["url"] if tracks else ""
        return self.save_journal_entry(
            user_id=user_str,
            date_str=date_str,
            mood=entry.get("mood", "🔥"),
            title=entry.get("title", ""),
            content_text=entry.get("content_text", ""),
            audio_path=primary_audio,
            audio_tracks=tracks,
            media_paths=entry.get("media_paths", []),
            entry_id=entry.get("entry_id")
        )

    def get_journal_entries(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieves chronological journal entries for Memory Album."""
        user_str = str(user_id)
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM journal_entries
                    WHERE user_id = ?
                    ORDER BY date_str DESC, created_at DESC
                    LIMIT ? OFFSET ?
                ''', (user_str, limit, offset))
                entries = []
                for r in cursor.fetchall():
                    item = dict(r)
                    try:
                        item["media_paths"] = json.loads(item.get("media_paths_json") or "[]")
                    except Exception:
                        item["media_paths"] = []
                    
                    try:
                        tracks = json.loads(item.get("audio_tracks_json") or "[]")
                    except Exception:
                        tracks = []
                    
                    # Fallback to legacy single audio_path if tracks array is empty
                    if not tracks and item.get("audio_path"):
                        tracks = [{
                            "id": "trk_legacy_1",
                            "url": item["audio_path"],
                            "duration": 0,
                            "created_at": item.get("created_at", time.time()),
                            "label": "Голосовая заметка"
                        }]
                    item["audio_tracks"] = tracks
                    entries.append(item)
                return entries

    def get_journal_entry_by_date(self, user_id: str, date_str: str) -> Optional[Dict[str, Any]]:
        """Finds entry for a specific date."""
        user_str = str(user_id)
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM journal_entries WHERE user_id = ? AND date_str = ?', (user_str, date_str))
                row = cursor.fetchone()
                if not row:
                    return None
                item = dict(row)
                try:
                    item["media_paths"] = json.loads(item.get("media_paths_json") or "[]")
                except Exception:
                    item["media_paths"] = []

                try:
                    tracks = json.loads(item.get("audio_tracks_json") or "[]")
                except Exception:
                    tracks = []
                
                if not tracks and item.get("audio_path"):
                    tracks = [{
                        "id": "trk_legacy_1",
                        "url": item["audio_path"],
                        "duration": 0,
                        "created_at": item.get("created_at", time.time()),
                        "label": "Голосовая заметка"
                    }]
                item["audio_tracks"] = tracks
                return item

    def delete_journal_entry(self, user_id: str, entry_id: str) -> bool:
        """Deletes a journal entry."""
        user_str = str(user_id)
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM journal_entries WHERE user_id = ? AND entry_id = ?', (user_str, entry_id))
                conn.commit()
                return cursor.rowcount > 0

    # --- ADVANCED HABIT STATISTICS & AI QUANT COACHING ---

    def get_habits_statistics(self, user_id: str, period: str = "week", lang: str = "ru") -> Dict[str, Any]:
        """
        Aggregates habit logs, focus time, consistency, and category metrics over Week (7d),
        Month (30d), and Year (365d), and enriches with AI Quant Coach analysis.
        """
        from src.domain.services.habit_ai_coach import HabitAICoachService
        user_str = str(user_id)
        period_clean = period.lower().strip()
        
        days_map = {"week": 7, "month": 30, "year": 365}
        days = days_map.get(period_clean, 7)
        
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=days - 1)
        start_str = start_date.isoformat()
        end_str = today.isoformat()

        habits = self.get_user_habits(user_str)
        habits_map = {h["habit_id"]: h for h in habits}
        total_active_habits = len(habits)

        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT habit_id, date_str, value, is_completed
                    FROM habit_logs
                    WHERE user_id = ? AND date_str >= ? AND date_str <= ?
                ''', (user_str, start_str, end_str))
                logs = [dict(r) for r in cursor.fetchall()]

        # Aggregations
        active_dates = set()
        completed_tasks_count = 0
        total_minutes_logged = 0
        habit_completion_counts: Dict[str, int] = {h["habit_id"]: 0 for h in habits}
        category_stats: Dict[str, Dict[str, Any]] = {
            "academic": {"title": "🧮 SAT & IELTS", "completed": 0, "minutes": 0, "icon": "📚"},
            "code": {"title": "💻 ML & AI Systems", "completed": 0, "minutes": 0, "icon": "💻"},
            "health": {"title": "🏃 Спорт & Сон", "completed": 0, "minutes": 0, "icon": "🏃"},
            "custom": {"title": "🎯 Личные задачи", "completed": 0, "minutes": 0, "icon": "🎯"}
        }

        for log in logs:
            h_id = log["habit_id"]
            h_meta = habits_map.get(h_id)
            is_done = bool(log.get("is_completed", 0))
            val = float(log.get("value", 0.0))

            if is_done:
                active_dates.add(log["date_str"])
                completed_tasks_count += 1
                habit_completion_counts[h_id] = habit_completion_counts.get(h_id, 0) + 1

            cat = h_meta.get("category", "custom") if h_meta else "custom"
            if cat not in category_stats:
                cat = "academic" if cat in ("sat", "ielts") else "custom"

            if is_done:
                category_stats[cat]["completed"] += 1

            if h_meta and h_meta.get("target_type") == "minutes":
                total_minutes_logged += int(val)
                category_stats[cat]["minutes"] += int(val)

        total_possible_tasks = max(1, days * total_active_habits)
        completion_rate = min(100.0, round((completed_tasks_count / total_possible_tasks) * 100.0, 1))
        consistency_score = min(100.0, round((len(active_dates) / max(1, days)) * 100.0, 1))

        # Identify Best & Weakest Habit
        best_id, best_count = max(habit_completion_counts.items(), key=lambda x: x[1], default=("", 0))
        weak_id, weak_count = min(habit_completion_counts.items(), key=lambda x: x[1], default=("", 0))

        best_habit = habits_map.get(best_id, {})
        weakest_habit = habits_map.get(weak_id, {})

        gamification = self.get_user_gamification(user_str)
        xp_earned_est = completed_tasks_count * 20 + (len(active_dates) * 50)

        raw_stats = {
            "period": period_clean,
            "days": days,
            "start_date": start_str,
            "end_date": end_str,
            "total_active_days": len(active_dates),
            "total_completed_tasks": completed_tasks_count,
            "total_possible_tasks": total_possible_tasks,
            "completion_rate": completion_rate,
            "consistency_score": consistency_score,
            "total_minutes_logged": total_minutes_logged,
            "total_hours_logged": round(total_minutes_logged / 60.0, 1),
            "xp_earned": xp_earned_est,
            "streak_current": gamification.get("streak_current", 0),
            "streak_best": gamification.get("streak_best", 0),
            "rank": gamification.get("rank", {}),
            "best_habit": {"id": best_id, "title": best_habit.get("title", ""), "completed": best_count, "icon": best_habit.get("icon", "🎯")},
            "weakest_habit": {"id": weak_id, "title": weakest_habit.get("title", ""), "completed": weak_count, "icon": weakest_habit.get("icon", "🎯")},
            "category_breakdown": category_stats
        }

        # Enrich with AI Quant Coach evaluation
        ai_insight = HabitAICoachService.generate_ai_coaching_insight(raw_stats, period=period_clean, lang=lang)
        raw_stats["ai_coach"] = ai_insight
        return raw_stats

    # --- ANTI-TILT & DISCIPLINE JOURNAL METHODS ---
    def record_anti_tilt_log(self, user_id: str, tag: str, pnl_usd: float = 0.0, followed_plan: bool = True, notes: str = "") -> int:
        """Records a new trader discipline/psychology journal log."""
        user_str = str(user_id)
        now_ts = time.time()
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO anti_tilt_logs (user_id, tag, pnl_usd, followed_plan, notes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (user_str, tag, pnl_usd, 1 if followed_plan else 0, notes, now_ts))
                conn.commit()
                return cursor.lastrowid

    def get_anti_tilt_logs(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetches chronological discipline logs for a user."""
        user_str = str(user_id)
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, user_id, tag, pnl_usd, followed_plan, notes, created_at
                    FROM anti_tilt_logs
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (user_str, limit))
                return [dict(row) for row in cursor.fetchall()]

    def set_user_last_tilt(self, user_id: str, timestamp: float) -> None:
        """Saves last tilt trigger timestamp for anti-tilt cooldown."""
        user_str = str(user_id)
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_anti_tilt_state (user_id, last_tilt_timestamp)
                    VALUES (?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET last_tilt_timestamp = excluded.last_tilt_timestamp
                ''', (user_str, timestamp))
                conn.commit()

    def get_user_last_tilt(self, user_id: str) -> float:
        """Gets last tilt timestamp."""
        user_str = str(user_id)
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT last_tilt_timestamp FROM user_anti_tilt_state WHERE user_id = ?', (user_str,))
                row = cursor.fetchone()
                return float(row[0]) if row else 0.0

    # --- STRATIFIED REGIME QUOTA PERSISTENCE ---
    def record_regime_trade(self, regime: str, symbol: str, is_win: bool, pnl_usd: float) -> int:
        """Records a forward-test trade under a specific market regime."""
        clean_regime = regime.upper().strip()
        clean_sym = symbol.upper().strip()
        now_ts = time.time()
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO regime_quota_logs (regime, symbol, is_win, pnl_usd, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (clean_regime, clean_sym, 1 if is_win else 0, float(pnl_usd), now_ts))
                conn.commit()
                return cursor.lastrowid

    def get_all_regime_trades(self) -> List[Dict[str, Any]]:
        """Returns all completed forward-test regime trades."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, regime, symbol, is_win, pnl_usd, created_at
                    FROM regime_quota_logs
                    ORDER BY created_at ASC
                ''')
                return [dict(row) for row in cursor.fetchall()]
