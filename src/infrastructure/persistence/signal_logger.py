import os
import sqlite3
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from threading import Lock

logger = logging.getLogger(__name__)

MODEL_VERSION = "v5.2"


class SignalLogger:
    """Thread-safe SQLite logger for trading signals with SHA256 deduplication."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None or db_path == "/root/mabuy2/data/mabuy_prod.db":
            if os.path.exists("/root/mabuy2/data"):
                db_path = "/root/mabuy2/data/mabuy_prod.db"
            else:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
                db_path = os.path.join(project_root, "data", "mabuy_prod.db")
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_time     TEXT    NOT NULL,
                    symbol          TEXT    NOT NULL,
                    direction       TEXT    NOT NULL CHECK(direction IN ('BULLISH','BEARISH')),
                    confidence      REAL    NOT NULL,
                    model_version   TEXT    NOT NULL DEFAULT 'v5.2',
                    entry_price     REAL    NOT NULL,
                    exit_price      REAL,
                    pnl_usd         REAL,
                    close_reason    TEXT,
                    signal_hash     TEXT    UNIQUE NOT NULL,
                    telegram_msg_id INTEGER,
                    live_order_id   TEXT,
                    status          TEXT    NOT NULL DEFAULT 'OPEN'
                                        CHECK(status IN ('OPEN','CLOSED','CANCELLED')),
                    created_at      TEXT    NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_time ON signal_log(signal_time);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_status ON signal_log(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_symbol ON signal_log(symbol);")
            conn.commit()

    @staticmethod
    def _make_hash(signal_time: str, symbol: str, direction: str, confidence: float) -> str:
        raw = f"{signal_time}|{symbol.upper()}|{direction.upper()}|{confidence:.4f}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def log_signal(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        entry_price: float,
        model_version: str = MODEL_VERSION,
        signal_time: Optional[str] = None,
        status: str = "OPEN",
        close_reason: Optional[str] = None,
    ) -> int:
        dir_clean = direction.upper() if direction else ""
        if dir_clean not in ("BULLISH", "BEARISH"):
            logger.debug("[SignalLogger] Skipping non-directional signal for %s (%s)", symbol, direction)
            return -1

        now_utc = datetime.now(timezone.utc).isoformat()
        signal_time = signal_time or now_utc
        sig_hash = self._make_hash(signal_time, symbol, dir_clean, confidence)

        valid_status = status if status in ("OPEN", "CLOSED", "CANCELLED") else "CANCELLED"

        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            try:
                cur.execute("""
                    INSERT INTO signal_log
                        (signal_time, symbol, direction, confidence, model_version,
                         entry_price, signal_hash, status, close_reason, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (signal_time, symbol.upper(), dir_clean, confidence, model_version,
                      entry_price, sig_hash, valid_status, close_reason, now_utc))
                conn.commit()
                signal_id = int(cur.lastrowid)
                logger.info(
                    "[SignalLogger] #%d %s | %s %s conf=%.1f%% entry=%.4f (reason=%s)",
                    signal_id, valid_status, symbol, direction, confidence * 100, entry_price, close_reason or "N/A"
                )
                return signal_id

            except sqlite3.IntegrityError:
                cur.execute("SELECT id FROM signal_log WHERE signal_hash=?", (sig_hash,))
                row = cur.fetchone()
                if row:
                    eid = int(row["id"])
                    logger.warning("[SignalLogger] Duplicate signal → existing #%d", eid)
                    return eid
                raise

    def set_live_order(self, signal_id: int, order_id: str, fill_price: float) -> None:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE signal_log
                SET live_order_id=?, entry_price=?
                WHERE id=?
            """, (str(order_id), float(fill_price), signal_id))
            conn.commit()
            if cur.rowcount == 0:
                logger.warning("[SignalLogger] set_live_order: signal #%d not found", signal_id)
            else:
                logger.info("[SignalLogger] #%d linked to Live Binance Order %s @ %.4f",
                            signal_id, order_id, fill_price)

    def set_telegram_msg_id(self, signal_id: int, msg_id: int) -> None:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE signal_log SET telegram_msg_id=? WHERE id=?", (msg_id, signal_id))
            conn.commit()

    def close_signal(self, signal_id: int, exit_price: float, pnl_usd: float, close_reason: str) -> bool:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE signal_log
                SET exit_price=?, pnl_usd=?, close_reason=?, status='CLOSED'
                WHERE id=? AND status='OPEN'
            """, (exit_price, pnl_usd, close_reason, signal_id))
            conn.commit()
            if cur.rowcount == 0:
                return False
            pnl_tag = f"+${pnl_usd:.2f}" if pnl_usd >= 0 else f"-${abs(pnl_usd):.2f}"
            logger.info("[SignalLogger] #%d CLOSED | %s | reason=%s", signal_id, pnl_tag, close_reason)
            return True

    def cancel_signal(self, signal_id: int, reason: str = "MANUAL") -> bool:
        with self._lock, self._conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE signal_log SET status='CANCELLED', close_reason=?
                WHERE id=? AND status='OPEN'
            """, (reason, signal_id))
            conn.commit()
            return cur.rowcount > 0

    def find_open_signal(self, symbol: str, direction: str) -> Optional[int]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id FROM signal_log
                WHERE symbol=? AND direction=? AND status='OPEN'
                ORDER BY id DESC LIMIT 1
            """, (symbol.upper(), direction.upper()))
            row = cur.fetchone()
            return int(row["id"]) if row else None

    def get_open_signals(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM signal_log WHERE status='OPEN' ORDER BY id ASC")
            return [dict(r) for r in cur.fetchall()]

    def get_stats(self, days: int = 30) -> Dict[str, Any]:
        with self._conn() as conn:
            cur = conn.cursor()
            params: tuple = ()
            where = ""
            if days > 0:
                cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
                where  = "WHERE signal_time >= ?"
                params = (cutoff,)

            cur.execute(f"""
                SELECT
                    COUNT(*)                                                    AS total_signals,
                    AVG(confidence)                                             AS avg_confidence,
                    SUM(CASE WHEN status='OPEN'                      THEN 1 ELSE 0 END) AS open_signals,
                    SUM(CASE WHEN status='CLOSED' AND pnl_usd IS NOT NULL THEN 1 ELSE 0 END) AS closed_signals,
                    SUM(CASE WHEN status='CLOSED' AND pnl_usd  > 0  THEN 1 ELSE 0 END) AS winning_signals,
                    SUM(CASE WHEN status='CLOSED' AND pnl_usd <= 0  THEN 1 ELSE 0 END) AS losing_signals,
                    SUM(CASE WHEN status='CLOSED' AND pnl_usd  > 0  THEN pnl_usd ELSE 0.0 END) AS gross_profit,
                    SUM(CASE WHEN status='CLOSED' AND pnl_usd  < 0  THEN ABS(pnl_usd) ELSE 0.0 END) AS gross_loss,
                    SUM(CASE WHEN status='CLOSED' AND pnl_usd IS NOT NULL THEN pnl_usd ELSE 0.0 END) AS total_pnl
                FROM signal_log {where}
            """, params)

            r = cur.fetchone()
            closed  = int(r["closed_signals"] or 0)
            winning = int(r["winning_signals"] or 0)
            losing  = int(r["losing_signals"]  or 0)
            gross_p = float(r["gross_profit"] or 0.0)
            gross_l = float(r["gross_loss"]   or 0.0)
            pf = round(gross_p / gross_l, 2) if gross_l > 0 else (999.0 if gross_p > 0 else 0.0)
            avg_win = (gross_p / winning) if winning > 0 else 0.0
            avg_loss = (gross_l / losing) if losing > 0 else 0.0
            payoff = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0

            return {
                "total_signals":   int(r["total_signals"]   or 0),
                "win_rate":        round(winning / closed * 100.0, 2) if closed > 0 else 0.0,
                "profit_factor":   pf,
                "gross_profit":    round(gross_p, 2),
                "gross_loss":      round(gross_l, 2),
                "payoff_ratio":    payoff,
                "total_pnl":       round(float(r["total_pnl"]       or 0.0), 2),
                "avg_confidence":  round(float(r["avg_confidence"]   or 0.0), 4),
                "open_signals":    int(r["open_signals"]    or 0),
                "closed_signals":  closed,
                "winning_signals": winning,
                "losing_signals":  losing,
            }

