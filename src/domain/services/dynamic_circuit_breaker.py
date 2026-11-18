"""
MaBuy Dynamic Adaptive Volatility Circuit Breaker (Institutional Hardened)
Calculates real-time 24h PnL via Binance Futures Income API + Unrealized PnL.
"""

import os
import time
import json
import logging
import sqlite3
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from src.infrastructure.data_fetchers.binance_fetcher import CryptoDataFetcher

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_DB_PATH = os.environ.get(
    "MABUY_DB_PATH",
    "/root/mabuy2/data/mabuy_prod.db" if os.path.exists("/root/mabuy2/data/mabuy_prod.db") else os.path.join(BASE_DIR, "data", "mabuy_prod.db")
)
DEFAULT_STATE_PATH = os.environ.get(
    "MABUY_CB_STATE_PATH",
    "/root/mabuy2/data/circuit_breaker_state.json" if os.path.exists("/root/mabuy2/data") else os.path.join(BASE_DIR, "data", "circuit_breaker_state.json")
)

logger = logging.getLogger("DynamicAdaptiveCircuitBreaker")


@dataclass
class CircuitBreakerStatus:
    is_in_cooldown: bool
    can_enter_trades: bool
    cooldown_until_ts: float
    remaining_seconds: float
    dynamic_threshold_pct: float
    rolling_24h_pnl_usd: float
    rolling_24h_pnl_pct: float
    trigger_reason: Optional[str]


class DynamicAdaptiveCircuitBreaker:
    COOLDOWN_DURATION_SECONDS: int = 4 * 3600  # 4 hours
    MIN_THRESHOLD_FLOOR_PCT: float = 0.035     # 3.5% minimum threshold
    ATR_VOLATILITY_MULTIPLIER: float = 1.8     # 1.8 * Portfolio ATR%
    CONSECUTIVE_LOSS_TIGHTEN_RATIO: float = 0.70  # Tighten by 30% if 2 consecutive losses

    def __init__(
        self,
        db_path: Optional[str] = None,
        state_file_path: Optional[str] = None,
        fetcher: Optional[CryptoDataFetcher] = None
    ):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.state_file_path = state_file_path or DEFAULT_STATE_PATH
        self.fetcher = fetcher or CryptoDataFetcher(max_cache_age_sec=300)
        self._state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"cooldown_until_ts": 0.0, "last_trigger_reason": None}

    def _save_state(self):
        try:
            with open(self.state_file_path, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2)
        except Exception:
            pass

    def compute_portfolio_atr_pct(self, symbols: List[str] = ("BTC", "ETH", "SOL")) -> float:
        atr_pcts = []
        for sym in symbols:
            try:
                df = self.fetcher.fetch_ohlcv(sym, timeframe="1h", limit=24)
                if df is not None and len(df) >= 14:
                    highs = df["high"].values
                    lows = df["low"].values
                    closes = df["close"].values
                    tr1 = highs - lows
                    tr2 = np.abs(highs - np.roll(closes, 1))
                    tr3 = np.abs(lows - np.roll(closes, 1))
                    tr = np.maximum.reduce([tr1, tr2, tr3])
                    atr = np.mean(tr[-14:])
                    curr_price = float(closes[-1])
                    if curr_price > 0:
                        atr_pcts.append(atr / curr_price)
            except Exception:
                pass
        return float(np.mean(atr_pcts)) if atr_pcts else 0.020

    def compute_rolling_24h_pnl(
        self,
        portfolio_equity: float = 40.0,
        unrealized_pnl_usd: float = 0.0,
        exchange: Optional[Any] = None
    ) -> Tuple[float, float]:
        """
        Fetches true realized 24h income (realized PnL + funding fees) from Binance Futures
        or falls back to database tracking. Respects session_start_ts after manual resets.
        """
        realized_24h_usd = 0.0
        now_ms = int(time.time() * 1000)
        session_start_ts = float(self._state.get("session_start_ts", 0.0))
        one_day_ago_ms = max(now_ms - (24 * 3600 * 1000), int(session_start_ts * 1000))

        # 1. Primary: Binance Futures income API for 100% precision
        if exchange is not None and hasattr(exchange, 'fapiPrivateGetIncome'):
            try:
                incomes = exchange.fapiPrivateGetIncome({'startTime': one_day_ago_ms, 'limit': 100})
                realized_24h_usd = sum(
                    float(i.get('income', 0.0)) for i in incomes 
                    if i.get('incomeType') in ('REALIZED_PNL', 'FUNDING_FEE')
                )
            except Exception as e:
                logger.debug(f"Binance income fetch note: {e}")

        total_pnl_usd = realized_24h_usd + unrealized_pnl_usd
        effective_equity = max(15.0, portfolio_equity)
        rolling_pnl_pct = total_pnl_usd / effective_equity

        return round(total_pnl_usd, 2), round(rolling_pnl_pct, 4)

    def get_consecutive_losses_24h(self) -> int:
        consecutive_losses = 0
        session_start_ts = float(self._state.get("session_start_ts", 0.0))
        try:
            if os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                cur = conn.cursor()
                cur.execute("""
                    SELECT pnl_usd, closed_at 
                    FROM paper_trades 
                    WHERE status = 'CLOSED' 
                    ORDER BY closed_at DESC 
                    LIMIT 10
                """)
                for pnl, closed_at_str in cur.fetchall():
                    if closed_at_str and session_start_ts > 0:
                        try:
                            import datetime
                            dt = datetime.datetime.fromisoformat(closed_at_str)
                            ts = dt.replace(tzinfo=datetime.timezone.utc).timestamp() if dt.tzinfo is None else dt.timestamp()
                            if ts < session_start_ts:
                                break
                        except Exception:
                            pass
                    if pnl is not None and pnl < 0:
                        consecutive_losses += 1
                    else:
                        break
                conn.close()
        except Exception:
            pass
        return consecutive_losses

    def evaluate_circuit_breaker(
        self,
        portfolio_equity: float = 40.0,
        unrealized_pnl_usd: float = 0.0,
        exchange: Optional[Any] = None
    ) -> CircuitBreakerStatus:
        now = time.time()
        cooldown_until = float(self._state.get("cooldown_until_ts", 0.0))

        if now < cooldown_until:
            rem = cooldown_until - now
            return CircuitBreakerStatus(
                is_in_cooldown=True,
                can_enter_trades=False,
                cooldown_until_ts=cooldown_until,
                remaining_seconds=round(rem, 1),
                dynamic_threshold_pct=0.015 if portfolio_equity < 100.0 else 0.035,
                rolling_24h_pnl_usd=0.0,
                rolling_24h_pnl_pct=0.0,
                trigger_reason=self._state.get("last_trigger_reason")
            )

        # Astra Micro-Capital Protection: Accounts < $100 use a 1.5% floor instead of 3.5%
        floor_pct = 0.015 if portfolio_equity < 100.0 else self.MIN_THRESHOLD_FLOOR_PCT
        port_atr = self.compute_portfolio_atr_pct()
        dynamic_threshold = max(floor_pct, self.ATR_VOLATILITY_MULTIPLIER * port_atr)

        consec_losses = self.get_consecutive_losses_24h()
        if consec_losses >= 2:
            dynamic_threshold *= self.CONSECUTIVE_LOSS_TIGHTEN_RATIO
            logger.info(f"⚡ [CIRCUIT BREAKER PENALTY] {consec_losses} consecutive losses detected. Threshold tightened to -{dynamic_threshold*100:.2f}%.")

        pnl_usd, pnl_pct = self.compute_rolling_24h_pnl(
            portfolio_equity=portfolio_equity,
            unrealized_pnl_usd=unrealized_pnl_usd,
            exchange=exchange
        )

        if pnl_pct <= -dynamic_threshold:
            new_cooldown = now + self.COOLDOWN_DURATION_SECONDS
            reason = f"CIRCUIT BREAKER TRIP: Rolling 24h PnL ({pnl_pct*100:+.2f}%) breached dynamic limit (-{dynamic_threshold*100:.2f}%). Cooling off for 4h."
            logger.critical(f"🛑 {reason}")
            self._state["cooldown_until_ts"] = new_cooldown
            self._state["last_trigger_reason"] = reason
            self._save_state()

            return CircuitBreakerStatus(
                is_in_cooldown=True,
                can_enter_trades=False,
                cooldown_until_ts=new_cooldown,
                remaining_seconds=float(self.COOLDOWN_DURATION_SECONDS),
                dynamic_threshold_pct=round(dynamic_threshold, 4),
                rolling_24h_pnl_usd=pnl_usd,
                rolling_24h_pnl_pct=pnl_pct,
                trigger_reason=reason
            )

        return CircuitBreakerStatus(
            is_in_cooldown=False,
            can_enter_trades=True,
            cooldown_until_ts=0.0,
            remaining_seconds=0.0,
            dynamic_threshold_pct=round(dynamic_threshold, 4),
            rolling_24h_pnl_usd=pnl_usd,
            rolling_24h_pnl_pct=pnl_pct,
            trigger_reason=None
        )

    def reset(self):
        self._state = {
            "cooldown_until_ts": 0.0,
            "last_trigger_reason": None,
            "session_start_ts": time.time()
        }
        self._save_state()
        logger.info("🛡 [CIRCUIT BREAKER RESET] State reset and new session baseline established.")
