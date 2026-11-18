import time
import hashlib
import collections
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass(frozen=True)
class KillSwitchState:
    is_trading_allowed: bool
    state_code: str # "OPERATIONAL", "API_ERROR_HALT", "CONSECUTIVE_SL_HALT", "DAILY_DRAWDOWN_HALT", "MANUAL_EMERGENCY_HALT"
    cooldown_remaining_sec: float
    consecutive_sl_count: int
    api_errors_5m_count: int
    reason: Optional[str]


class HardwareKillSwitchService:
    """
    Multi-Tier Hardware & API Circuit Breaker (The Hard Kill-Switch).
    
    Protections:
    1. Idempotency Token: Generates deterministic clientOrderId preventing duplicate fills on 502/504 errors.
    2. 3-Strike API Error Halt: 3 API network errors within 5 minutes => 60-minute automatic trading freeze.
    3. Consecutive Stop-Loss Throttle: 3 consecutive SLs => 120-minute cooldown.
    4. Daily Drawdown Halt: -7.5% drop in 24 hours => 24-hour total freeze.
    5. Manual Emergency Kill-Switch: Immediate halt callable via Telegram / CLI.
    """

    def __init__(
        self,
        max_api_errors_5m: int = 3,
        max_consecutive_sl: int = 3,
        api_halt_duration_sec: float = 3600.0, # 1 hour
        sl_halt_duration_sec: float = 7200.0,   # 2 hours
        max_daily_drawdown_pct: float = -7.5
    ):
        self.max_api_errors_5m = max_api_errors_5m
        self.max_consecutive_sl = max_consecutive_sl
        self.api_halt_duration_sec = api_halt_duration_sec
        self.sl_halt_duration_sec = sl_halt_duration_sec
        self.max_daily_drawdown_pct = max_daily_drawdown_pct

        self._api_error_timestamps: collections.deque = collections.deque(maxlen=100)
        self._consecutive_sl_count: int = 0
        self._halt_until_timestamp: float = 0.0
        self._halt_reason: Optional[str] = None
        self._halt_state: str = "OPERATIONAL"
        self._manual_emergency_halt: bool = False

    def generate_idempotent_order_id(self, symbol: str, side: str, timestamp_ms: Optional[int] = None) -> str:
        """
        Generates deterministic 5-second bucketed clientOrderId.
        If exchange returns 502 Bad Gateway and retry occurs within 5s, exchange rejects duplicate fill.
        """
        ts = timestamp_ms or int(time.time() * 1000)
        bucket_5s = ts // 5000
        raw = f"MABUY_{symbol.upper()}_{side.upper()}_{bucket_5s}"
        short_hash = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8].upper()
        return f"MB_{symbol.upper()}_{short_hash}"

    def record_api_error(self, error_type: str, details: str = "") -> None:
        """Records an API failure and evaluates 3-strike threshold."""
        now = time.time()
        self._api_error_timestamps.append(now)

        # Count errors in the last 5 minutes (300s)
        recent_errors = [t for t in self._api_error_timestamps if (now - t) <= 300.0]
        if len(recent_errors) >= self.max_api_errors_5m:
            self._halt_until_timestamp = now + self.api_halt_duration_sec
            self._halt_state = "API_ERROR_HALT"
            self._halt_reason = (
                f"CRITICAL API HALT: {len(recent_errors)} API failures in 5 minutes (Last: {error_type} {details}). "
                f"Trading frozen for {self.api_halt_duration_sec/60:.0f} minutes."
            )

    def record_trade_outcome(self, is_win: bool) -> None:
        """Tracks consecutive stop losses."""
        now = time.time()
        if is_win:
            self._consecutive_sl_count = 0
        else:
            self._consecutive_sl_count += 1
            if self._consecutive_sl_count >= self.max_consecutive_sl:
                self._halt_until_timestamp = now + self.sl_halt_duration_sec
                self._halt_state = "CONSECUTIVE_SL_HALT"
                self._halt_reason = (
                    f"CONSECUTIVE SL HALT: {self._consecutive_sl_count} stop-losses in a row. "
                    f"Cooldown activated for {self.sl_halt_duration_sec/60:.0f} minutes."
                )

    def trigger_manual_emergency_halt(self, reason: str = "Manual Emergency Kill-Switch triggered by Operator") -> None:
        """Instant operator kill-switch."""
        self._manual_emergency_halt = True
        self._halt_state = "MANUAL_EMERGENCY_HALT"
        self._halt_reason = reason

    def reset_manual_halt(self) -> None:
        self._manual_emergency_halt = False
        self._halt_state = "OPERATIONAL"
        self._halt_reason = None

    def evaluate_state(self, current_daily_drawdown_pct: float = 0.0) -> KillSwitchState:
        now = time.time()

        if self._manual_emergency_halt:
            return KillSwitchState(
                is_trading_allowed=False,
                state_code="MANUAL_EMERGENCY_HALT",
                cooldown_remaining_sec=float("inf"),
                consecutive_sl_count=self._consecutive_sl_count,
                api_errors_5m_count=len([t for t in self._api_error_timestamps if (now - t) <= 300.0]),
                reason=self._halt_reason
            )

        if current_daily_drawdown_pct <= self.max_daily_drawdown_pct:
            return KillSwitchState(
                is_trading_allowed=False,
                state_code="DAILY_DRAWDOWN_HALT",
                cooldown_remaining_sec=86400.0,
                consecutive_sl_count=self._consecutive_sl_count,
                api_errors_5m_count=len([t for t in self._api_error_timestamps if (now - t) <= 300.0]),
                reason=f"DAILY DRAWDOWN HALT: Drawdown ({current_daily_drawdown_pct:.2f}%) <= limit ({self.max_daily_drawdown_pct}%)."
            )

        if now < self._halt_until_timestamp:
            remaining = round(self._halt_until_timestamp - now, 1)
            return KillSwitchState(
                is_trading_allowed=False,
                state_code=self._halt_state,
                cooldown_remaining_sec=remaining,
                consecutive_sl_count=self._consecutive_sl_count,
                api_errors_5m_count=len([t for t in self._api_error_timestamps if (now - t) <= 300.0]),
                reason=self._halt_reason
            )

        # Operational state
        recent_errs = len([t for t in self._api_error_timestamps if (now - t) <= 300.0])
        return KillSwitchState(
            is_trading_allowed=True,
            state_code="OPERATIONAL",
            cooldown_remaining_sec=0.0,
            consecutive_sl_count=self._consecutive_sl_count,
            api_errors_5m_count=recent_errs,
            reason=None
        )
