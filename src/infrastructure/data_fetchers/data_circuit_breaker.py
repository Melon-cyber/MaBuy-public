"""
MaBuy V4 — Data Engineering Department
DataCircuitBreaker: Industrial Circuit Breaker for External Data Feeds (Binance API, RSS Feeds).
Prevents cascade failures, threads blocking, and protects live execution against stale feeds.
"""
import time
import threading
import logging
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "CLOSED"       # Normal operation: all requests go through
    OPEN = "OPEN"           # Tripped: bypass network requests, serve cache only
    HALF_OPEN = "HALF_OPEN" # Testing: single probe request to check recovery


class StaleDataException(Exception):
    """Raised when live trading requires fresh data, but circuit breaker is OPEN or feed is down."""
    pass


class DataCircuitBreaker:
    """
    Thread-safe, singleton-registry Circuit Breaker for market data feeds.
    """
    _instances: Dict[str, "DataCircuitBreaker"] = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        service_name: str,
        failure_threshold: int = 3,
        recovery_timeout_sec: float = 30.0
    ) -> "DataCircuitBreaker":
        """Singleton accessor per service domain (e.g., 'binance', 'rss')."""
        with cls._lock:
            if service_name not in cls._instances:
                cls._instances[service_name] = cls(
                    service_name=service_name,
                    failure_threshold=failure_threshold,
                    recovery_timeout_sec=recovery_timeout_sec
                )
            return cls._instances[service_name]

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 3,
        recovery_timeout_sec: float = 30.0
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_state_change = time.time()
        self._mutex = threading.Lock()

    def allow_request(self) -> bool:
        """
        Determines whether a live network request is permitted.
        Automatically transitions from OPEN to HALF_OPEN when cooldown expires.
        """
        with self._mutex:
            now = time.time()
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                if (now - self.last_state_change) >= self.recovery_timeout_sec:
                    logger.info(f"[{self.service_name}] Circuit cooldown expired. Entering HALF_OPEN probe state.")
                    self.state = CircuitState.HALF_OPEN
                    self.last_state_change = now
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                # In HALF_OPEN, allow probe
                return True
            return False

    def is_open(self) -> bool:
        with self._mutex:
            return self.state == CircuitState.OPEN

    def record_success(self):
        """Records a successful network request and resets the circuit to CLOSED."""
        with self._mutex:
            if self.state != CircuitState.CLOSED:
                logger.info(f"[{self.service_name}] Probe successful. Circuit Breaker reset to CLOSED.")
            self.failure_count = 0
            self.state = CircuitState.CLOSED
            self.last_state_change = time.time()

    def record_failure(self, error: Optional[Exception] = None):
        """Records a network failure. Trips the circuit to OPEN if threshold is exceeded."""
        with self._mutex:
            self.failure_count += 1
            now = time.time()
            err_msg = f" Error: {error}" if error else ""
            
            if self.state == CircuitState.HALF_OPEN:
                logger.warning(f"[{self.service_name}] Probe failed in HALF_OPEN state.{err_msg} Tripping circuit back to OPEN.")
                self.state = CircuitState.OPEN
                self.last_state_change = now
            elif self.failure_count >= self.failure_threshold:
                if self.state != CircuitState.OPEN:
                    logger.critical(
                        f"[{self.service_name}] {self.failure_count} consecutive failures reached.{err_msg} "
                        f"Tripping Circuit Breaker to OPEN for {self.recovery_timeout_sec}s."
                    )
                self.state = CircuitState.OPEN
                self.last_state_change = now

    def reset(self):
        """Manual reset for testing and recovery."""
        with self._mutex:
            self.failure_count = 0
            self.state = CircuitState.CLOSED
            self.last_state_change = time.time()
