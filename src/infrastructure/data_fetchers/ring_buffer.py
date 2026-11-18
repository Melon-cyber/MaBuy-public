from collections import deque
from typing import List, Optional
import threading

from src.domain.entities.tick import Tick

class TickRingBuffer:
    def __init__(self, maxlen: int = 30000):
        self._buffer = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add_tick(self, tick: Tick):
        with self._lock:
            self._buffer.append(tick)

    def get_snapshot(self) -> List[Tick]:
        """Returns a snapshot of the current buffer. Safe for concurrent access."""
        with self._lock:
            return list(self._buffer)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def is_full(self) -> bool:
        """Helper to check if we have enough ticks to start generating signals."""
        with self._lock:
            return len(self._buffer) == self._buffer.maxlen

# Global instance to be used by daemon and orchestrator
tick_buffer = TickRingBuffer(maxlen=2000)
