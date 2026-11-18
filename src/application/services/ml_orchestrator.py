import asyncio
import logging
from typing import Optional, Set

from src.domain.entities.tick import MLSignalRequest, MLSignalResponse
from src.infrastructure.data_fetchers.ring_buffer import tick_buffer
from src.domain.services.quant_analyzer import QuantAnalyzer
from src.infrastructure.persistence.sqlite_tracker import SQLiteTracker

logger = logging.getLogger(__name__)

# Global state to cache the latest signal for the FastAPI endpoints
_latest_signal: Optional[MLSignalResponse] = None

def get_latest_signal() -> Optional[MLSignalResponse]:
    return _latest_signal

class SignalBroadcaster:
    def __init__(self):
        self.queues: Set[asyncio.Queue] = set()
        
    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue()
        self.queues.add(q)
        return q
        
    def unsubscribe(self, q: asyncio.Queue):
        if q in self.queues:
            self.queues.remove(q)
            
    def broadcast(self, signal: MLSignalResponse, is_instant: bool = False):
        payload = {"signal": signal, "is_instant": is_instant}
        for q in self.queues:
            q.put_nowait(payload)

broadcaster = SignalBroadcaster()
# Initialize DB Tracker for saving history
db_tracker = SQLiteTracker()

async def orchestrator_worker(symbol: str = "btcusdt", interval_seconds: float = 1.0, min_ticks_required: int = 5000):
    """
    Periodically checks the ring buffer, calls the QuantAnalyzer, and caches the result.
    """
    global _latest_signal
    
    analyzer = QuantAnalyzer()
    logger.info(f"Started ML Orchestrator for {symbol} every {interval_seconds}s. Warmup target: {min_ticks_required} ticks.")
    
    last_direction = None
    
    while True:
        try:
            # 1. Wait strictly on a timer
            await asyncio.sleep(interval_seconds)
            
            # 2. Get snapshot
            ticks = tick_buffer.get_snapshot()
            
            if len(ticks) < min_ticks_required:
                if len(ticks) % 500 == 0 and len(ticks) > 0:
                    logger.info(f"⏳ Warmup in progress: {len(ticks)}/{min_ticks_required} ticks collected...")
                continue
                
            # 3. Form Request
            request = MLSignalRequest(
                symbol=symbol,
                ticks=ticks
            )
            
            # 4. Call ML Model
            response = analyzer.analyze(request)
            
            # 5. Cache result & Save to DB
            _latest_signal = response
            db_tracker.save_ml_signal(symbol, response.direction, response.confidence_score)
            
            # 6. Check for instant trigger (direction changed)
            if last_direction is not None and response.direction != last_direction:
                logger.info(f"⚡ Instant Trigger: Signal changed from {last_direction} to {response.direction}!")
                broadcaster.broadcast(response, is_instant=True)
            
            last_direction = response.direction
            
        except Exception as e:
            logger.error(f"Error in ML orchestrator: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(orchestrator_worker())
