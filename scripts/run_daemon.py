import asyncio
import logging
import sys
import os

# Add project root to PYTHONPATH so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.infrastructure.data_fetchers.ws_daemon import binance_ws_worker
from src.application.services.ml_orchestrator import orchestrator_worker

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("daemon_main")
    
    logger.info("Starting Background Daemon (WebSocket Data Ingestion & ML Orchestrator)...")
    
    # Run both the websocket ingestion and the ML orchestrator concurrently
    await asyncio.gather(
        binance_ws_worker(symbol="btcusdt"),
        orchestrator_worker(symbol="BTCUSDT", interval_seconds=1.0, min_ticks_required=100)
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDaemon stopped by user.")
