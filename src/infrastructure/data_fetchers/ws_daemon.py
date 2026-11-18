import asyncio
import json
import logging
import websockets
from websockets.exceptions import ConnectionClosed

from src.domain.entities.tick import Tick
from src.infrastructure.data_fetchers.ring_buffer import tick_buffer

logger = logging.getLogger(__name__)

BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/{}@trade"

async def binance_ws_worker(symbol: str = "btcusdt"):
    url = BINANCE_WS_URL.format(symbol)
    logger.info(f"Connecting to Binance WebSocket: {url}")
    
    while True:
        try:
            async with websockets.connect(url) as ws:
                logger.info(f"Connected to {url}")
                async for message in ws:
                    data = json.loads(message)
                    
                    # Binance Trade Payload mapping
                    # 't': trade_id, 'T': timestamp, 'p': price, 'q': quantity, 'm': is_buyer_maker
                    # If buyer is maker (m=True), the trade was initiated by a seller (side = -1)
                    # If buyer is NOT maker (m=False), the trade was initiated by a buyer (side = 1)
                    side = -1 if data.get("m", False) else 1
                    
                    tick = Tick(
                        trade_id=data["t"],
                        timestamp=data["T"],
                        price=float(data["p"]),
                        size=float(data["q"]),
                        side=side
                    )
                    
                    tick_buffer.add_tick(tick)
                    
        except ConnectionClosed:
            logger.warning("WebSocket connection closed. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error in WebSocket worker: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(binance_ws_worker())
