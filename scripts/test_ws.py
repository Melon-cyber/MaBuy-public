import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://localhost:8000/api/v1/signals/btcusdt/stream"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            for _ in range(10):
                message = await websocket.recv()
                print(f"Received: {message}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ws())
