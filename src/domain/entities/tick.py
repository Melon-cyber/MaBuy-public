from pydantic import BaseModel, Field

class Tick(BaseModel):
    trade_id: int = Field(description="ID сделки биржи")
    timestamp: int = Field(description="Unix timestamp в миллисекундах")
    price: float = Field(description="Цена сделки")
    size: float = Field(description="Объем сделки (в базовой монете)")
    side: int = Field(description="1 для Buy (Taker Buy), -1 для Sell (Maker Buy)")

class MLSignalRequest(BaseModel):
    symbol: str = Field(description="Торговая пара, например BTCUSDT")
    ticks: list[Tick] = Field(description="Батч последних N сырых тиков")

class MLSignalResponse(BaseModel):
    direction: str = Field(description="BULLISH, BEARISH или NEUTRAL")
    confidence_score: float = Field(description="Вероятность от 0.0 до 1.0", ge=0.0, le=1.0)
