from abc import ABC, abstractmethod
import pandas as pd


class BaseDataFetcher(ABC):
    """
    Abstract Port for Market Data Ingestion.
    Ensures data fetchers return standardized pandas DataFrames with columns:
    ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 1000) -> pd.DataFrame:
        """
        Fetches historical OHLCV kline candles for the given symbol and timeframe.
        """
        pass
