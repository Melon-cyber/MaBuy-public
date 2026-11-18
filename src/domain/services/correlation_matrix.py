import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from src.infrastructure.data_fetchers.binance_fetcher import CryptoDataFetcher
from src.presentation.telegram_bot.correlation_heatmap_generator import CorrelationHeatmapGenerator


class MarketCorrelationMatrixService:
    """
    Computes real-time pairwise return correlations and volatility covariance matrix
    across top cryptocurrency assets (BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX).
    Renders in-memory PNG Heatmap with systemic Black Swan crisis disclosure.
    """

    @staticmethod
    def get_correlation_matrix(lang: str = "ru", include_image: bool = True) -> Dict[str, Any]:
        symbols = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX"]
        fetcher = CryptoDataFetcher(max_cache_age_sec=3600)  # Cache 1d data for 1 hour

        # Fetch closing prices for the last 30 days (1d timeframe) - Rule of 3 Whales
        close_prices = {}
        for sym in symbols:
            try:
                df = fetcher.fetch_ohlcv(sym, timeframe="1d", limit=30)
                if df is not None and len(df) >= 15:
                    close_prices[sym] = df["close"].values
            except Exception as e:
                import logging
                logging.getLogger("MarketCorrelationMatrixService").warning(
                    f"Failed to fetch genuine candles for {sym}: {e}. Omitting asset from correlation matrix."
                )

        if len(close_prices) < 2:
            raise ValueError("Insufficient asset market data to compute correlation matrix (need at least 2 active assets).")

        active_symbols = list(close_prices.keys())

        # Create DataFrame and calculate log returns
        max_len = max(len(v) for v in close_prices.values())
        padded_prices = {k: np.pad(v, (max_len - len(v), 0), constant_values=np.nan) for k, v in close_prices.items()}
        df_closes = pd.DataFrame(padded_prices)
        
        returns = df_closes.pct_change().dropna()
        
        # Calculate Pearson correlation matrix
        corr_matrix = returns.corr().fillna(0.0).values
        
        pairs_summary = []
        for i in range(len(active_symbols)):
            for j in range(i + 1, len(active_symbols)):
                s1, s2 = active_symbols[i], active_symbols[j]
                val = corr_matrix[i, j]
                
                if val > 0.75:
                    status = "ВЫСОКАЯ (High)" if lang == "ru" else "HIGH CORRELATION"
                elif val > 0.50:
                    status = "УМЕРЕННАЯ (Moderate)" if lang == "ru" else "MODERATE"
                elif val < 0.20:
                    status = "ОЧЕНЬ НИЗКАЯ (Very Low)" if lang == "ru" else "VERY LOW"
                else:
                    status = "НИЗКАЯ (Low)" if lang == "ru" else "LOW CORRELATION"
                
                pairs_summary.append(f"• <b>{s1} / {s2}:</b> {val:.2f} ({status})")

        heatmap_bytes: Optional[bytes] = None
        if include_image:
            try:
                title = "КВАНТОВАЯ МАТРИЦА КОРРЕЛЯЦИЙ (30D)" if lang == "ru" else "QUANT CORRELATION MATRIX (30D)"
                heatmap_bytes = CorrelationHeatmapGenerator.generate_heatmap_bytes(active_symbols, corr_matrix, title=title)
            except Exception:
                heatmap_bytes = None

        return {
            "symbols": active_symbols,
            "matrix": corr_matrix.tolist(),
            "pairs_summary": pairs_summary,
            "heatmap_bytes": heatmap_bytes,
            "stress_disclosure": (
                "⚠️ <b>ВНИМАНИЕ:</b> Матрица отражает нормальный рынок. В периоды паники и сбросов (Black Swan) "
                "корреляции всех альткоинов к BTC стремятся к 1.0 (ρ → 1.0), и диверсификация перестает защищать."
                if lang == "ru" else
                "⚠️ <b>STRESS NOTE:</b> Matrix reflects normal conditions. During market crashes (Black Swan), "
                "all altcoin correlations to BTC surge towards 1.0 (ρ → 1.0), reducing diversification protection."
            )
        }
