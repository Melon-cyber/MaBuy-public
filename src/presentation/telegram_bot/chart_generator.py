import os
import io
import time
import matplotlib
matplotlib.use('Agg') # Non-interactive fast background renderer
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union

from src.domain.entities.prediction import PredictionResult


class TelegramChartGenerator:
    """
    High-performance, in-memory chart PNG generator for Telegram bot prediction cards.
    Optimized for sub-100ms rendering using pre-configured figure objects and BytesIO.
    """

    @staticmethod
    def generate_prediction_chart_bytes(result: PredictionResult) -> bytes:
        """Renders prediction chart directly into in-memory PNG bytes without disk I/O."""
        symbol = result.symbol.upper()
        current_price = result.current_price
        target_price = result.predicted_target_price
        direction = result.direction
        tf = getattr(result, "timeframe", "1h")

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=120)
        fig.patch.set_facecolor('#07090E')
        ax.set_facecolor('#0F1420')

        # Fetch genuine Binance historical price trajectory (Rule of 3 Whales: Zero Stubs)
        num_points = 18
        price_history = None
        try:
            from src.infrastructure.data_fetchers.binance_fetcher import CryptoDataFetcher
            fetcher = CryptoDataFetcher(max_cache_age_sec=30)
            df = fetcher.fetch_ohlcv(symbol, timeframe=tf, limit=num_points)
            if df is not None and len(df) >= 2:
                closes = df["close"].values
                if len(closes) < num_points:
                    pad = np.full(num_points - len(closes), closes[0])
                    price_history = np.concatenate([pad, closes])
                else:
                    price_history = closes[-num_points:]
        except Exception as e:
            import logging
            logging.getLogger("TelegramChartGenerator").warning(
                f"Could not fetch live Binance candles for chart #{symbol}: {e}"
            )

        if price_history is None or len(price_history) == 0:
            price_history = np.full(num_points, current_price)

        x_hist = np.arange(num_points)
        x_pred = num_points

        line_color = '#06B6D4'  # Cyan for historical
        target_color = '#10B981' if direction == "BULLISH" else ('#EF4444' if direction == "BEARISH" else '#F59E0B')

        # Plot Historical Price Line
        ax.plot(x_hist, price_history, color=line_color, linewidth=2.2, label='Historical Price')
        ax.fill_between(x_hist, price_history.min() * 0.998, price_history, color=line_color, alpha=0.10)

        # Plot Forecast Projection Line
        ax.plot([x_hist[-1], x_pred], [price_history[-1], target_price], color=target_color,
                linewidth=2.2, linestyle='--', label=f'Target ({tf.upper()})')

        # Target Forecast Dot
        ax.scatter([x_pred], [target_price], color=target_color, s=100, zorder=5, edgecolors='#FFFFFF', linewidths=1.2)

        # Annotations & Titles
        sign_str = "+" if result.predicted_log_return >= 0 else ""
        ret_pct = result.predicted_log_return * 100
        title_text = f"MaBuy ML Engine • #{symbol}/USDT ({tf.upper()})\nCurrent: ${current_price:,.2f}  ➔  Target: ${target_price:,.2f} ({sign_str}{ret_pct:.2f}%)"
        ax.set_title(title_text, fontsize=11, fontweight='bold', pad=10, color='#F8FAFC')

        # Grid and Axes styling
        ax.grid(True, linestyle=':', alpha=0.2, color='#64748B')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#262F45')
        ax.spines['bottom'].set_color('#262F45')
        ax.tick_params(colors='#94A3B8', labelsize=8)

        # Watermark
        ax.text(0.98, 0.04, "Powered by MaBuy ML Engine (PyTorch + GBDT)",
                transform=ax.transAxes, fontsize=7.5, color='#64748B',
                ha='right', va='bottom', fontstyle='italic')

        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buffer.seek(0)
        return buffer.getvalue()

    @staticmethod
    def generate_prediction_chart(result: PredictionResult, output_path: str = "temp_chart.png") -> str:
        """Legacy helper saving chart bytes to file path for compatibility."""
        png_bytes = TelegramChartGenerator.generate_prediction_chart_bytes(result)
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        return output_path
