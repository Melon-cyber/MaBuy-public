import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

class CorrelationHeatmapGenerator:
    """
    Renders an in-memory Cyber-Dark Correlation Heatmap matrix PNG.
    Features:
    - High-contrast Cyberpunk aesthetics (#07090E background, glowing values).
    - Clear Black Swan Crisis stress disclaimer.
    - Zero disk I/O, renders in <100ms.
    """

    @staticmethod
    def generate_heatmap_bytes(
        symbols: List[str],
        corr_matrix: np.ndarray,
        title: str = "MABUY ASSET CORRELATION MATRIX"
    ) -> bytes:
        plt.style.use('dark_background')
        n = len(symbols)

        fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=150, facecolor='#07090E')
        ax.set_facecolor('#07090E')

        # Customized colormap from deep indigo (#0a1128) to cyan (#00f2fe) to emerald (#00ff87)
        from matplotlib.colors import LinearSegmentedColormap
        colors = ["#0A1128", "#1C3144", "#0077B6", "#00B4D8", "#00F2FE", "#00FF87"]
        custom_cmap = LinearSegmentedColormap.from_list("cyber_heatmap", colors, N=256)

        cax = ax.imshow(corr_matrix, cmap=custom_cmap, vmin=-0.2, vmax=1.0, aspect='auto')

        # Add colorbar
        cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9, colors='#8E9BAE')
        cbar.set_label('Коэффициент корреляции (Pearson ρ)', color='#8E9BAE', fontsize=9, labelpad=8)

        # Set ticks & labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(symbols, fontsize=11, fontweight='bold', color='#FFFFFF')
        ax.set_yticklabels(symbols, fontsize=11, fontweight='bold', color='#FFFFFF')

        # Add numerical values inside cells
        for i in range(n):
            for j in range(n):
                val = corr_matrix[i, j]
                text_color = '#07090E' if val > 0.65 else '#FFFFFF'
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center",
                        color=text_color, fontsize=10, fontweight='bold')

        # Title & Subtitles
        ax.set_title(title, fontsize=14, fontweight='bold', color='#00F2FE', pad=22)

        # Stress Disclosure Footer Badge
        disclosure_text = "[STRESS NOTE] Обычный рынок. В моменты кризиса корреляции альткоинов к BTC стремятся к 1.0 (Black Swan)"
        fig.text(0.5, 0.02, disclosure_text, ha='center', fontsize=7.5, color='#FF9F43', style='italic')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08, top=0.92)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
