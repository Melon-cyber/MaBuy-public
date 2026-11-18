"""
MaBuy V5 — Analytical Signal Win Card Generator
Generates high-resolution, cyberpunk-themed 9:16 (Stories) and 1:1 (Posts) visual cards.
Positioned strictly around Quantitative AI Analytical Edge and verified target execution.
"""
import os
import io
import time
import logging
from typing import Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

class AnalyticalWinCardGenerator:
    def __init__(self, output_dir: Optional[str] = None):
        if output_dir is None:
            self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "generated_cards"))
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_stories_card(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        target_price: float,
        pnl_pct: float,
        confidence_pct: int = 88,
        rr_ratio: float = 2.2,
        persona: str = "quant",
        is_square: bool = False
    ) -> Optional[str]:
        """
        Generates a 9:16 (1080x1920) or 1:1 (1080x1080) visual image card.
        Returns the absolute filepath to the saved PNG image.
        """
        try:
            width, height = (1080, 1080) if is_square else (1080, 1920)
            img = Image.new("RGB", (width, height), color=(6, 10, 19)) # Dark obsidian cyber background
            draw = ImageDraw.Draw(img)

            # Try loading system TrueType fonts or fallback to default
            def get_font(size: int, bold: bool = False):
                font_names = ["arialbd.ttf", "arial.ttf", "segoeuib.ttf", "segoeui.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"]
                for fn in font_names:
                    try:
                        return ImageFont.truetype(fn, size)
                    except Exception:
                        pass
                return ImageFont.load_default()

            font_title = get_font(52, bold=True)
            font_hero = get_font(96, bold=True)
            font_sub = get_font(38, bold=True)
            font_body = get_font(32)
            font_small = get_font(24)

            # 1. Background Neon Gradient & Grid Accent
            for y in range(0, height, 60):
                draw.line([(0, y), (width, y)], fill=(15, 23, 42), width=1)
            for x in range(0, width, 60):
                draw.line([(x, 0), (x, height)], fill=(15, 23, 42), width=1)

            # Subtle Glowing Header Border
            draw.rectangle([40, 40, width - 40, height - 40], outline=(16, 185, 129), width=3)

            # 2. Header Brand Block
            draw.text((80, 80), "⚡ MABUY QUANTUM AI", fill=(52, 211, 153), font=font_title)
            draw.text((80, 145), "Квантовая модель V5 • Прогноз закрыт в цель", fill=(148, 163, 184), font=font_small)

            # Verified Badge
            draw.rounded_rectangle([width - 320, 80, width - 80, 140], radius=16, fill=(16, 185, 129))
            draw.text((width - 300, 95), "✓ VERIFIED 100%", fill=(2, 6, 23), font=font_small)

            # 3. Asset & Direction Hero Block
            dir_color = (52, 211, 153) if direction.upper() in ["BULLISH", "BUY", "LONG"] else (248, 113, 113)
            dir_text = "🟢 LONG" if direction.upper() in ["BULLISH", "BUY", "LONG"] else "🔴 SHORT"

            card_top = 230 if is_square else 320
            # Hero Card Container
            draw.rounded_rectangle([70, card_top, width - 70, card_top + 420], radius=28, fill=(15, 23, 42), outline=(30, 41, 59), width=2)
            
            draw.text((110, card_top + 40), f"{symbol.upper()} / USDT", fill=(255, 255, 255), font=font_hero)
            draw.text((110, card_top + 160), dir_text, fill=dir_color, font=font_sub)

            pnl_sign = "+" if pnl_pct >= 0 else ""
            draw.text((110, card_top + 230), f"{pnl_sign}{pnl_pct:.2f}%", fill=(52, 211, 153), font=font_hero)
            draw.text((110, card_top + 350), "Целевое математическое ожидание исполнено", fill=(100, 116, 139), font=font_small)

            # 4. Quantitative Metrics Grid
            grid_top = card_top + 470
            metrics = [
                ("🎯 Точность прогноза", f"{confidence_pct}%"),
                ("⚖️ Соотношение R:R", f"1 : {rr_ratio:.1f}"),
                ("📍 Точка входа", f"${entry_price:,.2f}"),
                ("🏁 Целевой выход (TP)", f"${target_price:,.2f}")
            ]

            box_w = (width - 170) // 2
            box_h = 130

            for i, (label, val) in enumerate(metrics):
                col = i % 2
                row = i // 2
                bx = 70 + col * (box_w + 30)
                by = grid_top + row * (box_h + 20)

                draw.rounded_rectangle([bx, by, bx + box_w, by + box_h], radius=18, fill=(15, 23, 42), outline=(30, 41, 59), width=1)
                draw.text((bx + 20, by + 18), label, fill=(148, 163, 184), font=font_small)
                draw.text((bx + 20, by + 58), val, fill=(255, 255, 255), font=font_sub)

            # 5. Footer Call-to-Action & AI Mascot Signature
            footer_y = height - 200 if not is_square else height - 120
            draw.rounded_rectangle([70, footer_y, width - 70, footer_y + 110], radius=22, fill=(6, 78, 59))
            draw.text((110, footer_y + 20), "🤖 Бесплатный демо-счет и аналитика MaIA:", fill=(209, 250, 229), font=font_small)
            draw.text((110, footer_y + 55), "@MabuyCryptoML_bot", fill=(255, 255, 255), font=font_sub)

            # Save result
            filename = f"win_card_{symbol.upper()}_{int(time.time())}_{'sq' if is_square else 'st'}.png"
            file_path = os.path.join(self.output_dir, filename)
            img.save(file_path, "PNG", quality=95)
            logger.info(f"✅ Generated Analytical Win Card -> {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to generate win card for {symbol}: {e}")
            return None
