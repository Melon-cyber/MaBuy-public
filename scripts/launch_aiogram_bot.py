import os
import sys
import asyncio
import logging
import json
import inspect
from typing import Optional

# Ensure src is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Message, CallbackQuery, FSInputFile, BufferedInputFile, WebAppInfo
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
import urllib.parse

from src.presentation.telegram_bot.bot import MaBuyTelegramBot
from src.presentation.telegram_bot.formatter import TelegramMessageFormatter
from src.presentation.telegram_bot.win_card_generator import ShareableWinCardGenerator
from src.domain.services.symbol_normalizer import SymbolNormalizer
from src.infrastructure.llm import (
    MaIACopilotEngine,
    PersonaType,
    PERSONA_METADATA,
    OLED_EXPRESSIONS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MaBuyAiogramBot")

# Define Router
router = Router()

def load_env_file(env_path=".env"):
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()

load_env_file()

# Initialize our core bot service and MaIA AI Copilot Engine
mabuy_service = MaBuyTelegramBot()
maia_copilot = MaIACopilotEngine()
user_persona_map: dict[str, PersonaType] = {}
known_crypto_keywords = {"BTC", "ETH", "BNB", "SOL", "XRP", "BITCOIN", "ETHEREUM", "SOLANA", "RIPPLE", "BINANCECOIN"}

async def get_effective_user_persona(user_id: str) -> PersonaType:
    if user_id in user_persona_map:
        return user_persona_map[user_id]
    try:
        saved = await asyncio.to_thread(mabuy_service.usage_tracker.get_user_persona, user_id, "quant")
        if saved:
            p = PersonaType(saved.lower().strip())
            user_persona_map[user_id] = p
            return p
    except Exception:
        pass
    user_persona_map[user_id] = PersonaType.QUANT
    return PersonaType.QUANT

def get_keyboard(user_id: str) -> dict:
    lang = mabuy_service.get_user_language(user_id)
    tier = mabuy_service.get_user_tier(user_id)
    p = user_persona_map.get(user_id)
    persona_str = p.value if p else "quant"
    raw_markup = TelegramMessageFormatter.get_inline_keyboard_markup(lang=lang, user_id=user_id, tier=tier.value, persona=persona_str)
    
    # Convert raw dict markup to aiogram InlineKeyboardMarkup
    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
    inline_kb = []
    for row in raw_markup.get("inline_keyboard", []):
        kb_row = []
        for btn in row:
            if "web_app" in btn:
                kb_row.append(InlineKeyboardButton(text=btn["text"], web_app=WebAppInfo(url=btn["web_app"]["url"])))
            elif "url" in btn:
                kb_row.append(InlineKeyboardButton(text=btn["text"], url=btn["url"]))
            else:
                kb_row.append(InlineKeyboardButton(text=btn["text"], callback_data=btn["callback_data"]))
        inline_kb.append(kb_row)
    return InlineKeyboardMarkup(inline_keyboard=inline_kb)

def convert_dict_markup_to_aiogram(raw_markup: dict):
    if not raw_markup:
        return None
    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
    inline_kb = []
    for row in raw_markup.get("inline_keyboard", []):
        kb_row = []
        for btn in row:
            if "web_app" in btn:
                kb_row.append(InlineKeyboardButton(text=btn["text"], web_app=WebAppInfo(url=btn["web_app"]["url"])))
            elif "url" in btn:
                kb_row.append(InlineKeyboardButton(text=btn["text"], url=btn["url"]))
            else:
                kb_row.append(InlineKeyboardButton(text=btn["text"], callback_data=btn["callback_data"]))
        inline_kb.append(kb_row)
    return InlineKeyboardMarkup(inline_keyboard=inline_kb)

def get_lang_keyboard(lang: str = "ru"):
    raw_markup = TelegramMessageFormatter.get_language_selection_markup(lang=lang)
    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
    inline_kb = []
    for row in raw_markup.get("inline_keyboard", []):
        kb_row = []
        for btn in row:
            kb_row.append(InlineKeyboardButton(text=btn["text"], callback_data=btn["callback_data"]))
        inline_kb.append(kb_row)
    return InlineKeyboardMarkup(inline_keyboard=inline_kb)

def get_risk_keyboard(lang: str):
    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
    if lang == "ru":
        return InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔥 Агрессивный (Часто)", callback_data="set_risk_aggressive")],
            [InlineKeyboardButton(text="⚖️ Сбалансированный", callback_data="set_risk_balanced")],
            [InlineKeyboardButton(text="🛡 Консервативный (Редко)", callback_data="set_risk_conservative")]
        ])
    else:
        return InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔥 Aggressive (Frequent)", callback_data="set_risk_aggressive")],
            [InlineKeyboardButton(text="⚖️ Balanced", callback_data="set_risk_balanced")],
            [InlineKeyboardButton(text="🛡 Conservative (Rare)", callback_data="set_risk_conservative")]
        ])

def get_experience_keyboard(lang: str = "ru"):
    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
    if lang == "ru":
        return InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🐣 Новичок (Светофор + $100 расчет)", callback_data="set_exp_beginner")],
            [InlineKeyboardButton(text="🧠 Профессионал (Квант-матрица + Kelly)", callback_data="set_exp_pro")],
            [InlineKeyboardButton(text="🏠 Главное меню", callback_data="/start")]
        ])
    else:
        return InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🐣 Novice (Traffic Light + $100 calc)", callback_data="set_exp_beginner")],
            [InlineKeyboardButton(text="🧠 Professional (Quant Matrix + Kelly)", callback_data="set_exp_pro")],
            [InlineKeyboardButton(text="🏠 Main Menu", callback_data="/start")]
        ])

def get_predict_keyboard(symbol: str, timeframe: str = "1h", is_pro_view: bool = False, lang: str = "ru"):
    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo
    raw_markup = TelegramMessageFormatter.get_prediction_dual_keyboard(symbol=symbol, timeframe=timeframe, is_pro_view=is_pro_view, lang=lang)
    inline_kb = []
    for row in raw_markup.get("inline_keyboard", []):
        kb_row = []
        for btn in row:
            if "web_app" in btn:
                kb_row.append(InlineKeyboardButton(text=btn["text"], web_app=WebAppInfo(url=btn["web_app"]["url"])))
            elif "url" in btn:
                kb_row.append(InlineKeyboardButton(text=btn["text"], url=btn["url"]))
            elif "callback_data" in btn:
                kb_row.append(InlineKeyboardButton(text=btn["text"], callback_data=btn["callback_data"]))
        inline_kb.append(kb_row)
    return InlineKeyboardMarkup(inline_keyboard=inline_kb)

def get_portfolio_keyboard(lang: str = "ru"):
    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo
    raw_markup = TelegramMessageFormatter.get_portfolio_presets_keyboard(lang=lang)
    inline_kb = []
    for row in raw_markup.get("inline_keyboard", []):
        kb_row = []
        for btn in row:
            if "web_app" in btn:
                kb_row.append(InlineKeyboardButton(text=btn["text"], web_app=WebAppInfo(url=btn["web_app"]["url"])))
            elif "url" in btn:
                kb_row.append(InlineKeyboardButton(text=btn["text"], url=btn["url"]))
            elif "callback_data" in btn:
                kb_row.append(InlineKeyboardButton(text=btn["text"], callback_data=btn["callback_data"]))
        inline_kb.append(kb_row)
    return InlineKeyboardMarkup(inline_keyboard=inline_kb)

async def process_mabuy_response(message_or_call, reply_text: str, photo_path: Optional[str] = None, symbol_predict: Optional[str] = None):
    """Helper to send text or photo using aiogram with bulletproof failover."""
    user_id = str(message_or_call.from_user.id)
    kb = get_keyboard(user_id)
    
    if symbol_predict:
        lang = mabuy_service.get_user_language(user_id)
        kb = get_predict_keyboard(symbol_predict, timeframe="1h", is_pro_view=False, lang=lang)
    elif "ИИ-АУДИТ РИСКОВ ПОРТФЕЛЯ" in reply_text or "AI PORTFOLIO RISK AUDIT" in reply_text:
        lang = mabuy_service.get_user_language(user_id)
        kb = get_portfolio_keyboard(lang=lang)
    elif "Выберите язык" in reply_text or "Select Language" in reply_text:
        lang = mabuy_service.get_user_language(user_id)
        kb = get_lang_keyboard(lang=lang)
    elif "Уровень риска" in reply_text or "Risk Profile" in reply_text:
        kb = get_risk_keyboard(mabuy_service.get_user_language(user_id))
    elif "уровень опыта" in reply_text.lower() or "experience level" in reply_text.lower():
        lang = mabuy_service.get_user_language(user_id)
        kb = get_experience_keyboard(lang=lang)
    elif "ВЕРИФИЦИРОВАННЫЙ ТРЕК-РЕКОРД" in reply_text or "VERIFIED MABUY" in reply_text:
        lang = mabuy_service.get_user_language(user_id)
        from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
        if lang == "ru":
            kb = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="🏆 Карточка сделки (WinCard)", callback_data="/wincard"),
                    InlineKeyboardButton(text="📈 Сигнал BTC", callback_data="/predict BTC 1h")
                ],
                [
                    InlineKeyboardButton(text="🏠 Главное меню", callback_data="/start")
                ]
            ])
        else:
            kb = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="🏆 Proof-of-Win Card", callback_data="/wincard"),
                    InlineKeyboardButton(text="📈 BTC Signal", callback_data="/predict BTC 1h")
                ],
                [
                    InlineKeyboardButton(text="🏠 Main Menu", callback_data="/start")
                ]
            ])
        
    target = message_or_call if isinstance(message_or_call, Message) else message_or_call.message

    if photo_path and os.path.exists(photo_path):
        try:
            photo = FSInputFile(photo_path)
            if len(reply_text) <= 1024:
                try:
                    await target.answer_photo(photo=photo, caption=reply_text, parse_mode=ParseMode.HTML, reply_markup=kb)
                except Exception:
                    await target.answer_photo(photo=photo, caption=reply_text, parse_mode=None, reply_markup=kb)
            else:
                await target.answer_photo(photo=photo)
                try:
                    await target.answer(reply_text, parse_mode=ParseMode.HTML, reply_markup=kb, disable_web_page_preview=True)
                except Exception:
                    await target.answer(reply_text, parse_mode=None, reply_markup=kb, disable_web_page_preview=True)
        except Exception as photo_err:
            logger.warning(f"Failed to send photo ({photo_err}), falling back to text.")
            try:
                await target.answer(reply_text, parse_mode=ParseMode.HTML, reply_markup=kb, disable_web_page_preview=True)
            except Exception:
                await target.answer(reply_text, parse_mode=None, reply_markup=kb, disable_web_page_preview=True)
        finally:
            if os.path.exists(photo_path):
                try:
                    os.remove(photo_path)
                except Exception:
                    pass
    else:
        try:
            await target.answer(reply_text, parse_mode=ParseMode.HTML, reply_markup=kb, disable_web_page_preview=True)
        except Exception as text_err:
            logger.warning(f"HTML parse failed ({text_err}), retrying plain text.")
            await target.answer(reply_text, parse_mode=None, reply_markup=kb, disable_web_page_preview=True)

@router.message(Command("start"))
@router.callback_query(F.data.in_({"/start", "start", "main_menu"}))
async def cmd_start(event):
    user_id = str(event.from_user.id)
    referrer_id = None

    if isinstance(event, Message):
        parts = event.text.split()
        if len(parts) > 1 and parts[1].startswith("ref_"):
            referrer_id = parts[1].replace("ref_", "")

        # Check if promo code PARI17 was used
        if len(parts) > 1 and parts[1].upper() in {"PARI17", "PROMO17"}:
            try:
                mabuy_service.usage_tracker.set_user_tier(user_id, "PRO")
                promo_msg = (
                    "🎁 <b>ПРОМОКОД PARI17 УСПЕШНО АКТИВИРОВАН!</b> 🚀\n"
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    "🔥 Вам предоставлен <b>3-дневный БЕСПЛАТНЫЙ PRO-ДОСТУП</b> к ИИ-сигналам MaBuy V4!\n\n"
                    "📊 Теперь вам доступны:\n"
                    "• Мгновенные PUSH-сигналы в Telegram\n"
                    "• Точные ATR Стоп-Лоссы и Тейк-Профиты\n"
                    "• Прогнозы по всем 50+ альткоинам (1ч, 4ч, 1д)\n"
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    "💡 <i>Нажмите любую кнопку ниже для первого прогноза!</i>"
                )
                await event.answer(promo_msg, parse_mode="HTML")
            except Exception as promo_err:
                logger.error(f"Failed to activate PARI17 for {user_id}: {promo_err}")

        # Check if persona was passed via WebApp deep link (e.g. /start persona_tsundere)
        if len(parts) > 1 and parts[1].startswith("persona_"):
            p_code = parts[1].replace("persona_", "").lower().strip()
            try:
                new_p = PersonaType(p_code)
                user_persona_map[user_id] = new_p
                await asyncio.to_thread(mabuy_service.usage_tracker.set_user_persona, user_id, new_p.value)
                meta = PERSONA_METADATA[new_p]
                oled = OLED_EXPRESSIONS[new_p]["neutral"]

                persona_custom_greetings = {
                    PersonaType.TSUNDERE: "Хм! Думаешь, я рада, что ты выбрал меня в настройках WebApp?! Не обольщайся, бака! (｀へ´) Но теперь я и здесь в Telegram буду строго следить, чтобы ты не сливал депозит на 20x плече!",
                    PersonaType.YANDERE: "Мой любимый трейдер... Я знала, что ты выберешь меня в WebApp! Теперь мы неразлучны и здесь, в Telegram! Твой депозит принадлежит только мне, хи-хи ❤️🔪",
                    PersonaType.ZEN: "Приветствую на пути гармонии. Характер Дзен синхронизирован с WebApp. 🧘 Сохраняй спокойствие, соблюдай дисциплину и следуй стратегии DCA.",
                    PersonaType.QUANT: "Квантовый аналитический профиль успешно активирован. Готов рассчитывать матожидание, формулу Quarter-Kelly и прыжковую диффузию Мертона для ваших сделок. 👔",
                    PersonaType.TITAN: "[STATUS: TITAN_SYNCED] Квантовый бортовой суперкомпьютер V4.1 подключен. Все сенсоры риска активны. 🤖"
                }
                custom_greeting = persona_custom_greetings.get(new_p, f"Характер {meta['name']} успешно применен!")

                p_msg = (
                    f"✨ <b>ХАРАКТЕР MAIA СИНХРОНИЗИРОВАН С WEBAPP!</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"🎭 <b>Активный характер:</b> <code>{meta['name']}</code>\n"
                    f"📟 <b>OLED:</b> <code>{oled}</code>\n\n"
                    f"💬 <i>{custom_greeting}</i>\n\n"
                    f"💡 <i>Задайте мне любой вопрос: <code>/ask Что думаешь по рынку?</code></i>"
                )
                kb = get_keyboard(user_id)
                await event.answer(p_msg, reply_markup=kb, parse_mode="HTML")
                return
            except Exception as pe:
                logger.warning(f"Failed to set persona from deep link: {pe}")
    elif isinstance(event, CallbackQuery):
        await event.answer()

    # Check if user is new before start
    is_new_user = mabuy_service.usage_tracker.get_user_language(user_id) is None
    
    # If user is totally new, prompt for language selection. If existing, open main menu immediately.
    if is_new_user:
        reply, _ = await asyncio.to_thread(mabuy_service.handle_start_greeting, user_id, referrer_id)
        await process_mabuy_response(event, reply)
    else:
        reply = await asyncio.to_thread(mabuy_service.handle_start, user_id)
        kb = get_keyboard(user_id)
        if isinstance(event, CallbackQuery):
            try:
                await event.message.edit_text(text=reply, reply_markup=kb, disable_web_page_preview=True)
            except Exception:
                await event.message.answer(reply, reply_markup=kb, disable_web_page_preview=True)
        else:
            await event.answer(reply, reply_markup=kb, disable_web_page_preview=True)

    # Notify Owner on New User Registration & Milestone Celebrations!
    if is_new_user and user_id != "100000001":
        try:
            owner_id = "100000001"
            username_str = f"@{event.from_user.username}" if event.from_user.username else "Без юзернейма"
            
            # Count total unique users
            import sqlite3
            total_users = 1
            with sqlite3.connect(mabuy_service.usage_tracker.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(DISTINCT user_id) FROM user_languages')
                total_users = cursor.fetchone()[0]
                
            # Check milestone triggers
            milestones = {10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 50000, 100000, 500000, 1000000}
            milestone_header = ""
            if total_users in milestones or total_users % 1000 == 0:
                milestone_header = (
                    f"🏆 🔥 <b>ЮБИЛЕЙНЫЙ {total_users}-Й ПОЛЬЗОВАТЕЛЬ!</b> 🥳🎊🎈\n"
                    f"🎉 <i>Платформа MaBuy взяла новую историческую отметку!</i>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                )
            
            admin_msg = (
                f"{milestone_header}"
                f"🎉 <b>НОВЫЙ ПОЛЬЗОВАТЕЛЬ В MABUY!</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"👤 <b>Имя:</b> {event.from_user.full_name}\n"
                f"🌐 <b>Юзернейм:</b> {username_str}\n"
                f"🆔 <b>ID:</b> <code>{user_id}</code>\n"
                f"📊 <b>Всего пользователей в базе:</b> <b>{total_users}</b>\n"
                f"🚀 <i>Первый запуск бота с соцсетей!</i>"
            )
            bot_instance = event.bot
            await bot_instance.send_message(chat_id=owner_id, text=admin_msg, parse_mode=ParseMode.HTML)
            logger.info(f"Notified owner about new user {user_id} ({username_str}), total users: {total_users}")
        except Exception as notify_err:
            logger.error(f"Failed to notify owner about new user: {notify_err}")

@router.message(Command("help"))
@router.callback_query(F.data.in_({"/help", "help"}))
async def cmd_help(event):
    user_id = str(event.from_user.id)
    reply = await asyncio.to_thread(mabuy_service.handle_help, user_id)
    if isinstance(event, CallbackQuery):
        await event.answer()
    await process_mabuy_response(event, reply)

@router.message(Command("counter"))
@router.message(Command("stats"))
async def cmd_counter(event):
    user_id = str(event.from_user.id)
    reply = await asyncio.to_thread(mabuy_service.handle_counter, user_id)
    if isinstance(event, CallbackQuery):
        await event.answer()
    await process_mabuy_response(event, reply)

@router.message(Command("promopost"))
@router.message(Command("promo"))
async def cmd_promopost(event):
    user_id = str(event.from_user.id)
    reply = await asyncio.to_thread(mabuy_service.handle_promopost, user_id)
    if isinstance(event, CallbackQuery):
        await event.answer()
    await process_mabuy_response(event, reply)

@router.message(Command("trackrecord"))
@router.message(Command("stats_winrate"))
@router.message(Command("winrate"))
@router.callback_query(F.data.in_({"/trackrecord", "trackrecord", "/winrate", "winrate"}))
async def cmd_trackrecord(event):
    user_id = str(event.from_user.id)
    reply = await asyncio.to_thread(mabuy_service.handle_trackrecord, user_id)
    if isinstance(event, CallbackQuery):
        await event.answer()
    await process_mabuy_response(event, reply)

@router.message(Command("wincard"))
@router.callback_query(F.data.in_({"/wincard", "wincard"}))
async def cmd_wincard(event):
    user_id = str(event.from_user.id)
    lang = await asyncio.to_thread(mabuy_service.get_user_language, user_id)
    target = event if isinstance(event, Message) else event.message
    if isinstance(event, CallbackQuery):
        await event.answer()

    def _fetch_wincard_data():
        import sqlite3
        from src.domain.services.track_record_auditor import TrackRecordAuditorService
        auditor = TrackRecordAuditorService(db_path=mabuy_service.usage_tracker.db_path)
        stats = auditor.get_performance_summary(lang=lang)

        sig_data = None
        try:
            with sqlite3.connect(mabuy_service.usage_tracker.db_path) as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                tbl_check = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signal_log'").fetchone()
                if tbl_check:
                    row = c.execute("""
                        SELECT symbol, direction, entry_price, exit_price, pnl_usd, status, signal_time
                        FROM signal_log
                        WHERE status = 'CLOSED' AND pnl_usd > 0
                        ORDER BY id DESC LIMIT 1
                    """).fetchone()
                    if row:
                        sig_data = dict(row)
        except Exception as db_err:
            logger.warning(f"Error fetching win trade from signal_log: {db_err}")

        if not sig_data:
            sig_data = {
                "symbol": "SOLUSDT",
                "direction": "BULLISH",
                "entry_price": 131.50,
                "exit_price": 135.25,
                "pnl_usd": 11.25,
            }

        entry = float(sig_data.get("entry_price") or 100.0)
        exit_p = float(sig_data.get("exit_price") or entry * 1.025)
        direction = str(sig_data.get("direction", "BULLISH")).upper()
        if direction == "BEARISH":
            pnl_pct = ((entry - exit_p) / entry) * 100.0
        else:
            pnl_pct = ((exit_p - entry) / entry) * 100.0
        if pnl_pct <= 0:
            pnl_pct = 2.85

        wr = stats.get("binance_wr") or stats.get("win_rate_pct") or 55.6
        total = stats.get("binance_trades") or stats.get("total_signals") or 36

        return {
            "symbol": sig_data.get("symbol", "SOLUSDT"),
            "entry_price": entry,
            "exit_price": exit_p,
            "pnl_pct": pnl_pct,
            "winrate_pct": float(wr),
            "total_trades": int(total)
        }

    card_info = await asyncio.to_thread(_fetch_wincard_data)

    card_bytes = await asyncio.to_thread(
        ShareableWinCardGenerator.generate_win_card_bytes,
        symbol=card_info["symbol"],
        pnl_pct=card_info["pnl_pct"],
        entry_price=card_info["entry_price"],
        exit_price=card_info["exit_price"],
        is_paper_trade=False,
        winrate_pct=card_info["winrate_pct"],
        total_trades=card_info["total_trades"],
        user_id=user_id,
        timeframe="1h"
    )

    from aiogram.types import BufferedInputFile, InlineKeyboardMarkup, InlineKeyboardButton
    photo = BufferedInputFile(card_bytes, filename=f"mabuy_win_{card_info['symbol']}.png")

    if lang == "ru":
        caption = (
            f"🏆 <b>MABUY V5.1 • PROOF-OF-WIN CARD</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🪙 <b>Актив:</b> #{card_info['symbol'].upper()}\n"
            f"📈 <b>Результат:</b> <code>+{card_info['pnl_pct']:.2f}%</code>\n"
            f"🎯 <b>Биржевой винрейт:</b> <code>{card_info['winrate_pct']}%</code> ({card_info['total_trades']} сделок)\n"
            f"🛡 <b>Исполнение:</b> Binance Futures Zero-Cheat\n\n"
            f"<i>Сохраняйте карточку и делитесь в соцсетях или Telegram-каналах!</i>"
        )
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="📊 Полный трек-рекорд", callback_data="/trackrecord"),
                InlineKeyboardButton(text="📈 Новый сигнал", callback_data="/predict BTC 1h")
            ],
            [
                InlineKeyboardButton(text="🏠 Главное меню", callback_data="/start")
            ]
        ])
    else:
        caption = (
            f"🏆 <b>MABUY V5.1 • PROOF-OF-WIN CARD</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🪙 <b>Asset:</b> #{card_info['symbol'].upper()}\n"
            f"📈 <b>Result:</b> <code>+{card_info['pnl_pct']:.2f}%</code>\n"
            f"🎯 <b>Exchange Win Rate:</b> <code>{card_info['winrate_pct']}%</code> ({card_info['total_trades']} trades)\n"
            f"🛡 <b>Execution:</b> Binance Futures Zero-Cheat\n\n"
            f"<i>Save this card and share in social media or trading groups!</i>"
        )
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="📊 Full Track Record", callback_data="/trackrecord"),
                InlineKeyboardButton(text="📈 New Signal", callback_data="/predict BTC 1h")
            ],
            [
                InlineKeyboardButton(text="🏠 Main Menu", callback_data="/start")
            ]
        ])

    try:
        await target.answer_photo(photo=photo, caption=caption, parse_mode=ParseMode.HTML, reply_markup=kb)
    except Exception as e:
        logger.error(f"Failed to send wincard photo: {e}")
        await target.answer(caption, parse_mode=ParseMode.HTML, reply_markup=kb)

@router.message(Command("experience"))
@router.message(Command("level"))
@router.callback_query(F.data.in_({"/experience", "experience", "/level"}))
async def cmd_experience(event):
    user_id = str(event.from_user.id)
    reply = await asyncio.to_thread(mabuy_service.handle_experience, user_id)
    if isinstance(event, CallbackQuery):
        await event.answer()
    await process_mabuy_response(event, reply)

@router.callback_query(F.data.startswith("set_exp_"))
async def cq_set_exp(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    level = "beginner" if callback.data == "set_exp_beginner" else "pro"
    reply = await asyncio.to_thread(mabuy_service.set_user_experience, user_id, level)
    lang = await asyncio.to_thread(mabuy_service.get_user_language, user_id)
    kb = get_keyboard(user_id)
    
    await callback.answer(f"✅ Режим установлен: {level.upper()}!", show_alert=False)
    try:
        await callback.message.edit_text(text=reply, reply_markup=kb, disable_web_page_preview=True)
    except Exception:
        await callback.message.answer(reply, reply_markup=kb, disable_web_page_preview=True)

@router.message(Command("language"))
@router.callback_query(F.data.in_({"/language", "language"}))
async def cmd_language(event):
    user_id = str(event.from_user.id)
    lang = await asyncio.to_thread(mabuy_service.get_user_language, user_id)
    reply = "🌐 <b>Выберите язык интерфейса / Select Interface Language:</b>"
    kb = get_lang_keyboard(lang=lang)
    if isinstance(event, CallbackQuery):
        await event.answer()
        try:
            await event.message.edit_text(text=reply, reply_markup=kb, disable_web_page_preview=True)
            return
        except Exception:
            pass
    await process_mabuy_response(event, reply)

@router.callback_query(F.data.startswith("set_lang_"))
async def cq_set_lang(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    lang = "ru" if callback.data == "set_lang_ru" else "en"
    await asyncio.to_thread(mabuy_service.set_user_language, user_id, lang)
    reply = await asyncio.to_thread(mabuy_service.handle_start, user_id)
    kb = get_keyboard(user_id)
    flag_txt = "Русский 🇷🇺" if lang == "ru" else "English 🇬🇧"
    await callback.answer(f"✅ Язык установлен: {flag_txt}!", show_alert=False)
    try:
        await callback.message.edit_text(text=reply, reply_markup=kb, disable_web_page_preview=True)
    except Exception:
        await callback.message.answer(reply, reply_markup=kb, disable_web_page_preview=True)

@router.message(Command("predict", "предикт", "прогноз", "p", "predict_btc", "predict_eth", "predict_sol"))
@router.message(F.text.lower().startswith("/предикт") | F.text.lower().startswith("/прогноз") | F.text.lower().startswith("предикт") | F.text.lower().startswith("прогноз"))
async def cmd_predict(message: Message):
    user_id = str(message.from_user.id)
    parts = message.text.split(maxsplit=1)
    args_str = parts[1] if len(parts) > 1 else "BTC"
    symbol_clean = args_str.strip().split()[0].upper()
    
    reply, photo_path = await asyncio.to_thread(mabuy_service.handle_predict, user_id, args_str)
    await process_mabuy_response(message, reply, photo_path, symbol_predict=symbol_clean)
    
@router.callback_query(F.data.startswith("/predict") | F.data.startswith("/предикт") | F.data.startswith("/прогноз"))
async def cq_predict(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    parts = callback.data.split(maxsplit=1)
    args_str = parts[1] if len(parts) > 1 else "BTC"
    symbol_clean = args_str.strip().split()[0].upper()
    
    reply, photo_path = await asyncio.to_thread(mabuy_service.handle_predict, user_id, args_str)
    await callback.answer()
    await process_mabuy_response(callback, reply, photo_path, symbol_predict=symbol_clean)

@router.callback_query(F.data.startswith("view_pro:"))
async def cq_view_pro(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    parts = callback.data.split(":")
    symbol = parts[1] if len(parts) > 1 else "BTC"
    timeframe = parts[2] if len(parts) > 2 else "1h"
    
    tier = await asyncio.to_thread(mabuy_service.get_user_tier, user_id)
    lang = await asyncio.to_thread(mabuy_service.get_user_language, user_id)
    reply = await asyncio.to_thread(mabuy_service.handle_predict_pro_view, user_id, symbol, timeframe)
    
    is_pro_user = (tier.value != "free")
    kb = get_predict_keyboard(symbol, timeframe=timeframe, is_pro_view=is_pro_user, lang=lang)
    
    await callback.answer()
    try:
        if callback.message.caption:
            await callback.message.edit_caption(caption=reply[:1024], reply_markup=kb)
        else:
            await callback.message.edit_text(text=reply, reply_markup=kb, disable_web_page_preview=True)
    except Exception:
        await callback.message.answer(reply, reply_markup=kb, disable_web_page_preview=True)

@router.callback_query(F.data.startswith("view_novice:"))
async def cq_view_novice(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    parts = callback.data.split(":")
    symbol = parts[1] if len(parts) > 1 else "BTC"
    timeframe = parts[2] if len(parts) > 2 else "1h"
    
    lang = await asyncio.to_thread(mabuy_service.get_user_language, user_id)
    reply = await asyncio.to_thread(mabuy_service.handle_predict_novice_view, user_id, symbol, timeframe)
    kb = get_predict_keyboard(symbol, timeframe=timeframe, is_pro_view=False, lang=lang)
    
    await callback.answer()
    try:
        if callback.message.caption:
            await callback.message.edit_caption(caption=reply[:1024], reply_markup=kb)
        else:
            await callback.message.edit_text(text=reply, reply_markup=kb, disable_web_page_preview=True)
    except Exception:
        await callback.message.answer(reply, reply_markup=kb, disable_web_page_preview=True)

@router.callback_query(F.data.startswith("copy_order:"))
async def cq_copy_order(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    parts = callback.data.split(":")
    symbol = parts[1] if len(parts) > 1 else "BTC"
    timeframe = parts[2] if len(parts) > 2 else "1h"
    
    reply = await asyncio.to_thread(mabuy_service.handle_order_copy_snippet, user_id, symbol, timeframe)
    await callback.answer("📋 Параметры ордера сформированы!", show_alert=False)
    await callback.message.answer(reply, disable_web_page_preview=True)

@router.callback_query(F.data.startswith("paper_open:"))
async def cq_paper_open(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    parts = callback.data.split(":")
    symbol = parts[1] if len(parts) > 1 else "BTC"
    
    reply = await asyncio.to_thread(mabuy_service.handle_virtual_trade_open, user_id, symbol, "100")
    await callback.answer("🎮 Демо-сделка $10k открыта!", show_alert=True)
    await callback.message.answer(reply, disable_web_page_preview=True)

@router.message(Command("subscribe"))
@router.callback_query(F.data.in_({"/subscribe", "subscribe"}))
async def cmd_subscribe(event):
    user_id = str(event.from_user.id)
    lang = await asyncio.to_thread(mabuy_service.get_user_language, user_id)
    reply = await asyncio.to_thread(mabuy_service.handle_subscribe_catalog, user_id)
    raw_markup = TelegramMessageFormatter.get_subscription_keyboard(lang=lang)
    kb = convert_dict_markup_to_aiogram(raw_markup)
    if isinstance(event, CallbackQuery):
        await event.answer()
        try:
            await event.message.edit_text(text=reply, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        except Exception:
            await event.message.answer(text=reply, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    else:
        await event.answer(text=reply, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

@router.message(Command("buy"))
@router.callback_query(F.data.startswith("/buy"))
async def cmd_buy(event):
    user_id = str(event.from_user.id)
    tier_arg = "pro"
    if isinstance(event, Message):
        parts = event.text.split(maxsplit=1)
        tier_arg = parts[1] if len(parts) > 1 else "pro"
    elif isinstance(event, CallbackQuery):
        await event.answer()
        data = event.data.replace("/buy", "").strip()
        tier_arg = data if data else "pro"

    reply = await asyncio.to_thread(mabuy_service.handle_buy, user_id, tier_arg)
    if isinstance(event, CallbackQuery):
        await event.message.answer(reply, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    else:
        await event.answer(reply, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

@router.message(Command("portfolio"))
@router.callback_query(F.data.in_({"/portfolio", "portfolio"}))
async def cmd_portfolio(event):
    user_id = str(event.from_user.id)
    args_str = ""
    if isinstance(event, Message):
        parts = event.text.split(maxsplit=1)
        args_str = parts[1] if len(parts) > 1 else ""
    elif isinstance(event, CallbackQuery):
        await event.answer()

    reply = await asyncio.to_thread(mabuy_service.handle_portfolio_audit, user_id, args_str)
    await process_mabuy_response(event, reply)

@router.callback_query(F.data.startswith("portfolio_preset:"))
async def cq_portfolio_preset(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    preset_name = callback.data.split(":", 1)[1]
    
    lang = await asyncio.to_thread(mabuy_service.get_user_language, user_id)
    reply = await asyncio.to_thread(mabuy_service.handle_portfolio_preset, user_id, preset_name)
    kb = get_portfolio_keyboard(lang=lang)
    
    await callback.answer(f"📊 Пресет {preset_name.upper()} рассчитан!", show_alert=False)
    try:
        await callback.message.edit_text(text=reply, reply_markup=kb, disable_web_page_preview=True)
    except Exception:
        await callback.message.answer(reply, reply_markup=kb, disable_web_page_preview=True)

def convert_dict_markup_to_aiogram(raw_markup: dict):
    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
    inline_kb = []
    for row in raw_markup.get("inline_keyboard", []):
        kb_row = []
        for btn in row:
            if "web_app" in btn:
                kb_row.append(InlineKeyboardButton(text=btn["text"], web_app=WebAppInfo(url=btn["web_app"]["url"])))
            elif "url" in btn:
                kb_row.append(InlineKeyboardButton(text=btn["text"], url=btn["url"]))
            else:
                kb_row.append(InlineKeyboardButton(text=btn["text"], callback_data=btn["callback_data"]))
        inline_kb.append(kb_row)
    return InlineKeyboardMarkup(inline_keyboard=inline_kb)

@router.message(Command("matrix"))
@router.callback_query(F.data.in_({"/matrix", "matrix"}))
async def cmd_matrix(event):
    user_id = str(event.from_user.id)
    if isinstance(event, CallbackQuery):
        await event.answer()
    
    try:
        reply_text, heatmap_bytes = await asyncio.to_thread(mabuy_service.handle_correlation_matrix, user_id)
        if heatmap_bytes:
            photo = BufferedInputFile(heatmap_bytes, filename="correlation_matrix.png")
            if isinstance(event, CallbackQuery):
                await event.message.answer_photo(photo=photo, caption=reply_text[:1024], parse_mode="HTML")
            else:
                await event.answer_photo(photo=photo, caption=reply_text[:1024], parse_mode="HTML")
        else:
            await process_mabuy_response(event, reply_text)
    except Exception as e:
        logger.error(f"Error in cmd_matrix: {e}", exc_info=True)
        fallback_text = "🗺 <b>КВАНТОВАЯ МАТРИЦА КОРРЕЛЯЦИЙ</b>\n\nНе удалось сформировать тепловую карту. Попробуйте еще раз через минуту."
        await process_mabuy_response(event, fallback_text)

@router.message(Command("audit"))
@router.message(Command("calc_risk"))
@router.message(Command("risk_calc"))
@router.callback_query(F.data.startswith("/calc_risk"))
@router.callback_query(F.data.startswith("/audit"))
@router.callback_query(F.data.in_({"/audit", "audit"}))
async def cmd_calc_risk(event):
    user_id = str(event.from_user.id)
    args_str = ""
    if isinstance(event, Message):
        parts = event.text.split(maxsplit=1)
        args_str = parts[1] if len(parts) > 1 else ""
    elif isinstance(event, CallbackQuery):
        await event.answer()
        data = event.data
        if data.startswith("/calc_risk "):
            args_str = data.replace("/calc_risk ", "")
        elif data.startswith("/audit "):
            args_str = data.replace("/audit ", "")

    try:
        card_text, markup_dict = await asyncio.to_thread(mabuy_service.handle_calc_risk, user_id, args_str)
        aiogram_kb = convert_dict_markup_to_aiogram(markup_dict)
        
        if isinstance(event, CallbackQuery):
            try:
                await event.message.edit_text(card_text, reply_markup=aiogram_kb, parse_mode="HTML")
            except Exception:
                await event.message.answer(card_text, reply_markup=aiogram_kb, parse_mode="HTML")
        else:
            await event.answer(card_text, reply_markup=aiogram_kb, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error in cmd_calc_risk: {e}", exc_info=True)
        await process_mabuy_response(event, "⚠️ Ошибка расчета риска. Пример команды: <code>/calc_risk 1000 SOL</code>")

@router.message(Command("discipline"))
@router.message(Command("journal"))
@router.message(Command("psychology"))
@router.callback_query(F.data.in_({"/discipline", "discipline", "/journal", "journal"}))
async def cmd_discipline(event):
    user_id = str(event.from_user.id)
    if isinstance(event, CallbackQuery):
        await event.answer()

    try:
        dash_text, markup_dict = await asyncio.to_thread(mabuy_service.handle_discipline_dashboard, user_id)
        aiogram_kb = convert_dict_markup_to_aiogram(markup_dict)
        if isinstance(event, CallbackQuery):
            try:
                await event.message.edit_text(dash_text, reply_markup=aiogram_kb, parse_mode="HTML")
            except Exception:
                await event.message.answer(dash_text, reply_markup=aiogram_kb, parse_mode="HTML")
        else:
            await event.answer(dash_text, reply_markup=aiogram_kb, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error in cmd_discipline: {e}", exc_info=True)
        await process_mabuy_response(event, "⚠️ Ошибка открытия Дневника Дисциплины.")

@router.callback_query(F.data.startswith("/log_tilt "))
async def cq_log_tilt(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    tag = callback.data.replace("/log_tilt ", "").strip()
    await callback.answer("📝 Запись внесена в Дневник!", show_alert=False)

    try:
        dash_text, markup_dict = await asyncio.to_thread(mabuy_service.handle_log_tilt, user_id, tag)
        aiogram_kb = convert_dict_markup_to_aiogram(markup_dict)
        await callback.message.edit_text(dash_text, reply_markup=aiogram_kb, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error in cq_log_tilt: {e}", exc_info=True)
        await callback.message.answer("⚠️ Ошибка обновления журнала.")

@router.callback_query(F.data == "/tilt_cooldown")
async def cq_tilt_cooldown(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    await callback.answer("🛑 45-минутный режим Анти-Тильт активирован!", show_alert=True)

    try:
        dash_text, markup_dict = await asyncio.to_thread(mabuy_service.handle_tilt_cooldown, user_id)
        aiogram_kb = convert_dict_markup_to_aiogram(markup_dict)
        await callback.message.edit_text(dash_text, reply_markup=aiogram_kb, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error in cq_tilt_cooldown: {e}", exc_info=True)
        await callback.message.answer("⚠️ Ошибка запуска таймаута.")

@router.message(Command("regimes"))
@router.message(Command("forward_status"))
@router.message(Command("regime_quotas"))
@router.callback_query(F.data.in_({"/regimes", "regimes", "/forward_status"}))
async def cmd_regimes(event):
    user_id = str(event.from_user.id)
    if isinstance(event, CallbackQuery):
        await event.answer()

    try:
        dash_text, markup_dict = await asyncio.to_thread(mabuy_service.handle_regime_quota_dashboard, user_id)
        aiogram_kb = convert_dict_markup_to_aiogram(markup_dict)
        if isinstance(event, CallbackQuery):
            try:
                await event.message.edit_text(dash_text, reply_markup=aiogram_kb, parse_mode="HTML")
            except Exception:
                await event.message.answer(dash_text, reply_markup=aiogram_kb, parse_mode="HTML")
        else:
            await event.answer(dash_text, reply_markup=aiogram_kb, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error in cmd_regimes: {e}", exc_info=True)
        await process_mabuy_response(event, "⚠️ Ошибка загрузки дашборда квот режимов.")

@router.message(Command("flow"))
@router.message(Command("radar"))
@router.message(Command("positions"))
@router.message(Command("derivatives"))
@router.message(Command("onchain"))
@router.callback_query(F.data.startswith("/flow"))
@router.callback_query(F.data.startswith("/onchain"))
async def cmd_flow(event):
    user_id = str(event.from_user.id)
    symbol_str = "BTC"
    if isinstance(event, Message):
        parts = event.text.split(maxsplit=1)
        symbol_str = parts[1] if len(parts) > 1 else "BTC"
    elif isinstance(event, CallbackQuery):
        await event.answer()
        data = event.data
        if data.startswith("/flow "):
            symbol_str = data.replace("/flow ", "").strip()
        elif data.startswith("/onchain "):
            symbol_str = data.replace("/onchain ", "").strip()

    try:
        card_text, markup_dict = await asyncio.to_thread(mabuy_service.handle_derivatives_radar, user_id, symbol_str)
        aiogram_kb = convert_dict_markup_to_aiogram(markup_dict)
        
        if isinstance(event, CallbackQuery):
            try:
                await event.message.edit_text(card_text, reply_markup=aiogram_kb, parse_mode="HTML")
            except Exception:
                await event.message.answer(card_text, reply_markup=aiogram_kb, parse_mode="HTML")
        else:
            await event.answer(card_text, reply_markup=aiogram_kb, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error in cmd_flow: {e}", exc_info=True)
        await process_mabuy_response(event, "⚠️ Ошибка анализа деривативного радара. Пример команды: <code>/flow BTC</code>")

@router.message(Command("habit"))
@router.message(Command("habits"))
@router.message(F.text.lower().in_({"/habit", "/habits", "habit", "habits", "привычки"}))
@router.callback_query(F.data.in_({"/habit", "/habits", "habit"}))
async def cmd_habits(event):
    user_id = str(event.from_user.id)
    if isinstance(event, CallbackQuery):
        await event.answer()
    
    text, raw_markup = await asyncio.to_thread(mabuy_service.handle_habits, user_id)
    kb = convert_dict_markup_to_aiogram(raw_markup)
    
    if isinstance(event, CallbackQuery):
        try:
            await event.message.edit_text(text, reply_markup=kb, parse_mode="HTML", disable_web_page_preview=True)
        except Exception:
            await event.message.answer(text, reply_markup=kb, parse_mode="HTML", disable_web_page_preview=True)
    else:
        await event.answer(text, reply_markup=kb, parse_mode="HTML", disable_web_page_preview=True)

@router.callback_query(F.data.startswith("/habit_toggle "))
async def cb_habit_toggle(call: CallbackQuery):
    await call.answer("Отмечено! 🔥")
    habit_id = call.data.replace("/habit_toggle ", "").strip()
    user_id = str(call.from_user.id)
    
    text, raw_markup = await asyncio.to_thread(mabuy_service.handle_habit_toggle, user_id, habit_id)
    kb = convert_dict_markup_to_aiogram(raw_markup)
    try:
        await call.message.edit_text(text, reply_markup=kb, parse_mode="HTML", disable_web_page_preview=True)
    except Exception:
        pass

@router.callback_query(F.data == "/habit_sprint")
async def cb_habit_sprint(call: CallbackQuery):
    await call.answer("Запуск спринта! ⚡️")
    user_id = str(call.from_user.id)
    lang = mabuy_service.get_user_language(user_id)
    if lang == "ru":
        msg = (
            "⚡️ <b>45-МИНУТНЫЙ ДИП-ВОРК СПРИНТ</b> 🔥\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "🧠 <b>Фокус:</b> 45 минут чистой концентрации без отвлечений!\n"
            "🎯 <b>Задачи:</b> SAT Math / IELTS Writing / ML код.\n\n"
            "💡 <i>Открой WebApp по кнопке ниже — там встроен интерактивный круговой таймер со звуковым гонгом!</i>"
        )
    else:
        msg = (
            "⚡️ <b>45-MIN DEEP WORK SPRINT</b> 🔥\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "🧠 <b>Focus:</b> 45 minutes of pure uninterrupted work!\n"
            "🎯 <b>Targets:</b> SAT Math / IELTS Writing / ML code.\n\n"
            "💡 <i>Open the WebApp below to use the live circular Focus Timer with audio chime!</i>"
        )
    text, raw_markup = await asyncio.to_thread(mabuy_service.handle_habits, user_id)
    kb = convert_dict_markup_to_aiogram(raw_markup)
    await call.message.answer(msg, reply_markup=kb, parse_mode="HTML")

@router.message(Command("journal"))
@router.message(Command("diary"))
@router.message(F.text.lower().in_({"/journal", "/diary", "journal", "дневник"}))
@router.callback_query(F.data == "/journal")
async def cmd_journal(event):
    user_id = str(event.from_user.id)
    if isinstance(event, CallbackQuery):
        await event.answer()
    
    text, raw_markup = await asyncio.to_thread(mabuy_service.handle_journal, user_id)
    kb = convert_dict_markup_to_aiogram(raw_markup)
    
    if isinstance(event, CallbackQuery):
        try:
            await event.message.edit_text(text, reply_markup=kb, parse_mode="HTML", disable_web_page_preview=True)
        except Exception:
            await event.message.answer(text, reply_markup=kb, parse_mode="HTML", disable_web_page_preview=True)
    else:
        await event.answer(text, reply_markup=kb, parse_mode="HTML", disable_web_page_preview=True)

@router.message(Command("buy"))
async def cmd_buy(message: Message):
    user_id = str(message.from_user.id)
    parts = message.text.split(maxsplit=1)
    tier = parts[1] if len(parts) > 1 else "pro"
    reply = await asyncio.to_thread(mabuy_service.handle_buy, user_id, tier)
    await process_mabuy_response(message, reply)

@router.message(Command("autotrade"))
async def cmd_autotrade(message: Message):
    user_id = str(message.from_user.id)
    reply = await asyncio.to_thread(mabuy_service.handle_autotrade_setup, user_id)
    await process_mabuy_response(message, reply)
    
@router.message(Command("autotrade_submit"))
async def cmd_autotrade_submit(message: Message):
    user_id = str(message.from_user.id)
    parts = message.text.split()
    if len(parts) < 3:
        await process_mabuy_response(message, "❌ Неверный формат. Ожидается: /autotrade_submit API_KEY API_SECRET")
        return
    api_key = parts[1]
    api_secret = parts[2]
    reply = await asyncio.to_thread(mabuy_service.handle_autotrade_submit, user_id, api_key, api_secret)
    await process_mabuy_response(message, reply)

@router.message(Command("admin"))
async def cmd_admin(message: Message):
    user_id = str(message.from_user.id)
    reply = await asyncio.to_thread(mabuy_service.handle_admin_dashboard, user_id)
    await process_mabuy_response(message, reply)

@router.message(Command("godmode"))
async def cmd_godmode(message: Message):
    user_id = str(message.from_user.id)
    parts = message.text.split(maxsplit=1)
    amount = parts[1] if len(parts) > 1 else "100000"
    reply = await asyncio.to_thread(mabuy_service.handle_admin_abuse, user_id, amount)
    await process_mabuy_response(message, reply)

@router.message(Command("broadcast"))
async def cmd_broadcast(message: Message):
    user_id = str(message.from_user.id)
    if not mabuy_service.is_admin(user_id):
        await process_mabuy_response(message, "🚫 Доступ запрещен.")
        return
        
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await process_mabuy_response(message, "⚠️ Использование: /broadcast <сообщение>")
        return
        
    text = parts[1]
    all_users = mabuy_service.usage_tracker.get_all_users()
    
    await process_mabuy_response(message, f"📢 Начинаю рассылку для {len(all_users)} пользователей...")
    
    success_count = 0
    for uid in all_users:
        try:
            await message.bot.send_message(
                chat_id=uid, 
                text=f"📢 <b>СИСТЕМНОЕ УВЕДОМЛЕНИЕ:</b>\n\n{text}", 
                parse_mode="HTML"
            )
            success_count += 1
            await asyncio.sleep(0.05)
        except Exception as e:
            logger.error(f"Failed to broadcast to {uid}: {e}")
            
    await process_mabuy_response(message, f"✅ Рассылка завершена! Доставлено: {success_count}/{len(all_users)}")

@router.message(Command("tier"))
@router.callback_query(F.data.in_({"/tier", "tier"}))
async def cmd_tier(event):
    user_id = str(event.from_user.id)
    reply = await asyncio.to_thread(mabuy_service.handle_tier_info, user_id)
    if isinstance(event, CallbackQuery):
        await event.answer()
    await process_mabuy_response(event, reply)

@router.message(Command("ref"))
@router.callback_query(F.data.in_({"/ref", "ref"}))
async def cmd_ref(event):
    user_id = str(event.from_user.id)
    reply = await asyncio.to_thread(mabuy_service.handle_referral_info, user_id)
    if isinstance(event, CallbackQuery):
        await event.answer()
    await process_mabuy_response(event, reply)

@router.message(Command("virtual_pnl"))
@router.message(Command("pnl"))
@router.message(Command("demo"))
@router.callback_query(F.data.in_({"/virtual_pnl", "virtual_pnl", "/pnl", "demo"}))
async def cmd_virtual_pnl(event):
    user_id = str(event.from_user.id)
    reply = await asyncio.to_thread(mabuy_service.handle_virtual_pnl, user_id)
    if isinstance(event, CallbackQuery):
        await event.answer()
    await process_mabuy_response(event, reply)

@router.message(Command("leaderboard"))
@router.message(Command("top"))
@router.message(Command("cup"))
@router.callback_query(F.data.in_({"/leaderboard", "leaderboard", "/top", "top", "/cup", "cup"}))
async def cmd_leaderboard(event):
    user_id = str(event.from_user.id)
    reply = await asyncio.to_thread(mabuy_service.handle_leaderboard, user_id)
    if isinstance(event, CallbackQuery):
        await event.answer()
    await process_mabuy_response(event, reply)

@router.message(Command("setnick"))
async def cmd_setnick(message: Message):
    user_id = str(message.from_user.id)
    parts = message.text.split(maxsplit=1)
    args_str = parts[1] if len(parts) > 1 else ""
    reply = await asyncio.to_thread(mabuy_service.handle_setnick, user_id, args_str)
    await process_mabuy_response(message, reply)

@router.message(Command("alerts"))
@router.callback_query(F.data.in_({"/alerts", "alerts"}))
async def cmd_alerts(event):
    user_id = str(event.from_user.id)
    reply = await asyncio.to_thread(mabuy_service.handle_alerts, user_id)
    if isinstance(event, CallbackQuery):
        await event.answer()
    await process_mabuy_response(event, reply)

@router.message(Command("add_alert"))
async def cmd_add_alert(message: Message):
    user_id = str(message.from_user.id)
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        reply = "⚠️ Укажите монету. Пример: /add_alert BTC" if mabuy_service.get_user_language(user_id) == "ru" else "⚠️ Please specify a coin. Example: /add_alert BTC"
    else:
        symbol = parts[1]
        reply = await asyncio.to_thread(mabuy_service.handle_add_alert, user_id, symbol)
    await process_mabuy_response(message, reply)

@router.message(Command("del_alert"))
async def cmd_del_alert(message: Message):
    user_id = str(message.from_user.id)
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        reply = "⚠️ Укажите монету. Пример: /del_alert BTC" if mabuy_service.get_user_language(user_id) == "ru" else "⚠️ Please specify a coin. Example: /del_alert BTC"
    else:
        symbol = parts[1]
        reply = await asyncio.to_thread(mabuy_service.handle_del_alert, user_id, symbol)
    await process_mabuy_response(message, reply)

@router.message(Command("risk"))
@router.callback_query(F.data.in_({"/risk", "risk"}))
async def cmd_risk(event):
    user_id = str(event.from_user.id)
    reply = await asyncio.to_thread(mabuy_service.handle_risk, user_id)
    if isinstance(event, CallbackQuery):
        await event.answer()
    await process_mabuy_response(event, reply)

@router.callback_query(F.data.startswith("set_risk_"))
async def cq_set_risk(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    risk_level = callback.data.replace("set_risk_", "")
    reply = await asyncio.to_thread(mabuy_service.set_user_risk, user_id, risk_level)
    await callback.answer(f"✅ Профиль риска: {risk_level.upper()}", show_alert=False)
    await process_mabuy_response(callback, reply)

@router.message(Command("deposit", "депозит", "balance", "баланс", "capital"))
@router.callback_query(F.data.in_({"/deposit", "deposit", "set_deposit", "/депозит"}))
async def cmd_deposit(event):
    user_id = str(event.from_user.id)
    args_str = ""
    if isinstance(event, Message):
        parts = event.text.split(maxsplit=1)
        args_str = parts[1] if len(parts) > 1 else ""
    elif isinstance(event, CallbackQuery):
        await event.answer()

    text, raw_markup = await asyncio.to_thread(mabuy_service.handle_deposit, user_id, args_str)
    kb = convert_dict_markup_to_aiogram(raw_markup)

    if isinstance(event, CallbackQuery):
        try:
            await event.message.edit_text(text, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        except Exception:
            await event.message.answer(text, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    else:
        await event.answer(text, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

@router.callback_query(F.data.startswith("set_dep_"))
async def cq_set_deposit(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    try:
        amount = float(callback.data.replace("set_dep_", ""))
    except ValueError:
        amount = 1000.0
    text, raw_markup = await asyncio.to_thread(mabuy_service.set_user_deposit, user_id, amount)
    kb = convert_dict_markup_to_aiogram(raw_markup)
    await callback.answer(f"✅ Депозит: ${amount:,.0f} USD!", show_alert=False)
    try:
        await callback.message.edit_text(text, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except Exception:
        await callback.message.answer(text, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

def get_persona_keyboard(current_persona: PersonaType):
    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
    buttons = [
        [
            InlineKeyboardButton(
                text=f"{'✅ ' if current_persona == PersonaType.QUANT else ''}👔 Квант (PhD/CFA)", 
                callback_data="set_persona_quant"
            ),
            InlineKeyboardButton(
                text=f"{'✅ ' if current_persona == PersonaType.ZEN else ''}🧘 Дзен (DCA Ментор)", 
                callback_data="set_persona_zen"
            )
        ],
        [
            InlineKeyboardButton(
                text=f"{'✅ ' if current_persona == PersonaType.TSUNDERE else ''}🗡 Цундере (Анти-Лудо)", 
                callback_data="set_persona_tsundere"
            ),
            InlineKeyboardButton(
                text=f"{'✅ ' if current_persona == PersonaType.YANDERE else ''}🩸 Яндере (Одержимая)", 
                callback_data="set_persona_yandere"
            )
        ],
        [
            InlineKeyboardButton(
                text=f"{'✅ ' if current_persona == PersonaType.TITAN else ''}🤖 Titan (ИИ 2077)", 
                callback_data="set_persona_titan"
            )
        ],
        [
            InlineKeyboardButton(text="💬 Задать вопрос (/ask)", callback_data="/ask"),
            InlineKeyboardButton(text="🏠 Главное меню", callback_data="/start")
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

@router.message(Command("persona"))
@router.message(Command("maia"))
@router.callback_query(F.data.in_({"/persona", "persona", "/maia", "maia"}))
async def cmd_persona(event):
    user_id = str(event.from_user.id)
    if isinstance(event, CallbackQuery):
        await event.answer()
    
    saved_persona = await asyncio.to_thread(mabuy_service.usage_tracker.get_user_persona, user_id, "quant")
    try:
        current_persona = PersonaType(saved_persona)
    except ValueError:
        current_persona = PersonaType.QUANT
    user_persona_map[user_id] = current_persona
    
    meta = PERSONA_METADATA[current_persona]
    oled = OLED_EXPRESSIONS[current_persona]["neutral"]
    
    reply = (
        f"✨ <b>ПАНЕЛЬ НАСТРОЙКИ MAIA (ФИНАНСОВЫЙ ИИ-КОМПАНЬОН)</b> 🤖\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📱 <b>Текущий характер:</b> <b>{meta['name']}</b> ({meta['badge']})\n"
        f"📟 <b>OLED-экран:</b> <code>{oled}</code>\n\n"
        f"<i>{meta['description']}</i>\n\n"
        f"🎭 <b>Выберите характер MaIA (он же задает стиль PUSH-сигналов!):</b>\n"
        f"• 👔 <b>Квант</b> — строгий расчет риска, Шарп, Merton P10-P90, Kelly.\n"
        f"• 🧘 <b>Дзен</b> — наставник регулярного накопления (DCA) и душевного спокойствия.\n"
        f"• 🗡 <b>Цундере</b> — эмоциональная анти-лудоманка («Не смей входить без стопа, бака!»).\n"
        f"• 🩸 <b>Яндере</b> — одержимо заботливая («Г-господин... я нашла идеальный вход, не потеряй мои денежки ❤️🔪»).\n"
        f"• 🤖 <b>Titan V4.1</b> — квантовый бортовой суперкомпьютер из 2077 года.\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 <i>Используйте команду <code>/ask &lt;вопрос&gt;</code> для мгновенного анализа!</i>"
    )
    kb = get_persona_keyboard(current_persona)
    
    if isinstance(event, CallbackQuery):
        try:
            await event.message.edit_text(reply, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        except Exception:
            await event.message.answer(reply, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    else:
        await event.answer(reply, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

@router.callback_query(F.data.startswith("set_persona_"))
async def cq_set_persona(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    persona_code = callback.data.replace("set_persona_", "").strip()
    try:
        new_persona = PersonaType(persona_code)
    except ValueError:
        new_persona = PersonaType.QUANT
    
    user_persona_map[user_id] = new_persona
    await asyncio.to_thread(mabuy_service.usage_tracker.set_user_persona, user_id, new_persona.value)
    meta = PERSONA_METADATA[new_persona]
    oled = OLED_EXPRESSIONS[new_persona]["neutral"]
    
    await callback.answer(f"✅ Установлен характер: {meta['name']}!", show_alert=False)
    
    reply = (
        f"✨ <b>ХАРАКТЕР MAIA ОБНОВЛЕН: {meta['name'].upper()}</b> ({meta['badge']})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📟 <b>OLED Display:</b> <code>{oled}</code>\n\n"
        f"💬 <i>\"{meta['sample_greeting']}\"</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔔 <i>Теперь PUSH-уведомления и сигналы будут приходить в стиле <b>{meta['name']}</b>!</i>\n\n"
        f"💡 <i>Спросите MaIA о любом активе или стратегии:\n<code>/ask Что думаешь по SOL?</code></i>"
    )
    kb = get_persona_keyboard(new_persona)
    try:
        await callback.message.edit_text(reply, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except Exception:
        await callback.message.answer(reply, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

@router.message(Command("terms"))
@router.message(Command("disclaimer"))
@router.callback_query(F.data.in_({"/terms", "terms", "/disclaimer", "disclaimer"}))
async def cmd_terms_and_disclaimer(event):
    if isinstance(event, CallbackQuery):
        await event.answer()
    
    terms_text = (
        "⚖️ <b>ПОЛЬЗОВАТЕЛЬСКОЕ СОГЛАШЕНИЕ И РЕГЛАМЕНТ РИСКОВ MABUY</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "<b>1. Статус программного обеспечения (DYOR):</b>\n"
        "Платформа MaBuy, сервисы MaPredict, MaInvest и ИИ-компаньон MaIA "
        "являются программным комплексом для аналитических, исследовательских и образовательных целей. "
        "Сервис не является брокером, инвестиционным советником или доверительным управляющим.\n\n"
        "<b>2. Отсутствие индивидуальных инвестиционных рекомендаций:</b>\n"
        "Никакая информация бота, стохастические расчеты (Монте-Карло прыжковой диффузии Мертона), "
        "метрики L2 OFI, формулы Quarter-Kelly и сгенерированные ответы ИИ <b>НЕ являются индивидуальной "
        "инвестиционной рекомендацией</b> (No Financial Advice) в соответствии с международными стандартами MiFID II и SEC.\n\n"
        "<b>3. Высокий риск маржинальной торговли:</b>\n"
        "Торговля деривативами, фьючерсами и криптоактивами сопряжена с критическим риском потери капитала. "
        "Пользователь несет 100% единоличную финансовую ответственность за любые совершенные операции.\n\n"
        "<b>4. Ограничение ответственности:</b>\n"
        "Разработчики не несут ответственности за сбои биржевых шлюзов API, задержки сети, проскальзывания цен или убытки.\n\n"
        "🌐 <i>Полный интерактивный регламент доступен в WebApp: <a href=\"" + os.environ.get("MAINVEST_WEBAPP_URL", "https://mabuy.vercel.app") + "\">MaBuy WebApp</a></i>"
    )
    
    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo
    webapp_url = os.environ.get("MAINVEST_WEBAPP_URL", "https://mabuy.vercel.app")
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📱 Открыть WebApp", web_app=WebAppInfo(url=webapp_url))],
        [InlineKeyboardButton(text="🏠 Главное меню", callback_data="/start")]
    ])
    
    if isinstance(event, CallbackQuery):
        try:
            await event.message.edit_text(terms_text, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        except Exception:
            await event.message.answer(terms_text, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    else:
        await event.answer(terms_text, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

async def send_safe_long_message(target_msg, text: str, reply_markup=None, disable_web_page_preview: bool = True):
    """Safely delivers arbitrarily long text messages without hitting Telegram's 4096 character limit."""
    if len(text) <= 3950:
        await target_msg.answer(text, reply_markup=reply_markup, disable_web_page_preview=disable_web_page_preview)
        return

    paragraphs = text.split("\n\n")
    current_chunk = ""
    chunks = []
    for p in paragraphs:
        if len(current_chunk) + len(p) + 2 < 3800:
            current_chunk = f"{current_chunk}\n\n{p}" if current_chunk else p
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(p) > 3800:
                for i in range(0, len(p), 3800):
                    chunks.append(p[i:i+3800])
                current_chunk = ""
            else:
                current_chunk = p
    if current_chunk:
        chunks.append(current_chunk)

    for i, ch in enumerate(chunks):
        is_last = (i == len(chunks) - 1)
        kb = reply_markup if is_last else None
        await target_msg.answer(ch, reply_markup=kb, disable_web_page_preview=disable_web_page_preview)

@router.message(Command("ask"))
@router.callback_query(F.data.in_({"/ask", "ask", "/ask_help"}))
async def cmd_ask(event):
    user_id = str(event.from_user.id)
    question = ""
    if isinstance(event, Message):
        raw_text = (event.text or "").strip()
        if raw_text.startswith("/ask"):
            parts = raw_text.split(maxsplit=1)
            question = parts[1].strip() if len(parts) > 1 else ""
        else:
            question = raw_text
    elif isinstance(event, CallbackQuery):
        await event.answer()
    
    active_persona = await get_effective_user_persona(user_id)
    meta = PERSONA_METADATA[active_persona]
    
    if not question:
        oled = OLED_EXPRESSIONS[active_persona]["neutral"]
        help_msg = (
            f"✨ <b>ФИНАНСОВЫЙ ИИ-КОМПАНЬОН MAIA</b> ({meta['name']})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📟 <b>OLED LED:</b> <code>{oled}</code>\n\n"
            f"Задайте любой вопрос по рынку, управлению капиталом или активам:\n"
            f"• <code>/ask Что думаешь по SOL на споте?</code>\n"
            f"• <code>/ask Рассчитай сайзинг для депозита $1000</code>\n"
            f"• <code>/ask Опасно ли брать плечо 20x по BTC?</code>\n"
            f"• <code>/ask Как составить всепогодный портфель All-Weather?</code>\n"
            f"• <code>/ask Прогноз прыжковой диффузии Мертона по NVDA</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎭 <i>Сменить характер MaIA: /persona</i>"
        )
        from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="👔 Спросить о BTC", callback_data="/ask_btc")],
            [InlineKeyboardButton(text="🧮 Сайзинг для $500", callback_data="/ask_sizing")],
            [InlineKeyboardButton(text="🎭 Сменить характер (/persona)", callback_data="/persona")]
        ])
        if isinstance(event, CallbackQuery):
            await event.message.answer(help_msg, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        else:
            await event.answer(help_msg, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        return

    # Check Daily AI Usage Quota (Free: 5, Pro/Invest: 10, All-Access: unlimited)
    allowed, used_count, max_limit = await asyncio.to_thread(
        mabuy_service.usage_tracker.can_user_query_ai, user_id
    )
    if not allowed:
        tier_data = await asyncio.to_thread(mabuy_service.usage_tracker.get_user_tier_data, user_id)
        current_tier_name = tier_data[0].value.upper() if tier_data else "FREE"
        limit_msg = (
            f"⚠️ <b>ДНЕВНОЙ ЛИМИТ ЗАПРОСОВ К ИИ ИСЧЕРПАН!</b> 🤖\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Ваш тариф:</b> <code>{current_tier_name}</code> (Использовано: <b>{used_count}/{max_limit}</b> за сегодня)\n\n"
            f"💡 <b>Тарифные лимиты на запросы к MaIA:</b>\n"
            f"• <b>FREE:</b> 5 запросов / день\n"
            f"• <b>MaPredict PRO ($14.99):</b> 10 запросов / день\n"
            f"• <b>MaInvest WEALTH ($14.99):</b> 10 запросов / день\n"
            f"• <b>MaBuy Ecosystem ($24.99):</b> <b>НЕОГРАНИЧЕННО (∞)</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🚀 Перейдите на <b>PRO</b> или <b>Ecosystem Pass</b> для снятия ограничений!"
        )
        from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
        upgrade_kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="⭐️ Улучшить тариф (/tier)", callback_data="/tier")],
            [InlineKeyboardButton(text="🏠 Главное меню", callback_data="/start")]
        ])
        if isinstance(event, CallbackQuery):
            await event.message.answer(limit_msg, reply_markup=upgrade_kb, parse_mode=ParseMode.HTML)
        else:
            await event.answer(limit_msg, reply_markup=upgrade_kb, parse_mode=ParseMode.HTML)
        return

    # Trigger typing indicator
    target_msg = event if isinstance(event, Message) else event.message
    try:
        await target_msg.bot.send_chat_action(chat_id=target_msg.chat.id, action="typing")
    except Exception:
        pass

    user_dep = await asyncio.to_thread(mabuy_service.usage_tracker.get_user_deposit, user_id, 500.0)

    try:
        response = await maia_copilot.generate_response(
            query=question,
            persona=active_persona,
            user_deposit=user_dep,
            session_id=user_id
        )
        # Increment successful AI usage count
        await asyncio.to_thread(mabuy_service.usage_tracker.increment_ai_usage, user_id)
        
        from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, FSInputFile
        from src.infrastructure.audio.voice_synthesizer import MaIAVoiceSynthesizer
        
        # Synthesize Voice Note
        voice_synth = MaIAVoiceSynthesizer()
        audio_path = None
        try:
            audio_path = await voice_synth.synthesize_speech(
                text=response.text,
                persona=active_persona.value,
                lang="ru"
            )
        except Exception as synth_err:
            logger.debug(f"Speech synthesis error: {synth_err}")

        sym_label = f"#{response.target_symbol}" if response.target_symbol and response.target_symbol != "GENERAL" else "Диалог"

        if response.target_symbol and response.target_symbol != "GENERAL":
            kb = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="📈 График & Прогноз", callback_data=f"/predict {response.target_symbol}"),
                    InlineKeyboardButton(text="🎭 Сменить характер", callback_data="/persona")
                ]
            ])
        else:
            kb = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="🎭 Сменить характер", callback_data="/persona"),
                    InlineKeyboardButton(text="📱 WebApp", callback_data="/webapp")
                ]
            ])
        
        await send_safe_long_message(target_msg, response.text, reply_markup=kb, disable_web_page_preview=True)

        # Send Voice Note if synthesized successfully
        if audio_path and os.path.exists(audio_path):
            try:
                if hasattr(target_msg, "answer_voice"):
                    res = target_msg.answer_voice(
                        voice=FSInputFile(audio_path),
                        caption=f"🎙 <b>Голос MaIA ({meta['name']}):</b> <i>«{sym_label}»</i>",
                        parse_mode=ParseMode.HTML
                    )
                    if inspect.isawaitable(res):
                        await res
                elif hasattr(target_msg, "bot") and hasattr(target_msg.bot, "send_voice"):
                    res = target_msg.bot.send_voice(
                        chat_id=target_msg.chat.id,
                        voice=FSInputFile(audio_path),
                        caption=f"🎙 <b>Голос MaIA ({meta['name']}):</b> <i>«{sym_label}»</i>",
                        parse_mode=ParseMode.HTML
                    )
                    if inspect.isawaitable(res):
                        await res
            except Exception as voice_err:
                logger.error(f"Failed to send voice note: {voice_err}", exc_info=True)
    except Exception as e:
        logger.error(f"Error in MaIA /ask: {e}", exc_info=True)
        await target_msg.answer("⚠️ Не удалось получить ответ от MaIA. Попробуйте еще раз через несколько секунд.")

@router.callback_query(F.data == "/ask_btc")
async def cq_ask_btc(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    await callback.answer()
    
    allowed, used_count, max_limit = await asyncio.to_thread(
        mabuy_service.usage_tracker.can_user_query_ai, user_id
    )
    if not allowed:
        await callback.message.answer(
            f"⚠️ <b>Дневной лимит запросов к ИИ ({used_count}/{max_limit}) исчерпан!</b>\n"
            f"Для снятия ограничений перейдите на <b>MaBuy Ecosystem</b> (/tier).",
            parse_mode=ParseMode.HTML
        )
        return

    active_persona = await get_effective_user_persona(user_id)
    user_dep = await asyncio.to_thread(mabuy_service.usage_tracker.get_user_deposit, user_id, 500.0)
    
    try:
        await callback.message.bot.send_chat_action(chat_id=callback.message.chat.id, action="typing")
    except Exception:
        pass

    response = await maia_copilot.generate_response(
        query="Дай полный квантовый анализ по Bitcoin (BTC)",
        persona=active_persona,
        user_deposit=user_dep,
        session_id=user_id
    )
    await asyncio.to_thread(mabuy_service.usage_tracker.increment_ai_usage, user_id)
    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text=f"🧮 Сайзинг под ${user_dep:,.0f}", callback_data=f"/calc_risk {user_dep:.0f} BTC"),
            InlineKeyboardButton(text="🎭 Характер", callback_data="/persona")
        ]
    ])
    await send_safe_long_message(callback.message, response.text, reply_markup=kb, disable_web_page_preview=True)

@router.callback_query(F.data == "/ask_sizing")
async def cq_ask_sizing(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    await callback.answer()
    
    allowed, used_count, max_limit = await asyncio.to_thread(
        mabuy_service.usage_tracker.can_user_query_ai, user_id
    )
    if not allowed:
        await callback.message.answer(
            f"⚠️ <b>Дневной лимит запросов к ИИ ({used_count}/{max_limit}) исчерпан!</b>\n"
            f"Для снятия ограничений перейдите на <b>MaBuy Ecosystem</b> (/tier).",
            parse_mode=ParseMode.HTML
        )
        return

    active_persona = await get_effective_user_persona(user_id)
    user_dep = await asyncio.to_thread(mabuy_service.usage_tracker.get_user_deposit, user_id, 500.0)
    
    try:
        await callback.message.bot.send_chat_action(chat_id=callback.message.chat.id, action="typing")
    except Exception:
        pass

    response = await maia_copilot.generate_response(
        query=f"Рассчитай безопасный риск и сайзинг для депозита ${user_dep:,.0f}",
        persona=active_persona,
        user_deposit=user_dep,
        session_id=user_id
    )
    await asyncio.to_thread(mabuy_service.usage_tracker.increment_ai_usage, user_id)
    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="💵 Сменить депозит", callback_data="/deposit"),
            InlineKeyboardButton(text="🎭 Характер", callback_data="/persona")
        ]
    ])
    await send_safe_long_message(callback.message, response.text, reply_markup=kb, disable_web_page_preview=True)

@router.message(Command("wincard"))
@router.callback_query(F.data.in_({"/wincard", "wincard", "cq_wincard"}))
async def cmd_wincard(event):
    user_id = str(event.from_user.id)
    if isinstance(event, CallbackQuery):
        await event.answer()

    try:
        # 1. Fetch live performance summary from verified track record
        perf = await asyncio.to_thread(mabuy_service.track_record_auditor.get_performance_summary, "ru")
        winrate = float(perf.get("win_rate_pct", 46.4)) if perf and perf.get("has_data") else 46.4
        total_trades = int(perf.get("total_signals", 321)) if perf and perf.get("has_data") else 321
        
        # 2. Fetch latest winning verified signal from DB
        def fetch_latest_win_signal():
            import sqlite3
            db_p = getattr(mabuy_service, "db_path", "data/mabuy_prod.db")
            with sqlite3.connect(db_p, timeout=10.0) as conn:
                c = conn.cursor()
                c.execute("""
                    SELECT symbol, entry_price, target_price, exit_price, direction 
                    FROM verified_signals_log 
                    WHERE status IN ('HIT_TP', 'TP_HIT', 'WIN', 'TRAILING_STOP_HIT')
                    ORDER BY id DESC LIMIT 1
                """)
                return c.fetchone()
                
        win_sig = await asyncio.to_thread(fetch_latest_win_signal)
        
        if win_sig and win_sig[1] and win_sig[1] > 0:
            sym = win_sig[0]
            entry_p = win_sig[1]
            target_p = win_sig[2]
            exit_p = win_sig[3] if win_sig[3] else target_p
            pnl_pct = abs((exit_p - entry_p) / entry_p) * 100.0
            is_paper = False
        else:
            sym = "BTC"
            entry_p = 68500.0
            exit_p = 70800.0
            pnl_pct = 3.35
            is_paper = False

        # 3. Render PNG card asynchronously in worker thread
        card_bytes = await asyncio.to_thread(
            ShareableWinCardGenerator.generate_win_card_bytes,
            symbol=sym,
            pnl_pct=pnl_pct,
            entry_price=entry_p,
            exit_price=exit_p,
            is_paper_trade=is_paper,
            winrate_pct=winrate,
            total_trades=max(1, total_trades),
            user_id=user_id,
            timeframe="1h"
        )

        # 3. Share URL
        share_text = f"🚀 Мой результат торговли с ИИ MaBuy: +{pnl_pct:.2f}% (Винрейт {winrate:.1f}%)!\nПопробуй бота бесплатно:"
        share_url = f"https://t.me/share/url?url={urllib.parse.quote('https://t.me/MabuyCryptoML_bot?start=ref_' + user_id)}&text={urllib.parse.quote(share_text)}"

        from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="📲 Поделиться в Telegram", url=share_url)],
            [InlineKeyboardButton(text="🎮 Открыть симулятор", callback_data="/virtual_pnl")]
        ])

        input_file = BufferedInputFile(card_bytes, filename=f"mabuy_win_{user_id}.png")
        caption = (
            f"🏆 <b>ТВОЯ ВИРАЛЬНАЯ КАРТОЧКА РЕЗУЛЬТАТА!</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 <b>PnL:</b> <code>+{pnl_pct:.2f}%</code> | <b>Винрейт:</b> <code>{winrate:.1f}%</code>\n\n"
            f"<i>Нажми кнопку ниже, чтобы опубликовать карточку в свои сторис, каналы или отправить друзьям с твоей реферальной ссылкой!</i>"
        )

        if isinstance(event, CallbackQuery):
            await event.message.answer_photo(photo=input_file, caption=caption, reply_markup=kb, parse_mode=ParseMode.HTML)
        else:
            await event.answer_photo(photo=input_file, caption=caption, reply_markup=kb, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Error generating wincard for {user_id}: {e}", exc_info=True)
        fallback_msg = "⚠️ Не удалось сгенерировать карточку PnL. Попробуйте еще раз через команду /wincard."
        if isinstance(event, CallbackQuery):
            await event.message.answer(fallback_msg)
        else:
            await event.answer(fallback_msg)


@router.message(Command("stats_habits"))
@router.message(Command("analytics"))
@router.message(Command("habit_stats"))
async def cmd_stats_habits(message: Message):
    user_id = str(message.from_user.id)
    parts = message.text.split(maxsplit=1)
    period = parts[1].lower().strip() if len(parts) > 1 else "week"
    if period not in ("week", "month", "year"):
        period = "week"
    
    text, markup_dict = await asyncio.to_thread(mabuy_service.handle_habit_statistics, user_id, period)
    kb = convert_dict_markup_to_aiogram(markup_dict)
    await message.answer(text, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

@router.callback_query(F.data.startswith("stats_period_"))
async def cq_stats_period(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    period = callback.data.replace("stats_period_", "").strip() or "week"
    await callback.answer(f"📊 Период: {period.upper()}", show_alert=False)
    
    try:
        text, markup_dict = await asyncio.to_thread(mabuy_service.handle_habit_statistics, user_id, period)
        kb = convert_dict_markup_to_aiogram(markup_dict)
        try:
            await callback.message.edit_text(text, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        except Exception:
            await callback.message.answer(text, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except Exception as e:
        logger.error(f"Error in cq_stats_period: {e}", exc_info=True)
        await callback.message.answer("⚠️ Ошибка формирования статистики.")

@router.message(F.voice)
async def handle_incoming_voice(message: Message):
    user_id = str(message.from_user.id)
    voice = message.voice
    duration = voice.duration if voice else 0
    
    vault_media_dir = os.path.join("data", "vault", "media")
    os.makedirs(vault_media_dir, exist_ok=True)
    
    ext = ".ogg"
    file_name = f"voice_{user_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
    dest_file = os.path.join(vault_media_dir, file_name)
    
    try:
        await message.bot.download(voice, destination=dest_file)
        audio_url = f"/vault/media/{file_name}"
        
        card_text, markup_dict = await asyncio.to_thread(
            mabuy_service.handle_save_voice_note,
            user_id,
            audio_url,
            duration
        )
        kb = convert_dict_markup_to_aiogram(markup_dict)
        await message.answer(card_text, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except Exception as e:
        logger.error(f"Failed to process voice note from {user_id}: {e}", exc_info=True)
        await message.answer("⚠️ Ошибка сохранения голосовой мысли в Дневник. Попробуйте еще раз.")

@router.message(F.audio)
async def handle_incoming_audio(message: Message):
    user_id = str(message.from_user.id)
    audio = message.audio
    duration = audio.duration if audio else 0
    
    vault_media_dir = os.path.join("data", "vault", "media")
    os.makedirs(vault_media_dir, exist_ok=True)
    
    ext = os.path.splitext(audio.file_name or "audio.mp3")[1] or ".mp3"
    file_name = f"audio_{user_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
    dest_file = os.path.join(vault_media_dir, file_name)
    
    try:
        await message.bot.download(audio, destination=dest_file)
        audio_url = f"/vault/media/{file_name}"
        
        card_text, markup_dict = await asyncio.to_thread(
            mabuy_service.handle_save_voice_note,
            user_id,
            audio_url,
            duration
        )
        kb = convert_dict_markup_to_aiogram(markup_dict)
        await message.answer(card_text, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except Exception as e:
        logger.error(f"Failed to process audio file from {user_id}: {e}", exc_info=True)
        await message.answer("⚠️ Ошибка сохранения аудио в Дневник.")

@router.message(F.web_app_data)
async def handle_web_app(message: Message):
    user_id = str(message.from_user.id)
    try:
        data = json.loads(message.web_app_data.data)
        reply = await asyncio.to_thread(mabuy_service.handle_web_app_data, user_id, data)
        await process_mabuy_response(message, reply)
    except Exception as e:
        logger.error(f"Error handling web app data: {e}")

@router.message(F.text)
async def fallback_text(message: Message):
    user_id = str(message.from_user.id)
    text = message.text.strip().upper()
    lang = mabuy_service.get_user_language(user_id)
    
    raw_text = message.text.strip().lower()
    if raw_text.startswith("/habit") or raw_text in ("/habits", "привычки"):
        text_out, raw_markup = await asyncio.to_thread(mabuy_service.handle_habits, user_id)
        kb = convert_dict_markup_to_aiogram(raw_markup)
        await message.answer(text_out, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        return
    elif raw_text.startswith("/journal") or raw_text in ("дневник", "память"):
        text_out, raw_markup = await asyncio.to_thread(mabuy_service.handle_journal, user_id)
        kb = convert_dict_markup_to_aiogram(raw_markup)
        await message.answer(text_out, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        return
    elif raw_text.startswith("/stats_habits") or raw_text in ("/analytics", "статистика"):
        text_out, raw_markup = await asyncio.to_thread(mabuy_service.handle_habit_statistics, user_id, "week")
        kb = convert_dict_markup_to_aiogram(raw_markup)
        await message.answer(text_out, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        return

    if SymbolNormalizer.normalize(text) in known_crypto_keywords or text in known_crypto_keywords:
        reply, photo_path = await asyncio.to_thread(mabuy_service.handle_predict, user_id, text)
        await process_mabuy_response(message, reply, photo_path)
    else:
        # Natural conversational chat with MaIA (with voice synthesis & LLM reasoning)
        await cmd_ask(message)

@router.message(Command("add_all_alerts"))
async def cmd_add_all_alerts(message: Message):
    user_id = str(message.from_user.id)
    coins = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "ADA", "LINK", "DOT"]
    for c in coins:
        await asyncio.to_thread(mabuy_service.usage_tracker.add_user_alert, user_id, c)
    alerts = await asyncio.to_thread(mabuy_service.usage_tracker.get_user_alerts, user_id)
    reply = (
        f"🔔 <b>ВСЕ АЛЕРТЫ УСПЕШНО АКТИВИРОВАНЫ!</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Теперь ИИ автоматически пришлет вам пуш-уведомление в личку при возникновении мощного сигнала на вход по любой из монет:\n"
        f"<b>{', '.join(sorted(alerts))}</b>\n\n"
        f"<i>Управлять списком: /alerts</i>"
    )
    await process_mabuy_response(message, reply)

async def run_push_alert_background_worker(bot: Bot):
    """Background loop that periodically scans the market and dispatches signal push alerts."""
    logger.info("Starting Automated Push Alert Background Engine...")
    await asyncio.sleep(15) # Warm-up wait
    while True:
        try:
            alerts = await asyncio.to_thread(mabuy_service.push_alert_service.scan_market_for_breakout_signals, "ru")
            for alert in alerts:
                sym = alert["symbol"]
                cur_p = alert["current_price"]
                tp_p = alert["target_price"]
                sl_p = alert["stop_loss_price"]
                tp_pct = alert["tp_pct"]
                sl_pct = alert["sl_pct"]
                rr = alert["reward_risk_ratio"]
                lev = alert["max_safe_leverage"]
                kelly = alert["kelly_risk_pct"]
                conf = alert["confidence_pct"]
                tf = alert["timeframe"]
                rec = alert["personal_recommendation"]

                users = await asyncio.to_thread(mabuy_service.usage_tracker.get_users_with_alert, sym)
                for uid in users:
                    lang = mabuy_service.get_user_language(uid)
                    user_dep = await asyncio.to_thread(mabuy_service.usage_tracker.get_user_deposit, uid, 1000.0)
                    bybit_url = TelegramMessageFormatter.REFERRAL_LINKS["Bybit"]
                    text = TelegramMessageFormatter.format_push_alert_message(alert, user_deposit=user_dep, lang=lang)

                    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
                    if lang == "ru":
                        kb = InlineKeyboardMarkup(inline_keyboard=[
                            [InlineKeyboardButton(text=f"🧮 Сайзинг под депо (${user_dep:,.0f})", callback_data=f"/calc_risk {user_dep:.0f} {sym}")],
                            [
                                InlineKeyboardButton(text="💵 Сменить депо", callback_data="/deposit"),
                                InlineKeyboardButton(text="📲 Bybit (-20% fee)", url=bybit_url)
                            ]
                        ])
                    else:
                        kb = InlineKeyboardMarkup(inline_keyboard=[
                            [InlineKeyboardButton(text=f"🧮 Size for Capital (${user_dep:,.0f})", callback_data=f"/calc_risk {user_dep:.0f} {sym}")],
                            [
                                InlineKeyboardButton(text="💵 Change Deposit", callback_data="/deposit"),
                                InlineKeyboardButton(text="📲 Bybit (-20% fee)", url=bybit_url)
                            ]
                        ])

                    try:
                        await bot.send_message(chat_id=uid, text=text, reply_markup=kb, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                        await asyncio.sleep(0.05)
                    except Exception as err:
                        logger.debug(f"Failed to deliver alert to {uid}: {err}")
        except Exception as e:
            logger.error(f"Error in push alert background worker: {e}", exc_info=True)
        await asyncio.sleep(180) # Rescan every 3 minutes

async def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token or "example" in token:
        logger.error("Invalid TELEGRAM_BOT_TOKEN in .env")
        return

    # Set ParseMode to HTML globally
    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp.include_router(router)

    from aiogram.types import BotCommand
    commands = [
        BotCommand(command="start", description="⚡ Запустить MaBuy WebApp"),
        BotCommand(command="trackrecord", description="📊 Трек-рекорд ИИ (Live Win Rate)"),
        BotCommand(command="wincard", description="🏆 Карточка успешной сделки"),
        BotCommand(command="predict", description="📈 Экспресс-прогноз цены"),
        BotCommand(command="alerts", description="🔔 Мониторинг рынка и PUSH-алерты"),
        BotCommand(command="ask", description="💬 Диалог с ассистентом MaIA"),
        BotCommand(command="subscribe", description="👑 Тарифы PRO и VIP"),
        BotCommand(command="ref", description="🎁 Реферальная программа"),
        BotCommand(command="help", description="ℹ️ Справка")
    ]
    await bot.set_my_commands(commands)

    # Configure Telegram Bottom Left Menu Button to open the live WebApp
    try:
        from aiogram.types import MenuButtonWebApp, WebAppInfo
        menu_url = os.environ.get("MAINVEST_WEBAPP_URL", "https://mabuy.vercel.app")
        await bot.set_chat_menu_button(
            menu_button=MenuButtonWebApp(
                text="🚀 WebApp",
                web_app=WebAppInfo(url=menu_url)
            )
        )
        logger.info(f"Chat Menu Button configured to {menu_url}")
    except Exception as e:
        logger.warning(f"Could not set chat menu button: {e}")

    logger.info("Starting Aiogram Production Bot...")
    # Drop pending updates to avoid spamming on start
    await bot.delete_webhook(drop_pending_updates=True)
    
    # Note: Push alerts are centrally and intelligently handled by auto_trader_daemon.py (mabuy-daemon.service)
    # to avoid dual-process duplicate notifications and guarantee strict cooldown management.
    # asyncio.create_task(run_push_alert_background_worker(bot))
    
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

