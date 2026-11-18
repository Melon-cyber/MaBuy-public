import os
import sys
import time
import json
import logging
import urllib.request
import urllib.parse
from typing import Optional, Dict, Any

# Ensure UTF-8 output formatting for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.presentation.telegram_bot.bot import MaBuyTelegramBot
from src.presentation.telegram_bot.formatter import TelegramMessageFormatter
from src.domain.services.symbol_normalizer import SymbolNormalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MaBuyLiveBot")


def load_env_file(env_path: str = ".env"):
    """Simple parser to load environment variables from .env file."""
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()


def send_telegram_message(token: str, chat_id: int, text: str, reply_markup: Optional[dict] = None):
    """Sends HTML formatted message to Telegram Chat via REST API with inline buttons."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        logger.error(f"Failed to send Telegram message to [{chat_id}]: {e}")
        return None


def send_telegram_photo(token: str, chat_id: int, photo_path: str, caption: str, reply_markup: Optional[dict] = None):
    """Sends chart PNG image photo with HTML caption via multipart/form-data POST."""
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    boundary = "----WebKitFormBoundaryMaBuyBot7MA4YWxkTrZu0gW"

    body = bytearray()

    def add_field(name, value):
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(f"{value}\r\n".encode("utf-8"))

    add_field("chat_id", str(chat_id))
    add_field("caption", caption)
    add_field("parse_mode", "HTML")
    if reply_markup:
        add_field("reply_markup", json.dumps(reply_markup))

    # Add photo file
    if os.path.exists(photo_path):
        filename = os.path.basename(photo_path)
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="photo"; filename="{filename}"\r\n'.encode("utf-8"))
        body.extend("Content-Type: image/png\r\n\r\n".encode("utf-8"))
        with open(photo_path, "rb") as f:
            body.extend(f.read())
        body.extend("\r\n".encode("utf-8"))

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            res = json.loads(resp.read().decode())
            if os.path.exists(photo_path):
                os.remove(photo_path)
            return res
    except Exception as e:
        logger.error(f"Failed to send Telegram photo to [{chat_id}]: {e}")
        return send_telegram_message(token, chat_id, caption, reply_markup=reply_markup)


def answer_callback_query(token: str, callback_query_id: str, text: Optional[str] = None):
    """Acknowledges inline button clicks to clear Telegram loading spinner."""
    url = f"https://api.telegram.org/bot{token}/answerCallbackQuery"
    payload = {"callback_query_id": callback_query_id}
    if text:
        payload["text"] = text

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.warning(f"Failed to answer callback query [{callback_query_id}]: {e}")


def poll_live_telegram_updates(token: str, bot_service: MaBuyTelegramBot):
    """
    Live Long-Polling Loop receiving real user updates from Telegram.
    Supports PNG Chart Card Rendering, Referral Engine, Portfolio Audits, Correlation Matrix, and Push Signals.
    """
    logger.info("Starting MaBuy Commercial Telegram Polling Engine...")
    offset = 0

    get_me_url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        with urllib.request.urlopen(get_me_url, timeout=10) as resp:
            me_data = json.loads(resp.read().decode())
            if me_data.get("ok"):
                bot_info = me_data["result"]
                logger.info(f"Connected to Telegram Bot: @{bot_info.get('username')} ({bot_info.get('first_name')})")
    except Exception as e:
        logger.error(f"Failed to connect to Telegram API with provided token: {e}")
    # Ensure any active webhooks are deleted before long polling starts
    try:
        del_wh_url = f"https://api.telegram.org/bot{token}/deleteWebhook?drop_pending_updates=True"
        with urllib.request.urlopen(del_wh_url, timeout=10) as resp:
            del_data = json.loads(resp.read().decode())
            logger.info(f"Cleared active webhooks for polling: {del_data}")
    except Exception as wh_err:
        logger.warning(f"Note on webhook cleanup: {wh_err}")

    known_crypto_keywords = {"BTC", "ETH", "BNB", "SOL", "XRP", "BITCOIN", "ETHEREUM", "SOLANA", "RIPPLE", "BINANCECOIN", "БИТКОИН", "ЭФИР", "СОЛАНА"}

    while True:
        try:
            url = f"https://api.telegram.org/bot{token}/getUpdates?offset={offset}&timeout=20"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=25) as response:
                result = json.loads(response.read().decode())

            if result.get("ok"):
                updates = result.get("result", [])
                for update in updates:
                    update_id = update["update_id"]
                    offset = max(offset, update_id + 1)

                    chat_id = None
                    text = None
                    user_id = None

                    if "callback_query" in update:
                        cb = update["callback_query"]
                        cb_id = cb["id"]
                        chat_id = cb["message"]["chat"]["id"]
                        text = cb.get("data", "")
                        user_id = str(cb["from"]["id"])
                        answer_callback_query(token, cb_id)

                    elif "message" in update and "text" in update["message"]:
                        msg = update["message"]
                        chat_id = msg["chat"]["id"]
                        text = msg["text"].strip()
                        user_id = str(msg["from"]["id"])

                    if not chat_id or not text or not user_id:
                        continue

                    logger.info(f"Processing command for User [{user_id}]: '{text}'")
                    clean_text = text.strip()
                    upper_text = clean_text.upper()
                    lang = bot_service.get_user_language(user_id)
                    reply_markup = TelegramMessageFormatter.get_inline_keyboard_markup(lang=lang)
                    photo_path_to_send = None

                    if clean_text == "set_lang_ru":
                        bot_service.set_user_language(user_id, "ru")
                        reply = bot_service.handle_start(user_id)
                        reply_markup = TelegramMessageFormatter.get_inline_keyboard_markup(lang="ru")

                    elif clean_text == "set_lang_en":
                        bot_service.set_user_language(user_id, "en")
                        reply = bot_service.handle_start(user_id)
                        reply_markup = TelegramMessageFormatter.get_inline_keyboard_markup(lang="en")

                    elif clean_text.startswith("/start"):
                        parts = clean_text.split()
                        referrer_id = None
                        if len(parts) > 1 and parts[1].startswith("ref_"):
                            referrer_id = parts[1].replace("ref_", "")
                        reply, reply_markup = bot_service.handle_start_greeting(user_id, referrer_id=referrer_id)

                    elif clean_text.startswith("/language") or clean_text.startswith("/lang"):
                        reply = "🌐 <b>Выберите язык / Select Language:</b>"
                        reply_markup = TelegramMessageFormatter.get_language_selection_markup()

                    elif clean_text.startswith("/matrix") or clean_text.startswith("/correlation"):
                        reply = bot_service.handle_correlation_matrix(user_id)

                    elif clean_text.startswith("/ref") or clean_text.startswith("/referral"):
                        reply = bot_service.handle_referral_info(user_id)

                    elif clean_text.startswith("/portfolio"):
                        parts = clean_text.split(maxsplit=1)
                        args_str = parts[1] if len(parts) > 1 else ""
                        reply = bot_service.handle_portfolio_audit(user_id, args_str)

                    elif clean_text.startswith("/alerts") or clean_text.startswith("/signals"):
                        reply = bot_service.handle_alerts(user_id)

                    elif clean_text.startswith("/add_alert"):
                        parts = clean_text.split(maxsplit=1)
                        sym = parts[1] if len(parts) > 1 else ""
                        reply = bot_service.handle_add_alert(user_id, sym) if sym else "⚠️ Укажите монету. Пример: /add_alert BTC"

                    elif clean_text.startswith("/del_alert"):
                        parts = clean_text.split(maxsplit=1)
                        sym = parts[1] if len(parts) > 1 else ""
                        reply = bot_service.handle_del_alert(user_id, sym) if sym else "⚠️ Укажите монету. Пример: /del_alert BTC"

                    elif clean_text.startswith("/admin"):
                        reply = bot_service.handle_admin_dashboard(user_id)

                    elif clean_text.startswith("/habit"):
                        if clean_text.startswith("/habit_toggle"):
                            h_id = clean_text.replace("/habit_toggle", "").strip()
                            reply, reply_markup = bot_service.handle_habit_toggle(user_id, h_id)
                        else:
                            reply, reply_markup = bot_service.handle_habits(user_id)

                    elif clean_text.startswith("/journal"):
                        reply, reply_markup = bot_service.handle_journal(user_id)

                    elif clean_text.startswith("/reset"):
                        reply = bot_service.reset_user_data(user_id)

                    elif clean_text.startswith("/subscribe"):
                        reply = bot_service.handle_subscribe_catalog(user_id)

                    elif clean_text.startswith("/buy") or clean_text.startswith("/upgrade"):
                        parts = clean_text.split(maxsplit=1)
                        tier = parts[1] if len(parts) > 1 else "pro"
                        reply = bot_service.handle_buy(user_id, tier)

                    elif clean_text.startswith("/confirm_pay"):
                        parts = clean_text.split(maxsplit=1)
                        inv_id = parts[1] if len(parts) > 1 else ""
                        reply = bot_service.handle_confirm_pay(user_id, inv_id)

                    elif clean_text.startswith("/predict") or clean_text.startswith("/предикт") or clean_text.startswith("/прогноз") or clean_text.startswith("/p"):
                        parts = clean_text.split(maxsplit=1)
                        args_str = parts[1] if len(parts) > 1 else "BTC"
                        reply, photo_path_to_send = bot_service.handle_predict(user_id, args_str)

                    elif clean_text.startswith("/tier"):
                        reply = bot_service.handle_tier_info(user_id)

                    elif clean_text.startswith("/help"):
                        reply = bot_service.handle_help(user_id)

                    elif SymbolNormalizer.normalize(upper_text) in known_crypto_keywords or upper_text in known_crypto_keywords:
                        reply, photo_path_to_send = bot_service.handle_predict(user_id, upper_text)

                    else:
                        reply = (
                            "🤖 Нажмите кнопку ниже или отправьте <code>/predict BTC</code> для мгновенного прогноза с графиком."
                            if lang == "ru" else
                            "🤖 Send <code>/predict BTC</code> or tap a button below to get an instant ML price forecast and chart."
                        )

                    if photo_path_to_send and os.path.exists(photo_path_to_send):
                        send_telegram_photo(token, chat_id, photo_path_to_send, reply, reply_markup=reply_markup)
                    else:
                        send_telegram_message(token, chat_id, reply, reply_markup=reply_markup)

        except Exception as e:
            logger.warning(f"Polling loop exception: {e}. Retrying in 3 seconds...")
            time.sleep(3)


import threading
import random

def channel_broadcaster_loop(token: str, bot_service: MaBuyTelegramBot, channel_id: str):
    logger.info(f"Starting Channel Broadcaster for {channel_id}...")
    symbols = ["BTC", "ETH", "SOL", "BNB", "XRP"]
    while True:
        try:
            symbol = random.choice(symbols)
            logger.info(f"Broadcasting to channel {channel_id}: {symbol}")
            
            text_reply, chart_path = bot_service.generate_channel_broadcast(symbol=symbol)
            
            send_telegram_photo(token, channel_id, chart_path, text_reply)
        except Exception as e:
            logger.error(f"Broadcaster exception: {e}")
        
        # Broadcast every 3 hours (10800 seconds)
        time.sleep(10800)


def main():
    load_env_file(".env")
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    channel_id = os.environ.get("TELEGRAM_CHANNEL_ID")

    if not token or "example" in token:
        logger.error("TELEGRAM_BOT_TOKEN is missing or invalid in .env")
        return

    bot_service = MaBuyTelegramBot()
    
    if channel_id and "@" in channel_id:
        threading.Thread(target=channel_broadcaster_loop, args=(token, bot_service, channel_id), daemon=True).start()

    poll_live_telegram_updates(token, bot_service)


if __name__ == "__main__":
    main()
