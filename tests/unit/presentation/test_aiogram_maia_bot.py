import os
os.environ["ENCRYPTION_KEY"] = "UN_5bf8rBxMiQOudLLjBJz42sY6vWznie0oxcjcKLYQ="
os.environ["TELEGRAM_BOT_TOKEN"] = "123456789:ABCDefGhIJKlmNoPQRsTUVwxyZ"
import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from aiogram.types import Message, CallbackQuery, User, Chat

from scripts.launch_aiogram_bot import (
    cmd_persona,
    cq_set_persona,
    cmd_ask,
    cq_ask_btc,
    cq_ask_sizing,
    user_persona_map,
    PersonaType,
)

class TestAiogramMaIABot(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.user = User(id=100000001, is_bot=False, first_name="Admin", username="owner")
        self.chat = Chat(id=100000001, type="private")

    async def test_cmd_persona_message(self):
        message = MagicMock(spec=Message)
        message.from_user = self.user
        message.chat = self.chat
        message.answer = AsyncMock()

        await cmd_persona(message)
        message.answer.assert_awaited_once()
        args, kwargs = message.answer.call_args
        self.assertIn("ПАНЕЛЬ НАСТРОЙКИ MAIA", args[0])
        self.assertIn("OLED-экран", args[0])

    async def test_cq_set_persona_zen(self):
        cq = MagicMock(spec=CallbackQuery)
        cq.from_user = self.user
        cq.data = "set_persona_zen"
        cq.answer = AsyncMock()
        cq.message = MagicMock(spec=Message)
        cq.message.edit_text = AsyncMock()

        await cq_set_persona(cq)
        self.assertEqual(user_persona_map.get("100000001"), PersonaType.ZEN)
        cq.answer.assert_awaited_once()
        cq.message.edit_text.assert_awaited_once()
        args, kwargs = cq.message.edit_text.call_args
        self.assertIn("ДЗЕН", args[0])

    async def test_cmd_ask_with_query(self):
        message = MagicMock(spec=Message)
        message.from_user = self.user
        message.chat = self.chat
        message.text = "/ask Что думаешь по SOL?"
        message.bot = MagicMock()
        message.bot.send_chat_action = AsyncMock()
        message.answer = AsyncMock()

        await cmd_ask(message)
        self.assertGreaterEqual(message.answer.await_count, 1)
        last_call_args = message.answer.call_args_list[-1]
        self.assertIn("DYOR", last_call_args[0][0])

    async def test_cmd_ask_without_query_shows_help(self):
        message = MagicMock(spec=Message)
        message.from_user = self.user
        message.chat = self.chat
        message.text = "/ask"
        message.bot = MagicMock()
        message.bot.send_chat_action = AsyncMock()
        message.answer = AsyncMock()

        await cmd_ask(message)
        message.answer.assert_awaited_once()
        args, kwargs = message.answer.call_args
        self.assertIn("ФИНАНСОВЫЙ ИИ-КОМПАНЬОН MAIA", args[0])

    async def test_cq_ask_btc_and_sizing(self):
        cq = MagicMock(spec=CallbackQuery)
        cq.from_user = self.user
        cq.data = "/ask_btc"
        cq.answer = AsyncMock()
        cq.message = MagicMock(spec=Message)
        cq.message.chat = self.chat
        cq.message.bot = MagicMock()
        cq.message.bot.send_chat_action = AsyncMock()
        cq.message.answer = AsyncMock()

        await cq_ask_btc(cq)
        self.assertGreaterEqual(cq.message.answer.await_count, 1)
        last_call_args = cq.message.answer.call_args_list[-1]
        self.assertIn("DYOR", last_call_args[0][0])

if __name__ == "__main__":
    unittest.main()
