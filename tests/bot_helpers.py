"""Shared helpers for tests that drive BotHandlers with fake Telegram objects."""

from unittest.mock import AsyncMock, Mock

from src.bot.handlers import BotHandlers
from src.config import Config


def make_handlers(**config_values):
    """Build BotHandlers with mocked dependencies and a real Config."""
    document_store = Mock()
    document_store.list_documents.return_value = [{"id": 1, "filename": "a.txt"}]
    handlers = BotHandlers(
        pdf_processor=Mock(),
        text_chunker=Mock(),
        document_store=document_store,
        embedding_generator=Mock(),
        retriever=Mock(),
        llm_client=Mock(),
        config=Config(telegram_bot_token="test", **config_values),
    )
    return handlers


def make_update(user_id=1, text="hello"):
    """Build a fake Telegram update whose reply_text records every reply."""
    reply = AsyncMock()
    reply.return_value.delete = AsyncMock()
    update = Mock()
    update.effective_user.id = user_id
    update.effective_chat.id = 99
    update.message.text = text
    update.message.reply_text = reply
    update.effective_message.reply_text = reply
    return update


def make_context():
    context = Mock()
    context.bot.send_chat_action = AsyncMock()
    context.args = []
    return context


def reply_texts(update):
    return [call.args[0] for call in update.message.reply_text.call_args_list]
