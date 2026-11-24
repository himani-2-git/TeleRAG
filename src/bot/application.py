"""Builds the Telegram application: network timeouts, concurrency and shutdown logging."""

import logging

from telegram.ext import Application

logger = logging.getLogger(__name__)

NETWORK_TIMEOUT_SECONDS = 30.0


async def _log_shutdown(application: Application) -> None:
    """Runs after the application has stopped. Stopping first waits for running handlers."""
    logger.info("Bot stopped. All requests that were running have finished.")


def build_application(config, bot_handlers) -> Application:
    """Create the Application and attach all handlers.

    concurrent_updates lets the bot process several updates at the same time, so one slow
    answer does not block /list, uploads or the questions of other users. The LLM itself is
    still used by one question at a time (see BotHandlers.llm_semaphore).

    Shutdown: when the bot receives Ctrl+C or SIGTERM, python-telegram-bot stops fetching new
    updates and waits until the updates already being processed are finished.
    """
    application = (
        Application.builder()
        .token(config.telegram_bot_token)
        .concurrent_updates(config.concurrent_updates)
        .connect_timeout(NETWORK_TIMEOUT_SECONDS)
        .read_timeout(NETWORK_TIMEOUT_SECONDS)
        .write_timeout(NETWORK_TIMEOUT_SECONDS)
        .pool_timeout(NETWORK_TIMEOUT_SECONDS)
        .get_updates_read_timeout(NETWORK_TIMEOUT_SECONDS)
        .post_shutdown(_log_shutdown)
        .build()
    )
    bot_handlers.register_handlers(application)
    return application
