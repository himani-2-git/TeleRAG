"""Command handlers for Telegram bot (/start, /help, /list, /delete, /clear, /addurl)."""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any

from telegram import Update
from telegram.ext import ContextTypes

from ...logging_setup import with_request_id
from ..health import format_health_report, run_health_checks

logger = logging.getLogger(__name__)


class CommandHandlersMixin:
    """Mixin providing slash-command handlers for BotHandlers."""

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send welcome message and overview of capabilities and commands."""
        welcome_message = (
            "👋 Welcome to PDF RAG Chatbot!\n\n"
            "I can help you analyze PDF, Markdown, and text documents, as well as web pages, "
            "and answer questions about them.\n\n"
            "📤 Send me a PDF, .md, or .txt file\n"
            "🌐 Send /addurl <url> to ingest a static web page\n"
            "🔗 Or send a direct link to a PDF\n"
            "❓ Ask questions about your uploaded documents\n\n"
            "Commands:\n"
            "/list - Show your uploaded documents\n"
            "/delete 2 or /delete part-of-name - Delete document (precedence: exact name, then list number, then unique substring)\n"
            "/clear - Delete all your documents\n"
            "/addurl <url> - Ingest a web page\n"
            "/help - Show this message"
        )
        await update.message.reply_text(welcome_message)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help information."""
        await self.start_command(update, context)

    @staticmethod
    def _format_upload_date(raw_date: Any) -> str:
        """Format document upload timestamp in local time as '21 Oct 2026, 22:51'."""
        if not raw_date:
            return "Unknown"
        if isinstance(raw_date, datetime):
            dt = raw_date
        else:
            try:
                date_str = str(raw_date).replace("Z", "+00:00")
                dt = datetime.fromisoformat(date_str)
            except (ValueError, TypeError):
                return str(raw_date)

        if dt.tzinfo is not None:
            dt = dt.astimezone()
        return dt.strftime("%d %b %Y, %H:%M")

    async def list_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List all documents currently ingested for the user."""
        user_id = update.effective_user.id
        try:
            documents = await asyncio.to_thread(self.document_store.list_documents, user_id)
            if not documents:
                await update.message.reply_text("📭 You haven't uploaded any documents yet.")
                return

            message = "📚 Your uploaded documents:\n\n"
            for i, doc in enumerate(documents, start=1):
                size_mb = doc["file_size"] / (1024 * 1024)
                uploaded_str = self._format_upload_date(doc.get("upload_date"))
                message += f"{i}. 📄 {doc['filename']}\n"
                message += f"   Size: {size_mb:.2f} MB | Chunks: {doc['chunk_count']}\n"
                message += f"   Uploaded: {uploaded_str}\n\n"

            await update.message.reply_text(message)
        except Exception as e:
            logger.exception("Failed to list documents: %s", e)
            await update.message.reply_text("❌ Failed to retrieve document list.")

    async def delete_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Delete a specified document for the user."""
        user_id = update.effective_user.id
        if not context.args:
            await update.message.reply_text(
                "❌ Please specify a document to delete: /delete 2 or /delete part-of-name"
            )
            return

        raw_arg = " ".join(context.args).strip()
        target = raw_arg.strip(" <>\"'").strip()
        if not target:
            await update.message.reply_text(
                "❌ Please specify a document to delete: /delete 2 or /delete part-of-name"
            )
            return

        try:
            documents = await asyncio.to_thread(self.document_store.list_documents, user_id)
            if not documents:
                await update.message.reply_text(
                    "Document not found. Send /list to see names and numbers."
                )
                return

            exact_matches = [d for d in documents if d["filename"].lower() == target.lower()]
            if len(exact_matches) == 1:
                target_doc = exact_matches[0]
            elif len(exact_matches) > 1:
                matches_list = "\n".join(f"• {d['filename']}" for d in exact_matches)
                await update.message.reply_text(
                    f"Multiple documents match '{target}':\n{matches_list}\n"
                    "Please specify the exact name or document number from /list."
                )
                return
            elif target.isdigit() and 1 <= int(target) <= len(documents):
                target_doc = documents[int(target) - 1]
            else:
                substring_matches = [
                    d for d in documents if target.lower() in d["filename"].lower()
                ]
                if len(substring_matches) == 1:
                    target_doc = substring_matches[0]
                elif len(substring_matches) > 1:
                    matches_list = "\n".join(f"• {d['filename']}" for d in substring_matches)
                    await update.message.reply_text(
                        f"Multiple documents match '{target}':\n{matches_list}\n"
                        "Please specify the exact name or document number from /list."
                    )
                    return
                else:
                    await update.message.reply_text(
                        "Document not found. Send /list to see names and numbers."
                    )
                    return

            deleted = await asyncio.to_thread(
                self.document_store.delete_document, user_id, target_doc["filename"]
            )
            if deleted:
                await update.message.reply_text(f"✅ Deleted document: {target_doc['filename']}")
            else:
                await update.message.reply_text(
                    "Document not found. Send /list to see names and numbers."
                )
        except Exception as e:
            logger.exception("Failed to delete document: %s", e)
            await update.message.reply_text("❌ Failed to delete document.")

    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Clear all documents for the user."""
        user_id = update.effective_user.id
        try:
            count = await asyncio.to_thread(self.document_store.clear_user_documents, user_id)
            if count > 0:
                await update.message.reply_text(f"✅ Deleted {count} document(s).")
            else:
                await update.message.reply_text("📭 You don't have any documents to delete.")
        except Exception as e:
            logger.exception("Failed to clear documents: %s", e)
            await update.message.reply_text("❌ Failed to clear documents.")

    @with_request_id
    async def addurl_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Ingest a web page specified by URL."""
        if not context.args:
            await update.message.reply_text("❌ Please specify a URL: /addurl <url>")
            return

        url = context.args[0].strip()
        await self._handle_web_url_ingest(update, context, url)

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show whether the database, embeddings and LLM server work. Admins only."""
        admin_ids = self.config.admin_user_ids or self.config.allowed_user_ids
        if not admin_ids:
            await update.message.reply_text(
                "ℹ️ /status is turned off. Set ADMIN_USER_IDS in the .env file to use it."
            )
            return
        if update.effective_user.id not in admin_ids:
            await update.message.reply_text("🔒 This command is only for the bot admin.")
            return

        try:
            checks = await asyncio.wait_for(
                asyncio.to_thread(
                    run_health_checks,
                    self.document_store,
                    self.embedding_generator,
                    self.llm_client,
                ),
                timeout=30,
            )
        except asyncio.TimeoutError:
            await update.message.reply_text("⏳ The status check took too long. Try again later.")
            return

        uptime_seconds = time.monotonic() - self.started_at
        await update.message.reply_text(
            format_health_report(checks, uptime_seconds, self.active_requests)
        )
