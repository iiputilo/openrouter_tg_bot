import os
import io
import base64
import asyncio
import logging
import tempfile
import traceback
from typing import List

import openrouter
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.helpers import escape_markdown
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    filters,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

load_dotenv()

TELEGRAM_BOT_KEY = os.getenv("TELEGRAM_BOT_TOKEN")

MAX_MESSAGE_LENGTH = 4096
MAX_PDF_CHARS = int(os.getenv("MAX_PDF_CHARS", "20000000"))
BOT_REQUEST_TIMEOUT = float(os.getenv("BOT_REQUEST_TIMEOUT", "3600"))


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = getattr(context, "error", None)

    if err is not None:
        tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
        logging.error("Unhandled exception while processing update:\n%s", tb)
    else:
        logging.error("Unhandled exception while processing update (no context.error)")

    chat = getattr(update, "effective_chat", None)
    if chat and getattr(chat, "id", None):
        try:
            await context.bot.send_message(
                chat_id=chat.id,
                text="Internal error occurred. Try again later",
            )
        except Exception:
            logging.warning("Failed to notify user about the error", exc_info=True)


async def _run_with_timeout(coro, timeout_s: float):
    return await asyncio.wait_for(coro, timeout=timeout_s)


async def _safe_delete(msg) -> None:
    try:
        await msg.delete()
    except Exception:
        pass


def chunk_plain(text: str, limit: int = 4096):
    t = text or ""
    i, n = 0, len(t)
    while i < n:
        j = min(n, i + limit)
        if j < n:
            k = t.rfind("\n", i, j)
            if k != -1 and k >= i + limit - 512:
                j = k
        yield t[i:j]
        i = j


async def send_plain_text(bot, chat_id: int, text: str) -> None:
    await bot.send_message(chat_id=chat_id, text=text or "")


async def _download_to_bytes(context: ContextTypes.DEFAULT_TYPE, file_id: str) -> bytes:
    tg_file = await context.bot.get_file(file_id)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        await tg_file.download_to_drive(custom_path=tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


async def _file_to_data_uri(context: ContextTypes.DEFAULT_TYPE, file_id: str, mime: str) -> str:
    data = await _download_to_bytes(context, file_id)
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _extract_pdf_text(pdf_bytes: bytes, limit_chars: int = 20000) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("Missing dependency: pypdf") from exc

    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts: List[str] = []
    total = 0

    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt:
            parts.append(txt)
            total += len(txt)
            if total >= limit_chars:
                break

    text = "\n\n".join(parts).strip()
    if len(text) > limit_chars:
        text = text[:limit_chars]
    return text


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    base = (
        "This is a bot who can help you with access to the OpenRouter API\n"
        "Default model is ***openai\\/gpt\\-5***\n"
        "To see help use /help command"
    )
    await update.message.reply_text(base, parse_mode=ParseMode.MARKDOWN_V2)

    if not openrouter.has_api_key(update.effective_user.id):
        await update.message.reply_text(
            "You haven't set OpenRouter API key\n"
            "Send `/change\\_api\\_key <your\\_key>` to save your key",
            parse_mode=ParseMode.MARKDOWN_V2,
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "📖 ***Command help:***\n"
        "/start \\- start a bot\n"
        "/reset \\- reset message history\n"
        "/set\\_history \\<n\\> \\- set history size \\(messages\\)\n"
        "/history \\- show current history size\n"
        "/switch\\_model \\- switch to the another OpenRouter LLM\n"
        "/change\\_api\\_key \\<key\\> \\- set or change your OpenRouter API key\n"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN_V2)


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    openrouter.reset_history(update.effective_user.id)
    await update.message.reply_text("Message history cleared")


async def change_api_key_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tg_id = update.effective_user.id
    new_key = " ".join(context.args).strip() if context.args else ""
    if not new_key:
        await update.message.reply_text(
            "Using: `/change\\_api\\_key <your\\_key>`",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    openrouter.set_user_api_key(tg_id, new_key)
    await update.message.reply_text("API key was saved")


async def set_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tg_id = update.effective_user.id
    raw = (context.args[0] if context.args else "").strip()
    if not raw:
        await update.message.reply_text(
            "Using: `/set\\_history <n>`",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    try:
        n = int(raw)
    except ValueError:
        await update.message.reply_text(
            "n must be an integer\\",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    saved = openrouter.set_user_max_history(tg_id, n)
    await update.message.reply_text(
        f"History size set to {saved} messages",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tg_id = update.effective_user.id
    n = openrouter.get_user_max_history(tg_id)
    await update.message.reply_text(
        f"Current history size: {n} messages",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def switch_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tg_id = update.effective_user.id
    current = openrouter.get_user_model(tg_id)
    safe_model = escape_markdown(current, version=2)

    buttons = [
        [InlineKeyboardButton(m, callback_data=f"set_model:{m}")]
        for m in openrouter.available_models
    ]
    markup = InlineKeyboardMarkup(buttons)

    text = f"Your current model is ***{safe_model}***\nSelect another model:"
    await update.message.reply_text(
        text,
        reply_markup=markup,
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def set_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    _, model = query.data.split(":", 1)
    openrouter.set_user_model(query.from_user.id, model)
    safe_model = escape_markdown(model, version=2)
    await query.edit_message_text(
        f"Model switched to ***{safe_model}***",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def _process_media_group(context: ContextTypes.DEFAULT_TYPE) -> None:
    job = context.job
    chat_id = job.chat_id
    group_id = job.data["group_id"]

    album_store = context.chat_data.get("albums", {})
    group = album_store.pop(group_id, None)
    if not group:
        return

    items = group["items"]
    tg_id = group["user_id"]

    if not openrouter.has_api_key(tg_id):
        await send_plain_text(
            context.bot,
            chat_id,
            "You haven't set OpenRouter API key.\n"
            "Send /change_api_key <your_key> to save your key",
        )
        return

    texts: List[str] = []
    image_uris: List[str] = []

    for it in items:
        if it.get("text"):
            texts.append(it["text"])

        if it.get("kind") == "image":
            try:
                uri = await _file_to_data_uri(context, it["file_id"], it["mime"])
                image_uris.append(uri)
            except Exception as exc:
                logging.warning("Failed to process media in album: %s", exc)

        if it.get("kind") == "pdf":
            try:
                pdf_bytes = await _download_to_bytes(context, it["file_id"])
                pdf_text = _extract_pdf_text(pdf_bytes, limit_chars=MAX_PDF_CHARS)
                if pdf_text:
                    texts.append(f"[PDF]\n{pdf_text}")
                else:
                    texts.append("[PDF] (no extractable text)")
            except Exception as exc:
                logging.warning("Failed to process pdf in album: %s", exc)
                texts.append("[PDF] (failed to extract text)")

    user_text = "\n".join([t for t in texts if t]).strip()

    processing_msg = await context.bot.send_message(
        chat_id=chat_id,
        text="OpenRouter is processing your request…",
    )

    try:
        response_text = await _run_with_timeout(
            openrouter.get_response(user_text if user_text else None, tg_id, image_uris or None),
            BOT_REQUEST_TIMEOUT,
        )
        await _safe_delete(processing_msg)

        for part in chunk_plain(response_text, MAX_MESSAGE_LENGTH):
            await send_plain_text(context.bot, chat_id, part)

    except asyncio.TimeoutError:
        await _safe_delete(processing_msg)
        await context.bot.send_message(chat_id=chat_id, text="Timeout. Try again later.")
    except TelegramError as exc:
        logging.warning("Telegram error: %s", exc)
        await _safe_delete(processing_msg)
    except Exception as exc:
        logging.error("Failed to process media group: %s", exc, exc_info=True)
        await _safe_delete(processing_msg)
        await context.bot.send_message(chat_id=chat_id, text="OpenRouter error. Try again later")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = (update.message.text or update.message.caption or "").strip()

    if update.message.media_group_id:
        mgid = update.message.media_group_id
        album_store = context.chat_data.setdefault("albums", {})
        group = album_store.setdefault(
            mgid, {"items": [], "job": None, "user_id": update.effective_user.id}
        )

        text = (update.message.caption or update.message.text or "").strip()

        if update.message.photo:
            photo = update.message.photo[-1]
            group["items"].append(
                {"kind": "image", "file_id": photo.file_id, "mime": "image/jpeg", "text": text}
            )
        elif update.message.document and update.message.document.mime_type:
            mt = update.message.document.mime_type
            if mt.startswith("image/"):
                group["items"].append(
                    {"kind": "image", "file_id": update.message.document.file_id, "mime": mt, "text": text}
                )
            elif mt == "application/pdf":
                group["items"].append(
                    {"kind": "pdf", "file_id": update.message.document.file_id, "mime": mt, "text": text}
                )

        if group["job"]:
            group["job"].schedule_removal()

        group["job"] = context.job_queue.run_once(
            _process_media_group,
            when=1.0,
            chat_id=update.effective_chat.id,
            data={"group_id": mgid},
        )
        return

    if not openrouter.has_api_key(update.effective_user.id):
        await send_plain_text(
            context.bot,
            update.effective_chat.id,
            "You haven't set OpenRouter API key.\n"
            "Send /change_api_key <your_key> to save your key",
        )
        return

    image_uris: List[str] = []
    pdf_text_block: str = ""

    if update.message.photo:
        photo = update.message.photo[-1]
        try:
            image_uris.append(await _file_to_data_uri(context, photo.file_id, "image/jpeg"))
        except Exception as exc:
            logging.warning("Failed to process photo: %s", exc)

    if update.message.document and update.message.document.mime_type:
        mt = update.message.document.mime_type

        if mt.startswith("image/"):
            try:
                image_uris.append(await _file_to_data_uri(context, update.message.document.file_id, mt))
            except Exception as exc:
                logging.warning("Failed to process image document: %s", exc)
        elif mt == "application/pdf":
            try:
                pdf_bytes = await _download_to_bytes(context, update.message.document.file_id)
                pdf_text = _extract_pdf_text(pdf_bytes, limit_chars=MAX_PDF_CHARS)
                if pdf_text:
                    pdf_text_block = f"[PDF]\n{pdf_text}"
                else:
                    pdf_text_block = "[PDF] (no extractable text)"
            except Exception as exc:
                logging.warning("Failed to process pdf document: %s", exc)
                pdf_text_block = "[PDF] (failed to extract text)"

    processing_msg = await update.message.reply_text("OpenRouter is processing your request…")

    try:
        combined_text = "\n\n".join([t for t in [user_text, pdf_text_block] if t]).strip()

        if not combined_text and not image_uris:
            await _safe_delete(processing_msg)
            await update.message.reply_text("Attach an image, PDF, or send a message")
            return

        response_text = await _run_with_timeout(
            openrouter.get_response(
                combined_text if combined_text else None,
                update.effective_user.id,
                image_uris or None,
            ),
            BOT_REQUEST_TIMEOUT,
        )
        await _safe_delete(processing_msg)

        for part in chunk_plain(response_text, MAX_MESSAGE_LENGTH):
            await send_plain_text(context.bot, update.effective_chat.id, part)

    except asyncio.TimeoutError:
        await _safe_delete(processing_msg)
        await update.message.reply_text("Timeout. Try again later.")
    except TelegramError as exc:
        logging.warning("Telegram error: %s", exc)
        await _safe_delete(processing_msg)
    except Exception as exc:
        logging.error("Unhandled error in handle_message: %s", exc, exc_info=True)
        await _safe_delete(processing_msg)
        await update.message.reply_text("OpenRouter error. Try again later.")


def main() -> None:
    if not TELEGRAM_BOT_KEY:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var")

    app = ApplicationBuilder().token(TELEGRAM_BOT_KEY).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("switch_model", switch_model_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("change_api_key", change_api_key_command))
    app.add_handler(CommandHandler("set_history", set_history_command))
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CallbackQueryHandler(set_model_callback, pattern=r"^set_model:"))

    message_filter = (filters.TEXT | filters.PHOTO | filters.Document.IMAGE | filters.Document.PDF) & ~filters.COMMAND
    app.add_handler(MessageHandler(message_filter, handle_message))

    app.add_error_handler(error_handler)
    app.run_polling()


if __name__ == "__main__":
    main()
