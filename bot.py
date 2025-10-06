import os
import logging
import base64
import tempfile
from typing import List
import re
import openrouter
from dotenv import load_dotenv
from telegram.error import TelegramError
from telegram.constants import ParseMode
from telegram.helpers import escape_markdown
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, CallbackQueryHandler, filters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

load_dotenv()
TELEGRAM_BOT_KEY = os.getenv("TELEGRAM_BOT_TOKEN")
MAX_MESSAGE_LENGTH = 4096

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logging.exception("Error while processing exception")
    if update and getattr(update, "effective_chat", None):
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Internal error occurred. Try again later"
        )

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'This is a bot who can help you with access to OpenRouter API\nDefault model is ***openai\\/gpt\\-5***\n'
        'To see help use /help command',
        parse_mode=ParseMode.MARKDOWN_V2
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
    📖 ***Command help:***
    /start \\- start a bot
    /reset \\- reset message history
    /about \\- project information
    /switch\\_model \\- switch to another OpenRouter LLM
    """
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN_V2)

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    openrouter.reset_history(update.effective_user.id)
    await update.message.reply_text("Message history cleared")

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        ''
    )

async def switch_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tg_id = update.effective_user.id
    current = openrouter.get_user_model(tg_id)
    safe_model = escape_markdown(current, version=2)
    buttons = [
        [InlineKeyboardButton(m, callback_data=f"set_model:{m}")]
        for m in openrouter.AVAILABLE_MODELS
    ]
    markup = InlineKeyboardMarkup(buttons)
    text = f"Your current model is ***{safe_model}***\nSelect another model:"
    await update.message.reply_text(
        text,
        reply_markup=markup,
        parse_mode=ParseMode.MARKDOWN_V2
    )

async def set_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    _, model = query.data.split(":", 1)
    openrouter.set_user_model(query.from_user.id, model)
    safe_model = escape_markdown(model, version=2)
    await query.edit_message_text(
        f"Model switched to ***{safe_model}***",
        parse_mode=ParseMode.MARKDOWN_V2
    )

async def _file_to_data_uri(context: ContextTypes.DEFAULT_TYPE, file_id: str, mime: str) -> str:
    tg_file = await context.bot.get_file(file_id)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        await tg_file.download_to_drive(custom_path=tmp_path)
        with open(tmp_path, 'rb') as f:
            data = f.read()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    b64 = base64.b64encode(data).decode('ascii')
    return f"data:{mime};base64,{b64}"

async def _process_media_group(context: ContextTypes.DEFAULT_TYPE):
    job = context.job
    chat_id = job.chat_id
    group_id = job.data["group_id"]

    album_store = context.chat_data.get("albums", {})
    group = album_store.pop(group_id, None)
    if not group:
        return

    items = group["items"]
    tg_id = group["user_id"]

    texts = []
    image_uris = []
    for it in items:
        if it.get("text"):
            texts.append(it["text"])
        try:
            uri = await _file_to_data_uri(context, it["file_id"], it["mime"])
            image_uris.append(uri)
        except Exception as e:
            logging.warning("Failed to process media in album: %s", e)

    user_text = "\n".join([t for t in texts if t]).strip()

    processing_msg = await context.bot.send_message(chat_id=chat_id, text="OpenRouter is processing your request…")
    try:
        response_text = await openrouter.get_response(user_text if user_text else None, tg_id, image_uris or None)
        await processing_msg.delete()

        safe = escape_markdown(response_text, version=2)
        safe = re.sub(r'\\\*\\\*(.+?)\\\*\\\*', r'***\1***', safe, flags=re.S)
        safe = safe.replace(r'\_', '_').replace(r'\`', '```')

        for i in range(0, len(safe), MAX_MESSAGE_LENGTH):
            part = safe[i:i + MAX_MESSAGE_LENGTH]
            await context.bot.send_message(chat_id=chat_id, text=part, parse_mode=ParseMode.MARKDOWN_V2)
    except TelegramError as e:
        logging.warning("Error happened while deleting system message: %s", e)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = (update.message.text or update.message.caption or "").strip()

    if update.message.media_group_id:
        mgid = update.message.media_group_id
        album_store = context.chat_data.setdefault("albums", {})
        group = album_store.setdefault(mgid, {"items": [], "job": None, "user_id": update.effective_user.id})

        text = (update.message.caption or update.message.text or "").strip()
        if update.message.photo:
            photo = update.message.photo[-1]
            group["items"].append({"file_id": photo.file_id, "mime": "image/jpeg", "text": text})
        elif update.message.document and update.message.document.mime_type and update.message.document.mime_type.startswith("image/"):
            mime = update.message.document.mime_type
            group["items"].append({"file_id": update.message.document.file_id, "mime": mime, "text": text})

        if group["job"]:
            group["job"].schedule_removal()
        group["job"] = context.job_queue.run_once(
            _process_media_group,
            when=1.0,
            chat_id=update.effective_chat.id,
            data={"group_id": mgid},
        )
        return

    image_uris: List[str] = []

    if update.message.photo:
        photo = update.message.photo[-1]
        try:
            image_uris.append(await _file_to_data_uri(context, photo.file_id, "image/jpeg"))
        except Exception as e:
            logging.warning("Failed to process photo: %s", e)

    if update.message.document and update.message.document.mime_type and update.message.document.mime_type.startswith("image/"):
        try:
            mime = update.message.document.mime_type
            image_uris.append(await _file_to_data_uri(context, update.message.document.file_id, mime))
        except Exception as e:
            logging.warning("Failed to process image document: %s", e)

    processing_msg = await update.message.reply_text("OpenRouter is processing your request…")

    try:
        if not user_text and not image_uris:
            await processing_msg.delete()
            await update.message.reply_text("Attach an image or send a message.")
            return

        response_text = await openrouter.get_response(user_text if user_text else None, update.effective_user.id, image_uris or None)
        await processing_msg.delete()

        safe = escape_markdown(response_text, version=2)
        safe = re.sub(r'\\\*\\\*(.+?)\\\*\\\*', r'***\1***', safe, flags=re.S)
        safe = safe.replace(r'\_', '_').replace(r'\`', '```')

        for i in range(0, len(safe), MAX_MESSAGE_LENGTH):
            part = safe[i:i + MAX_MESSAGE_LENGTH]
            await update.message.reply_text(part, parse_mode=ParseMode.MARKDOWN_V2)

    except TelegramError as e:
        logging.warning("Error happened while deleting system message: %s", e)

def main():
    logging.basicConfig(level=logging.INFO)
    app = ApplicationBuilder().token(TELEGRAM_BOT_KEY).build()
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('about', about_command))
    app.add_handler(CommandHandler('switch_model', switch_model_command))
    app.add_handler(CommandHandler('reset', reset_command))
    app.add_handler(CallbackQueryHandler(set_model_callback, pattern=r"^set_model:"))
    message_filter = (filters.TEXT | filters.PHOTO | filters.Document.IMAGE) & ~filters.COMMAND
    app.add_handler(MessageHandler(message_filter, handle_message))
    app.add_error_handler(error_handler)
    app.run_polling()

if __name__ == "__main__":
    main()