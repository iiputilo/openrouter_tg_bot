import os
import logging
import openrouter
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import TelegramError
from telegram.constants import ParseMode
from telegram.helpers import escape_markdown
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

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    processing_msg = await update.message.reply_text("OpenRouter is processing your request…")

    try:
        response_text = await openrouter.get_response(user_text, update.effective_user.id)
        await processing_msg.delete()

        safe = escape_markdown(response_text, version=2)
        safe = (
            safe
            .replace(r'\*\*', '***')
            .replace(r'\_', '_')      # _курсив_
            .replace(r'\`', '```')      # `код`
        )

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
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    app.run_polling()

if __name__ == "__main__":
    main()
