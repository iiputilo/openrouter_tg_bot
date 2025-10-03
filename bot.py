import os
import logging
import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

load_dotenv()
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TELEGRAM_BOT_KEY = os.getenv("TELEGRAM_BOT_TOKEN")
MAX_MESSAGE_LENGTH = 4096

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": user_text}]
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(OPENROUTER_API_URL, json=payload, headers=headers)
        data = response.json()
    ai_text = data["choices"][0]["message"]["content"]

    # разбиваем на части по MAX_MESSAGE_LENGTH
    for i in range(0, len(ai_text), MAX_MESSAGE_LENGTH):
        part = ai_text[i:i + MAX_MESSAGE_LENGTH]
        await update.message.reply_text(part)

def main():
    logging.basicConfig(level=logging.INFO)
    app = ApplicationBuilder().token(TELEGRAM_BOT_KEY).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
