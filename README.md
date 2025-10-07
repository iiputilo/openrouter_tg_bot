# OpenRouter Telegram Bot

A small Telegram bot for the OpenRouter API. Supports text and image prompts (including albums), per‑user model selection, and token cost estimation.

## Features
- Text and multimodal prompts (single images and media albums).
- Per‑user model selection: `openai/gpt-5` (default), `anthropic/claude-sonnet-4.5`.
- Cost estimation based on OpenRouter `usage` tokens.
- Persistent per‑user state: history, selected model, API key.
- Splits long replies to Telegram’s 4096 char limit; safe MarkdownV2 formatting.