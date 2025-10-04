import os
import httpx
import json
import uuid
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

load_dotenv()
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
USER_MAP_FILE = Path('user_map.json')
MAX_HISTORY_MESSAGES = 15
DEFAULT_MODEL_NAME = "gpt-3.5-turbo"
AVAILABLE_MODELS = ["gpt-3.5-turbo", "anthropic/claude-sonnet-4.5"]

def _load_user_map() -> Dict[str, Any]:
    if not USER_MAP_FILE.exists():
        return {}
    try:
        content = USER_MAP_FILE.read_text(encoding='utf-8')
        data = json.loads(content) if content.strip() else {}
        changed = False
        for k, v in list(data.items()):
            if isinstance(v, str):
                data[k] = {"uid": v, "history": [], "model": DEFAULT_MODEL_NAME}
                changed = True
            elif isinstance(v, dict):
                v.setdefault("uid", str(uuid.uuid4()))
                v.setdefault("history", [])
                v.setdefault("model", DEFAULT_MODEL_NAME)
        if changed:
            _save_user_map(data)
        return data
    except json.JSONDecodeError as e:
        logging.warning(f"Wrong JSON format in {USER_MAP_FILE}: {e}")
        return {}

def _save_user_map(state: Dict[str, Any]) -> None:
    USER_MAP_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')

user_map: Dict[str, Any] = _load_user_map()

def _ensure_user_record(tg_id: int) -> Dict[str, Any]:
    key = str(tg_id)
    rec = user_map.get(key)
    if not rec:
        rec = {"uid": str(uuid.uuid4()), "history": [], "model": DEFAULT_MODEL_NAME}
        user_map[key] = rec
        _save_user_map(user_map)
    else:
        rec.setdefault("uid", str(uuid.uuid4()))
        rec.setdefault("history", [])
        rec.setdefault("model", DEFAULT_MODEL_NAME)
    return rec

def set_user_model(tg_id: int, model_name: str) -> None:
    rec = _ensure_user_record(tg_id)
    rec["model"] = model_name
    _save_user_map(user_map)

def get_user_model(tg_id: int) -> str:
    rec = _ensure_user_record(tg_id)
    return rec.get("model", DEFAULT_MODEL_NAME)

def get_user_id(tg_id: int) -> str:
    return _ensure_user_record(tg_id)["uid"]

def reset_history(tg_id: int) -> None:
    rec = _ensure_user_record(tg_id)
    rec["history"] = []
    _save_user_map(user_map)

def _trim_history(history: List[Dict[str, str]], max_messages: int) -> List[Dict[str, str]]:
    return history[-max_messages:] if len(history) > max_messages else history

async def get_response(user_text: str, tg_id: int) -> str:
    rec = _ensure_user_record(tg_id)
    uid = rec["uid"]
    model = rec["model"]
    logging.info(f'Сообщение от {uid}: {user_text}')

    rec["history"].append({"role": "user", "content": user_text})
    rec["history"] = _trim_history(rec["history"], MAX_HISTORY_MESSAGES)
    _save_user_map(user_map)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": rec["history"]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(OPENROUTER_API_URL, json=payload, headers=headers, timeout=520.0)
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]

    rec["history"].append({"role": "assistant", "content": content})
    rec["history"] = _trim_history(rec["history"], MAX_HISTORY_MESSAGES)
    _save_user_map(user_map)

    return content
