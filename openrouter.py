import httpx
import json
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
USER_MAP_FILE = Path('user_map.json')
MAX_HISTORY_MESSAGES = 15
DEFAULT_MODEL_NAME = "openai/gpt-5"
AVAILABLE_MODELS = ["openai/gpt-5", "anthropic/claude-sonnet-4.5"]

MODEL_PRICING = {
    "openai/gpt-5": [1.25, 10],
    "anthropic/claude-sonnet-4.5": [3, 15]
}

def _load_user_map() -> Dict[str, Any]:
    if not USER_MAP_FILE.exists():
        return {}
    try:
        content = USER_MAP_FILE.read_text(encoding='utf-8')
        data = json.loads(content) if content.strip() else {}
        changed = False
        for k, v in list(data.items()):
            if isinstance(v, str):
                data[k] = {"uid": v, "history": [], "model": DEFAULT_MODEL_NAME, "api_key": None}
                changed = True
            elif isinstance(v, dict):
                v.setdefault("uid", str(uuid.uuid4()))
                v.setdefault("history", [])
                v.setdefault("model", DEFAULT_MODEL_NAME)
                v.setdefault("api_key", None)
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
        rec = {"uid": str(uuid.uuid4()), "history": [], "model": DEFAULT_MODEL_NAME, "api_key": None}
        user_map[key] = rec
        _save_user_map(user_map)
    else:
        rec.setdefault("uid", str(uuid.uuid4()))
        rec.setdefault("history", [])
        rec.setdefault("model", DEFAULT_MODEL_NAME)
        rec.setdefault("api_key", None)
    return rec

def set_user_model(tg_id: int, model_name: str) -> None:
    rec = _ensure_user_record(tg_id)
    rec["model"] = model_name
    _save_user_map(user_map)

def get_user_model(tg_id: int) -> str:
    rec = _ensure_user_record(tg_id)
    return rec.get("model", DEFAULT_MODEL_NAME)

def set_user_api_key(tg_id: int, api_key: str) -> None:
    rec = _ensure_user_record(tg_id)
    rec["api_key"] = (api_key or "").strip()
    _save_user_map(user_map)

def get_user_api_key(tg_id: int) -> Optional[str]:
    rec = _ensure_user_record(tg_id)
    key = (rec.get("api_key") or "").strip()
    return key if key else None

def has_api_key(tg_id: int) -> bool:
    return bool(get_user_api_key(tg_id))

def get_user_id(tg_id: int) -> str:
    return _ensure_user_record(tg_id)["uid"]

def reset_history(tg_id: int) -> None:
    rec = _ensure_user_record(tg_id)
    rec["history"] = []
    _save_user_map(user_map)

def _trim_history(history: List[Dict[str, Any]], max_messages: int) -> List[Dict[str, Any]]:
    return history[-max_messages:] if len(history) > max_messages else history

async def get_response(user_text: Optional[str], tg_id: int, image_data_uris: Optional[List[str]] = None) -> str:
    rec = _ensure_user_record(tg_id)
    model = rec["model"]
    api_key = get_user_api_key(tg_id)
    if not api_key:
        return "Configuration error: OpenRouter API key wasn't set\\. Use the command `/change\\_api\\_key <key>`"

    if image_data_uris:
        parts: List[Dict[str, Any]] = []
        if user_text:
            parts.append({"type": "text", "text": user_text})
        for uri in image_data_uris:
            parts.append({"type": "image_url", "image_url": {"url": uri}})
        user_message = {"role": "user", "content": parts}
    else:
        user_message = {"role": "user", "content": [{"type": "text", "text": user_text or ""}]}

    rec["history"].append(user_message)
    rec["history"] = _trim_history(rec["history"], MAX_HISTORY_MESSAGES)
    _save_user_map(user_map)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/iiputilo/openrouter_tg_bot",
        "X-Title": "openrouter_tg_bot"
    }
    payload = {
        "model": model,
        "messages": rec["history"]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(OPENROUTER_API_URL, json=payload, headers=headers, timeout=520.0)

    if response.status_code >= 400:
        try:
            err_json = response.json()
            err_msg = (
                err_json.get("error", {}).get("message")
                or err_json.get("message")
                or json.dumps(err_json, ensure_ascii=False)
            )
        except Exception:
            err_msg = response.text
        logger.error("OpenRouter API error %s: %s", response.status_code, err_msg)
        return f"OpenRouter error {response.status_code}: {err_msg}"

    data = response.json()

    raw_content = data["choices"][0]["message"]["content"]
    if isinstance(raw_content, list):
        content = "\n".join(
            part.get("text", "")
            for part in raw_content
            if isinstance(part, dict)
        ).strip()
    else:
        content = str(raw_content)

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    price_per_1m_input, price_per_1m_output = MODEL_PRICING.get(model, [0, 0])

    cost_input = prompt_tokens / 10**6 * price_per_1m_input
    cost_output = completion_tokens / 10**6 * price_per_1m_output
    total_cost = cost_input + cost_output

    rec["history"].append({"role": "assistant", "content": content})
    rec["history"] = _trim_history(rec["history"], MAX_HISTORY_MESSAGES)
    _save_user_map(user_map)

    return (
        f"{content}\n\n---\n"
        f"Input cost: ${cost_input:.6f}\n"
        f"Output cost: ${cost_output:.6f}\n"
        f"**Total cost: ${total_cost:.6f}**"
    )