import httpx
import json
import os
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_RESPONSES_URL = "https://openrouter.ai/api/v1/responses"

DATA_DIR = Path(os.getenv("APP_DATA_DIR", Path(__file__).parent))
DATA_DIR.mkdir(parents=True, exist_ok=True)

USER_MAP_FILE = DATA_DIR / "user_map.json"
MODEL_PRICING_FILE = DATA_DIR / "model_pricing.json"

DEFAULT_MODEL_NAME = "openai/gpt-5"

DEFAULT_MODEL_PRICING: Dict[str, List[float]] = {
    "openai/gpt-5": [1.25, 10.0],
}

MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "30"))
MIN_USER_HISTORY_MESSAGES = int(os.getenv("MIN_USER_HISTORY_MESSAGES", "1"))
MAX_USER_HISTORY_MESSAGES = int(os.getenv("MAX_USER_HISTORY_MESSAGES", "200"))

OPENROUTER_CONNECT_TIMEOUT = float(os.getenv("OPENROUTER_CONNECT_TIMEOUT", "30"))
OPENROUTER_READ_TIMEOUT = float(os.getenv("OPENROUTER_READ_TIMEOUT", "300"))
OPENROUTER_WRITE_TIMEOUT = float(os.getenv("OPENROUTER_WRITE_TIMEOUT", "60"))
OPENROUTER_POOL_TIMEOUT = float(os.getenv("OPENROUTER_POOL_TIMEOUT", "60"))

OPENROUTER_TIMEOUT = httpx.Timeout(
    connect=OPENROUTER_CONNECT_TIMEOUT,
    read=OPENROUTER_READ_TIMEOUT,
    write=OPENROUTER_WRITE_TIMEOUT,
    pool=OPENROUTER_POOL_TIMEOUT,
)

WEB_SEARCH_DEFAULT = False
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))
WEB_SEARCH_MAX_OUTPUT_TOKENS = int(os.getenv("WEB_SEARCH_MAX_OUTPUT_TOKENS", "9000"))


def _load_model_pricing() -> Dict[str, List[float]]:
    if not MODEL_PRICING_FILE.exists():
        MODEL_PRICING_FILE.write_text(
            json.dumps(DEFAULT_MODEL_PRICING, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return DEFAULT_MODEL_PRICING.copy()

    try:
        content = MODEL_PRICING_FILE.read_text(encoding="utf-8")
        data = json.loads(content) if content.strip() else {}
        if not isinstance(data, dict):
            raise ValueError("model_pricing must be an object")

        result: Dict[str, List[float]] = {}
        for key, value in data.items():
            if isinstance(value, list) and len(value) == 2:
                result[str(key)] = [float(value[0]), float(value[1])]

        if not result:
            raise ValueError("model_pricing is empty")

        return result
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Incorrect pricing file %s: %s", MODEL_PRICING_FILE, exc)
        MODEL_PRICING_FILE.write_text(
            json.dumps(DEFAULT_MODEL_PRICING, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return DEFAULT_MODEL_PRICING.copy()


model_pricing: Dict[str, List[float]] = _load_model_pricing()
available_models: List[str] = list(model_pricing.keys())


def _load_user_map() -> Dict[str, Any]:
    if not USER_MAP_FILE.exists():
        return {}
    try:
        content = USER_MAP_FILE.read_text(encoding="utf-8")
        data = json.loads(content) if content.strip() else {}
        changed = False

        for k, v in list(data.items()):
            if isinstance(v, str):
                data[k] = {
                    "uid": v,
                    "history": [],
                    "model": DEFAULT_MODEL_NAME,
                    "api_key": None,
                    "max_history": MAX_HISTORY_MESSAGES,
                    "web_search": WEB_SEARCH_DEFAULT,
                }
                changed = True
            elif isinstance(v, dict):
                v.setdefault("uid", str(uuid.uuid4()))
                v.setdefault("history", [])
                v.setdefault("model", DEFAULT_MODEL_NAME)
                v.setdefault("api_key", None)
                v.setdefault("max_history", MAX_HISTORY_MESSAGES)
                v.setdefault("web_search", WEB_SEARCH_DEFAULT)

        if changed:
            _save_user_map(data)

        return data
    except json.JSONDecodeError as exc:
        logger.warning("Wrong JSON format in %s: %s", USER_MAP_FILE, exc)
        return {}


def _save_user_map(state: Dict[str, Any]) -> None:
    USER_MAP_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


user_map: Dict[str, Any] = _load_user_map()


def _ensure_user_record(tg_id: int) -> Dict[str, Any]:
    key = str(tg_id)
    rec = user_map.get(key)

    if not rec:
        rec = {
            "uid": str(uuid.uuid4()),
            "history": [],
            "model": DEFAULT_MODEL_NAME,
            "api_key": None,
            "max_history": MAX_HISTORY_MESSAGES,
            "web_search": WEB_SEARCH_DEFAULT,
        }
        user_map[key] = rec
        _save_user_map(user_map)
        return rec

    rec.setdefault("uid", str(uuid.uuid4()))
    rec.setdefault("history", [])
    rec.setdefault("model", DEFAULT_MODEL_NAME)
    rec.setdefault("api_key", None)
    rec.setdefault("max_history", MAX_HISTORY_MESSAGES)
    rec.setdefault("web_search", WEB_SEARCH_DEFAULT)
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


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def set_user_max_history(tg_id: int, max_messages: int) -> int:
    rec = _ensure_user_record(tg_id)
    max_messages = _clamp_int(int(max_messages), MIN_USER_HISTORY_MESSAGES, MAX_USER_HISTORY_MESSAGES)
    rec["max_history"] = max_messages
    rec["history"] = _trim_history(rec["history"], max_messages)
    _save_user_map(user_map)
    return max_messages


def get_user_max_history(tg_id: int) -> int:
    rec = _ensure_user_record(tg_id)
    try:
        v = int(rec.get("max_history", MAX_HISTORY_MESSAGES))
    except Exception:
        v = MAX_HISTORY_MESSAGES
    return _clamp_int(v, MIN_USER_HISTORY_MESSAGES, MAX_USER_HISTORY_MESSAGES)


def _trim_history(history: List[Dict[str, Any]], max_messages: int) -> List[Dict[str, Any]]:
    return history[-max_messages:] if len(history) > max_messages else history


def set_user_web_search(tg_id: int, enabled: bool) -> None:
    rec = _ensure_user_record(tg_id)
    rec["web_search"] = bool(enabled)
    _save_user_map(user_map)


def get_user_web_search(tg_id: int) -> bool:
    rec = _ensure_user_record(tg_id)
    return bool(rec.get("web_search", WEB_SEARCH_DEFAULT))


def _extract_text_from_responses(result: Dict[str, Any]) -> str:
    out = result.get("output")
    if isinstance(out, list):
        chunks: List[str] = []
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        t = part.get("text", "")
                        if t:
                            chunks.append(str(t))
        text = "\n".join(chunks).strip()
        if text:
            return text

    for key in ("output_text", "text", "content"):
        v = result.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return ""


def _extract_usage_tokens_from_responses(result: Dict[str, Any]) -> Tuple[int, int]:
    usage = result.get("usage")
    if isinstance(usage, dict):
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        if isinstance(pt, int) and isinstance(ct, int):
            return pt, ct

        it = usage.get("input_tokens")
        ot = usage.get("output_tokens")
        if isinstance(it, int) and isinstance(ot, int):
            return it, ot

    for k_pt, k_ct in (
        ("prompt_tokens", "completion_tokens"),
        ("input_tokens", "output_tokens"),
    ):
        pt = result.get(k_pt)
        ct = result.get(k_ct)
        if isinstance(pt, int) and isinstance(ct, int):
            return pt, ct

    return 0, 0


def _extract_text_from_chat_completions(data: Dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    raw = choices[0].get("message", {}).get("content", "")
    if isinstance(raw, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in raw
            if isinstance(part, dict) and part.get("text")
        ).strip()
    return str(raw or "").strip()


def _extract_usage_tokens_from_chat_completions(data: Dict[str, Any]) -> Tuple[int, int]:
    usage = data.get("usage", {}) or {}
    pt = usage.get("prompt_tokens", 0) or 0
    ct = usage.get("completion_tokens", 0) or 0
    return int(pt), int(ct)


def _history_to_responses_text(history: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        content = msg.get("content")

        text_parts: List[str] = []
        if isinstance(content, str):
            if content.strip():
                text_parts.append(content.strip())
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    t = part.get("text")
                    if isinstance(t, str) and t.strip():
                        text_parts.append(t.strip())
                elif ptype == "image_url":
                    text_parts.append("[image]")
        else:
            s = str(content or "").strip()
            if s:
                text_parts.append(s)

        text = "\n".join(text_parts).strip()
        if text:
            lines.append(f"{role}: {text}")

    return "\n\n".join(lines).strip()


def _make_user_message(user_text: Optional[str], image_data_uris: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    text_in = (user_text or "").strip()
    if text_in:
        parts.append({"type": "text", "text": text_in})
    for uri in (image_data_uris or []):
        if uri:
            parts.append({"type": "image_url", "image_url": {"url": uri}})
    if not parts:
        return None
    return {"role": "user", "content": parts}


def _calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Tuple[float, float, float]:
    price_per_1m_input, price_per_1m_output = model_pricing.get(model, [0.0, 0.0])
    cost_input = (prompt_tokens / 10**6) * float(price_per_1m_input)
    cost_output = (completion_tokens / 10**6) * float(price_per_1m_output)
    return cost_input, cost_output, cost_input + cost_output


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "HTTP-Referer": "https://github.com/iiputilo/openrouter_tg_bot",
        "X-Title": "openrouter_tg_bot",
    }


async def get_response(
    user_text: Optional[str],
    tg_id: int,
    image_data_uris: Optional[List[str]] = None,
) -> str:
    rec = _ensure_user_record(tg_id)

    api_key = get_user_api_key(tg_id)
    if not api_key:
        return (
            "Configuration error: OpenRouter API key wasn't set\n"
            "Use the command `/change\\_api\\_key <key>`"
        )

    model = rec.get("model", DEFAULT_MODEL_NAME)
    user_max_history = get_user_max_history(tg_id)

    user_message = _make_user_message(user_text, image_data_uris)
    if not user_message:
        return "Send a text message or an image"

    rec["history"].append(user_message)
    rec["history"] = _trim_history(rec["history"], user_max_history)
    _save_user_map(user_map)

    if get_user_web_search(tg_id):
        payload = {
            "model": model,
            "input": _history_to_responses_text(rec["history"]),
            "plugins": [{"id": "web", "max_results": WEB_SEARCH_MAX_RESULTS}],
            "max_output_tokens": WEB_SEARCH_MAX_OUTPUT_TOKENS,
        }

        try:
            async with httpx.AsyncClient(timeout=OPENROUTER_TIMEOUT) as client:
                response = await client.post(
                    OPENROUTER_RESPONSES_URL,
                    json=payload,
                    headers=_headers(api_key),
                )
        except httpx.TimeoutException as exc:
            logger.error("OpenRouter timeout: %s", exc)
            return "OpenRouter timeout\\. Try again later"
        except httpx.HTTPError as exc:
            logger.error("OpenRouter network error: %s", exc)
            return "OpenRouter network error\\. Try again later"

        if response.status_code >= 400:
            try:
                err_json = response.json()
                err_msg = (
                    err_json.get("error", {}).get("message")
                    or err_json.get("message")
                    or json.dumps(err_json, ensure_ascii=False)
                )
            except Exception:
                err_msg = (response.text or "").strip()
            logger.error("OpenRouter API error %s: %s", response.status_code, err_msg)
            return f"OpenRouter error {response.status_code}: {err_msg}"

        try:
            data: Dict[str, Any] = response.json()
        except json.JSONDecodeError:
            snippet = (response.text or "")[:500]
            ctype = response.headers.get("content-type", "")
            logger.error(
                "Invalid JSON from OpenRouter (%s, %s)\\. First 500 chars: %r",
                response.status_code,
                ctype,
                snippet,
            )
            return "OpenRouter returned an unexpected response\\. Please try again"

        content = _extract_text_from_responses(data)
        if not content:
            logger.error("OpenRouter responses empty text: %s", json.dumps(data)[:800])
            return "OpenRouter returned empty response\\. Please try again"

        prompt_tokens, completion_tokens = _extract_usage_tokens_from_responses(data)
        cost_input, cost_output, total_cost = _calc_cost(model, prompt_tokens, completion_tokens)

        rec["history"].append({"role": "assistant", "content": content})
        rec["history"] = _trim_history(rec["history"], user_max_history)
        _save_user_map(user_map)

        return (
            f"{content}\n\n---\n"
            f"Input cost: ${cost_input:.6f}\n"
            f"Output cost: ${cost_output:.6f}\n"
            f"Total cost: ${total_cost:.6f}"
        )

    payload2 = {
        "model": model,
        "messages": rec["history"],
    }

    try:
        async with httpx.AsyncClient(timeout=OPENROUTER_TIMEOUT) as client:
            response2 = await client.post(
                OPENROUTER_API_URL,
                json=payload2,
                headers=_headers(api_key),
            )
    except httpx.TimeoutException as exc:
        logger.error("OpenRouter timeout: %s", exc)
        return "OpenRouter timeout\\. Try again later"
    except httpx.HTTPError as exc:
        logger.error("OpenRouter network error: %s", exc)
        return "OpenRouter network error\\. Try again later"

    if response2.status_code >= 400:
        try:
            err_json2 = response2.json()
            err_msg2 = (
                err_json2.get("error", {}).get("message")
                or err_json2.get("message")
                or json.dumps(err_json2, ensure_ascii=False)
            )
        except Exception:
            err_msg2 = (response2.text or "").strip()
        logger.error("OpenRouter API error %s: %s", response2.status_code, err_msg2)
        return f"OpenRouter error {response2.status_code}: {err_msg2}"

    try:
        data2: Dict[str, Any] = response2.json()
    except json.JSONDecodeError:
        snippet2 = (response2.text or "")[:500]
        ctype2 = response2.headers.get("content-type", "")
        logger.error(
            "Invalid JSON from OpenRouter (%s, %s)\\. First 500 chars: %r",
            response2.status_code,
            ctype2,
            snippet2,
        )
        return "OpenRouter returned an unexpected response\\. Please try again"

    content2 = _extract_text_from_chat_completions(data2)
    if not content2:
        logger.error("OpenRouter chat empty text: %s", json.dumps(data2)[:800])
        return "OpenRouter returned empty response\\. Please try again"

    prompt_tokens2, completion_tokens2 = _extract_usage_tokens_from_chat_completions(data2)
    cost_input2, cost_output2, total_cost2 = _calc_cost(model, prompt_tokens2, completion_tokens2)

    rec["history"].append({"role": "assistant", "content": content2})
    rec["history"] = _trim_history(rec["history"], user_max_history)
    _save_user_map(user_map)

    return (
        f"{content2}\n\n---\n"
        f"Input cost: ${cost_input2:.6f}\n"
        f"Output cost: ${cost_output2:.6f}\n"
        f"Total cost: ${total_cost2:.6f}"
    )
