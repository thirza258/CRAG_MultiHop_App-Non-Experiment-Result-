"""The list of models a user can pick from, fetched live from OpenRouter.

OpenRouter publishes two *separate* public catalogs, and the distinction matters:

* ``GET /api/v1/models`` — chat/completion models (hundreds of them).
* ``GET /api/v1/embeddings/models`` — embedding models (a few dozen).

Embedding models are **not** in the first list, so a picker built from it alone
would offer no valid embedding ids at all. Neither endpoint needs a key, so this
works before the user has entered one.

Responses are cached in-process for :data:`CACHE_TTL_SECONDS`; if OpenRouter is
unreachable the last good response is served past its TTL, and only if there has
never been one do we fall back to :data:`FALLBACK_CATALOG` — a small hand-kept
list that keeps the picker usable offline.
"""

import logging
import threading
import time

import requests

logger = logging.getLogger(__name__)

CHAT_MODELS_URL = "https://openrouter.ai/api/v1/models"
EMBEDDING_MODELS_URL = "https://openrouter.ai/api/v1/embeddings/models"

#: The catalog changes on the order of days; an hour keeps it fresh without
#: making OpenRouter part of this app's request path.
CACHE_TTL_SECONDS = 3600

#: Kept short — this runs inside a user-facing GET.
REQUEST_TIMEOUT_SECONDS = 10

#: Enough to pick from when OpenRouter cannot be reached and nothing is cached.
#: Ids verified against the live catalogs; the defaults the pipeline ships with
#: are deliberately included so the fallback can always express them.
FALLBACK_CATALOG = {
    "chat": [
        {"id": "mistralai/mistral-nemo", "name": "Mistral: Mistral Nemo"},
        {"id": "qwen/qwen3-30b-a3b-instruct-2507", "name": "Qwen: Qwen3 30B A3B Instruct"},
        {"id": "google/gemini-2.0-flash-001", "name": "Google: Gemini 2.0 Flash"},
        {"id": "openai/gpt-4o-mini", "name": "OpenAI: GPT-4o mini"},
        {"id": "anthropic/claude-3.5-sonnet", "name": "Anthropic: Claude 3.5 Sonnet"},
        {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Meta: Llama 3.3 70B Instruct"},
        {"id": "deepseek/deepseek-chat", "name": "DeepSeek: DeepSeek V3"},
    ],
    "embedding": [
        {"id": "google/gemini-embedding-2-preview", "name": "Google: Gemini Embedding 2 (preview)"},
        {"id": "openai/text-embedding-3-small", "name": "OpenAI: text-embedding-3-small"},
        {"id": "openai/text-embedding-3-large", "name": "OpenAI: text-embedding-3-large"},
        {"id": "baai/bge-m3", "name": "BAAI: bge-m3"},
        {"id": "qwen/qwen3-embedding-4b", "name": "Qwen: Qwen3 Embedding 4B"},
        {"id": "mistralai/mistral-embed-2312", "name": "Mistral: Mistral Embed"},
    ],
}

_cache = {"payload": None, "fetched_at": 0.0}
_lock = threading.Lock()


def _price(pricing, field: str):
    """One pricing field as a float, or None when absent/unparseable.

    OpenRouter sends prices as strings ("0.000000834") and omits fields that do
    not apply, so this has to tolerate both.
    """
    if not isinstance(pricing, dict):
        return None
    raw = pricing.get(field)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _summarize(entry) -> dict:
    """Keep only what the picker renders.

    The raw catalog is ~650KB; the trimmed form is small enough to cache in the
    browser and cheap to filter as the user types.
    """
    if not isinstance(entry, dict):
        return {}

    model_id = entry.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        return {}

    pricing = entry.get("pricing")
    prompt_price = _price(pricing, "prompt")
    completion_price = _price(pricing, "completion")

    summary = {
        "id": model_id.strip(),
        "name": entry.get("name") if isinstance(entry.get("name"), str) else model_id,
        "context_length": entry.get("context_length"),
        "prompt_price": prompt_price,
        "completion_price": completion_price,
        # ":free" in the id is OpenRouter's own marker, but a zero prompt price
        # is the fact that actually matters to someone spending their own key.
        "free": model_id.endswith(":free") or prompt_price == 0,
    }
    return summary


def _fetch(url: str) -> list:
    """One catalog endpoint to a list of trimmed entries. Raises on failure."""
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()

    entries = data.get("data") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"unexpected catalog shape from {url}")

    models = [summary for summary in (_summarize(e) for e in entries) if summary]
    if not models:
        raise ValueError(f"no usable models in the response from {url}")

    models.sort(key=lambda m: m["id"])
    return models


def get_catalog(force_refresh: bool = False) -> dict:
    """``{"chat": [...], "embedding": [...], "source": ..., "fetched_at": ...}``.

    Never raises: a caller rendering a settings panel should get a usable list
    even when OpenRouter is down.
    """
    now = time.time()

    cached = _cache.get("payload")
    if cached and not force_refresh and (now - _cache["fetched_at"]) < CACHE_TTL_SECONDS:
        return {**cached, "source": "cache"}

    with _lock:
        # Another thread may have refreshed while this one waited.
        cached = _cache.get("payload")
        if cached and not force_refresh and (time.time() - _cache["fetched_at"]) < CACHE_TTL_SECONDS:
            return {**cached, "source": "cache"}

        chat, embedding, errors = [], [], []

        try:
            chat = _fetch(CHAT_MODELS_URL)
        except Exception as e:
            errors.append(f"chat: {e}")
            logger.warning("[ModelCatalog] chat catalog fetch failed: %s", e)

        try:
            embedding = _fetch(EMBEDDING_MODELS_URL)
        except Exception as e:
            errors.append(f"embedding: {e}")
            logger.warning("[ModelCatalog] embedding catalog fetch failed: %s", e)

        # A partial refresh keeps the half that worked and reuses the cached
        # half — better than discarding a good list because the other call
        # timed out.
        if cached:
            chat = chat or cached.get("chat") or []
            embedding = embedding or cached.get("embedding") or []

        if not chat and not embedding:
            logger.warning(
                "[ModelCatalog] serving the bundled fallback catalog (%s)",
                "; ".join(errors) or "no data",
            )
            return {
                "chat": list(FALLBACK_CATALOG["chat"]),
                "embedding": list(FALLBACK_CATALOG["embedding"]),
                "source": "fallback",
                "fetched_at": now,
                "errors": errors,
            }

        chat = chat or list(FALLBACK_CATALOG["chat"])
        embedding = embedding or list(FALLBACK_CATALOG["embedding"])

        payload = {"chat": chat, "embedding": embedding, "fetched_at": now}
        _cache["payload"] = payload
        _cache["fetched_at"] = now

        result = {**payload, "source": "openrouter"}
        if errors:
            result["errors"] = errors
        return result


def reset_cache() -> None:
    """Drop the cached catalog. Exists for tests."""
    with _lock:
        _cache["payload"] = None
        _cache["fetched_at"] = 0.0
