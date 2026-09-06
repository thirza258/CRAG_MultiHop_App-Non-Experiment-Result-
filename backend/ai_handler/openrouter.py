"""Shared OpenRouter client construction.

Every component that talks to OpenRouter used to build its own ``OpenAI`` client
in ``__init__`` from ``os.getenv("OPENROUTER_API_KEY")``, and two of them raised
when that variable was missing. That had two consequences worth removing:

1. A deployment where users bring their own keys and the server has none could
   not even *construct* the pipeline — startup died on the first retriever.
2. The key was frozen at construction, so a per-request key was impossible on a
   singleton pipeline.

So clients are built here, on demand, from the key that applies to the current
request (see :mod:`common.runtime.context`), and cached by key fingerprint so a
call chain does not pay for a new client per step.
"""

import logging
import threading
from typing import Optional

from openai import OpenAI

from common.runtime import api_keys as api_keys_module
from common.runtime.context import openrouter_api_key

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: OpenRouter attributes traffic to an app with these; they are not credentials.
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://crag.nevatal.tech",
    "X-Title": "CRAG MultiHop RAG",
}


class MissingAPIKeyError(RuntimeError):
    """No OpenRouter key for this request — neither the user's nor the server's.

    Raised at call time rather than construction time so the failure lands in
    the stage that needed the key, where the pipeline's per-stage guards can
    degrade instead of taking the whole query down.
    """


#: Small bound: enough that a handful of concurrent users each reuse a client,
#: small enough that a stream of distinct keys cannot grow this without limit.
_MAX_CACHED_CLIENTS = 32

_CLIENT_CACHE = {}
_CACHE_LOCK = threading.Lock()


def openrouter_client(api_key: Optional[str] = None) -> OpenAI:
    """An OpenAI-SDK client pointed at OpenRouter, for the effective key.

    ``api_key`` is for callers that already hold one (the Celery indexing path);
    everything else should pass nothing and let the request context decide.
    """
    key = (api_key or openrouter_api_key() or "").strip()
    if not key:
        raise MissingAPIKeyError(
            "No OpenRouter API key available. Add one in the chat settings, or "
            "set OPENROUTER_API_KEY on the server."
        )

    cache_key = api_keys_module.fingerprint(key)

    client = _CLIENT_CACHE.get(cache_key)
    if client is not None:
        return client

    with _CACHE_LOCK:
        # Re-check inside the lock: two threads can miss simultaneously.
        client = _CLIENT_CACHE.get(cache_key)
        if client is not None:
            return client

        if len(_CLIENT_CACHE) >= _MAX_CACHED_CLIENTS:
            # Plain clear rather than an LRU: clients are cheap to rebuild and
            # this only happens when many distinct keys are in flight.
            logger.info("[OpenRouter] client cache full — clearing")
            _CLIENT_CACHE.clear()

        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=key,
            default_headers=dict(OPENROUTER_HEADERS),
        )
        _CLIENT_CACHE[cache_key] = client
        logger.info("[OpenRouter] built client for key %s", cache_key)

    return client


def has_openrouter_key() -> bool:
    """Whether this request could talk to OpenRouter at all."""
    return bool((openrouter_api_key() or "").strip())
