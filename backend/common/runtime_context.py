"""Request-scoped model choice and credentials.

The pipeline is a process-wide singleton: the reranker and the CRAG evaluator are
transformer models that take seconds to load, so rebuilding a pipeline per query
is not an option. But a user now picks their own LLM, their own embedding model
and supplies their own API keys, all of which have to change *per query* on that
shared object.

Mutating the singleton's attributes would race between concurrent queries — user
A's key would answer user B's question. Instead the per-query choices live in a
:class:`contextvars.ContextVar`, and every component resolves the model and key
it needs at **call** time via the helpers here. A ``ContextVar`` is per-thread,
which matches how the request actually runs: the websocket consumer hands each
query to its own thread, and one query is one synchronous call chain inside it.

Rules that keep this honest:

* An empty string means "the user did not choose" — the component's own config
  default applies, then the environment. It never means "use nothing".
* ``embedding_model`` is always *the model that matches the vectors the request
  is about to touch*. On the query path the pipeline pins it to whatever the
  target collection was indexed with; on the index path it is the user's choice.
  Dense retrieval cannot be allowed to embed a query with a model of a different
  dimension than the collection it is searching.
* Callers must use :func:`use_runtime`, which resets the context afterwards. The
  REST path runs on a pooled server thread, so a leaked context would hand the
  next request on that thread somebody else's key.
"""

import logging
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Dict

from common import api_keys as api_keys_module

logger = logging.getLogger(__name__)

#: OpenRouter ids look like ``vendor/model-name`` or ``vendor/model-name:free``;
#: bare OpenAI ids like ``gpt-4o`` are also valid. Anything else is a client bug
#: (or an injection attempt against the URL we build), so it is dropped.
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:-]{0,127}$")

MAX_MODEL_ID_LENGTH = 128


def normalize_model_id(value, default: str = "") -> str:
    """Coerce one model id, falling back to ``default`` for anything invalid."""
    if not isinstance(value, str):
        return default

    candidate = value.strip()
    if not candidate:
        return default

    if len(candidate) > MAX_MODEL_ID_LENGTH:
        logger.warning(
            "[Runtime] ignoring over-long model id (%d chars)", len(candidate)
        )
        return default

    if not _MODEL_ID_RE.match(candidate):
        logger.warning("[Runtime] ignoring malformed model id %r", candidate[:60])
        return default

    return candidate


@dataclass(frozen=True)
class RuntimeSettings:
    """What one request chose. Immutable so it cannot drift mid-query."""

    llm_model: str = ""
    embedding_model: str = ""
    keys: Dict[str, str] = field(default_factory=dict)

    def with_embedding_model(self, model: str) -> "RuntimeSettings":
        """A copy pinned to ``model`` — used when the collection's embedding
        model overrides the user's pick."""
        return replace(self, embedding_model=normalize_model_id(model, ""))

    def key(self, name: str) -> str:
        """The key for ``name``: the request's, else the server environment."""
        return api_keys_module.resolve(name, self.keys)

    def to_dict(self) -> dict:
        """JSON-safe form, for handing settings to a worker process."""
        return {
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "keys": dict(self.keys or {}),
        }

    @classmethod
    def from_dict(cls, raw) -> "RuntimeSettings":
        """Rebuild from :meth:`to_dict`, re-validating everything."""
        if not isinstance(raw, dict):
            return cls()
        return cls(
            llm_model=normalize_model_id(raw.get("llm_model"), ""),
            embedding_model=normalize_model_id(raw.get("embedding_model"), ""),
            keys=api_keys_module.normalize_api_keys(raw.get("keys")),
        )

    def describe(self) -> str:
        """Log-safe summary. Never includes a key value."""
        return (
            f"llm={self.llm_model or 'default'} "
            f"embedding={self.embedding_model or 'default'} "
            f"{api_keys_module.describe(self.keys or {})}"
        )


#: The settings that apply when nobody set any — every component's own config
#: default plus the server environment, i.e. exactly the historic behaviour.
EMPTY_RUNTIME = RuntimeSettings()

_RUNTIME: ContextVar[RuntimeSettings] = ContextVar("rag_runtime", default=EMPTY_RUNTIME)


def current_runtime() -> RuntimeSettings:
    """The settings for the request on this thread, or the empty defaults."""
    try:
        value = _RUNTIME.get()
    except LookupError:  # pragma: no cover - ContextVar has a default
        return EMPTY_RUNTIME
    return value if isinstance(value, RuntimeSettings) else EMPTY_RUNTIME


@contextmanager
def use_runtime(settings: RuntimeSettings):
    """Install ``settings`` for the duration of the block, then restore.

    The reset is the whole point: without it the next request served by this
    thread inherits these credentials.
    """
    if not isinstance(settings, RuntimeSettings):
        settings = EMPTY_RUNTIME
    token = _RUNTIME.set(settings)
    try:
        yield settings
    finally:
        try:
            _RUNTIME.reset(token)
        except ValueError:
            # reset() refuses a token created in a different context (e.g. the
            # block was entered and exited across an await boundary). Clearing
            # is still better than leaving another request's key installed.
            _RUNTIME.set(EMPTY_RUNTIME)


# ── Resolution helpers used by the components ────────────────────────────────
# Each takes the component's own configured default, so "the user did not choose"
# keeps the exact behaviour that component had before this feature existed.


def resolve_llm_model(default: str) -> str:
    """The chat model for this request."""
    return current_runtime().llm_model or default


def resolve_embedding_model(default: str) -> str:
    """The embedding model for this request.

    On the query path this is pinned to the target collection's model, so a
    component must never substitute its own default once a runtime is installed
    with a value.
    """
    return current_runtime().embedding_model or default


def resolve_key(name: str) -> str:
    """The API key for ``name`` — the request's, else the environment."""
    return current_runtime().key(name)


def openrouter_api_key() -> str:
    return resolve_key("openrouter")


def news_api_key() -> str:
    return resolve_key("news")


def request_secrets() -> dict:
    """The request's own keys, for scrubbing outbound text.

    Only the user-supplied ones: the server's environment key is not something
    the client could have leaked to us, and adding it would mean hashing it into
    every error path for no gain.
    """
    return dict(current_runtime().keys or {})


def server_has_key(name: str) -> bool:
    """Whether the deployment itself has a key for ``name`` configured."""
    env_name = api_keys_module.ENV_FALLBACKS.get(name)
    if not env_name:
        return False
    return bool((os.getenv(env_name) or "").strip())
