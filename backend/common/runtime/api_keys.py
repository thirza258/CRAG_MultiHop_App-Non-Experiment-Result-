"""Per-request API keys ("bring your own key").

The chat UI lets a user paste their own OpenRouter and NewsAPI keys instead of
spending the server's. Those keys ride on the query websocket (and the REST
payload) as a ``KEYS`` object beside ``CONFIG``, live only for the duration of
the request, and are **never** written to the database.

Why ``KEYS`` is a separate wire field rather than more entries in ``CONFIG``:
``CONFIG`` is logged verbatim and summarised into status events, so a secret
inside it would end up in the log file and on the client's screen. Keeping the
two apart means ``CONFIG`` stays freely loggable and every secret goes through
:func:`redact` / :func:`scrub` on the way out.

The contract mirrors :mod:`common.runtime.config`: **absent, partial or
malformed input normalises to "no key supplied"**, which falls back to the
server's environment. Nothing in here raises.
"""

import hashlib
import logging
import os
import re

logger = logging.getLogger(__name__)

#: Canonical key names. Everything downstream indexes the normalised dict with
#: these, so adding a provider means adding it here and in _ALIASES.
KEY_FIELDS = ("openrouter", "news")

#: Environment variable each key falls back to when the client sends none.
ENV_FALLBACKS = {
    "openrouter": "OPENROUTER_API_KEY",
    "news": "NEWS_API_KEY",
}

#: Clients spell these differently depending on which layer built the payload,
#: so accept the obvious variants instead of silently dropping a real key.
_ALIASES = {
    "openrouter": (
        "openrouter",
        "openrouter_api_key",
        "openrouter_key",
        "or_api_key",
    ),
    "news": (
        "news",
        "news_api",
        "news_api_key",
        "newsapi",
        "newsapi_key",
    ),
}

#: Long enough for any provider's key, short enough that a pasted document
#: cannot be smuggled through as one.
MAX_KEY_LENGTH = 400

#: A key goes into an HTTP header (OpenRouter) and a query string (NewsAPI).
#: Anything outside this set could smuggle a newline into a header or break the
#: URL, so a value containing one is dropped rather than sanitised — a mangled
#: key would only fail later with a confusing 401.
_ALLOWED_KEY_CHARS = re.compile(r"^[A-Za-z0-9._~+/=:-]+$")

#: What a redacted value looks like. Deliberately not the empty string, so a
#: reader of the log can tell "key was supplied" from "key was absent".
PLACEHOLDER = "***redacted***"


def _clean(value) -> str:
    """One raw value to a usable key, or "" when it is not one.

    Rejects rather than repairs: a key we had to strip characters out of is not
    the user's key any more, and would surface as an opaque auth failure.
    """
    if not isinstance(value, str):
        return ""

    candidate = value.strip()
    if not candidate:
        return ""

    if len(candidate) > MAX_KEY_LENGTH:
        logger.warning(
            "[ApiKeys] ignoring an over-long key value (%d chars, max %d)",
            len(candidate),
            MAX_KEY_LENGTH,
        )
        return ""

    if not _ALLOWED_KEY_CHARS.match(candidate):
        # No fingerprint in this log line: the value is not a key we trust, and
        # hashing arbitrary user input here would be noise.
        logger.warning("[ApiKeys] ignoring a key value containing unexpected characters")
        return ""

    return candidate


def normalize_api_keys(raw) -> dict:
    """Return ``{field: key-or-empty-string}`` for every name in KEY_FIELDS.

    Unknown keys are dropped and an empty string always means "use the server's
    environment". Never raises — a malformed KEYS block must not fail a query.
    """
    keys = {field: "" for field in KEY_FIELDS}

    if raw is None:
        return keys

    if not isinstance(raw, dict):
        logger.warning(
            "[ApiKeys] ignoring non-dict KEYS of type %s", type(raw).__name__
        )
        return keys

    # Case-insensitive lookup so a client sending OPENROUTER_API_KEY works.
    lowered = {}
    for name, value in raw.items():
        if isinstance(name, str):
            lowered.setdefault(name.strip().lower(), value)

    for field, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                cleaned = _clean(lowered[alias])
                if cleaned:
                    keys[field] = cleaned
                    break

    return keys


def resolve(field: str, keys: dict = None) -> str:
    """The key to actually use for ``field``: the request's, else the server's.

    Returns "" when neither exists; callers decide whether that is fatal (an
    embedding call) or merely a skipped stage (a news lookup).
    """
    if keys:
        supplied = keys.get(field)
        if isinstance(supplied, str) and supplied.strip():
            return supplied.strip()

    env_name = ENV_FALLBACKS.get(field)
    if not env_name:
        return ""
    return (os.getenv(env_name) or "").strip()


def fingerprint(key: str) -> str:
    """A stable, log-safe identifier for a key.

    Used both as a client-cache key and in log lines. A hash rather than the
    usual last-four-characters, because these are other people's credentials
    and the log file has a longer life than the request.
    """
    if not isinstance(key, str) or not key.strip():
        return "none"
    digest = hashlib.sha256(key.strip().encode("utf-8")).hexdigest()
    return f"sha256:{digest[:12]}"


def describe(keys: dict) -> str:
    """Short summary of which keys a request supplied, for logs."""
    if not isinstance(keys, dict):
        return "keys=invalid"
    parts = []
    for field in KEY_FIELDS:
        value = keys.get(field)
        parts.append(f"{field}={'own' if value else 'server'}")
    return " ".join(parts)


def secret_values(*sources) -> list:
    """Every secret string found in the given key dicts, longest first.

    Longest first matters for :func:`scrub`: replacing a short key that happens
    to be a prefix of a longer one would leave the remainder of the longer key
    visible.
    """
    found = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        for value in source.values():
            if isinstance(value, str) and len(value.strip()) >= 8:
                found.add(value.strip())
    return sorted(found, key=len, reverse=True)


def scrub(text, *key_sources) -> str:
    """Replace any supplied secret appearing in ``text`` with the placeholder.

    Provider SDKs habitually echo the failing request — headers included — into
    their exception messages, and those messages are returned to the client and
    written to the log. This is the last gate before either.
    """
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""

    cleaned = text
    for secret in secret_values(*key_sources):
        if secret in cleaned:
            cleaned = cleaned.replace(secret, PLACEHOLDER)
    return cleaned


def redact(payload):
    """A copy of a wire payload safe to log.

    Any key whose *name* looks like a credential is replaced wherever it appears
    in the structure, so this stays correct even if a future client tucks a
    secret somewhere unexpected. Falls back to a type name if the payload turns
    out not to be traversable.
    """
    try:
        return _redact_value(payload, depth=0)
    except Exception as e:  # pragma: no cover - defensive, this only ever logs
        logger.warning("[ApiKeys] could not redact payload for logging: %s", e)
        return f"<unloggable {type(payload).__name__}>"


#: Substrings that mark a field name as holding a credential.
_SECRET_NAME_HINTS = ("key", "token", "secret", "password", "authorization")


def _looks_secret(name) -> bool:
    if not isinstance(name, str):
        return False
    lowered = name.lower()
    # "keys" (the container) is handled by recursing into it, but a bare
    # "api_key"/"openrouter_key" leaf is a secret.
    return any(hint in lowered for hint in _SECRET_NAME_HINTS)


def _redact_value(value, depth: int):
    # A cap keeps a hostile payload from turning a log call into a stack
    # overflow; nothing legitimate on this wire nests anywhere near this deep.
    if depth > 6:
        return "<...>"

    if isinstance(value, dict):
        out = {}
        for name, inner in value.items():
            if _looks_secret(name) and not isinstance(inner, (dict, list, tuple)):
                out[name] = PLACEHOLDER if inner not in (None, "") else inner
            elif isinstance(name, str) and name.strip().lower() == "keys":
                # The KEYS container itself: report presence, never values.
                out[name] = _redact_keys_container(inner)
            else:
                out[name] = _redact_value(inner, depth + 1)
        return out

    if isinstance(value, (list, tuple)):
        return [_redact_value(item, depth + 1) for item in value]

    return value


def _redact_keys_container(value):
    if not isinstance(value, dict):
        return PLACEHOLDER if value else value
    return {
        name: (PLACEHOLDER if inner not in (None, "") else inner)
        for name, inner in value.items()
    }
