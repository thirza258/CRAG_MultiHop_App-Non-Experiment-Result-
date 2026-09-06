"""Handing request-scoped model choices and credentials to a background worker.

Document indexing runs in a Celery worker — a separate process — so the
``ContextVar`` that carries the user's embedding-model choice and their API keys
cannot reach it. The obvious fix is to pass them as task arguments, and that is
the wrong one: task arguments are persisted in the broker queue, copied into the
result backend, re-enqueued verbatim on every retry, and rendered into Celery's
own log lines when a task fails. Credentials should not be in any of those.

So the settings are written to one short-lived Redis entry under a random token,
and the task receives only the token. Losing the entry is not an error — the
worker then falls back to the server's own configuration, which is exactly the
behaviour indexing had before users could bring their own keys.

The entry is read rather than consumed, because ``build_index_task`` retries with
backoff and each attempt needs it. Expiry does the cleanup.
"""

import json
import logging
import secrets
from dataclasses import replace

import redis

from common.runtime.config import is_default, normalize_pipeline_config
from common.runtime.context import RuntimeSettings
from ragreader.settings import REDIS_HOST, REDIS_PORT

logger = logging.getLogger(__name__)

#: Comfortably longer than indexing plus build_index_task's retry backoff
#: (60s + 120s + 240s), so a retried document still finds its settings.
TTL_SECONDS = 2 * 60 * 60

_KEY_PREFIX = "rag:runtime:"

#: Separate database from the status pub/sub channels (db 0) so a stray
#: ``KEYS *`` while debugging status events cannot list credentials.
_REDIS_DB = 1


def _client():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=_REDIS_DB,
        socket_connect_timeout=5,
        socket_timeout=5,
    )


def stash_runtime(settings: RuntimeSettings):
    """Store ``settings`` and return a token, or None if there is nothing to store.

    Returns None both when the request expressed no preferences (nothing to hand
    over) and when Redis is unavailable — the caller passes the token through
    unconditionally and the worker treats a missing one as "use the server's
    configuration".
    """
    if not isinstance(settings, RuntimeSettings):
        return None

    # Nothing chosen and no key brought: the worker's own environment already
    # produces this exact behaviour, so skip the round-trip.
    #
    # params has to be part of this test, not just the models and the keys. The
    # chunking controls live there and nowhere else, so a user who changes only
    # the chunk size would otherwise get no token and have their choice quietly
    # ignored by the worker.
    if (
        not settings.llm_model
        and not settings.embedding_model
        and not any((settings.keys or {}).values())
        and is_default(settings.params)
    ):
        return None

    token = secrets.token_urlsafe(24)
    client = None
    try:
        client = _client()
        client.set(
            f"{_KEY_PREFIX}{token}",
            json.dumps(settings.to_dict()),
            ex=TTL_SECONDS,
        )
        logger.info("[RuntimeHandoff] stashed request settings (%s)", settings.describe())
        return token
    except Exception as e:
        logger.warning(
            "[RuntimeHandoff] could not stash request settings, the worker will "
            "use the server configuration: %s",
            e,
        )
        return None
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def load_runtime(token) -> RuntimeSettings:
    """The settings for ``token``, or empty defaults when unavailable.

    Never raises: a background job must not fail because a bookkeeping lookup
    did. Empty defaults mean the worker's environment applies.
    """
    if not token or not isinstance(token, str):
        return RuntimeSettings()

    client = None
    try:
        client = _client()
        raw = client.get(f"{_KEY_PREFIX}{token}")
        if raw is None:
            logger.info(
                "[RuntimeHandoff] no stashed settings for this task (expired?) — "
                "using the server configuration"
            )
            return RuntimeSettings()
        settings = RuntimeSettings.from_dict(json.loads(raw))
        # RuntimeSettings cannot import the config schema without a cycle, so
        # the params it carries are re-validated here — the worker is a
        # separate process and must not trust a payload it did not build.
        return replace(settings, params=normalize_pipeline_config(settings.params))
    except Exception as e:
        logger.warning(
            "[RuntimeHandoff] could not load stashed settings, using the server "
            "configuration: %s",
            e,
        )
        return RuntimeSettings()
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def discard_runtime(token) -> None:
    """Delete a token's entry. Best-effort; expiry is the real cleanup."""
    if not token or not isinstance(token, str):
        return
    client = None
    try:
        client = _client()
        client.delete(f"{_KEY_PREFIX}{token}")
    except Exception as e:
        logger.debug("[RuntimeHandoff] could not discard token: %s", e)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
