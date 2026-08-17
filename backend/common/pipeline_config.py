"""Per-query pipeline configuration.

The chat UI lets a user switch individual pipeline stages off, cap how many
multi-hop hops are allowed, and choose which corpus to search. A query carries
that choice as a ``CONFIG`` object on the websocket message (or in the REST
payload); everything in this module exists to turn whatever the client sent
into a safe, fully-populated dict.

The contract that matters: **absent, partial or malformed input normalises to
:data:`DEFAULT_PIPELINE_CONFIG`**, which is the exact composition the pipeline
used before this feature existed. A client that knows nothing about config
keeps working unchanged.
"""

import logging

logger = logging.getLogger(__name__)

MIN_HOPS = 1
MAX_HOPS = 3

RETRIEVER_CHOICES = ("both", "dense", "sparse")
CORPUS_CHOICES = ("auto", "user", "base")

DEFAULT_PIPELINE_CONFIG = {
    # Decompose the question into follow-up hops.
    "use_multi_hop": True,
    # Hop ceiling; ignored when use_multi_hop is False.
    "max_hops": MAX_HOPS,
    # Self-grade retrieved context and correct/escalate when it is weak.
    "use_corrective": True,
    # Reorder merged candidates with the local cross-encoder.
    "use_reranker": True,
    # Which base retrievers feed the merge step.
    "retrievers": "both",
    # "auto" keeps the historic behaviour: the user's own collection when it
    # has anything in it, otherwise the shared base corpus.
    "corpus": "auto",
}

_TRUTHY = {"1", "true", "yes", "on", "t"}
_FALSY = {"0", "false", "no", "off", "f", ""}


def _as_bool(value, default: bool) -> bool:
    """Accept real booleans (what JSON gives us) and common string spellings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUTHY:
            return True
        if lowered in _FALSY:
            return False
    if isinstance(value, int):  # 0/1 from a loosely-typed client
        return bool(value)
    return default


def _as_int(value, default: int, low: int, high: int) -> int:
    """Coerce then clamp. Out-of-range is clamped, not rejected — a client
    asking for 99 hops gets the maximum rather than an error."""
    if isinstance(value, bool):  # bool is an int subclass; not a hop count
        return default
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError covers float("inf"), which int() refuses to convert.
        return default
    return max(low, min(high, number))


def _as_choice(value, choices: tuple, default: str) -> str:
    if isinstance(value, str) and value.strip().lower() in choices:
        return value.strip().lower()
    return default


def normalize_pipeline_config(raw) -> dict:
    """Return a complete, valid config dict for any input.

    Unknown keys are dropped. Never raises — a malformed config must not be
    able to fail a query, it just falls back to the defaults.
    """
    config = dict(DEFAULT_PIPELINE_CONFIG)

    if raw is None:
        return config

    if not isinstance(raw, dict):
        logger.warning(
            "[PipelineConfig] ignoring non-dict config of type %s", type(raw).__name__
        )
        return config

    config["use_multi_hop"] = _as_bool(
        raw.get("use_multi_hop"), DEFAULT_PIPELINE_CONFIG["use_multi_hop"]
    )
    config["use_corrective"] = _as_bool(
        raw.get("use_corrective"), DEFAULT_PIPELINE_CONFIG["use_corrective"]
    )
    config["use_reranker"] = _as_bool(
        raw.get("use_reranker"), DEFAULT_PIPELINE_CONFIG["use_reranker"]
    )
    config["max_hops"] = _as_int(
        raw.get("max_hops"), DEFAULT_PIPELINE_CONFIG["max_hops"], MIN_HOPS, MAX_HOPS
    )
    config["retrievers"] = _as_choice(
        raw.get("retrievers"), RETRIEVER_CHOICES, DEFAULT_PIPELINE_CONFIG["retrievers"]
    )
    config["corpus"] = _as_choice(
        raw.get("corpus"), CORPUS_CHOICES, DEFAULT_PIPELINE_CONFIG["corpus"]
    )

    return config


def is_default(config) -> bool:
    """True when the config asks for the historic full pipeline."""
    return normalize_pipeline_config(config) == DEFAULT_PIPELINE_CONFIG


def describe(config) -> str:
    """Short human-readable summary, for logs and status events."""
    resolved = normalize_pipeline_config(config)
    parts = [f"retrievers={resolved['retrievers']}", f"corpus={resolved['corpus']}"]
    parts.append(
        f"multi_hop={'x' + str(resolved['max_hops']) if resolved['use_multi_hop'] else 'off'}"
    )
    parts.append(f"corrective={'on' if resolved['use_corrective'] else 'off'}")
    parts.append(f"reranker={'on' if resolved['use_reranker'] else 'off'}")
    return " ".join(parts)
