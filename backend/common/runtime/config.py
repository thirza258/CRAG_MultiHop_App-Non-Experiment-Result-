"""Per-query pipeline configuration.

The chat UI lets a user switch individual pipeline stages off, size the
retrieval, pick the models, and choose which corpus to search. A query carries
that choice as a ``CONFIG`` object on the websocket message (or in the REST
payload); everything in this module exists to turn whatever the client sent
into a safe, fully-populated dict.

The contract that matters: **absent, partial or malformed input normalises to
:data:`DEFAULT_PIPELINE_CONFIG`**, which is the exact composition the pipeline
used before this feature existed. A client that knows nothing about config
keeps working unchanged.

Two sentinel conventions carry "the user did not choose", and they matter:

* ``""`` for a model id or a named strategy.
* ``None`` for a number.

Both mean *defer to whatever the deployment configured in
``rag/rag_service.py``* — deliberately not "use the value written here". Giving
these concrete defaults would silently override every deployment's own tuning
with one hard-coded number, which is the opposite of configurable.

Booleans are the exception: they carry real defaults, all of them matching the
historic behaviour, because "on" and "off" leave no room for a third state.

API keys are **not** part of this object — they travel in a separate ``KEYS``
field handled by :mod:`common.runtime.api_keys`. Everything here is logged verbatim and
summarised into status events shown to the user, so nothing secret may live in
it.
"""

import logging

from common.runtime.context import normalize_model_id

logger = logging.getLogger(__name__)

MIN_HOPS = 1
MAX_HOPS = 3

RETRIEVER_CHOICES = ("both", "dense", "sparse")
CORPUS_CHOICES = ("auto", "user", "base")
STRICTNESS_CHOICES = ("lenient", "balanced", "strict")
CHUNK_STRATEGY_CHOICES = ("fixed", "paragraph", "semantic", "recursive")

# ── Retrieval sizing bounds ──────────────────────────────────────────────────
MIN_TOP_K, MAX_TOP_K = 1, 20
MIN_TEMPERATURE, MAX_TEMPERATURE = 0.0, 2.0
MIN_CHUNK_SIZE, MAX_CHUNK_SIZE = 100, 4000
MIN_CHUNK_OVERLAP, MAX_CHUNK_OVERLAP = 0, 1000
MIN_EXPANSION_QUERIES, MAX_EXPANSION_QUERIES = 1, 5

#: Corrective grading thresholds per strictness preset.
#:
#: Exposed as a preset rather than three free floats on purpose. The grader
#: normalises cosine similarity from [-100, 100] to [0, 1]
#: (``CRAGEvaluator._normalize``), so real relevance scores cluster in roughly
#: 0.5-0.95 and the useful band between "accept this chunk" and "escalate to
#: the web" is only a few hundredths wide. Three independent sliders in a
#: 0.04-wide band, where the decision logic also requires upper > lower, is a
#: footgun dressed as a feature: set upper below lower and every chunk grades
#: "incorrect" and every query hits external search.
#:
#: "balanced" reproduces the values rag/rag_service.py has always shipped.
CORRECTIVE_THRESHOLDS = {
    "lenient": {"upper": 0.88, "lower": 0.84, "strip": 0.85},
    "balanced": {"upper": 0.91, "lower": 0.87, "strip": 0.88},
    "strict": {"upper": 0.94, "lower": 0.90, "strip": 0.91},
}

DEFAULT_PIPELINE_CONFIG = {
    # ── Stage composition ────────────────────────────────────────────────────
    # Decompose the question into follow-up hops.
    "use_multi_hop": True,
    # Hop ceiling; ignored when use_multi_hop is False. None defers to the
    # deployment's multi_hop_config, so a deployment that tuned it down is not
    # silently pushed back up to the schema default by an untouched request.
    "max_hops": None,
    # Self-grade retrieved context and correct/escalate when it is weak.
    "use_corrective": True,
    # Reorder merged candidates with the local cross-encoder.
    "use_reranker": True,
    # Score the finished answer with the RAGAS judge. Off saves two LLM-judge
    # calls plus an embedding pass per query, and costs only the two numbers
    # shown under the answer.
    "use_evaluation": True,
    # Which base retrievers feed the merge step.
    "retrievers": "both",
    # "auto" keeps the historic behaviour: the user's own collection when it
    # has anything in it, otherwise the shared base corpus.
    "corpus": "auto",

    # ── Corrective sub-stages (ignored when use_corrective is False) ─────────
    # Let weak local context escalate to the web at all.
    "use_external_search": True,
    # The two external sources, individually.
    "use_wikipedia": True,
    "use_news": True,
    # LLM-rewritten keywords, reformulations and alternative phrasings. Off
    # also disables the ambiguous-resolution retries, which are built on them.
    "use_query_expansion": True,
    # How many alternative phrasings the ambiguous retry may try.
    "expansion_queries": None,
    # CRAG's strip-level refinement of a chunk that graded ambiguous.
    "use_knowledge_refinement": True,
    # How readily a chunk is accepted rather than escalated. "" defers to the
    # thresholds the deployment set in crag_config, so a deployment that tuned
    # them is never silently overridden by a preset.
    "corrective_strictness": "",

    # ── Retrieval sizing ─────────────────────────────────────────────────────
    # Chunks each base retriever returns.
    "top_k": None,
    # Chunks kept after the merge/rerank step — what the LLM actually reads.
    "rerank_top_k": None,
    # Passages the external (Wikipedia / news) search returns when corrective
    # escalates to the web. Separate from top_k: these are fetched and scored
    # in-memory per query, not read from the index.
    "external_top_k": None,
    # Drop English stop words before BM25 tokenisation.
    "remove_stop_words": True,

    # ── Generation ───────────────────────────────────────────────────────────
    # OpenRouter chat model for every generative step (answer, hop decisions,
    # query expansion). "" means "whatever each stage is configured with",
    # i.e. the historic per-stage models.
    "llm_model": "",
    # Sampling temperature for the *answer*. The pipeline's own decision calls
    # (hop bridging, keyword extraction) stay deterministic regardless: making
    # those random would change which documents are retrieved, not how the
    # answer reads.
    "temperature": None,

    # ── Embedding / indexing ─────────────────────────────────────────────────
    # OpenRouter embedding model. "" means the pipeline's configured default.
    #
    # This one only takes effect where it *can*: a collection's vectors were
    # produced by one specific model, and embedding a query with a different one
    # would compare vectors of different dimensions. So the pipeline pins dense
    # retrieval to whatever the target collection was indexed with and this
    # value applies when indexing new documents. See
    # AppRAGPipeline._resolve_collection_for and _build_index.
    "embedding_model": "",
    # How uploaded documents are split. Applies at index time only, for the
    # same reason: the chunks in a collection are already split.
    "chunk_strategy": "",
    "chunk_size": None,
    "chunk_overlap": None,

    # ── Evaluation detail (ignored when use_evaluation is False) ─────────────
    # The judge models stay separate from the answering model by default: a
    # judge that is the same model as the one being judged scores its own
    # output. "" keeps the configured judge.
    "evaluation_llm_model": "",
    "evaluation_embedding_model": "",
    # The two RAGAS metrics, individually.
    "eval_answer_relevancy": True,
    "eval_faithfulness": True,
}

_TRUTHY = {"1", "true", "yes", "on", "t"}
_FALSY = {"0", "false", "no", "off", "f", ""}

#: Every boolean field, so they can be coerced in one pass.
_BOOL_FIELDS = (
    "use_multi_hop",
    "use_corrective",
    "use_reranker",
    "use_evaluation",
    "use_external_search",
    "use_wikipedia",
    "use_news",
    "use_query_expansion",
    "use_knowledge_refinement",
    "remove_stop_words",
    "eval_answer_relevancy",
    "eval_faithfulness",
)

#: Model-id fields: validated, "" means defer.
_MODEL_FIELDS = ("llm_model", "embedding_model", "evaluation_llm_model", "evaluation_embedding_model")

#: Optional integers as (field, low, high). None means defer.
_OPTIONAL_INT_FIELDS = (
    ("top_k", MIN_TOP_K, MAX_TOP_K),
    ("rerank_top_k", MIN_TOP_K, MAX_TOP_K),
    ("external_top_k", MIN_TOP_K, MAX_TOP_K),
    ("max_hops", MIN_HOPS, MAX_HOPS),
    ("chunk_size", MIN_CHUNK_SIZE, MAX_CHUNK_SIZE),
    ("chunk_overlap", MIN_CHUNK_OVERLAP, MAX_CHUNK_OVERLAP),
    ("expansion_queries", MIN_EXPANSION_QUERIES, MAX_EXPANSION_QUERIES),
)


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


def _as_optional_int(value, low: int, high: int):
    """Same as :func:`_as_int` but ``None`` means "defer to the server"."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return max(low, min(high, number))


def _as_optional_float(value, low: float, high: float):
    """A clamped float, or ``None`` for "defer to the server"."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    # NaN survives float() and compares false against everything, which would
    # get it straight past the clamp and into an API request.
    if number != number:
        return None
    return max(low, min(high, number))


def _as_choice(value, choices: tuple, default: str) -> str:
    if isinstance(value, str) and value.strip().lower() in choices:
        return value.strip().lower()
    return default


def _as_optional_choice(value, choices: tuple) -> str:
    """A named choice, or ``""`` for "defer to the server"."""
    return _as_choice(value, choices, "")


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

    for field in _BOOL_FIELDS:
        config[field] = _as_bool(raw.get(field), DEFAULT_PIPELINE_CONFIG[field])

    config["retrievers"] = _as_choice(
        raw.get("retrievers"), RETRIEVER_CHOICES, DEFAULT_PIPELINE_CONFIG["retrievers"]
    )
    config["corpus"] = _as_choice(
        raw.get("corpus"), CORPUS_CHOICES, DEFAULT_PIPELINE_CONFIG["corpus"]
    )
    config["corrective_strictness"] = _as_optional_choice(
        raw.get("corrective_strictness"), STRICTNESS_CHOICES
    )
    config["chunk_strategy"] = _as_optional_choice(
        raw.get("chunk_strategy"), CHUNK_STRATEGY_CHOICES
    )

    # A model id is interpolated into an API request, so it is validated rather
    # than passed through; anything malformed falls back to the default.
    for field in _MODEL_FIELDS:
        config[field] = normalize_model_id(
            raw.get(field), DEFAULT_PIPELINE_CONFIG[field]
        )

    for field, low, high in _OPTIONAL_INT_FIELDS:
        config[field] = _as_optional_int(raw.get(field), low, high)

    config["temperature"] = _as_optional_float(
        raw.get("temperature"), MIN_TEMPERATURE, MAX_TEMPERATURE
    )

    # Overlap must leave something to overlap with. A chunker given
    # overlap >= size either loops forever or emits one chunk per character,
    # so this is corrected rather than passed on.
    if config["chunk_size"] is not None and config["chunk_overlap"] is not None:
        if config["chunk_overlap"] >= config["chunk_size"]:
            corrected = max(MIN_CHUNK_OVERLAP, config["chunk_size"] // 4)
            logger.warning(
                "[PipelineConfig] chunk_overlap=%s is not smaller than chunk_size=%s "
                "— using %s",
                config["chunk_overlap"],
                config["chunk_size"],
                corrected,
            )
            config["chunk_overlap"] = corrected

    return config


def corrective_thresholds(config):
    """The grading thresholds for this config's strictness preset.

    Returns ``None`` when the config expressed no preference — the caller then
    keeps whatever the deployment configured. A returned triple always satisfies
    ``upper > strip > lower``, which the grader's decision logic depends on.
    """
    resolved = normalize_pipeline_config(config)
    preset = resolved["corrective_strictness"]
    if not preset:
        return None
    return dict(CORRECTIVE_THRESHOLDS.get(preset, CORRECTIVE_THRESHOLDS["balanced"]))


def is_default(config) -> bool:
    """True when the config asks for the historic full pipeline."""
    return normalize_pipeline_config(config) == DEFAULT_PIPELINE_CONFIG


def describe(config) -> str:
    """Short human-readable summary, for logs and status events.

    Only the parts that differ from the defaults are spelled out past the core
    stage line — this runs on every query and a 25-field dump would bury the
    thing the reader is looking for.
    """
    resolved = normalize_pipeline_config(config)

    parts = [f"retrievers={resolved['retrievers']}", f"corpus={resolved['corpus']}"]
    if not resolved["use_multi_hop"]:
        hops = "off"
    else:
        hops = f"x{resolved['max_hops']}" if resolved["max_hops"] else "on"
    parts.append(f"multi_hop={hops}")
    parts.append(f"corrective={'on' if resolved['use_corrective'] else 'off'}")
    parts.append(f"reranker={'on' if resolved['use_reranker'] else 'off'}")
    parts.append(f"evaluation={'on' if resolved['use_evaluation'] else 'off'}")
    parts.append(f"llm={resolved['llm_model'] or 'default'}")
    parts.append(f"embedding={resolved['embedding_model'] or 'default'}")

    # Everything else only when the user actually changed it.
    changed = [
        f"{field}={resolved[field]}"
        for field in DEFAULT_PIPELINE_CONFIG
        if field not in (
            "retrievers", "corpus", "use_multi_hop", "max_hops", "use_corrective",
            "use_reranker", "use_evaluation", "llm_model", "embedding_model",
        )
        and resolved[field] != DEFAULT_PIPELINE_CONFIG[field]
    ]
    parts.extend(changed)

    return " ".join(parts)
