import logging
from dataclasses import dataclass, field
from typing import Any

from pipeline.base_pipeline import BasePipeline

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Ranked retrieval result from a single pipeline."""
    pipeline_name: str
    ranked_chunks: list[dict[str, Any]]   
    error: str | None = None


@dataclass
class PooledResult:
    """Final output of CandidatePooler.pool()."""
    rrf_ranked_chunks: list[dict[str, Any]]   # RRF-merged, best-first
    rrf_chunk_ids: list[str]                  # convenience: IDs only
    per_pipeline: dict[str, PipelineResult]   # raw results keyed by name
    query: str = ""


# ── RRF implementation ───────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
    id_key: str = "chunk_id",
) -> list[dict[str, Any]]:
    """
    Standard RRF over multiple ranked lists of chunk dicts.

    Score formula: sum(1 / (k + rank_i))  for each list i that contains
    the chunk, where rank_i is 1-based.

    Args:
        ranked_lists: Each element is an ordered list of chunk dicts.
                      Each dict must contain `id_key`.
        k:            RRF constant (default 60, per the original paper).
        id_key:       Dict key used to identify a chunk.

    Returns:
        Single merged list, sorted by descending RRF score.
        Each dict is the original chunk dict enriched with `rrf_score`.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict[str, Any]] = {}  # id → best chunk dict seen

    for ranked_list in ranked_lists:
        for rank_0, chunk in enumerate(ranked_list):
            cid = chunk.get(id_key)
            if cid is None:
                logger.warning(f"Chunk missing '{id_key}' key — skipped: {chunk}")
                continue

            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank_0 + 1)

            # Keep the first-seen copy of the chunk payload; later pipelines
            # may have slightly different score fields but the text is the same.
            if cid not in chunk_map:
                chunk_map[cid] = chunk

    merged = []
    for cid, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        entry = dict(chunk_map[cid])  # shallow copy — don't mutate originals
        entry["rrf_score"] = round(rrf_score, 6)
        merged.append(entry)

    return merged


# ── CandidatePooler ──────────────────────────────────────────────────────────

class CandidatePooler:
    """
    Retrieves candidates from every registered pipeline, fuses them with RRF,
    and returns the merged ranking as a pseudo-ground-truth for evaluation.

    Usage
    -----
    pooler = CandidatePooler(k=60)
    pooler.register("hybrid", hybrid_pipeline)
    pooler.register("dense",  dense_pipeline)
    pooler.register("sparse", sparse_pipeline)

    result = pooler.pool(document_id="...", query="What is X?")

    # Use RRF IDs as ground truth
    evaluate_chunks(retrieved_ids, set(result.rrf_chunk_ids))
    """

    def __init__(self, k: int = 60, top_n: int | None = None):
        """
        Args:
            k:     RRF constant. Higher → more weight to mid-rank hits.
            top_n: If set, truncate the final RRF list to the top-N chunks.
        """
        self.k = k
        self.top_n = top_n
        self._pipelines: dict[str, BasePipeline] = {}

    # ── Registration ────────────────────────────────────────────────────────

    def register(self, name: str, pipeline: BasePipeline) -> "CandidatePooler":
        """Register a pipeline under a unique name. Returns self for chaining."""
        if name in self._pipelines:
            logger.warning(f"CandidatePooler: overwriting existing pipeline '{name}'")
        self._pipelines[name] = pipeline
        return self

    def unregister(self, name: str) -> None:
        self._pipelines.pop(name, None)

    @property
    def pipeline_names(self) -> list[str]:
        return list(self._pipelines.keys())

    # ── Core retrieval ───────────────────────────────────────────────────────

    def _retrieve_from_pipeline(
        self,
        name: str,
        pipeline: BasePipeline,
        query: str,
    ) -> PipelineResult:
        """
        Call the pipeline's underlying retriever directly (bypasses LLM generation).
        Falls back to _run_core if the pipeline exposes it.
        """
        try:
            # All three pipeline types expose rag.retrieve() via _run_core.
            # We replicate the retrieval-only path here to avoid LLM calls.
            optimized_query = pipeline.optimize_query(query)

            # Try direct retrieval (avoids paying for LLM generation)
            if hasattr(pipeline, "rag") and hasattr(pipeline.rag, "retrieve"):
                raw = pipeline.rag.retrieve(optimized_query) or pipeline.rag.retrieve(query)
            else:
                # Fallback: full _run_core (will call LLM — less efficient)
                logger.warning(
                    f"Pipeline '{name}' has no rag.retrieve(); "
                    "falling back to _run_core (LLM will be invoked)."
                )
                core_result = pipeline._run_core(pipeline.get_document_from_query(query), query)
                raw = core_result.get("retrieved_docs", [])

            ranked = raw or []
            logger.info(f"[{name}] retrieved {len(ranked)} chunks.")
            return PipelineResult(pipeline_name=name, ranked_chunks=ranked)

        except Exception as e:
            logger.error(f"[{name}] retrieval failed: {e}", exc_info=True)
            return PipelineResult(pipeline_name=name, ranked_chunks=[], error=str(e))

    # ── Public API ───────────────────────────────────────────────────────────

    def pool(
        self,
        query: str,
        document_id: str | None = None,
        username: str | None = None,
        ensure_init: bool = True,
    ) -> PooledResult:
        """
        Retrieve from all registered pipelines and fuse results with RRF.

        Args:
            query:       The user / evaluation query.
            document_id: If provided, calls pipeline.get_document(document_id)
                         before retrieval (ensures the right doc is loaded).
            username:    Alternative to document_id — used when pipelines are
                         keyed by user.
            ensure_init: If True, calls pipeline.init() when memory is empty.

        Returns:
            PooledResult with rrf_ranked_chunks and per-pipeline raw results.
        """
        if not self._pipelines:
            raise RuntimeError("No pipelines registered. Call .register() first.")

        per_pipeline: dict[str, PipelineResult] = {}
        ranked_lists: list[list[dict[str, Any]]] = []

        for name, pipeline in self._pipelines.items():

            # Optionally ensure the index is loaded into memory
            if ensure_init and username:
                try:
                    doc_in_memory = (
                        hasattr(pipeline, "rag")
                        and hasattr(pipeline.rag, "dense_engine")
                        and len(getattr(pipeline.rag.dense_engine, "documents", []) or []) > 0
                    )
                    if not doc_in_memory:
                        logger.info(f"[{name}] Index not in memory — calling init()...")
                        pipeline.init(username)
                except Exception as e:
                    logger.error(f"[{name}] init() failed: {e}")

            result = self._retrieve_from_pipeline(name, pipeline, query)
            per_pipeline[name] = result

            if result.ranked_chunks:
                ranked_lists.append(result.ranked_chunks)

        if not ranked_lists:
            logger.warning("All pipelines returned empty results.")
            return PooledResult(
                rrf_ranked_chunks=[],
                rrf_chunk_ids=[],
                per_pipeline=per_pipeline,
                query=query,
            )

        rrf_chunks = reciprocal_rank_fusion(ranked_lists, k=self.k)

        if self.top_n is not None:
            rrf_chunks = rrf_chunks[: self.top_n]

        rrf_ids = [c["chunk_id"] for c in rrf_chunks if "chunk_id" in c]

        logger.info(
            f"CandidatePooler: {len(rrf_chunks)} unique chunks after RRF "
            f"from {len(ranked_lists)} pipelines."
        )

        return PooledResult(
            rrf_ranked_chunks=rrf_chunks,
            rrf_chunk_ids=rrf_ids,
            per_pipeline=per_pipeline,
            query=query,
        )

    def pool_as_ground_truth(
        self,
        query: str,
        username: str | None = None,
        top_n: int | None = None,
    ) -> set[str]:
        """
        Convenience wrapper: returns just the RRF chunk IDs as a set,
        ready to drop into evaluate_chunks(retrieved_ids, ground_truth_ids).

        Args:
            top_n: Override self.top_n for this call only.
        """
        result = self.pool(query=query, username=username)
        ids = result.rrf_chunk_ids
        if top_n is not None:
            ids = ids[:top_n]
        return set(ids)