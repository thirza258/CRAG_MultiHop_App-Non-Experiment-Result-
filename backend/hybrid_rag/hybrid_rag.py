from typing import List, Dict, Any
from collections import defaultdict
from sparse_rag.sparse_rag import SparseRAG
from dense_rag.dense_rag import DenseRAG
import torch
import numpy as np
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel, AutoModelForCausalLM
import json
from common.memory import _local_model_path
from common.runtime.context import resolve_param
from emitter.status import NULL_EMITTER
from collections import defaultdict

_RERANKER_CACHE: Dict[str, Any] = {}
import logging

logger = logging.getLogger(__name__)

class HybridRAG:
    def __init__(self, config: Dict[str, Any]):
        self._rerank_only = config.get("rerank_only", False)
        if not self._rerank_only:
            dense_config = {
                "embedding_model": config.get("embedding_model", "openai/text-embedding-3-small"),
                "llm_model":       config.get("llm_model", "google/gemini-flash-2.0"),
                "collection_name": config.get("dense_collection_name", "dense_rag"),
                "top_k":           config.get("top_k", 5),
            }
            sparse_config = {
                "remove_stop_words": config.get("remove_stop_words", True),
                "collection_name":   config.get("sparse_collection_name", "sparse_rag"),
                "top_k":             config.get("top_k", 5),
            }
            self.dense_rag  = DenseRAG(dense_config)
            self.sparse_rag = SparseRAG(sparse_config)

        self.reranker_model_name = config.get("reranker_model", "jina")
        self._configured_final_top_k = config.get("retrieval_top_k", 5)
        self._configured_top_k       = config.get("top_k", 5)
        self.documents: List[str] = []

        # Callers inject the real emitter through set_emitter(); until then a
        # no-op emitter keeps self.emitter.emit(...) safe on every code path.
        self.emitter = NULL_EMITTER

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Reranker device: {self.device} | CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Loading reranker model: {self.reranker_model_name!r}")
        try:
            self._load_reranker()
        except Exception as exc:
            # A missing or corrupt local reranker snapshot must not make the whole
            # RAG pipeline unconstructible — retrieval still works, it just keeps
            # the merged candidate order instead of a reranked one.
            self._model = None
            logger.warning(
                f"[HybridRAG] Reranker {self.reranker_model_name!r} could not be loaded ({exc}) — "
                f"continuing without reranking",
                exc_info=True,
            )

    @property
    def top_k(self) -> int:
        """Candidate budget, matching what the base retrievers were asked for."""
        return int(resolve_param("top_k", getattr(self, "_configured_top_k", 5)))

    @top_k.setter
    def top_k(self, value) -> None:
        self._configured_top_k = value

    @property
    def final_top_k(self) -> int:
        """How many chunks survive the merge/rerank — what the LLM actually reads.

        Its own knob rather than sharing top_k: retrieving widely and then
        keeping a few is the normal shape of a rerank, so the two budgets are
        independently useful.
        """
        return int(
            resolve_param("rerank_top_k", getattr(self, "_configured_final_top_k", 5))
        )

    @final_top_k.setter
    def final_top_k(self, value) -> None:
        self._configured_final_top_k = value

    def _load_reranker(self) -> None:
        cache_key = self.reranker_model_name

        if cache_key in _RERANKER_CACHE:
            logger.info(f"Cache hit — reusing loaded reranker: {self.reranker_model_name}")
            for attr, val in _RERANKER_CACHE[cache_key].items():
                setattr(self, attr, val)
            return

        logger.info("Loading fresh — will cache for next iteration...")
        model_path = _local_model_path(self.reranker_model_name)

        self._model = (
            AutoModel
            .from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        )
        self._model.eval()
        _RERANKER_CACHE[cache_key] = {"_model": self._model}

    @staticmethod
    def _extract_text_only(chunk: str) -> str:
        if "Text:" in chunk:
            return chunk.split("Text:", 1)[1].strip()
        return chunk.strip()

    @staticmethod
    def _normalize_meta(meta: Dict) -> Dict:
        """Normalize NaN/None values to empty string for stable keying."""
        return {
            k: ("" if (v is None or (isinstance(v, float) and np.isnan(v)) or str(v).lower() == "nan") else v)
            for k, v in meta.items()
        }

    @staticmethod
    def _meta_to_key(meta: Dict, chunk_text: str = "") -> str:
        """
        Build dedup key based on source type:
        - External (has url): url + chunk_index  → keeps each chunk from same page separate
        - User doc (has document_id): document_id + chunk_index → keeps each chunk separate
        - Fallback: md5 of content → nothing wrongly collapsed
        """
        if meta.get("url"):
            return json.dumps({
                "url":         meta["url"].strip().rstrip("/").lower(),
                "chunk_index": meta.get("chunk_index", 0),
            }, sort_keys=True)

        if meta.get("document_id") is not None:
            return json.dumps({
                "document_id": meta["document_id"],
                "chunk_index": meta.get("chunk_index", 0),
            }, sort_keys=True)

        # Fallback — hash the content so nothing is wrongly merged
        import hashlib
        return hashlib.md5(chunk_text.encode("utf-8", errors="ignore")).hexdigest()


    def _deduplicate_new_chunks(
        self,
        retrieved_chunks: List[str],
        metadatas:        List[Dict],
        seen_keys:        set,
    ) -> tuple[List[str], List[Dict]]:
        new_chunks, new_metas = [], []

        for chunk, meta in zip(retrieved_chunks, metadatas):
            meta = self._normalize_meta(meta)
            key  = self._meta_to_key(meta, chunk)   # <-- pass chunk text for fallback

            if key in seen_keys:
                continue

            seen_keys.add(key)

            text_only = self._extract_text_only(chunk)
            enriched  = (
                f"Title: {meta.get('title', '')}\n"
                f"Author: {meta.get('author', '')}\n"
                f"Source: {meta.get('source', meta.get('username', ''))}\n"
                f"Category: {meta.get('category', 'user_document')}\n\n"
                f"Text: {text_only}"
            )

            new_chunks.append(enriched)
            new_metas.append(meta)

        return new_chunks, new_metas

    def retrieve(self, query: str, keyword: str = None, where_filter: Dict = None):
        dense_results,  dense_metadatas  = self.dense_rag.retrieve(query, None, where_filter)
        sparse_results, sparse_metadatas = self.sparse_rag.retrieve(query, keyword, where_filter)
        return self._merge_and_rerank(query, dense_results, dense_metadatas, sparse_results, sparse_metadatas)

    def retrieve_from_precomputed(
        self,
        query:         str,
        dense_chunks:  List[str],
        sparse_chunks: List[str],
        dense_metas:   List[Dict] = None,
        sparse_metas:  List[Dict] = None,
        *,
        use_reranker:  bool = True,
    ) -> tuple[List[str], List[Dict], str]:
        if isinstance(dense_chunks,  str): dense_chunks  = json.loads(dense_chunks)
        if isinstance(sparse_chunks, str): sparse_chunks = json.loads(sparse_chunks)
        if isinstance(dense_metas,   str): dense_metas   = json.loads(dense_metas)
        if isinstance(sparse_metas,  str): sparse_metas  = json.loads(sparse_metas)

        dense_chunks  = dense_chunks  or []
        sparse_chunks = sparse_chunks or []
        dense_metas   = dense_metas   or []
        sparse_metas  = sparse_metas  or []

        logger.debug(
            f"retrieve_from_precomputed | "
            f"dc={len(dense_chunks)} dm={len(dense_metas)} "
            f"sc={len(sparse_chunks)} sm={len(sparse_metas)}"
        )

        if not dense_chunks and not sparse_chunks:
            return [], [], "ERROR: no retrieved chunks in both dense and sparse"
        if not dense_metas and not sparse_metas:
            return [], [], "ERROR: no retrieved metas in both dense and sparse"

        if not dense_chunks or not dense_metas:
            logger.warning(f"dense side empty (dc={len(dense_chunks)} dm={len(dense_metas)}) — using sparse only")
            all_chunks = sparse_chunks
            all_metas  = sparse_metas
        elif not sparse_chunks or not sparse_metas:
            logger.warning(f"sparse side empty (sc={len(sparse_chunks)} sm={len(sparse_metas)}) — using dense only")
            all_chunks = dense_chunks
            all_metas  = dense_metas
        else:
            all_chunks = dense_chunks + sparse_chunks
            all_metas  = dense_metas  + sparse_metas

        # ── Dedup ─────────────────────────────────────────────────────────────────
        seen_keys: set = set()
        merged_chunks, merged_metas = self._deduplicate_new_chunks(
            all_chunks, all_metas, seen_keys
        )

        if not merged_chunks:
            logger.warning("0 chunks after dedup — falling back to raw input")
            seen, fallback_chunks, fallback_metas = set(), [], []
            for c, m in zip(all_chunks, all_metas):
                if c not in seen:
                    seen.add(c)
                    fallback_chunks.append(c)
                    fallback_metas.append(m)
            if not fallback_chunks:
                return [], [], "ERROR: no chunks after dedup and fallback"
            merged_chunks, merged_metas = fallback_chunks, fallback_metas

        # ── Rerank ────────────────────────────────────────────────────────────────
        # Opting out is a caller decision (user-facing toggle), so it keeps the
        # merge order rather than reporting a reranker problem.
        if not use_reranker:
            logger.info(
                f"[HybridRAG] Reranking disabled by caller — returning first "
                f"{self.final_top_k} of {len(merged_chunks)} merged chunks"
            )
            return (
                merged_chunks[:self.final_top_k],
                merged_metas[:self.final_top_k],
                "ok (rerank disabled)",
            )

        reranked_indices, rerank_status = self._rerank(query, merged_chunks)
        reranked_indices = reranked_indices[:self.final_top_k]

        if not reranked_indices:
            return (
                merged_chunks[:self.final_top_k],
                merged_metas[:self.final_top_k],
                "ERROR: reranker returned empty indices",
            )

        reranked_docs  = [merged_chunks[i] for i in reranked_indices]
        reranked_metas = [merged_metas[i]  for i in reranked_indices]

        return reranked_docs, reranked_metas, rerank_status

    def _merge_and_rerank(
        self,
        query:       str,
        dense_docs:  List[str], dense_metas:  List[Dict],
        sparse_docs: List[str], sparse_metas: List[Dict],
    ) -> tuple[List[str], List[Dict], str]:

        # ── 1. Dedup & merge ──────────────────────────────────────────────────────
        seen_keys: set = set()
        merged_chunks, merged_metas = self._deduplicate_new_chunks(
            dense_docs + sparse_docs,
            dense_metas + sparse_metas,
            seen_keys,
        )

        if not merged_chunks:
            logger.warning("_merge_and_rerank: 0 chunks after dedup — returning raw input")
            seen, fallback_chunks, fallback_metas = set(), [], []
            for c, m in zip(dense_docs + sparse_docs, dense_metas + sparse_metas):
                if c not in seen:
                    seen.add(c)
                    fallback_chunks.append(c)
                    fallback_metas.append(m)
            return (
                fallback_chunks[:self.final_top_k],
                fallback_metas[:self.final_top_k],
                "ERROR: no chunks after dedup",
            )

        self.emitter.emit("reranking_pipeline", f"Merged {len(dense_docs)} dense and {len(sparse_docs)} sparse chunks into {len(merged_chunks)} unique candidates")
        reranked_indices, rerank_status = self._rerank(query, merged_chunks)
        reranked_indices = reranked_indices[:self.final_top_k]

        if not reranked_indices:
            return (
                merged_chunks[:self.final_top_k],
                merged_metas[:self.final_top_k],
                "ERROR: reranker returned empty indices",
            )

        reranked_docs  = [merged_chunks[i] for i in reranked_indices]
        reranked_metas = [merged_metas[i]  for i in reranked_indices]

        return reranked_docs, reranked_metas, rerank_status

    def _rerank(self, query: str, candidates: List[str]) -> tuple[List[int], str]:
        if not candidates:
            return [], "ok"
        # Construction is allowed to succeed without a reranker (see __init__), so a
        # missing model is an expected state here — never worth retrying the load.
        if getattr(self, "_model", None) is None:
            logger.warning("[HybridRAG] Reranker model unavailable — keeping original candidate order")
            return list(range(len(candidates))), "ok (reranking skipped: reranker unavailable)"
        try:
            return self._rerank_jina(query, candidates), "ok"
        except Exception as exc:
            logger.error(f"Reranking error (jina): {exc}", exc_info=True)
        # Callers index into the candidate list, so the fallback must be
        # indices (original order), not the documents themselves.
        return list(range(len(candidates))), "ERROR: reranking failed, returning original order"

    def _rerank_jina(self, query: str, candidates: List[str]) -> List[int]:
        results = self._model.rerank(
            query,
            candidates,
            top_n=len(candidates),
        )
        return [r["index"] for r in results]
    
    def set_emitter(self, emitter):
        self.emitter = emitter