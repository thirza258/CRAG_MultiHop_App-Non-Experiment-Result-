from collections import defaultdict
from typing import Any, List, Dict
import inspect
import json
from ai_handler.llm import OpenRouterLLM
from common.runtime.context import resolve_param
from emitter.status import NULL_EMITTER
import logging
import hashlib

logger = logging.getLogger(__name__)

# ai_handler.llm reports exhausted retries by RETURNING "<Provider> Error (...)"
# instead of raising, so those bodies have to be recognised as failures here.
_LLM_ERROR_PREFIXES = ("OpenRouter Error", "OpenAI Error")

BRIDGE_PROMPT = """
You are helping with multi-hop question answering.

Original question: {question}
Retrieved passages so far:
{passages}

Decide:
1. If the passages contain enough info to answer the question, reply exactly: SUFFICIENT
2. If not, reply exactly: NEED: <specific entity or concept to search next>

One line only. No explanation.
"""

class MultiHopRetriever:
    def __init__(self, retriever, config: Dict[str, Any]):
        self.retriever = retriever
        self.llm = OpenRouterLLM(
            model=config.get("llm_model", "google/gemini-3-flash-preview")
        )
        self._configured_max_hops = config.get("max_hops", 3)
        self._configured_top_k = config.get(
            "top_k",
            getattr(retriever, "top_k", 5)
        )
        # Only CorrectiveRAG owns an evaluator. Corrective mode is going to become
        # switchable, and a bare Dense/SparseRAG wrapped directly has none — so the
        # re-rank stays optional instead of exploding at construction time.
        self.evaluator = getattr(retriever, "evaluator", None)
        # The pipeline injects the real emitter via set_emitter() later; until then
        # status calls must be no-ops rather than AttributeError.
        self.emitter = NULL_EMITTER
        # Probed once: CorrectiveRAG.retrieve takes seen_urls, Dense/SparseRAG do not.
        self._inner_accepts_seen_urls = self._accepts_seen_urls(retriever)

    @property
    def max_hops(self) -> int:
        """This request's hop ceiling.

        Read per call rather than held as state. It used to be assigned onto
        this object by the pipeline before each query, which on the shared
        single-instance pipeline meant two concurrent queries asking for
        different hop counts raced: whichever wrote last set the ceiling for
        both.
        """
        return int(resolve_param(
            "max_hops", getattr(self, "_configured_max_hops", 3)
        ))

    @max_hops.setter
    def max_hops(self, value) -> None:
        self._configured_max_hops = value

    @property
    def top_k(self) -> int:
        """The hop's chunk budget for this request.

        Read per call rather than captured at construction: the wrapped
        retriever resolves the same value, and truncating to a stale copy here
        would quietly discard chunks the inner layer was asked to return.
        """
        return int(resolve_param("top_k", getattr(self, "_configured_top_k", 5)))

    @top_k.setter
    def top_k(self, value) -> None:
        self._configured_top_k = value

    @staticmethod
    def _accepts_seen_urls(retriever) -> bool:
        """
        True when the wrapped retriever's retrieve() can take a seen_urls keyword.
        """
        retrieve = getattr(retriever, "retrieve", None)

        if retrieve is None:
            return False

        try:
            params = inspect.signature(retrieve).parameters
        except (TypeError, ValueError):
            # Introspection-hostile callables (C-implemented, exotic wrappers): assume
            # the narrower signature so the hop still runs instead of TypeError-ing.
            logger.warning("[MultiHop] Could not inspect retrieve(); assuming no seen_urls support.")
            return False

        if "seen_urls" in params:
            return True

        # A **kwargs retriever swallows anything we hand it.
        return any(
            param.kind is inspect.Parameter.VAR_KEYWORD
            for param in params.values()
        )

    def _extract_bridge_or_done(
        self,
        original_query: str,
        passages: List[str]
    ) -> str | None:
        """
        Returns None if context is sufficient,
        or a new search string (bridge entity) if not.
        """
        passages_text = "\n\n".join(passages)

        prompt = BRIDGE_PROMPT.format(
            question=original_query,
            passages=passages_text
        )

        logger.info("\n" + "=" * 80)
        logger.info("[Bridge Prompt]")
        logger.info(prompt)
        logger.info("=" * 80)

        try:
            response = self.llm.generate(prompt)
        except Exception as exc:
            # No bridge decision means no further hop — the chunks already gathered
            # are still usable, so stop cleanly instead of failing the whole query.
            logger.error(f"[Bridge] LLM call failed, stopping retrieval: {exc}")
            return None

        logger.info("[Bridge Response]")
        logger.info(response)
        logger.info("=" * 80 + "\n")

        if not isinstance(response, str) or not response.strip():
            logger.info("[Decision] Empty LLM response. Stopping retrieval.")
            return None

        if response.startswith(_LLM_ERROR_PREFIXES):
            # An error body is not a decision, and must never be treated as a
            # bridge entity to search for on the next hop.
            logger.error(f"[Decision] LLM returned an error body. Stopping retrieval: {response}")
            return None

        if response.startswith("SUFFICIENT"):
            logger.info("[Decision] Context is sufficient. Stopping retrieval.")
            return None

        elif response.startswith("NEED:"):
            bridge_query = response.replace("NEED:", "").strip()
            logger.info(f"[Decision] Need another hop. Next query: {bridge_query}")
            return bridge_query

        logger.info("[Decision] Unexpected response format. Stopping retrieval.")
        return None

    @staticmethod
    def _extract_text_only(chunk: str) -> str:
        """Extract only the 'Text:' part from enriched chunk."""
        if "Text:" in chunk:
            return chunk.split("Text:", 1)[1].strip()
        return chunk.strip()

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Strip trailing slash, fragment, and query params for stable dedup."""
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url.strip().rstrip("/"))
        return urlunparse(parsed._replace(query="", fragment="")).lower()

    @staticmethod
    def _meta_to_key(meta: Dict, chunk_text: str = "") -> str:
        if meta.get("url"):
            url = MultiHopRetriever._normalize_url(meta["url"])
            return json.dumps({
                "url":         url,
                "chunk_index": meta.get("chunk_index", 0),
            }, sort_keys=True)

        if meta.get("document_id") is not None:
            return json.dumps({
                "document_id": meta["document_id"],
                "chunk_index": meta.get("chunk_index", 0),
            }, sort_keys=True)

        return hashlib.md5(chunk_text.encode()).hexdigest()

    def _deduplicate_new_chunks(
        self,
        retrieved_chunks: List[str],
        metadatas: List[Dict],
        seen_keys: set
    ) -> tuple[List[str], List[Dict]]:
        grouped = defaultdict(list)
        meta_lookup = {}

        for chunk, meta in zip(retrieved_chunks, metadatas):
            key = self._meta_to_key(meta, chunk) 

            if key in seen_keys:
                continue

            text_only = self._extract_text_only(chunk)
            grouped[key].append(text_only)
            meta_lookup[key] = meta

        new_chunks, new_metas = [], []

        for key, text_list in grouped.items():
            meta = meta_lookup[key]
            combined_text = "\n\n".join(text_list)

            enriched_chunk = (
                f"Title: {meta.get('title', '')}\n"
                f"Author: {meta.get('author', '')}\n"
                f"Source: {meta.get('source', meta.get('username', ''))}\n"
                f"Category: {meta.get('category', 'user_document')}\n\n"
                f"Text: {combined_text}"
            )

            seen_keys.add(key)
            new_chunks.append(enriched_chunk)
            new_metas.append(meta)

        return new_chunks, new_metas

    def multi_hop_retrieve(
        self,
        query: str,
        keyword: str = None,
        where_filter: Dict = None
    ) -> tuple[List[str], List[Dict]]:
        all_chunks = []
        all_metadata = []
        seen_keys = set() 
        seen_urls = set() 
        current_query = query

        for hop in range(self.max_hops):
            logger.info(f"\n[Hop {hop + 1}/{self.max_hops}]")
            self.emitter.emit("multi_hop_retrieval", f"Starting hop {hop + 1}")

            retrieve_kwargs = {
                "query":        current_query,
                "keyword":      keyword,
                "where_filter": where_filter,
            }
            if self._inner_accepts_seen_urls:
                retrieve_kwargs["seen_urls"] = seen_urls

            try:
                retrieved_chunks, metadatas = self.retriever.retrieve(**retrieve_kwargs)
                retrieved_chunks = retrieved_chunks[:self.top_k]
                metadatas  = metadatas[:self.top_k]
            except Exception as exc:
                # A dead hop must not throw away what earlier hops already found.
                logger.error(
                    f"[Hop {hop + 1}] retrieval failed, keeping the "
                    f"{len(all_chunks)} chunks gathered so far: {exc}"
                )
                self.emitter.emit(
                    "multi_hop_retrieval",
                    f"Hop {hop + 1} retrieval failed. Continuing with chunks gathered so far."
                )
                break

            self.emitter.emit("multi_hop_retrieval", f"Retrieved {len(retrieved_chunks)} chunks in hop {hop + 1}")

            if not retrieved_chunks:
                logger.info("No chunks retrieved. Stopping.")
                break

            new_chunks, new_metas = self._deduplicate_new_chunks(
                retrieved_chunks, metadatas, seen_keys
            )

            if not new_chunks:
                logger.info("No new unique chunks after deduplication. Stopping.")
                self.emitter.emit("multi_hop_retrieval", "No new unique chunks found. Stopping.")
                break

            all_chunks.extend(new_chunks)
            all_metadata.extend(new_metas)

            bridge = self._extract_bridge_or_done(
                original_query=query,
                passages=all_chunks
            )
            self.emitter.emit("multi_hop_retrieval", f"Bridge decision: {'Done' if bridge is None else 'Next query: ' + bridge}")

            if bridge is None:
                break

            current_query = bridge
        
        top_chunks = all_chunks
        top_metas = all_metadata
        
        # Without an evaluator there is nothing to rank by, so the whole pool is
        # handed on — HybridRAG re-ranks the merged result downstream anyway.
        if self.evaluator is not None and len(all_chunks) > self.top_k:
            try:
                scores = self.evaluator.score_docs(query, all_chunks)

                ranked = sorted(
                    zip(scores, all_chunks, all_metadata),
                    key=lambda x: x[0],
                    reverse=True
                )

                top_chunks = [c for _, c, _ in ranked[:self.top_k]]
                top_metas  = [m for _, _, m in ranked[:self.top_k]]

                logger.info(f"[multi_hop_retrieve] pool={len(all_chunks)}, returning top-{self.top_k} after re-rank")
                logger.info(f"[multi_hop_retrieve] top scores: {[round(s,3) for s,_,_ in ranked[:self.top_k]]}")
            except Exception as exc:
                # Scoring needs the embedding model; unranked chunks still answer the
                # question, so fall back to retrieval order instead of returning nothing.
                logger.error(f"[multi_hop_retrieve] re-rank failed, returning first {self.top_k} chunks unranked: {exc}")
                top_chunks = all_chunks[:self.top_k]
                top_metas  = all_metadata[:self.top_k]

        return top_chunks, top_metas
    
    def retrieve(self, query: str, keyword: str = None, where_filter: Dict = None) -> tuple[List[str], List[Dict]]:
        return self.multi_hop_retrieve(query, keyword, where_filter)
    
    def set_max_hops(self, max_hops: int) -> None:
        """
        Hop count is going to be user-configurable; clamp so the loop always runs once.
        """
        try:
            hops = int(max_hops)
        except (TypeError, ValueError):
            # A malformed user setting should not break retrieval — keep the current value.
            logger.warning(f"[MultiHop] Ignoring invalid max_hops={max_hops!r}, keeping {self.max_hops}")
            return

        self.max_hops = max(1, hops)
        logger.info(f"[MultiHop] max_hops set to {self.max_hops}")

    def set_emitter(self, emitter):
        self.emitter = emitter
        self.retriever.set_emitter(emitter)