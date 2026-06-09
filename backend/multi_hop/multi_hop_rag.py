from collections import defaultdict
from typing import Any, List, Dict
import json
from ai_handler.llm import OpenRouterLLM
import logging
import hashlib

logger = logging.getLogger(__name__)

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
        self.max_hops = config.get("max_hops", 3)
        self.top_k = config.get(
            "top_k",
            getattr(retriever, "top_k", 5)
        )
        self.evaluator    = retriever.evaluator

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

        response = self.llm.generate(prompt)

        logger.info("[Bridge Response]")
        logger.info(response)
        logger.info("=" * 80 + "\n")

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

            retrieved_chunks, metadatas = self.retriever.retrieve(
                query=current_query,
                keyword=keyword,
                where_filter=where_filter,
                seen_urls=seen_urls
            )
            
            retrieved_chunks = retrieved_chunks[:self.top_k]
            metadatas  = metadatas[:self.top_k]

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

            for idx, (chunk, meta) in enumerate(zip(new_chunks, new_metas), start=1):
                all_chunks.append(chunk)
                all_metadata.append(meta)

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
        
        if len(all_chunks) > self.top_k:
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

        return top_chunks, top_metas
    
    def retrieve(self, query: str, keyword: str = None, where_filter: Dict = None) -> tuple[List[str], List[Dict]]:
        return self.multi_hop_retrieve(query, keyword, where_filter)
    
    def set_emitter(self, emitter):
        self.emitter = emitter
        self.retriever.set_emitter(emitter)