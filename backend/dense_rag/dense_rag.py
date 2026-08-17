import os
import numpy as np
from typing import List, Dict, Any, Optional
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from chroma.chroma_settings import get_chroma_client
from emitter.status import NULL_EMITTER
import logging
from math import ceil

logger = logging.getLogger(__name__)

class DenseRAG:
    def __init__(self, config: Dict[str, Any]):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables.")
        
        self.config = config
        self.embedding_model = config.get('embedding_model', 'openai/text-embedding-3-small')
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        self.llm_model = config.get('llm_model', "google/gemini-3-flash-preview")
        self.top_k = config.get('top_k', 5)
        self.documents: List[str] = []

        # The pipeline injects the real emitter later via set_emitter(); until it
        # does, retrieve() must still be callable instead of raising AttributeError.
        self.emitter = NULL_EMITTER

        self.collection = get_chroma_client(collection_name=config.get('collection_name', 'default_corpus'))

    def _get_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        try:
            # Callers (_build_index) align embeddings with their
            # input positionally, so coerce odd entries instead of dropping them.
            cleaned_texts = []
            coerced = 0
            for text in texts or []:
                if isinstance(text, str):
                    cleaned_texts.append(text.replace("\n", " "))
                    continue
                coerced += 1
                cleaned_texts.append("" if text is None else str(text).replace("\n", " "))
        except Exception as e:
            logger.error(f"[DENSE] Could not normalise input texts for embedding: {e}")
            return []

        if coerced:
            # Positions are preserved, but an embedded placeholder would otherwise be
            # indistinguishable from real content — say so loudly instead.
            logger.warning(f"[DENSE] Warning: {coerced} of {len(cleaned_texts)} inputs were not text and had to be coerced")

        if not cleaned_texts:
            logger.warning("[DENSE] No texts to embed — returning no embeddings.")
            return []

        # Callers derive batch_size from the input size (min(len(texts), 50)), so a
        # zero/negative value can reach us and range() refuses a 0 step.
        if not batch_size or batch_size < 1:
            batch_size = len(cleaned_texts)

        all_embeddings = []
        logger.info(f"[DENSE] Fetching embeddings for {len(cleaned_texts)} texts using model {self.embedding_model}...")

        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i : i + batch_size]
            try:
                
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.embedding_model,
                    encoding_format="float"
                )
                    
                if not response.data:
                    logger.warning(f"[DENSE] Warning: No embedding data received for batch {i // batch_size}")
                    continue
                    
                all_embeddings.extend([data.embedding for data in response.data])

                logger.info(f"[DENSE] Embedded batch {i // batch_size + 1} / {ceil(len(cleaned_texts) / batch_size)}")

            except Exception as e:
                logger.error(f"[DENSE] Error fetching embeddings for batch {i // batch_size}: {e}")
                continue

        if len(all_embeddings) != len(cleaned_texts):
            logger.warning(f"[DENSE] Warning: expected {len(cleaned_texts)} embeddings, got {len(all_embeddings)}")

        return all_embeddings
        
    def retrieve(self, query: str, keyword: str = None, where_filter: Dict = None) -> tuple[List[str], List[Dict]]:
        # AppRAGPipeline treats an empty retrieval as survivable but not an
        # exception, so every failure mode below degrades to ([], []).
        try:
            logger.info(f"[RETRIEVE] Query: '{str(query)[:80]}' | where_filter={where_filter} | keyword={keyword}")
            # set_emitter() may not have run yet (or at all, for a bare instance).
            getattr(self, "emitter", NULL_EMITTER).emit(
                "dense_retrieval", f"Starting retrieval for query: '{str(query)[:80]}'"
            )

            collection_name = getattr(self.collection, "name", "unknown")

            # count() is the first ChromaDB round-trip — it is where an unreachable
            # server shows up, so it gets its own guard and its own log line.
            try:
                collection_count = self.collection.count()
            except Exception as e:
                logger.error(f"[RETRIEVE] Could not count collection '{collection_name}' (ChromaDB unreachable?): {e}")
                return [], []

            logger.info(f"[RETRIEVE] Collection '{collection_name}' has {collection_count} documents.")

            if not collection_count:
                logger.warning(f"[RETRIEVE] Collection '{collection_name}' is empty. Returning empty result.")
                return [], []

            logger.info(f"[RETRIEVE] Embedding query using model '{self.embedding_model}'...")
            query_embeddings = self._get_embeddings([query])
            if not query_embeddings:
                logger.warning(f"[RETRIEVE] Embedding returned empty for query: '{str(query)[:80]}'")
                return [], []
            logger.debug(f"[RETRIEVE] Query embedded successfully. Vector dim={len(query_embeddings[0])}")

            count = collection_count
            if where_filter:
                available_docs = (self.collection.get(where=where_filter) or {}).get("documents") or []
                available = len(available_docs)
                logger.info(f"[RETRIEVE] Docs matching where_filter: {available} / {collection_count}")
                if available == 0:
                    logger.warning(f"[RETRIEVE] No documents match where_filter={where_filter}. Falling back to full count.")
                    where_filter = None
                count = available if available > 0 else count

            n_results = min(self.top_k, count)
            logger.info(f"[RETRIEVE] Querying ChromaDB | n_results={n_results} (top_k={self.top_k}, count={count})")

            if n_results < 1:
                logger.warning(f"[RETRIEVE] n_results={n_results} (top_k={self.top_k}) — nothing to ask ChromaDB for.")
                return [], []

            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas"]
            ) or {}

            # A malformed/partial payload (missing or empty "documents"/"metadatas")
            # must read as "nothing found" rather than KeyError/IndexError.
            docs_payload  = results.get("documents") or []
            metas_payload = results.get("metadatas") or []
            # query() nests one row per query embedding; anything else (a flat or
            # None payload) is not something we can safely unpack.
            first_docs  = docs_payload[0]  if docs_payload  else None
            first_metas = metas_payload[0] if metas_payload else None
            docs  = list(first_docs)  if isinstance(first_docs,  (list, tuple)) else []
            metas = list(first_metas) if isinstance(first_metas, (list, tuple)) else []

            if not docs:
                logger.warning(f"[RETRIEVE] ChromaDB returned no documents for query: '{str(query)[:80]}'")
                return [], []

            # Downstream code zips/indexes metas against docs (and calls .items() on
            # each meta), so pad and coerce instead of letting it fail later.
            metas = [meta if isinstance(meta, dict) else {} for meta in metas]
            if len(metas) < len(docs):
                logger.warning(f"[RETRIEVE] Metadata shorter than documents ({len(metas)} < {len(docs)}) — padding.")
                metas.extend({} for _ in range(len(docs) - len(metas)))
            metas = metas[:len(docs)]

            logger.info(f"[RETRIEVE] Retrieved {len(docs)} chunks from '{collection_name}'.")
            logger.debug(f"[RETRIEVE] Metadata: {metas}")

            return docs, metas

        except Exception as e:
            logger.error(f"[RETRIEVE] Dense retrieval failed for query '{str(query)[:80]}': {e}", exc_info=True)
            return [], []

    def set_collection(self, collection_name: str):
        """Swap collection at runtime without reinitializing."""
        self.collection_name = collection_name
        self.collection = get_chroma_client(collection_name=collection_name)
        logger.info(f"[DenseRAG] Switched to collection: {collection_name}")
        
    def set_emitter(self, emitter):
        self.emitter = emitter
        