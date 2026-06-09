from typing import List
import logging

logger = logging.getLogger(__name__)

class FallbackRetriever:
    """
    Chain of Responsibility:
      user_collection → dataset_collection → MultiHopRetriever
    """

    def __init__(
        self,
        user_collection,
        dataset_collection,
        multi_hop_retriever,      # MultiHopRetriever wrapping CorrectiveRAG
        n_results: int = 5,
    ):
        self.user_collection = user_collection
        self.dataset_collection = dataset_collection
        self.multi_hop_retriever = multi_hop_retriever
        self.n_results = n_results

        # Define the chain explicitly — easy to reorder or extend
        self.chain = [
            ("user",       self._query_user_collection),
            ("dataset",    self._query_dataset_collection),
            ("multi_hop",  self._query_multi_hop),
        ]

    def retrieve(self, query: str) -> dict:
        for source_name, retriever_fn in self.chain:
            logger.info(f"[Fallback] Trying: {source_name}...")
            chunks = retriever_fn(query)

            if chunks:
                logger.info(f"[Fallback] Hit on: {source_name} ({len(chunks)} chunks)")
                return {"chunks": chunks, "source": source_name}

            logger.info(f"[Fallback] Miss on: {source_name}, escalating...")

        # Should never reach here since multi_hop always returns something
        logger.warning("[Fallback] All stages exhausted, returning empty.")
        return {"chunks": [], "source": "none"}

    # --- Individual retriever fns ---

    def _query_user_collection(self, query: str) -> List[str]:
        return self._query_chroma(self.user_collection, query)

    def _query_dataset_collection(self, query: str) -> List[str]:
        return self._query_chroma(self.dataset_collection, query)

    def _query_multi_hop(self, query: str) -> List[str]:
        """
        MultiHopRetriever decomposes the query into sub-queries,
        runs CorrectiveRAG on each, and merges the results.
        """
        try:
            # Adjust this to match your MultiHopRetriever's actual method signature
            result = self.multi_hop_retriever.retrieve(query)

            # Normalize — MultiHop may return dicts or strings
            if isinstance(result, list):
                return [
                    r["text"] if isinstance(r, dict) else r
                    for r in result if r
                ]
            return []
        except Exception as e:
            logger.error(f"[Fallback] MultiHop failed: {e}")
            return []

    def _query_chroma(self, collection, query: str) -> List[str]:
        try:
            results = collection.query(
                query_texts=[query],
                n_results=self.n_results
            )
            documents = results.get("documents", [[]])[0]
            return [doc for doc in documents if doc and doc.strip()]
        except Exception as e:
            logger.warning(f"[Fallback] Chroma query failed: {e}")
            return []