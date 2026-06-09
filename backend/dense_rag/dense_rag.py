import os
import numpy as np
from typing import List, Dict, Any, Optional
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from rag.base_rag import BaseRAG
from chroma.chroma_settings import get_chroma_client
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

        self.collection = get_chroma_client(collection_name=config.get('collection_name', 'default_corpus'))
        
    def _get_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        cleaned_texts = [text.replace("\n", " ") for text in texts]
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

        if len(all_embeddings) != len(texts):
            logger.warning(f"[DENSE] Warning: expected {len(texts)} embeddings, got {len(all_embeddings)}")

        return all_embeddings
        
    def index_documents(self, documents: List[str]) -> None:
        self.documents = list(documents)
        embeddings_list = self._get_embeddings(self.documents)
        if not embeddings_list:
            return
            
    def retrieve(self, query: str, keyword: str = None, where_filter: Dict = None) -> tuple[List[str], List[Dict]]:
        logger.info(f"[RETRIEVE] Query: '{query[:80]}' | where_filter={where_filter} | keyword={keyword}")
        self.emitter.emit("dense_retrieval", f"Starting retrieval for query: '{query[:80]}'")

        collection_count = self.collection.count()
        logger.info(f"[RETRIEVE] Collection '{self.collection.name}' has {collection_count} documents.")

        if collection_count == 0:
            logger.warning(f"[RETRIEVE] Collection '{self.collection.name}' is empty. Returning empty result.")
            return [], []

        logger.info(f"[RETRIEVE] Embedding query using model '{self.embedding_model}'...")
        query_embeddings = self._get_embeddings([query])
        if not query_embeddings:
            logger.warning(f"[RETRIEVE] Embedding returned empty for query: '{query[:80]}'")
            return [], []
        logger.debug(f"[RETRIEVE] Query embedded successfully. Vector dim={len(query_embeddings[0])}")

        count = self.collection.count()
        if where_filter:
            available_docs = self.collection.get(where=where_filter)["documents"]
            available = len(available_docs)
            logger.info(f"[RETRIEVE] Docs matching where_filter: {available} / {collection_count}")
            if available == 0:
                logger.warning(f"[RETRIEVE] No documents match where_filter={where_filter}. Falling back to full count.")
                where_filter = None
            count = available if available > 0 else count

        n_results = min(self.top_k, count)
        logger.info(f"[RETRIEVE] Querying ChromaDB | n_results={n_results} (top_k={self.top_k}, count={count})")

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas"]
        )

        docs  = results["documents"][0]
        metas = results["metadatas"][0]
        logger.info(f"[RETRIEVE] Retrieved {len(docs)} chunks from '{self.collection.name}'.")
        logger.debug(f"[RETRIEVE] Metadata: {metas}")

        return docs, metas
    
    def set_collection(self, collection_name: str):
        """Swap collection at runtime without reinitializing."""
        self.collection_name = collection_name
        self.collection = get_chroma_client(collection_name=collection_name)
        logger.info(f"[DenseRAG] Switched to collection: {collection_name}")
        
    def set_emitter(self, emitter):
        self.emitter = emitter
        