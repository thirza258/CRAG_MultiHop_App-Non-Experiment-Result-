import pickle
import os
import re
from typing import List, Dict, Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from rag.base_rag import BaseRAG
from chroma.chroma_settings import get_chroma_client
import logging
import numpy as np

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
logger = logging.getLogger(__name__)

class SparseRAG:
    def __init__(self, config: Dict[str, Any]):
        self.bm25 = None
        self.documents: List[str] = []
        self.metadatas: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []
        self.top_k = config.get("top_k", 5)
        self._index_loaded = False

        self.collection = get_chroma_client(collection_name=config.get('collection_name', 'sparse_rag_bm25'))
        
        if config.get("remove_stop_words", True):
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = set()
            
    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        tokens = word_tokenize(text)
        return [w for w in tokens if w not in self.stop_words]
    
    def _load_index_from_chroma(self) -> bool:
        """Lazy-load BM25 index from ChromaDB once."""
        if self._index_loaded:
            return True

        count = self.collection.count()
        if count == 0:
            logger.warning("Warning: No documents indexed.")
            return False

        logger.info(f"Loading {count} chunks from ChromaDB into BM25...")
        all_docs = self.collection.get()
        
        self.documents = all_docs["documents"]
        self.metadatas = all_docs["metadatas"]
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._index_loaded = True
        logger.info("BM25 index ready.")
        return True
        
    def index_documents(self, documents: List[str]) -> None:
        if not documents:
            logger.warning("No documents to index.")
            return
        
        self.documents = list(documents)
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.documents]
        
        try:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            self._index_loaded = True

            logger.info("Indexing complete.")
        except Exception as e:
            logger.error(f"Error during indexing: {e}")

    def retrieve(self, query: str, keyword: str = None, where_filter: Dict = None) -> tuple[List[str], List[Dict]]:
        self.emitter.emit("sparse_retrieval", f"Starting retrieval for query: '{query[:80]}'")
        if not self._load_index_from_chroma():
            return [], []
        
        if self.collection.count() == 0:
            logger.warning("Warning: No documents indexed.")
            return [], []
        
        search_term = keyword if keyword else query
        
        if where_filter:
            filtered_result = self.collection.get(where=where_filter)
            filtered_docs = filtered_result["documents"]
            filtered_metas = filtered_result["metadatas"]
            if not filtered_docs:
                logger.warning("Warning: No documents matched the filter.")
                return [], []
            tokenized_corpus = [self._tokenize(doc) for doc in filtered_docs]
            bm25 = BM25Okapi(tokenized_corpus)
        else:
            filtered_docs = self.documents
            filtered_metas = self.metadatas
            bm25 = self.bm25
            
        
        tokenized_query = self._tokenize(search_term)
        scores = np.array(bm25.get_scores(tokenized_query))

        top_k = min(self.top_k, len(filtered_docs))
        top_indices = scores.argsort()[-top_k:][::-1]

        return [filtered_docs[i] for i in top_indices], [filtered_metas[i] for i in top_indices]
    
    def set_collection(self, collection_name: str):
        """Swap collection at runtime without reinitializing."""
        self.collection_name = collection_name
        self.collection = get_chroma_client(collection_name=collection_name)
        logger.info(f"[SparseRAG] Switched to collection: {collection_name}")
        
    def set_emitter(self, emitter):
        self.emitter = emitter