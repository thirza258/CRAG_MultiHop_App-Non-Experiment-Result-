import pickle
import os
import re
from typing import List, Dict, Any
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from chroma.chroma_settings import get_chroma_client
from common.nltk_setup import ensure_nltk_data
from common.runtime.context import resolve_param
from emitter.status import NULL_EMITTER
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ensure_nltk_data() downloads whatever is missing and raises if it cannot. Importing
# this module must not be the thing that takes the app down: _tokenize and the
# stop-word list below both degrade on their own when the data never arrives.
try:
    ensure_nltk_data()
except Exception as exc:
    logger.warning(f"[SPARSE] NLTK data unavailable ({exc}) — falling back to plain tokenization")

class SparseRAG:
    def __init__(self, config: Dict[str, Any]):
        self.bm25 = None
        self.documents: List[str] = []
        self.metadatas: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []
        self._configured_top_k = config.get("top_k", 5)
        self._index_loaded = False
        # Which stop-word setting the loaded index was tokenised with. The query
        # is tokenised the same way at search time, so the two have to agree or
        # BM25 is matching against terms the corpus no longer contains.
        self._index_stop_words: bool = None

        # The pipeline injects the real emitter later via set_emitter(); until it
        # does, retrieve() must still be callable instead of raising AttributeError.
        self.emitter = NULL_EMITTER

        self.collection = get_chroma_client(collection_name=config.get('collection_name', 'sparse_rag_bm25'))

        self._configured_remove_stop_words = bool(config.get("remove_stop_words", True))
        # Loaded on first use rather than here: a request can switch stop-word
        # removal on even when the deployment configured it off, so the list
        # cannot be decided at construction — and a deployment that leaves it
        # off should never touch the NLTK corpus at all.
        self._stop_word_list = None

    def _load_stop_words(self) -> set:
        """The English stop-word list, read from NLTK once and cached."""
        cached = getattr(self, "_stop_word_list", None)
        if cached is not None:
            return cached

        try:
            self._stop_word_list = set(stopwords.words('english'))
        except Exception as e:
            # stopwords.words() reads the NLTK corpus from disk and raises
            # LookupError when it is missing. BM25 still ranks without
            # stop-word removal, so this must not kill retrieval.
            logger.warning(f"[SPARSE] Stop-word list unavailable ({e}) — indexing without stop-word removal")
            self._stop_word_list = set()

        return self._stop_word_list

    @property
    def top_k(self) -> int:
        """How many chunks to return for this request."""
        return int(resolve_param("top_k", getattr(self, "_configured_top_k", 5)))

    @top_k.setter
    def top_k(self, value) -> None:
        # Assignment sets the *configured* default; a request still overrides it.
        self._configured_top_k = value

    @property
    def remove_stop_words(self) -> bool:
        """Whether this request wants English stop words dropped."""
        return bool(
            resolve_param(
                "remove_stop_words",
                getattr(self, "_configured_remove_stop_words", True),
            )
        )

    @property
    def stop_words(self) -> set:
        """The stop words to actually strip, given this request's choice.

        Empty when removal is off, and the NLTK corpus is not read at all in
        that case.
        """
        if not self.remove_stop_words:
            return set()
        return self._load_stop_words()

    @stop_words.setter
    def stop_words(self, value) -> None:
        self._stop_word_list = set(value or ())

    def _tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            # ChromaDB can hand back None documents, and callers tokenize the raw
            # query — neither should blow up the whole retrieval.
            if text is not None:
                logger.warning(f"[SPARSE] Cannot tokenize {type(text).__name__} — treating as empty")
            return []

        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        try:
            tokens = word_tokenize(text)
        except Exception as e:
            # word_tokenize needs the punkt data; whitespace splitting keeps BM25
            # usable (the text is already stripped of punctuation above).
            logger.warning(f"[SPARSE] word_tokenize unavailable ({e}) — falling back to whitespace split")
            tokens = text.split()

        stop_words = getattr(self, "stop_words", set())
        return [w for w in tokens if w not in stop_words]

    @staticmethod
    def _clean_documents(documents: List[str]) -> List[str]:
        """Keep positions intact while making every entry a string — a None chunk
        from ChromaDB would otherwise crash tokenization here or the reranker later."""
        return [doc if isinstance(doc, str) else "" for doc in (documents or [])]

    @staticmethod
    def _align_metas(documents: List[str], metadatas: List[Dict]) -> List[Dict]:
        """Metas are indexed by document position, so a short/None metadata list
        from ChromaDB has to be padded rather than raise IndexError later."""
        metadatas = [
            meta if isinstance(meta, dict) else {}
            for meta in (metadatas or [])
        ]
        if len(metadatas) < len(documents):
            metadatas.extend({} for _ in range(len(documents) - len(metadatas)))
        return metadatas[:len(documents)]

    def _load_index_from_chroma(self) -> bool:
        """Lazy-load BM25 index from ChromaDB once.

        Returns False (never raises) when ChromaDB is unreachable or the corpus
        cannot be turned into a BM25 index — the caller reads that as "no results".
        """
        # A cached index tokenised under a different stop-word setting cannot
        # be searched with this request's tokenisation, so it is rebuilt rather
        # than silently mismatched.
        if getattr(self, "_index_loaded", False) and (
            getattr(self, "_index_stop_words", None) != self.remove_stop_words
        ):
            logger.info(
                "[SPARSE] Stop-word setting changed (%s -> %s) — rebuilding the BM25 index",
                getattr(self, "_index_stop_words", None),
                self.remove_stop_words,
            )
            self._index_loaded = False

        if self._index_loaded:
            return True

        try:
            count = self.collection.count()
            if not count:
                logger.warning("Warning: No documents indexed.")
                return False

            logger.info(f"Loading {count} chunks from ChromaDB into BM25...")
            all_docs = self.collection.get() or {}

            documents = self._clean_documents(all_docs.get("documents"))
            if not documents:
                logger.warning("[SPARSE] ChromaDB reported documents but returned none.")
                return False

            self.documents = documents
            self.metadatas = self._align_metas(documents, all_docs.get("metadatas"))
            self.tokenized_corpus = [self._tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        except Exception as e:
            # ChromaDB down, or a corpus BM25Okapi refuses (e.g. no usable tokens
            # at all, which divides by zero inside rank_bm25).
            logger.error(f"[SPARSE] Could not build BM25 index from ChromaDB: {e}", exc_info=True)
            return False

        self._index_loaded = True
        self._index_stop_words = self.remove_stop_words
        logger.info("BM25 index ready.")
        return True

    def retrieve(self, query: str, keyword: str = None, where_filter: Dict = None) -> tuple[List[str], List[Dict]]:
        # AppRAGPipeline treats an empty retrieval as survivable but not an
        # exception, so every failure mode below degrades to ([], []).
        try:
            # set_emitter() may not have run yet (or at all, for a bare instance).
            getattr(self, "emitter", NULL_EMITTER).emit(
                "sparse_retrieval", f"Starting retrieval for query: '{str(query)[:80]}'"
            )

            if not self._load_index_from_chroma():
                return [], []

            # Redundant on the first call, but the index is cached across calls: a
            # collection emptied in the meantime must still read as "no results".
            try:
                if not self.collection.count():
                    logger.warning("Warning: No documents indexed.")
                    return [], []
            except Exception as e:
                logger.error(f"[SPARSE] Could not count collection (ChromaDB unreachable?): {e}")
                return [], []

            search_term = keyword if keyword else query

            if where_filter:
                filtered_result = self.collection.get(where=where_filter) or {}
                filtered_docs = self._clean_documents(filtered_result.get("documents"))
                if not filtered_docs:
                    logger.warning("Warning: No documents matched the filter.")
                    return [], []
                filtered_metas = self._align_metas(filtered_docs, filtered_result.get("metadatas"))
                tokenized_corpus = [self._tokenize(doc) for doc in filtered_docs]
                try:
                    bm25 = BM25Okapi(tokenized_corpus)
                except Exception as e:
                    # rank_bm25 divides by the corpus size / term count, so a corpus
                    # with no usable tokens raises instead of scoring.
                    logger.error(f"[SPARSE] Could not build BM25 index for filtered corpus: {e}")
                    return [], []
            else:
                filtered_docs = self.documents
                filtered_metas = self._align_metas(self.documents, self.metadatas)
                bm25 = self.bm25

            if not filtered_docs or bm25 is None:
                logger.warning("[SPARSE] No usable BM25 corpus for this query. Returning empty result.")
                return [], []

            tokenized_query = self._tokenize(search_term)
            scores = np.array(bm25.get_scores(tokenized_query))

            top_k = min(self.top_k, len(filtered_docs), len(scores))
            # A top_k of 0 turns scores.argsort()[-0:] into "the whole corpus,
            # reversed", so bail out instead of returning everything.
            if top_k < 1:
                logger.warning(f"[SPARSE] top_k resolved to {top_k} — returning empty result.")
                return [], []

            # A cached BM25 index can outlive the document list it was built from,
            # so drop any index the current corpus cannot answer for.
            top_indices = [i for i in scores.argsort()[-top_k:][::-1] if i < len(filtered_docs)]

            return [filtered_docs[i] for i in top_indices], [filtered_metas[i] for i in top_indices]

        except Exception as e:
            logger.error(f"[SPARSE] Sparse retrieval failed for query '{str(query)[:80]}': {e}", exc_info=True)
            return [], []

    def set_collection(self, collection_name: str):
        """Swap collection at runtime without reinitializing."""
        self.collection_name = collection_name
        self.collection = get_chroma_client(collection_name=collection_name)
        # The BM25 index is built from the collection's contents and cached
        # behind _index_loaded, which nothing reset here. On a shared pipeline
        # that meant the first collection queried after startup kept answering
        # for every later one — including another user's documents.
        self._index_loaded = False
        self.bm25 = None
        self.documents = []
        self.metadatas = []
        self.tokenized_corpus = []
        logger.info(f"[SparseRAG] Switched to collection: {collection_name}")
        
    def set_emitter(self, emitter):
        self.emitter = emitter