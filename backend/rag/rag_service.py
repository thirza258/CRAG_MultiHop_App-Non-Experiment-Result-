from pipeline.app_pipeline import AppRAGPipeline


class RAGRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.engine = None
        self.initialize_engine()
        self._initialized = True

    def initialize_engine(self):
        """
        Initialize a single RAG pipeline.
        """

        crag_config = {
            "llm_model": "mistralai/mistral-nemo",
            "embedding_model": "google/gemini-embedding-2-preview",
            "upper_threshold": 0.91,
            "lower_threshold": 0.87,
            "strip_threshold": 0.88,
            "top_k": 4,
            "external_chunk_size": 1000,
            "external_chunk_overlap": 200,
        }

        multi_hop_config = {
            "llm_model": "mistralai/mistral-nemo",
            "max_hops": 3,
            "top_k": 4,
        }

        dense_config = {
            "collection_name": "ragreader_collection",
            "embedding_model": "google/gemini-embedding-2-preview",
            "top_k": 4,
        }

        sparse_config = {
            "collection_name": "ragreader_collection",
            "top_k": 4,
            "remove_stop_words": True,
        }

        hybrid_config = {
            "rerank_only": True,
            "reranker_model": "jinaai/jina-reranker-v3",
            "top_k": 4,
            "retrieval_top_k": 4,
        }

        instance_config = {
            "llm_model": "qwen/qwen3-30b-a3b-instruct-2507",
            "collection_name": "ragreader_collection",
            "dense_config": dense_config,
            "sparse_config": sparse_config,
            "hybrid_config": hybrid_config,
            "crag_config": crag_config,
            "multi_hop_config": multi_hop_config,
            "evaluation_llm_model": "google/gemma-4-26b-a4b-it",
            "evaluation_embedding_model": "openai/text-embedding-3-small"
        }

        try:
            self.engine = AppRAGPipeline(instance_config)
            print("✅ RAG Engine initialized")
        except Exception as e:
            print(f"❌ Error initializing RAG pipeline: {e}")

        print("--- RAG ENGINE READY ---")

    def get_engine(self, method=None, model=None):
        """
        Retrieve the initialized RAG pipeline.

        `method`/`model` are accepted for backwards compatibility with older
        call sites but ignored — a single shared pipeline serves all requests.
        If startup initialization failed (e.g. ChromaDB or the model cache
        was briefly unavailable), retry once per call instead of staying
        broken until the container restarts.
        """
        if self.engine is None:
            self.initialize_engine()
        if self.engine is None:
            raise ValueError(
                "RAG engine is not initialized (see startup logs for the "
                "original error). It will be retried on the next request."
            )
        return self.engine


# Global singleton instance
rag_registry = RAGRegistry()

_registry = None


def get_registry():
    global _registry
    if _registry is None:
        _registry = RAGRegistry()
    return _registry