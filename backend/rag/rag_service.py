from pipeline.app_pipeline import AppRAGPipeline
from rag.config import load_pipeline_config
from threading import RLock


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
        self._lock = RLock()
        self._initialized = True

    def initialize_engine(self):
        """
        Initialize a single RAG pipeline.
        """

        instance_config = load_pipeline_config()

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
        with self._lock:
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