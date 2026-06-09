from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union

class BaseRAG(ABC):
    """
    Abstract Base Class for all RAG methodologies.
    Enforces a standard structure: Index -> Retrieve -> Generate.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration (API keys, DB paths, model names).
        """
        self.config = config

    @abstractmethod
    def index_documents(self, documents: List[str]) -> None:
        """
        Step 1: Ingest data.
        - Sparse: Build BM25 index.
        - Dense: Create embeddings and store in Vector DB.
        - Graph: Extract entities and build Knowledge Graph.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str) -> List[str]:
        """
        Step 2: Get relevant context.
        Returns a list of strings (chunks/documents).
        """
        pass
    
    @abstractmethod
    def get_retrieved_scores(self, query: str) -> Dict[str, List[float]]:
        """
        Optional: Return relevance scores or metadata for retrieved documents.
        Useful for debugging and for advanced fusion techniques.
        """
        pass


    def run(self, query: str) -> str:
        """
        The main execution pipeline.
        """
        pass