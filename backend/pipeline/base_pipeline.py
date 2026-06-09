from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from router.models import Document, GuestUser
from router.models import Job
import logging
from ai_handler.llm import OpenAILLM, GeminiLLM, ClaudeLLM
import os
import glob
import hashlib
import json
from evaluation.models import Chunk

logger = logging.getLogger(__name__)

class BasePipeline(ABC):
    """
    The High-Level Controller.
    It orchestrates the RAG Engine (retrieval), the LLM (generation),
    and the Reranker (refinement).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = None         
        self.reranker = None    

    def get_document(self, username: str) -> Document | None:
        try:
            user = GuestUser.objects.filter(username=username).first()
            if not user:
                return None
            return Document.objects.filter(user=user).last()
        except Exception as e:
            logger.error(f"Error getting document for {username}: {e}")
            return None

    def optimize_query(self, query: str) -> str:
        """
        Optimizes the query for better retrieval.
        Returns ONLY the optimized string.
        """
        prompt = (
            "You are a query optimization tool for a Vector Database. "
            "Your task is to rewrite the user's input into a single, keyword-rich sentence "
            "that is optimized for cosine similarity search."
            "\n\n"
            "Rules:\n"
            "1. Output ONLY the rewritten query.\n"
            "2. Do NOT provide explanations, bullet points, or numbering.\n"
            "3. Do NOT use quotes around the output.\n"
            "4. Keep the language the same as the input.\n"
            "\n"
            f"Input: {query}\n"
            "Output:"
        )
        
        raw_response = self.llm.prompt_generate(prompt)
        
        optimized_query = self._validate_and_clean_query(raw_response, query)
        
        logger.info(f"Original Query: '{query}' -> Optimized: '{optimized_query}'")
        return optimized_query

    def _validate_and_clean_query(self, response: str, original_query: str) -> str:
        """
        Ensures the response is a clean, single-line string.
        If the LLM hallucinates or fails, fallback to the original query.
        """
        if not response:
            return original_query

        cleaned = response.strip()

        cleaned = cleaned.replace('"', '').replace("'", "")

        if "\n" in cleaned:
            cleaned = cleaned.split("\n")[0]

        prefixes = ["Here is", "Optimized query:", "Answer:"]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()

        if len(cleaned) > 200: 
            logger.warning(f"Optimization failed (result too long). Fallback to original.")
            return original_query

        return cleaned
    
    def _initialize_llm(self, model_name: str):
        if model_name.startswith("gpt-") or model_name.startswith("text-"):
            from ai_handler.llm import OpenAILLM
            return OpenAILLM(
                model=model_name,
                temperature=self.config.get("temperature", 0.0)
            )
        elif model_name.startswith("google"):
            from ai_handler.llm import GeminiLLM
            return GeminiLLM(
                model=model_name,
                temperature=self.config.get("temperature", 0.0)
            )
        elif model_name.startswith("anthropic"):
            from ai_handler.llm import ClaudeLLM
            return ClaudeLLM(
                model=model_name,
                temperature=self.config.get("temperature", 0.0)
            )
        else:
            raise ValueError(f"Unsupported LLM model: {model_name}")
        
    def is_initialized(self, username):
        """Check if this engine variant is already initialized for the user"""
        user_vector_dir = os.path.join(self.vector_store_root, username)
        
        if not os.path.exists(user_vector_dir):
            return False
        
        pattern = f"{username}_*_{self.method}_*.pkl"
        
        matching_files = glob.glob(os.path.join(user_vector_dir, pattern))
        
        return len(matching_files) > 0
    
    def get_retrieved_docs(self, query: str) -> List[str]:
        """
        Retrieves relevant documents using the RAG engine.
        """
        optimized_query = self.optimize_query(query)
        retrieved_docs = self.rag.retrieve(optimized_query)
        
        if not retrieved_docs:
            logger.warning("No documents retrieved with optimized query. Retrying with original query.")
            retrieved_docs = self.rag.retrieve(query)
        
        return retrieved_docs
    
    def get_retrieved_scores(self, query: str) -> Dict[str, List[float]]:
        """
        Get relevance scores for retrieved documents.
        """
        optimized_query = self.optimize_query(query)
        return self.rag.get_retrieved_scores(optimized_query)
    
    def _get_chunk_config(self) -> dict:
        return {
            "strategy": self.chunker.strategy,
            "chunk_size": self.chunker.chunk_size,
            "overlap": self.chunker.overlap,
        }

    def _get_config_hash(self) -> str:
        config = self._get_chunk_config()
        return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()

    def _sync_chunks(self, document, chunks: List[str]) -> List[Dict]:
        """
        Syncs chunks to DB with config-awareness:
        - Reuses existing chunks if text + config match
        - Creates new chunks if config changed
        - Deletes stale chunks no longer in current chunking result
        """
        config_hash = self._get_config_hash()
        config_meta = self._get_chunk_config()
        
        existing_qs = Chunk.objects.filter(document=document, config_hash=config_hash)
        existing_map = {c.text: c for c in existing_qs}  
        
        all_chunks_qs = Chunk.objects.filter(document=document)
        config_changed = all_chunks_qs.exists() and not existing_qs.exists()
        
        if config_changed:
            logger.info(f"Chunk config changed for document {document.id}. Replacing old chunks.")
            all_chunks_qs.delete()
            existing_map = {}

        incoming_texts = set(chunks)
        stale_chunks = [c for text, c in existing_map.items() if text not in incoming_texts]
        
        if stale_chunks:
            stale_ids = [c.id for c in stale_chunks]
            logger.info(f"Removing {len(stale_ids)} stale chunks for document {document.id}")
            Chunk.objects.filter(id__in=stale_ids).delete()
            for c in stale_chunks:
                existing_map.pop(c.text, None)

        new_texts = [text for text in chunks if text not in existing_map]
        if new_texts:
            created_chunks = Chunk.objects.bulk_create([
                Chunk(
                    document=document,
                    text=text,
                    metadata=config_meta,
                    config_hash=config_hash,
                )
                for text in new_texts
            ])
            logger.info(f"Created {len(created_chunks)} new chunks for document {document.id}")
            for chunk in created_chunks:
                existing_map[chunk.text] = chunk
        else:
            logger.info(f"All {len(chunks)} chunks reused from DB (config unchanged).")

        # Preserve original chunk order
        return [
            {"text": text, "chunk_id": existing_map[text].id}
            for text in chunks
            if text in existing_map
        ]
        

    @abstractmethod
    def _save_state(self, path: str):
        """Saves the vector index to disk."""
        pass

    @abstractmethod
    def _load_state(self, path: str) -> bool:
        """Loads the vector index from disk."""
        pass

    @abstractmethod
    def run(self, username:str, query: str) -> Dict[str, Any]:
        """
        The main execution flow:
        Query -> Retrieve -> Generate Answer.
        Replaces your 'parse_response'.
        """
        pass

    @abstractmethod
    def init(self, username: str) -> bool:
        """
        Initializes the pipeline by loading the state from the database
        .
        """
        pass
    
    @abstractmethod
    def init_job(self, username: str, job=None) -> bool:
        """
        Initializes the pipeline with optional job progress tracking.
        """
        pass
    
    
    @abstractmethod
    def _build_index(self, username: str, document: Document) -> bool:
        """
        Builds the vector index for the given document and user.
        """
        pass
    
    