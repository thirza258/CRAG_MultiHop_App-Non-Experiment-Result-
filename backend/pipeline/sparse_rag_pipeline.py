import os
import pickle
import logging
import uuid

from typing import Dict, Any
from pipeline.base_pipeline import BasePipeline
from sparse_rag.sparse_rag import SparseRAG
from ai_handler.llm import OpenAILLM
from common.chunker import DocumentChunker
from utils.insert_file import DataLoader 
from router.models import (
    Conversation,
    Document, 
    GuestUser, 
    VectorStore, 
    DocumentVector
)

from evaluation.models import (
    Chunk,
    GroundTruthChunk,
    GroundTruthResponse
)
from evaluation.eval import evaluate_chunks, evaluate_response
logger = logging.getLogger(__name__)

class SparseRAGPipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.rag = SparseRAG(config)

        self.llm = self._initialize_llm(config.get("llm_model", "openai"))

        self.chunker = DocumentChunker(
            strategy=config.get("chunk_strategy", "paragraph"),
            chunk_size=config.get("chunk_size", 500),
            overlap=config.get("overlap", 50),
            embedding_client=None
        )

        self.loader = DataLoader()

        self.vector_store_root = config.get("vector_store_path", "./vector_stores")
        os.makedirs(self.vector_store_root, exist_ok=True)

    def _save_state(self, path: str):
        data = {
            "documents": self.rag.documents,
            "bm25": self.rag.bm25,
            "metadata": self.rag.document_metadata
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _load_state(self, path: str) -> bool:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.rag.documents = data.get("documents", [])
            self.rag.bm25 = data.get("bm25", None)
            self.rag.document_metadata = data.get("metadata", [])
            return True
        except Exception as e:
            logger.error(f"Error loading state from {path}: {e}")
            return False
        
    def _build_index(self, username: str, document) -> str:
        """
        Internal function that performs Sparse indexing.
        Returns the saved vectorstore path.
        """

        logger.info("Creating new index (Sparse)...")

        if not document.extracted_text_path:
            raise ValueError("Document has no text source path.")

        logger.info(f"Loading text from {document.extracted_text_path}")

        raw_text = self.loader.load(document.extracted_text_path)
        chunks = self.chunker.chunk(raw_text)
        
        chunks_with_ids = self._sync_chunks(document, chunks)
        
        self.rag.index_documents(chunks_with_ids)

        file_name = f"{username}_{document.pk}_sparse_{uuid.uuid4().hex[:6]}.pkl"
        save_path = os.path.join(self.vector_store_root, file_name)

        self._save_state(save_path)

        vs, _ = VectorStore.objects.get_or_create(base_path=self.vector_store_root)

        DocumentVector.objects.create(
            document=document,
            vectorstore=vs,
            vectorstore_location=save_path,
            document_location=document.extracted_text_path,
            status="ready",
            method="sparse"
        )

        logger.info("Sparse index creation complete.")

        return save_path

    def init(self, username: str) -> bool:
        """
        Initializes Sparse RAG synchronously.
        """

        logger.info(f"Initializing Sparse RAG for {username}...")

        document = self.get_document(username)
        if not document:
            raise ValueError(f"No document found for user: {username}")

        doc_vector = DocumentVector.objects.filter(
            document=document,
            status="ready",
            method="sparse"
        ).last()

        if doc_vector:
            logger.info("Existing index found. Loading into memory.")

            success = self._load_state(doc_vector.vectorstore_location)
            if not success:
                raise RuntimeError("Index record exists but file load failed.")

            return True

        self._build_index(username, document)

        logger.info("Initialization Complete.")
        return True

    def _run_core(self, document: Document, query: str) -> Dict[str, Any]:
        """
        Shared core logic for run() and run_analysis().
        Handles init guard, retrieval, and LLM generation.
        """
        # Init guard — sparse checks rag.documents directly
        if not self.rag.documents or len(self.rag.documents) == 0:
            logger.warning("No documents found in memory. Initializing...")

            doc_vector = DocumentVector.objects.filter(
                document=document,
                status="ready",
                method="sparse"
            ).last()

            if not doc_vector:
                raise ValueError("No ready index found for this document.")

            success = self._load_state(doc_vector.vectorstore_location)
            if not success:
                raise RuntimeError("Index record exists but file load failed.")

            if not self.rag.documents or len(self.rag.documents) == 0:
                raise RuntimeError("State loaded from disk, but memory is still empty. The .pkl file might be corrupt or empty.")

        # Retrieval
        optimized_query = self.optimize_query(query)

        retrieved_docs = self.rag.retrieve(optimized_query)

        if not retrieved_docs:
            retrieved_docs = self.rag.retrieve(query)

        # No results fallback
        if not retrieved_docs:
            logger.warning(f"No relevant documents found for query: {query}")
            answer = self.llm.rag_generate(query, context="")
            return {
                "answer": answer,
                "context": [],
                "chunk_ids": [],
                "retrieved_docs": [],
            }

        # LLM generation
        context_str = "\n\n".join(doc["text"] for doc in retrieved_docs)
        answer = self.llm.rag_generate(optimized_query, context_str)

        return {
            "answer": answer,
            "context": [
                {
                    "text": doc["text"],
                    "chunk_id": doc["chunk_id"],
                    "score": doc.get("score"),
                }
                for doc in retrieved_docs
            ],
            "chunk_ids": [doc["chunk_id"] for doc in retrieved_docs],
            "retrieved_docs": retrieved_docs,  # internal handoff for run_analysis
        }


    def run(self, username: str, query: str) -> Dict[str, Any]:
        """
        Retrieves relevant documents and generates an answer using Sparse RAG.
        """
        logger.info(f"Running Sparse RAG for {username}...")

        document = self.get_document(username)
        if not document:
            raise ValueError(f"No document found for user: {username}")

        result = self._run_core(document, query)
        result.pop("retrieved_docs", None)
        return result


    def run_analysis(self, document_id: str, conversation_id: str) -> Dict[str, Any]:
        """
        Same as run() but also evaluates retrieved chunks and answer against ground truth.
        """
        logger.info(f"Running Sparse Analysis for conversation {conversation_id}...")

        document = Document.objects.get(id=document_id)
        conversation = Conversation.objects.get(id=conversation_id)

        result = self._run_core(document, conversation.query)

        retrieved_docs = result.pop("retrieved_docs", [])

        if not retrieved_docs:
            result["evaluation"] = {}
            return result

        retrieved_ids = set(result["chunk_ids"])

        ground_truth_qs = GroundTruthChunk.objects.filter(conversation=conversation)

        if not ground_truth_qs.exists():
            logger.warning(f"No ground truth chunks for conversation {conversation_id}")

        ground_truth_ids = set(
            ground_truth_qs.values_list("chunk_id", flat=True)
        )
        
        evaluation_chunks_results = evaluate_chunks(retrieved_ids, ground_truth_ids)

        ground_truth_response = GroundTruthResponse.objects.filter(conversation=conversation).first()
        if ground_truth_response:
            evaluation_response_result = evaluate_response(result["answer"], ground_truth_response.response)
        else:
            logger.warning(f"No ground truth response for conversation {conversation_id}")

        result["evaluation"] = {
            "chunk_evaluation": evaluation_chunks_results,
            "response_evaluation": evaluation_response_result if ground_truth_response else {}
        }
        return result
    
    def init_job(self, username: str, job=None) -> bool:
        """
        Initializes Sparse RAG with job progress tracking.
        """

        logger.info(f"Initializing Sparse RAG for {username}...")

        if job:
            job.progress = 10
            job.save()

        document = self.get_document(username)
        if not document:
            raise ValueError(f"No document found for user: {username}")

        doc_vector = DocumentVector.objects.filter(
            document=document,
            status="ready",
            method="sparse"
        ).last()

        if doc_vector:
            logger.info("Existing index found. Loading into memory.")

            success = self._load_state(doc_vector.vectorstore_location)
            if not success:
                raise RuntimeError("Index record exists but file load failed.")

            if job:
                job.progress = 80
                job.save()

            return True

        if job:
            job.progress = 20
            job.save()

        self._build_index(username, document)

        if job:
            job.progress = 90
            job.save()

        logger.info("Initialization Complete.")
        return True