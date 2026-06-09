import os
import pickle
import logging
import uuid
from typing import Dict, Any, List

from pipeline.base_pipeline import BasePipeline

from common.chunker import DocumentChunker
from hybrid_rag.hybrid_rag import HybridRAG  
from utils.insert_file import DataLoader

from router.models import (
    Conversation,
    Document, 
    GuestUser, 
    VectorStore, 
    DocumentVector
)

from evaluation.models import Chunk, GroundTruthChunk, GroundTruthResponse
from evaluation.eval import evaluate_chunks, evaluate_response

logger = logging.getLogger(__name__)

class HybridRAGPipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.rag = HybridRAG(config)
        
        self.llm = self._initialize_llm(config.get("llm_model", "openai"))
        embedding_client = getattr(self.rag.dense_engine, 'client', None)
        
        self.chunker = DocumentChunker(
            strategy=config.get("chunk_strategy", "paragraph"),
            chunk_size=config.get("chunk_size", 500),
            overlap=config.get("overlap", 50),
            embedding_client=embedding_client
        )
        self.loader = DataLoader()

        self.vector_store_root = config.get("vector_store_path", "./vector_stores")
        os.makedirs(self.vector_store_root, exist_ok=True)

    def _save_state(self, path: str):
        """
        Saves the state of both the Sparse and Dense engines.
        """
        sparse_docs = list(getattr(self.rag.sparse_engine, "documents", []))
        dense_docs = list(getattr(self.rag.dense_engine, "documents", []))
        
        if not sparse_docs or not dense_docs:
            raise RuntimeError(
                f"Cannot save empty state — "
                f"sparse: {len(sparse_docs)} docs, dense: {len(dense_docs)} docs"
            )

        data = {
            "sparse": {
                "documents": sparse_docs,
                "bm25": getattr(self.rag.sparse_engine, "bm25", None),
                "tokenized_corpus": getattr(self.rag.sparse_engine, "tokenized_corpus", []),
                "metadata": getattr(self.rag.sparse_engine, "document_metadata", [])
            },
            "dense": {
                "documents": dense_docs,
                "vectors": list(getattr(self.rag.dense_engine, "document_vectors", [])),
                "metadata": getattr(self.rag.dense_engine, "document_metadata", [])
            }
        }

        # Verify BM25 was actually built
        if data["sparse"]["bm25"] is None:
            raise RuntimeError("BM25 index is None — sparse engine did not index correctly.")
        
        if not data["dense"]["vectors"]:
            raise RuntimeError("Dense vectors are empty — dense engine did not index correctly.")

        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"State saved: {len(sparse_docs)} sparse docs, {len(dense_docs)} dense docs")
        except Exception as e:
            logger.error(f"Error saving pickle state: {e}")
            raise


    def _load_state(self, path: str) -> bool:
        """
        Restores the state of both engines from disk.
        """
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            if "sparse" in data:
                self.rag.sparse_engine.documents = data["sparse"].get("documents") or []
                self.rag.sparse_engine.tokenized_corpus = data["sparse"].get("tokenized_corpus") or []
                self.rag.sparse_engine.document_metadata = data["sparse"].get("metadata") or []
                bm25 = data["sparse"].get("bm25")
                if bm25 is not None:
                    self.rag.sparse_engine.bm25 = bm25
                else:
                    logger.warning("BM25 not in pickle, rebuilding from tokenized_corpus...")
                    corpus = getattr(self.rag.sparse_engine, "tokenized_corpus", [])
                    if corpus:
                        from rank_bm25 import BM25Okapi
                        self.rag.sparse_engine.bm25 = BM25Okapi(corpus)
                    else:
                        logger.error("Cannot rebuild BM25 — tokenized_corpus is also empty.")
                        return False

            if "dense" in data:
                # Use `or []` to safely handle None values from old corrupt pickles
                self.rag.dense_engine.documents = data["dense"].get("documents") or []
                self.rag.dense_engine.document_vectors = data["dense"].get("vectors") or []
                self.rag.dense_engine.document_metadata = data["dense"].get("metadata") or []

            sparse_ok = len(getattr(self.rag.sparse_engine, "documents", []) or []) > 0
            dense_ok = len(getattr(self.rag.dense_engine, "documents", []) or []) > 0
            bm25_ok = getattr(self.rag.sparse_engine, "bm25", None) is not None
            vectors_ok = len(getattr(self.rag.dense_engine, "document_vectors", []) or []) > 0

            if not all([sparse_ok, dense_ok, bm25_ok, vectors_ok]):
                logger.error(
                    f"State load incomplete — "
                    f"sparse_docs: {sparse_ok}, dense_docs: {dense_ok}, "
                    f"bm25: {bm25_ok}, vectors: {vectors_ok}"
                )
                return False

            logger.info(
                f"State loaded — "
                f"{len(self.rag.sparse_engine.documents)} sparse docs, "
                f"{len(self.rag.dense_engine.documents)} dense docs"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading state from {path}: {e}")
            return False

    def _build_index(self, username: str, document) -> str:
        """
        Internal function that builds the Hybrid index.
        Returns the saved vectorstore path.
        """

        logger.info("Creating new Hybrid index...")

        if not document.extracted_text_path:
            raise ValueError("Document has no text source path.")

        logger.info(f"Loading text from {document.extracted_text_path}")

        raw_text = self.loader.load(document.extracted_text_path)

        chunks = self.chunker.chunk(raw_text)

        chunks_with_ids = self._sync_chunks(document, chunks)
        
        self.rag.index_documents(chunks_with_ids)

        file_name = f"{username}_{document.pk}_hybrid_{uuid.uuid4().hex[:6]}.pkl"
        save_path = os.path.join(self.vector_store_root, file_name)

        self._save_state(save_path)

        vs, _ = VectorStore.objects.get_or_create(base_path=self.vector_store_root)

        DocumentVector.objects.create(
            document=document,
            vectorstore=vs,
            vectorstore_location=save_path,
            document_location=document.extracted_text_path,
            status="ready",
            method="hybrid"
        )

        logger.info("Hybrid index creation complete.")

        return save_path
    
    def init(self, username: str) -> bool:
        """
        Initializes Hybrid RAG synchronously.
        """

        logger.info(f"Initializing Hybrid RAG for {username}...")

        document = self.get_document(username)
        if not document:
            raise ValueError(f"No document found for user: {username}")

        doc_vector = DocumentVector.objects.filter(
            document=document,
            status="ready",
            method="hybrid"
        ).last()

        if doc_vector:
            logger.info("Existing index found. Loading into memory.")
            logger.info(f"Loading state from {doc_vector.vectorstore_location}")

            success = self._load_state(doc_vector.vectorstore_location)
            if success:
                return True

            logger.warning("Corrupt or outdated index found. Deleting and re-indexing...")
            try:
                if os.path.exists(doc_vector.vectorstore_location):
                    os.remove(doc_vector.vectorstore_location)
                doc_vector.delete()
            except Exception as e:
                logger.error(f"Failed to clean up bad index: {e}")

        self._build_index(username, document)

        logger.info("Hybrid Initialization Complete.")
        return True
    
    def _run_core(self, document: Document, query: str) -> Dict[str, Any]:
        """
        Handles init guard, retrieval, and LLM generation.
        """
        if not self.rag.dense_engine.documents or len(self.rag.dense_engine.documents) == 0:
            logger.warning("No documents found in memory. Initializing...")

            success = self.init(document.user.username)
            if not success:
                doc_vector = DocumentVector.objects.filter(
                    document=document,
                    status="ready",
                    method="hybrid"
                ).first()
                if doc_vector:
                    self._load_state(doc_vector.vectorstore_location)
                else:
                    raise ValueError("Initialization failed.")

            if not self.rag.dense_engine.documents or len(self.rag.dense_engine.documents) == 0:
                raise RuntimeError("State loaded from disk, but memory is still empty.")

        optimized_query = self.optimize_query(query)
        logger.info(f"Optimized Query: {optimized_query}")

        retrieved_docs = self.rag.retrieve(optimized_query)

        if not retrieved_docs:
            retrieved_docs = self.rag.retrieve(query)

        if not retrieved_docs:
            logger.warning(f"No relevant documents found for query: {query}")
            answer = self.llm.rag_generate(query, context="")
            return {
                "answer": answer,
                "context": [],
                "chunk_ids": [],
                "retrieved_docs": [],
            }

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
            "retrieved_docs": retrieved_docs, 
        }


    def run(self, username: str, query: str) -> Dict[str, Any]:
        """
        Retrieves relevant documents and generates an answer using Hybrid RAG.
        """
        logger.info(f"Running Hybrid Chat for {username}...")

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
        logger.info(f"Running Hybrid Analysis for conversation {conversation_id}...")

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
        Initializes Hybrid RAG with job progress tracking.
        """

        logger.info(f"Initializing Hybrid RAG for {username}...")

        if job:
            job.progress = 10
            job.save()

        document = self.get_document(username)
        if not document:
            raise ValueError(f"No document found for user: {username}")

        doc_vector = DocumentVector.objects.filter(
            document=document,
            status="ready",
            method="hybrid"
        ).last()

        if doc_vector:
            logger.info("Existing index found. Loading into memory.")

            if job:
                job.progress = 80
                job.save()

            logger.info(f"Loading state from {doc_vector.vectorstore_location}")

            success = self._load_state(doc_vector.vectorstore_location)
            if success:
                return True

            logger.warning("Corrupt or outdated index found. Deleting and re-indexing...")
            try:
                if os.path.exists(doc_vector.vectorstore_location):
                    os.remove(doc_vector.vectorstore_location)
                doc_vector.delete()
            except Exception as e:
                logger.error(f"Failed to clean up bad index: {e}")

        if job:
            job.progress = 20
            job.save()

        self._build_index(username, document)

        if job:
            job.progress = 90
            job.save()

        logger.info("Hybrid Initialization Complete.")
        return True