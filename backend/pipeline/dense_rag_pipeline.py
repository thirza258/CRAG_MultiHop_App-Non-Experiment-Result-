from importlib.resources import path
from os import path
import os
import pickle
import logging
import uuid
from typing import Dict, Any 

from pipeline.base_pipeline import BasePipeline
from common.chunker import DocumentChunker
from dense_rag.dense_rag import DenseRAG
from utils.insert_file import DataLoader

from router.models import (
    Conversation,
    Document, 
    VectorStore, 
    DocumentVector
)
from evaluation.models import (
    Chunk,
    GroundTruthChunk,
    GroundTruthResponse
)

from evaluation.eval import evaluate_chunks, evaluate_response
from django.conf import settings
import os

logger = logging.getLogger(__name__)

class DenseRAGPipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.rag = DenseRAG(config)
        self.llm = self._initialize_llm(config.get("llm_model", "openai"))
        self.chunker = DocumentChunker(
            strategy=config.get("chunk_strategy", "paragraph"),
            chunk_size=config.get("chunk_size", 500),
            overlap=config.get("overlap", 50),
            embedding_client=self.rag.client 
        )
        self.loader = DataLoader()

        self.vector_store_root = config.get("vector_store_path", "./vector_stores")
        os.makedirs(self.vector_store_root, exist_ok=True)

    def _save_state(self, path: str):
        data = {
            "documents": self.rag.documents,
            "vectors": self.rag.document_vectors,
            "metadata": self.rag.document_metadata
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _load_state(self, path: str) -> bool:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            self.rag.documents = data.get('documents', [])
            self.rag.document_vectors = data.get('vectors', [])
            self.rag.document_metadata = data.get("metadata", []) 
            return True
        except Exception as e:
            logger.error(f"Error loading state from {path}: {e}")
            return False
        
    def _build_index(self, username: str, document: Document) -> str:
        """
        Internal function that performs the heavy indexing work.
        Returns the path of the saved vector store.
        """

        logger.info("Creating new index (Embedding)...")

        if not document.extracted_text_path:
            raise ValueError("Document has no text source path.")

        logger.info(f"Loading text from {document.extracted_text_path}")
        raw_text = self.loader.load(document.extracted_text_path)

        chunks = self.chunker.chunk(raw_text)
        
        chunks_with_ids = self._sync_chunks(document, chunks)
        
        self.rag.index_documents(chunks_with_ids)

        file_name = f"{username}_{document.pk}_dense_{uuid.uuid4().hex[:6]}.pkl"
        save_path = os.path.join(self.vector_store_root, file_name)

        self._save_state(save_path)

        vs, _ = VectorStore.objects.get_or_create(base_path=self.vector_store_root)

        DocumentVector.objects.create(
            document=document,
            vectorstore=vs,
            vectorstore_location=save_path,
            document_location=document.extracted_text_path,
            status="ready",
            method="dense"
        )

        logger.info("Index creation complete.")

        return save_path

    def init(self, username: str) -> bool:
        """
        Prepares the vector store for the user.
        1. Checks DB for existing ready index.
        2. If missing, loads text, chunks, embeds, and saves to disk.
        3. Loads data into memory for this instance.
        """

        logger.info(f"Initializing Chat for {username}...")

        document = self.get_document(username)
        if not document:
            raise ValueError(f"No document found for user: {username}")

        doc_vector = DocumentVector.objects.filter(
            document=document,
            status="ready",
            method="dense"
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

    def _save_chunks(self, chunks: list) -> str:
        """
        Save chunked text into the Django project root folder.
        """

        content = []
        print("Saving chunks...")

        for i, chunk in enumerate(chunks, 1):
            content.append(f"===== CHUNK {i} =====\n")
            content.append(chunk.strip())
            content.append("\n\n")

        final_text = "".join(content)

        path = os.path.join(settings.BASE_DIR, "chunks.txt")

        with open(path, "w", encoding="utf-8") as f:
            f.write(final_text)
            
        print("CWD:", os.getcwd())
        print("Saving to:", path)

        return path
    
    def _run_core(self, document: Document, query: str) -> Dict[str, Any]:
        """
        Handles init guard, retrieval, and LLM generation.
        """
        if not self.rag.documents or len(self.rag.documents) == 0:
            logger.warning("No documents found in memory. Initializing...")

            doc_vector = DocumentVector.objects.filter(
                document=document,
                status="ready",
                method="dense"
            ).last()

            if not doc_vector:
                raise ValueError("No ready index found for this document.")

            success = self._load_state(doc_vector.vectorstore_location)
            if not success:
                raise RuntimeError("Index record exists but file load failed.")

            if not self.rag.documents or len(self.rag.documents) == 0:
                raise RuntimeError("State loaded from disk, but memory is still empty.")

        optimized_query = self.optimize_query(query)
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
        Retrieves relevant documents and generates an answer.
        """
        logger.info(f"Running Chat for {username}...")

        document = self.get_document(username)
        if not document:
            raise ValueError(f"No document found for user: {username}")

        result = self._run_core(document, query)

        result.pop("retrieved_docs", None)
        return result


    def run_analysis(self, document_id: str, conversation_id: str) -> Dict[str, Any]:
        """
        evaluates retrieved chunks and answer against ground truth.
        """
        try:
            logger.info(f"Running Analysis for conversation {conversation_id}...")

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
        except Exception as e:
            logger.error(f"Error in run_analysis: {e}")
            return {"error": str(e)}

    
    def init_job(self, username: str, job=None) -> bool:
        """
        job: Instance of Job model with progess fro websocket updates
        """
        logger.info(f"Initializing Dense RAG for {username}...")

        if job:
            job.progress = 10
            job.save()

        document = self.get_document(username)
        if not document:
            raise ValueError(f"No document found for user: {username}")

        doc_vector = DocumentVector.objects.filter(
            document=document,
            status="ready",
            method="dense"
        ).last()

        if doc_vector:
            logger.info("Existing index found. Loading into memory.")

            if job:
                job.progress = 80
                job.save()

            success = self._load_state(doc_vector.vectorstore_location)

            if not success:
                raise RuntimeError("Index record exists but file load failed.")

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