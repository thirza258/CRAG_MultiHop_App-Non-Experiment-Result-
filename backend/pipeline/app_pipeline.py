import json
import math

import redis

from ai_handler.llm import OpenRouterLLM
from ai_handler.wrapper import (
    llm_langchain_wrapper,
    embeddings_langchain_wrapper
)
from evaluation.eval import ragas_llm_as_a_judge_generation_evaluation
from dense_rag.dense_rag import DenseRAG
from sparse_rag.sparse_rag import SparseRAG
from ragreader.settings import REDIS_HOST, REDIS_PORT

from corrective.corrective_rag import CorrectiveRAG
from multi_hop.multi_hop_rag import MultiHopRetriever
from hybrid_rag.hybrid_rag import HybridRAG
from common.chunker import DocumentChunker
from common.dataset_settings import convert_data_response_and_dataset_to_dataset
from utils.insert_file import DataLoader
import os
from router.models import (
    Document,
    DocumentVector,
    UserCollection,
    DocumentChunk,
)
from chroma.chroma_settings import get_chroma_client, insert_chunk_to_chromadb
from typing import Dict, Any
from emitter.status import StatusEmitter, NULL_EMITTER


import logging

logger = logging.getLogger(__name__)

class AppRAGPipeline():
    def __init__(self, config):
        self.config = config

        self.dense_rag  = DenseRAG(config["dense_config"])
        self.sparse_rag = SparseRAG(config["sparse_config"])

        # CorrectiveRAG wraps each base retriever
        self.dense_corrective_rag  = CorrectiveRAG(self.dense_rag,  config["crag_config"])
        self.sparse_corrective_rag = CorrectiveRAG(self.sparse_rag, config["crag_config"])

        # MultiHop wraps CorrectiveRAG — this is your final fallback
        self.dense_multi_hop  = MultiHopRetriever(self.dense_corrective_rag,  config["multi_hop_config"])
        self.sparse_multi_hop = MultiHopRetriever(self.sparse_corrective_rag, config["multi_hop_config"])
        
        self.hybrid_rag = HybridRAG(config["hybrid_config"])
        
        self.llm_client = OpenRouterLLM(model=config["llm_model"])

        # Shared dataset collection
        self.dataset_collection = get_chroma_client(
            collection_name=config.get("collection_name", "dataset_collection")
        )
        
        self.chunker = DocumentChunker()
        self.loader = DataLoader()
    
    def _save_state(self, path: str):
        """
        Saves the state of both the Sparse and Dense engines.
        """
        pass

    def _load_state(self, path: str) -> bool:
        """
        Restores the state of both engines from disk.
        """
        pass

    def _build_index(self, username: str, document: Document):
        user_collection, _ = UserCollection.objects.get_or_create(
            user=document.user,
            defaults={"collection_name": f"user_{username}_collection"}
        )

        raw_text = self.loader.load(document.extracted_text_path)
        chunks = self.chunker.chunk(raw_text)

        ids, texts, metadatas = [], [], []
        chunk_records = []

        for i, chunk in enumerate(chunks):
            chroma_id = f"{username}_{document.pk}_chunk_{i}"

            ids.append(chroma_id)
            texts.append(chunk)
            metadatas.append({
                "document_id": document.pk,      # <-- key: tag for deletion later
                "username": username,
                "chunk_index": i,
            })

            chunk_records.append(DocumentChunk(
                document=document,
                user_collection=user_collection,
                chroma_id=chroma_id,
                chunk_index=i,
            ))
            
        embeddings = self.dense_rag._get_embeddings(texts, min(len(texts), 50))

        # Insert into ChromaDB
        insert_chunk_to_chromadb(
            collection_name=user_collection.collection_name,
            chunks=texts,
            metadata=metadatas,
            embeddings=embeddings,
            batch_size=100
        )

        # Insert into Django DB
        DocumentChunk.objects.bulk_create(chunk_records)

        # Update chunk count
        user_collection.chunk_count += len(chunks)
        user_collection.save()
           
    def _resolve_collection(self, username: str) -> tuple[str, str]:
        """
        Returns (collection_name, source) based on user collection count.
        """
        try:
            user_col_record = UserCollection.objects.get(user__username=username)
            user_collection = get_chroma_client(
                collection_name=user_col_record.collection_name
            )

            if user_collection.count() > 0:
                logger.info(f"[Pipeline] Using user_collection: {user_col_record.collection_name}")
                return user_col_record.collection_name, "user_collection"

        except UserCollection.DoesNotExist:
            logger.info(f"[Pipeline] No UserCollection record found for {username}")

        dataset_col_name = self.config.get("collection_name", "dataset_collection")
        logger.info(f"[Pipeline] Falling back to dataset_collection: {dataset_col_name}")
        return dataset_col_name, "dataset_collection"


    def _run_core(self, query: str, username: str, conversation_id: int) -> dict:
        # Resolve which collection to use
        try:
            emitter = self._build_emitter(conversation_id)
            
            collection_name, source = self._resolve_collection(username)

            self.dense_rag.set_collection(collection_name)
            self.sparse_rag.set_collection(collection_name)

            logger.info(f"[Pipeline] Both RAG engines pointed at: {collection_name}")
            
            self.dense_multi_hop.set_emitter(emitter)
            self.sparse_multi_hop.set_emitter(emitter)
            self.hybrid_rag.set_emitter(emitter)

            retriever_dense, metadata_dense   = self.dense_multi_hop.retrieve(query)
            retriever_sparse, metadata_sparse = self.sparse_multi_hop.retrieve(query)
            
            logger.info(f"[Pipeline] Dense retrieved {len(retriever_dense)} chunks; Sparse retrieved {len(retriever_sparse)} chunks")
            
            chunks, reranked_metas, status = self.hybrid_rag.retrieve_from_precomputed(
                query=query,
                dense_chunks=retriever_dense,
                sparse_chunks=retriever_sparse,
                dense_metas=metadata_dense,
                sparse_metas=metadata_sparse,
            )
           
            if status != "ok":
                logger.warning(f"[Pipeline] HybridRAG retrieval status: {status}")

            answer = self._generate_answer(query, chunks)
            emitter.emit("answer_generation", "Answer generated by LLM")
            
            emitter.emit("evaluation_start", "Starting evaluation of generated answer and retrieved chunks")
            answer_relevancy, faithfulness = self.evaluate(query, chunks, answer)
            logger.info(f"[Pipeline] Evaluation results - Answer Relevancy: {answer_relevancy}, Faithfulness: {faithfulness}")

            return {
                "answer": answer,
                "source": source,
                "context": [
                    {
                        "text": chunk,
                        "metadata": meta,
                    }
                    for chunk, meta in zip(chunks, reranked_metas)
                ],
                "evaluation": {
                    "answer_relevancy": answer_relevancy,
                    "faithfulness": faithfulness,
                }
            }
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}", exc_info=True)
            return {
                "answer": "Sorry, something went wrong while processing your request.",
                "source": None,
                "context": [],
            }
        
    def _generate_answer(self, query: str, chunks: list) -> str:
        """
        Feeds the retrieved chunks into the LLM to generate an answer.
        Handles chunks as either plain strings or dicts with a text field.
        """
        normalized = []
        for chunk in chunks:
            if isinstance(chunk, str):
                normalized.append(chunk)
            elif isinstance(chunk, dict):
                # Adjust key to match whatever your chunk dicts actually use
                text = chunk.get("text") or chunk.get("content") or chunk.get("page_content") or ""
                normalized.append(text)
            else:
                logger.warning(f"[Pipeline] Unexpected chunk type: {type(chunk)} — skipping")

        combined_text = "\n\n".join(normalized)

        prompt = (
            f"Answer the question based on the following retrieved context.\n\n"
            f"Context:\n{combined_text}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        return self.llm_client._call_api(prompt)
            
        
    def run_analysis(self, document_id: str, conversation_id: str) -> Dict[str, Any]:
        """
        Same as run() but also evaluates retrieved chunks and answer against ground truth.
        """
        pass
    
    def run(self, username: str, query: str, conversation_id: int) -> Dict[str, Any]:
        """
        Main entry point for running the RAG pipeline.
        """
        logger.info(f"Running RAG pipeline for user: {username} with query: {query}")

        return self._run_core(query, username, conversation_id)
    
    def evaluate(self, query: str, retrieved_chunks: list, generated_response: str) -> dict:
        """
        Evaluates the generated answer and retrieved chunks against the ground truth.
        """
        try:
            converted_dataset = convert_data_response_and_dataset_to_dataset(
                query=query,
                retrieved_chunks=retrieved_chunks,
                generated_response=generated_response
            )
            evaluation_result = ragas_llm_as_a_judge_generation_evaluation(
                dataset=converted_dataset,
                llm_judge=llm_langchain_wrapper(self.config["evaluation_llm_model"]),
                judge_embeddings=embeddings_langchain_wrapper(self.config["evaluation_embedding_model"])
            )
            answer_relevancy = evaluation_result["answer_relevancy"].iloc[0] if not evaluation_result.empty else None
            faithfulness = evaluation_result["faithfulness"].iloc[0] if not evaluation_result.empty else None
            return answer_relevancy, faithfulness
        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
            return None, None
    
    def _build_emitter(self, conversation_id):
        if not conversation_id:
            return NULL_EMITTER
        r = redis.Redis(
            host=REDIS_HOST,   
            port=REDIS_PORT,   
            db=0
        )
        def push(event):
            r.publish(f"rag:status:{conversation_id}", json.dumps(event))
        return StatusEmitter(callback=push)
        
    