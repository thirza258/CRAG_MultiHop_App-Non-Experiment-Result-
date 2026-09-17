import uuid

import chromadb
from openai import OpenAI
from django.conf import settings
from router.models import Document, UserCollection, DocumentChunk
from math import ceil
from typing import List, Dict, Any
from common.chunker import DocumentChunker


import logging

logger = logging.getLogger(__name__)

CHROMA_DIR = "./chroma_db"

def get_chroma_client(collection_name: str):
    client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
    # get_or_create: the app must not crash just because a collection has
    # not been populated yet (e.g. base dataset indexing disabled/failed).
    return client.get_or_create_collection(name=collection_name, embedding_function=None)

def get_client():
    return chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)

def create_chroma_collection(collection_name: str):
    client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
    try:
        collection = client.create_collection(name=collection_name, embedding_function=None)
        logger.info(f"Created ChromaDB collection: {collection_name}")
        return collection
    except chromadb.errors.CollectionAlreadyExistsError:
        logger.warning(f"Collection already exists: {collection_name}, retrieving existing collection.")
        return client.get_collection(name=collection_name, embedding_function=None)

def insert_chunk_to_chromadb(
    collection_name: str,
    chunks: List[str],
    metadata: List[Dict[str, str]],
    embeddings: List[List[float]] = None,
    batch_size: int = 100,
    ids: List[str] = None,
) -> bool:
    
    try:
        if not chunks or len(metadata) != len(chunks):
            raise ValueError("Chunks and metadata must be nonempty and aligned.")
        if ids is not None and len(ids) != len(chunks):
            raise ValueError("Chunk IDs must match the chunks.")
        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError("Embeddings must match the chunks.")
        collection = get_client().get_or_create_collection(
            name=collection_name,
            embedding_function=None
        )

        total_batches = ceil(len(chunks) / batch_size)
        failed_batches = []

        for i in range(0, len(chunks), batch_size):
            batch_num = i // batch_size + 1

            batch_chunks = chunks[i:i + batch_size]
            batch_metadatas = metadata[i:i + batch_size]
            batch_ids = ids[i:i + batch_size] if ids is not None else [str(uuid.uuid4()) for _ in batch_chunks]

            add_kwargs = {
                "ids": batch_ids,
                "documents": batch_chunks,
                "metadatas": batch_metadatas,
            }

            if embeddings is not None:
                add_kwargs["embeddings"] = embeddings[i:i + batch_size]

            try:
                collection.upsert(**add_kwargs)

                print(f"[OK] Inserted batch {batch_num}/{total_batches}")

            except Exception as batch_error:
                print(
                    f"[ERROR] Failed batch {batch_num}/{total_batches}: "
                    f"{batch_error}"
                )

                failed_batches.append({
                    "batch": batch_num,
                    "start_index": i,
                    "error": str(batch_error)
                })

                continue

        print(f"\nFinished insertion.")
        print(f"Failed batches: {len(failed_batches)}")

        return len(failed_batches) == 0

    except Exception as e:
        print(f"[FATAL] Error initializing ChromaDB insertion: {e}")
        return False
