from typing import List, Dict
import uuid
import numpy as np
import pandas as pd
from datasets import Dataset
from chroma.chroma_settings import get_chroma_client, insert_chunk_to_chromadb
from common.chunker import DocumentChunker

chunker = DocumentChunker("recursive", chunk_size=1000, overlap=200)

def extract_ground_truth_metadata(evidence_list: List[Dict]) -> List[Dict]:
    """Extract only the fields that exist in ChromaDB metadata."""
    return [
        {
            "title": e["title"],
            "author": e["author"],
            "source": e["source"],
            "url": e["url"],
            "category": e["category"],
        }
        for e in evidence_list
    ]
    
def extract_fact_from_query(evidence_list):
    """Extract facts from the evidence list in numbered format."""
    facts = []

    for idx, e in enumerate(evidence_list, start=1):
        fact = e.get("fact", "").strip()
        if fact:
            facts.append(
                f"Fact {idx}: {fact}\n"
                f"title: {e.get('title', 'N/A')}, author: {e.get('author', 'N/A')}, "
                f"source: {e.get('source', 'N/A')}, category: {e.get('category', 'N/A')}"
            )

    return "\n".join(facts)

def concatenate_chunks_metadata(chunks: List[str], metadatas: List[Dict]) -> List[Dict]:
    """Concatenate chunk text with its metadata for better LLM evaluation."""
    concatenated = []
    for chunk, meta in zip(chunks, metadatas):
        meta_copy = meta.copy()
        meta_copy["document"] = chunk
        concatenated.append(meta_copy)
    return concatenated

def convert_data_response_and_dataset_to_dataset(query, retrieved_chunks, generated_response):
    try:
        return Dataset.from_dict({
            "question":    [query],
            "contexts":    [retrieved_chunks if isinstance(retrieved_chunks, list) else [retrieved_chunks]],
            "answer":      [generated_response],
        })
    except Exception as e:
        print(f"Error in convert_data_response_and_dataset_to_dataset: {e}")
        return Dataset.from_dict({
            "question":    [],
            "contexts":    [],
            "answer":      [],
        })
        
def convert_corpus_to_metadata(title: str, author: str, source: str, url: str, category: str, published_at: str) -> Dict[str, str]:
    def sanitize(value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        return str(value)
    
    return {
        "published_at": sanitize(published_at),
        "title":    sanitize(title),
        "author":   sanitize(author),
        "source":   sanitize(source),
        "url":      sanitize(url),
        "category": sanitize(category),
    }


    
def prepare_corpus(corpus: pd.DataFrame):
    """Returns chunks and metadatas without inserting."""
    all_documents = []
    all_metadatas = []

    for _, row in corpus.iterrows():
        metadata = convert_corpus_to_metadata(
            title=row["title"],
            author=row["author"],
            source=row["source"],
            url=row["url"],
            category=row["category"],
            published_at=row["published_at"]
        )
        chunks = chunker.chunk(row["body"])

        for chunk_idx, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = chunk_idx
            enriched_chunk = (
                f"Title: {metadata['title']}\n"
                f"Author: {metadata['author']}\n"
                f"Source: {metadata['source']}\n"
                f"Category: {metadata['category']}\n\n"
                f"Text: {chunk}"
            )
            all_documents.append(enriched_chunk)
            all_metadatas.append(chunk_metadata)

    return all_documents, all_metadatas

def process_corpus_entry(corpus: pd.DataFrame, collection_name: str, embeddings=None) -> None:
    """Prepares and inserts into ChromaDB."""
    all_documents, all_metadatas = prepare_corpus(corpus)
    success = insert_chunk_to_chromadb(collection_name, all_documents, all_metadatas, embeddings)
    if not success:
        print(f"Failed to insert chunks into ChromaDB.")
        