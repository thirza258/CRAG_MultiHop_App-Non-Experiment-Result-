---
title: Corrective Multi-Hop RAG with Reranking | CRAG MultiHop RAG
description: Chat with your PDFs, URLs and notes. Dense + BM25 retrieval, self-grading Corrective RAG, up to 3 query hops, local reranking and faithfulness scores on every answer.
image: https://crag.nevatal.tech/og-image.png
---

# Corrective Multi-Hop RAG with Reranking

CRAG MultiHop RAG is an intelligent retrieval-augmented generation application designed to eliminate hallucinations in question answering through corrective self-grading, multi-hop query decomposition, hybrid retrieval, and local reranking.

## Core Features

- **Hybrid Retrieval**: Combines semantic Dense vector search with BM25 keyword matching for high recall and precision.
- **Self-Grading Corrective RAG**: Automatically evaluates retrieved document passages for relevance before generation, filtering out noisy context.
- **Multi-Hop Query Decomposition**: Breaks complex, multi-step queries into sub-questions (up to 3 hops) to gather dispersed evidence across documents.
- **Local Reranking**: Re-orders candidate passages using local cross-encoder models for maximal contextual alignment.
- **Faithfulness & Relevance Scoring**: Employs RAGAs metrics to assess response fidelity against source texts and ensure high answer quality.
- **Multi-Source Ingestion**: Ingests PDFs, web pages via URL extraction, and raw text notes.

## How to Use

1. **Upload Documents**: Provide PDFs, submit web URLs, or paste notes directly into the workspace.
2. **Indexing**: The system chunks and indexes content into dense vector embeddings and BM25 indices.
3. **Ask Queries**: Pose complex questions; the pipeline automatically performs query expansion and multi-hop retrieval as needed.
4. **Inspect Metrics**: View confidence scores, relevance grading, and source passage citations with every response.

```json
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "CRAG MultiHop RAG",
  "url": "https://crag.nevatal.tech/",
  "applicationCategory": "DeveloperApplication",
  "description": "Chat with your PDFs, URLs and notes. Dense + BM25 retrieval, self-grading Corrective RAG, up to 3 query hops, local reranking and faithfulness scores on every answer.",
  "offers": {
    "@type": "Offer",
    "price": "0",
    "priceCurrency": "USD"
  }
}
```
