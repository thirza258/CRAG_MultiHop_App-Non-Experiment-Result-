from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from rank_bm25 import BM25Okapi
import wikipedia
import numpy as np
import re
import unicodedata
import requests
from typing import List, Dict, Any
from .query_expander import QueryExpander
from ai_handler.openrouter import MissingAPIKeyError, openrouter_client
from common.runtime import api_keys as api_keys_module
from common.runtime.context import (
    news_api_key,
    request_secrets,
    resolve_embedding_model,
    resolve_param,
)
import logging

logger = logging.getLogger(__name__)

class ExternalSearcher:
    def __init__(self, config):
        wikipedia.set_user_agent("RAGReader/2.0 (thirzahmad@gmail.com)")

        self.chunk_size = config.get("external_chunk_size", 400)
        self.chunk_overlap = config.get("external_chunk_overlap", 100)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self._configured_top_k = config.get("external_top_k", 5)

        self.query_expander = QueryExpander(config)
        # Configured fallback; read through active_embedding_model so a request
        # that picked its own model is honoured.
        self.embedding_model = config.get('embedding_model', 'openai/text-embedding-3-small')

        self.method = config.get("method", "dense")

    @property
    def top_k(self) -> int:
        """Passages this request keeps from the web.

        Resolved per call for the same reason the embedding model is: one
        searcher instance serves every concurrent query.
        """
        return int(resolve_param(
            "external_top_k", getattr(self, "_configured_top_k", 5)
        ))

    @property
    def client(self) -> OpenAI:
        """Client for whichever OpenRouter key applies to this request.

        Built per call rather than in __init__ so that a deployment without a
        server key can still construct the pipeline — users bring their own.
        """
        return openrouter_client()

    @property
    def active_embedding_model(self) -> str:
        """Embedding model for this request.

        Unlike dense retrieval this only ever scores freshly fetched external
        text against the query in-memory — no stored vectors — so any model is
        dimensionally safe here.
        """
        return resolve_embedding_model(self.embedding_model)

    def _clean_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[[a-zA-Z ]{1,30}\]', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        lines = [line for line in text.splitlines() if len(line.strip()) > 10]
        text = '\n'.join(lines)
        
        return text.strip()
    def _chunk_on_the_fly(self, text: str, source_meta: Dict) -> tuple[List[str], List[Dict]]:
        text   = self._clean_text(text)   
        chunks = self.splitter.split_text(text)
        metas = [
            {
                **source_meta,
                "chunk_index": i,
                "chunk_total": len(chunks),
                "source_type": "external"
            }
            for i, _ in enumerate(chunks)
        ]

        return chunks, metas
    
    def _get_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]] | None:
        """
        Returns a flat list of embeddings in the same order as `texts`,
        or None if any batch fails.
        """
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        all_embeddings = []
        embedding_model = self.active_embedding_model

        try:
            client = self.client
        except MissingAPIKeyError as e:
            # None means "no relevance filter available"; the caller then keeps
            # all chunks rather than losing the external results entirely.
            logger.info(f"Cannot embed external chunks — {e}")
            return None

        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i : i + batch_size]
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=embedding_model
                )
                if not response.data or len(response.data) != len(batch):
                    logger.info(f"Warning: batch {i // batch_size} returned {len(response.data or [])} embeddings, expected {len(batch)} — aborting")
                    return None 
                
                batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.info(
                    f"Error fetching embeddings for batch {i // batch_size}: "
                    f"{api_keys_module.scrub(str(e), request_secrets())}"
                )
                return None 

        if len(all_embeddings) != len(texts):
            logger.info(f"Warning: expected {len(texts)} embeddings, got {len(all_embeddings)}")
            return None

        return all_embeddings

    def _filter_chunks_by_relevance(
        self,
        query: str,
        chunks: List[str],
        metas: List[Dict],
        top_k: int = None
    ) -> tuple[List[str], List[Dict]]:
        if not chunks:
            return [], []

        k = top_k or self.top_k

        if len(chunks) <= k:
            return chunks, metas

        if self.method == "BM25":
            try:                
                def tokenize(text):
                    return re.findall(r'\w+', text.lower())
                
                tokenized_chunks = [tokenize(chunk) for chunk in chunks]
                bm25 = BM25Okapi(tokenized_chunks)
                tokenized_query = tokenize(query)
                
                scores = bm25.get_scores(tokenized_query)
                top_idx = np.argsort(scores)[::-1][:k].tolist()
            except ImportError:
                logger.info("Warning: rank_bm25 package not installed. Falling back to TF-IDF.")
                self.method = "TF-IDF"  


        if self.method == "TF-IDF":
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            vectorizer = TfidfVectorizer()
            all_texts = [query] + chunks
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            query_vec = tfidf_matrix[0]
            chunk_vecs = tfidf_matrix[1:]
            
            scores = (chunk_vecs * query_vec.T).toarray().flatten()
            top_idx = np.argsort(scores)[::-1][:k].tolist()

        elif self.method not in ["BM25", "TF-IDF"]:
            all_texts = [query] + chunks
            all_embeddings = self._get_embeddings(all_texts, batch_size=100)

            if all_embeddings is None:
                logger.info("Warning: embedding failed — skipping relevance filter, returning all chunks")
                return chunks, metas

            query_emb  = np.array(all_embeddings[0])          
            chunk_embs = np.array(all_embeddings[1:])           

            scores  = chunk_embs @ query_emb                 
            top_idx = np.argsort(scores)[::-1][:k].tolist()

        # Retrieve mapped text/metas from the sorted indexes
        filtered_chunks = [chunks[i] for i in top_idx]
        filtered_metas  = [metas[i]  for i in top_idx]

        return filtered_chunks, filtered_metas

    def search_wikipedia(self, query: str) -> tuple[List[str], List[Dict]]:
        try:
            wikipedia_keywords = self.query_expander.retrieve_wikipedia_keyword_from_query(query)
            results = wikipedia.search(", ".join(wikipedia_keywords), results=3) 
            all_chunks, all_metas = [], []

            for title in results:
                try:
                    page = wikipedia.page(title)
                    chunks, metas = self._chunk_on_the_fly(
                        page.content[:3000],
                        {"url": page.url, "title": title,
                         "source": "wikipedia", "category": "external"}
                    )
                    all_chunks.extend(chunks)
                    all_metas.extend(metas)
                except wikipedia.exceptions.PageError:
                    continue
                except wikipedia.exceptions.DisambiguationError:
                    continue
                
            
            all_chunks, all_metas = self._filter_chunks_by_relevance(query, all_chunks, all_metas)
            logger.info(f"Wikipedia search retrieved {len(all_chunks)} chunks before relevance filtering.")
            return all_chunks, all_metas
        except Exception as e:
            logger.info(f"Wikipedia search error: {e}")
            return [], []

    def search_news(self, query: str) -> tuple[List[str], List[Dict]]:
        # The user's own NewsAPI key when they supplied one, else the server's.
        # Absent either, news is simply skipped: corrective retrieval still has
        # Wikipedia, so this is a narrower result, not a failure.
        api_key = news_api_key()
        if not api_key:
            logger.info(
                "No NewsAPI key available (none supplied with the request and "
                "NEWS_API_KEY is unset) — skipping the news search."
            )
            return [], []

        keywords = self.query_expander.retrieve_news_keyword_from_query(query)

        url = "https://newsapi.org/v2/everything"

        params = {
            "q": " ".join(keywords), 
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": 1,
            "apiKey": api_key
        }

        try:
            all_chunks, all_metas = [], []

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            articles = data.get("articles", [])

            if not articles:
                logger.info("No news articles found for the query.")
                return [], []

            article = articles[0]

            title            = article.get("title")    or ""
            description      = article.get("description") or ""
            content_snippet  = article.get("content")  or ""

            raw_text = f"Title: {title}\nDescription: {description}\nContent Snippet: {content_snippet}"

            chunk, metas = self._chunk_on_the_fly(
                raw_text,
                {"url": article.get("url", ""), "title": title,
                "source": article.get("source", {}).get("name", ""),
                "category": "external"}
            )
            all_chunks.extend(chunk)
            all_metas.extend(metas)
            
            all_chunks, all_metas = self._filter_chunks_by_relevance(query, all_chunks, all_metas)

            return all_chunks, all_metas

        except Exception as e:
            # requests renders the failing URL into its exceptions, and the key
            # is a query parameter on that URL.
            logger.info(
                f"News search error: {api_keys_module.scrub(str(e), request_secrets())}"
            )
            return [], []
