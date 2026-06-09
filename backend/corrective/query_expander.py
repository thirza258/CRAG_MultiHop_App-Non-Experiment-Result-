from typing import List, Dict, Any
from pydantic import BaseModel, field_validator, model_validator
from openai import OpenAI
import os

class KeywordList(BaseModel):
    keywords: List[str]

    @field_validator("keywords", mode="before")
    @classmethod
    def parse_if_string(cls, v):
        if isinstance(v, str):
            v = [kw.strip() for kw in v.split(",")]
        return v

    @field_validator("keywords")
    @classmethod
    def clean_and_filter(cls, v):
        cleaned = [kw.strip().strip("\"'") for kw in v if kw.strip()]
        return cleaned

    @model_validator(mode="after")
    def enforce_limits(self) -> "KeywordList":
        if not self.keywords:
            raise ValueError("LLM returned no usable keywords")
        self.keywords = self.keywords[:3] 
        return self

    def to_search_string(self) -> str:
        return ", ".join(self.keywords)


class QueryExpander:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.model = config.get("llm_model", "google/gemini-3-flash-preview")

    def _call(self, prompt: str, max_tokens: int = 100) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    def retrieve_keyword_from_query(self, query: str) -> List[str]:
        prompt = (
            f"Extract 3-5 main keywords from the following query for retrieval purposes. "
            f"Return only the keywords separated by commas, nothing else.\n\nQuery: {query}\n\nKeywords:"
        )
        keywords_str = self._call(prompt, max_tokens=50)
        return [kw.strip() for kw in keywords_str.split(",") if kw.strip()]

    def retrieve_wikipedia_keyword_from_query(self, query: str) -> List[str]:
        prompt = (
            f"Extract 2-3 keywords from the following query for Wikipedia search. "
            f"Keywords should be named entities or concepts likely to have Wikipedia pages. "
            f"Return only the keywords separated by commas, nothing else.\n\nQuery: {query}\n\nWikipedia Keywords:"
        )
        raw = self._call(prompt, max_tokens=50)
        return KeywordList(keywords=raw).keywords

    def retrieve_news_keyword_from_query(self, query: str) -> List[str]:
        prompt = (
            f"Extract 2-3 keywords from the following query for news article search. "
            f"Keywords should be entities or topics likely covered in recent news. "
            f"Return only the keywords separated by commas, nothing else.\n\nQuery: {query}\n\nNews Keywords:"
        )
        raw = self._call(prompt, max_tokens=50)
        return KeywordList(keywords=raw).keywords

    def to_keyword(self, query: str) -> str:
        keywords = self.retrieve_keyword_from_query(query)
        return " ".join(keywords)

    def expand_multiple(self, query: str, n: int = 3) -> List[str]:
        prompt = (
            f"Generate {n} different ways to search for the same information "
            f"as this query. Each should approach the topic from a different angle. "
            f"Return only the queries, one per line, no numbering.\n\nQuery: {query}"
        )
        lines = self._call(prompt, max_tokens=200).split("\n")
        return [l.strip() for l in lines if l.strip()][:n]

    def reformulate(self, query: str) -> str:
        prompt = (
            f"Reformulate this search query to find the same information "
            f"using different words. Return only the reformulated query.\n\nQuery: {query}"
        )
        return self._call(prompt, max_tokens=100)

    def hypothetical_document(self, query: str) -> str:
        prompt = (
            f"Write a short factual paragraph that would directly answer this question. "
            f"Be concise and specific.\n\nQuestion: {query}"
        )
        return self._call(prompt, max_tokens=150)