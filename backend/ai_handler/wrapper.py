import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import Dict
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def llm_langchain_wrapper(model_name: str):
    api_key = os.getenv("OPENROUTER_API_KEY")
    langchain_llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1", 
        model=model_name,
        temperature=0,
    )
    return LangchainLLMWrapper(langchain_llm)


def embeddings_langchain_wrapper(model_name: str):
    api_key = os.getenv("OPENROUTER_API_KEY")
    langchain_embeddings = OpenAIEmbeddings(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=model_name,
    )
    return LangchainEmbeddingsWrapper(langchain_embeddings)

