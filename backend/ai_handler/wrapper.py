"""RAGAS judge wrappers.

These build the LangChain clients RAGAS grades answers with. Two deliberate
choices here:

* The **key** is resolved from the current request, so a user who brought their
  own key pays for their own evaluation instead of silently spending the
  server's.
* The **models** are not. They stay whatever the pipeline configured as its
  judge (``evaluation_llm_model`` / ``evaluation_embedding_model``) rather than
  following the user's model pick — a judge that is the same model as the one
  being judged scores its own output, which is exactly the bias the separate
  judge configuration exists to avoid.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ai_handler.openrouter import OPENROUTER_BASE_URL, MissingAPIKeyError
from common.runtime.context import openrouter_api_key


def _judge_api_key() -> str:
    """The key the judge should spend, or raise if there is none to spend."""
    key = openrouter_api_key()
    if not key:
        raise MissingAPIKeyError(
            "No OpenRouter API key available for answer evaluation."
        )
    return key


def llm_langchain_wrapper(model_name: str):
    langchain_llm = ChatOpenAI(
        api_key=_judge_api_key(),
        base_url=OPENROUTER_BASE_URL,
        model=model_name,
        temperature=0,
    )
    return LangchainLLMWrapper(langchain_llm)


def embeddings_langchain_wrapper(model_name: str):
    langchain_embeddings = OpenAIEmbeddings(
        api_key=_judge_api_key(),
        base_url=OPENROUTER_BASE_URL,
        model=model_name,
    )
    return LangchainEmbeddingsWrapper(langchain_embeddings)
