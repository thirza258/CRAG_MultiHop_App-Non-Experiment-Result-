from common.prompt_builder import vote_prompt, rag_prompt, prompt_generator
from abc import ABC, abstractmethod
from typing import Optional
import logging
import os
import time
from openai import OpenAI

from ai_handler.openrouter import MissingAPIKeyError, openrouter_client
from common.runtime import api_keys as api_keys_module
from common.runtime.context import request_secrets, resolve_llm_model

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3


def _chat_with_retry(client: OpenAI, provider: str, model: str, prompt: str, temperature: float) -> str:
    """Call the chat API, retrying transient failures with backoff."""
    last_error = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            last_error = e
            # SDK exceptions quote the failing request, which can include the
            # Authorization header — so scrub before this reaches a log file.
            logger.warning(
                "%s call failed (attempt %d/%d, model=%s): %s",
                provider, attempt, _MAX_ATTEMPTS, model,
                api_keys_module.scrub(str(e), request_secrets()),
            )
            if attempt < _MAX_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))

    # Exhausted retries are reported by *returning* the error rather than
    # raising (callers treat an "<Provider> Error" body as a soft failure), and
    # this string can end up in front of the user — scrub it too.
    return api_keys_module.scrub(
        f"{provider} Error ({model}): {last_error}", request_secrets()
    )

class BaseLLM(ABC):
    def __init__(self, model: str, temperature: float = 0.0, api_key: Optional[str] = None):
        # The *configured* model. What a given call actually uses may be the
        # per-request choice instead — see OpenRouterLLM.active_model.
        self.model = model
        self.temperature = temperature
        self.api_key = api_key

    @abstractmethod
    def _call_api(self, prompt: str, temperature: float = None) -> str:
        """
        Abstract method that child classes must implement.
        This handles the specific API call to the provider.

        ``temperature`` overrides the instance's own for one call. It is passed
        only by the answer-generation path: the pipeline's internal decision
        calls (hop bridging, keyword extraction) stay deterministic, because
        making those random changes which documents get retrieved rather than
        how the answer reads.
        """
        pass

    def generate(self, prompt: str) -> str:
        """Standard text generation."""
        return self._call_api(prompt)

    def rag_generate(self, query: str, context: str) -> str:
        """Generates an answer based on RAG context."""
        formatted_prompt = rag_prompt(query, context)
        return self._call_api(formatted_prompt)

    def prompt_generate(self, query: str) -> str:
        """Generates/Optimizes a search query."""
        formatted_prompt = prompt_generator(query)
        return self._call_api(formatted_prompt)

    def vote_generate(self, query: str, chunk: str, response: str) -> str:
        """Generates a vote (Yes/No) for validity."""
        formatted_prompt = vote_prompt(query, chunk, response)
        return self._call_api(formatted_prompt)


class OpenAILLM(BaseLLM):
    """
    Direct OpenAI models. The per-request model choice is an OpenRouter id, so
    it deliberately does *not* apply here — only the key is resolved late.
    """
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0, api_key: str = ""):
        super().__init__(model, temperature, api_key)

    @property
    def client(self) -> OpenAI:
        # Built per call rather than in __init__: a deployment where users bring
        # their own keys has none at construction time, and raising there used to
        # make the whole pipeline unconstructible.
        return OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))

    def _call_api(self, prompt: str, temperature: float = None) -> str:
        return _chat_with_retry(
            self.client, "OpenAI", self.model, prompt,
            self.temperature if temperature is None else temperature,
        )


class OpenRouterLLM(BaseLLM):
    """
    Base class for any model routed via OpenRouter.

    Both the key and the model are resolved at call time from the request
    context, so one shared instance can serve queries that chose different
    models with different credentials. ``self.model`` stays the fallback for
    requests that expressed no preference.
    """
    def __init__(self, model: str, temperature: float = 0.0, api_key: str = ""):
        super().__init__(model, temperature, api_key)

    @property
    def client(self) -> OpenAI:
        """Cached client for whichever key applies to this request."""
        return openrouter_client(self.api_key or None)

    @property
    def active_model(self) -> str:
        """The model this call should use: the request's choice, else ours."""
        return resolve_llm_model(self.model)

    def _call_api(self, prompt: str, temperature: float = None) -> str:
        model = self.active_model
        try:
            client = self.client
        except MissingAPIKeyError as e:
            # Same soft-failure contract as an exhausted retry: callers already
            # recognise an "OpenRouter Error" body and degrade around it.
            logger.error("[OpenRouter] %s", e)
            return f"OpenRouter Error ({model}): {e}"
        return _chat_with_retry(
            client, "OpenRouter", model, prompt,
            self.temperature if temperature is None else temperature,
        )


class ClaudeLLM(OpenRouterLLM):
    """
    Anthropic models via OpenRouter.
    Default model: anthropic/claude-3.5-sonnet
    """
    def __init__(self, model: str = "anthropic/claude-3.5-sonnet", temperature: float = 0.0, api_key: str = ""):
        super().__init__(model, temperature, api_key)


class GeminiLLM(OpenRouterLLM):
    """
    Google models via OpenRouter.
    Default model: google/gemini-pro-1.5
    """
    def __init__(self, model: str = "google/gemini-pro-1.5", temperature: float = 0.0, api_key: str = ""):
        super().__init__(model, temperature, api_key)
