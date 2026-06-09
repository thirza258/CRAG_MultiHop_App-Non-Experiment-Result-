from langchain_openai import ChatOpenAI
from common.prompt_builder import vote_prompt, rag_prompt, prompt_generator
from abc import ABC, abstractmethod
from typing import Optional
import os
from openai import OpenAI

class BaseLLM(ABC):
    def __init__(self, model: str, temperature: float = 0.0, api_key: Optional[str] = None):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """
        Abstract method that child classes must implement.
        This handles the specific API call to the provider.
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
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0, api_key: str = ""):
        super().__init__(model, temperature, api_key)
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def _call_api(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            return f"OpenAI Error: {e}"


class OpenRouterLLM(BaseLLM):
    """
    Base class for any model routed via OpenRouter.
    It uses the OpenAI SDK but points to the OpenRouter URL.
    """
    def __init__(self, model: str, temperature: float = 0.0, api_key: str = ""):
        super().__init__(model, temperature, api_key)
        
        # OpenRouter Configuration
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            default_headers={
                "HTTP-Referer": "https://rag.nevatal.tech", 
                "X-Title": "RagReader"
            }
        )

    def _call_api(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            return f"OpenRouter Error ({self.model}): {e}"


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