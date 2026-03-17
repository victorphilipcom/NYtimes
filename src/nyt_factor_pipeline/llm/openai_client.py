"""Thin wrapper around the OpenAI ChatCompletion API for sparse labeling tasks."""

from __future__ import annotations

from nyt_factor_pipeline.config import get_settings
from nyt_factor_pipeline.logging_utils import get_logger

log = get_logger(__name__)


class OpenAIClient:
    """Wrapper for OpenAI chat completions. Lazy-loads the client."""

    def __init__(self, model: str | None = None):
        self._client = None
        self._model = model or get_settings().openai_chat_model

    def _get_client(self):
        if self._client is None:
            settings = get_settings()
            if not settings.has_openai:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. LLM features require an OpenAI key."
                )
            import openai
            self._client = openai.OpenAI(api_key=settings.openai_api_key)
        return self._client

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> str:
        """Send a chat completion request. Returns the assistant message content."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    @property
    def model(self) -> str:
        return self._model
