"""
Groq-backed evaluator model for DeepEval.

DeepEval's LLM-based metrics (relevancy, faithfulness, hallucination) need a
"judge" model. By default DeepEval reaches for OpenAI, which this project
forbids. This module wraps Groq's Chat Completions endpoint in a
``DeepEvalBaseLLM`` subclass so every metric is judged by Groq/Llama3.

DeepEval calls ``generate`` with a Pydantic ``schema`` when it wants structured
output and expects an INSTANCE of that schema back. We use the ``instructor``
library (recommended by the DeepEval docs for API-based models) to coerce Groq's
reply into the schema -- this both guarantees valid JSON and avoids the
"Object of type Groq is not JSON serializable" error that arises when the raw
client leaks into DeepEval's result serialization.
"""

from __future__ import annotations

from typing import Any

import instructor
from deepeval.models.base_model import DeepEvalBaseLLM
from groq import Groq
from pydantic import BaseModel

import config

# Retry on transient Groq errors (notably 429 rate limits). The Groq SDK honours
# Retry-After and backs off exponentially up to this many attempts.
_MAX_RETRIES = config.GROQ_MAX_RETRIES


class GroqEvaluatorLLM(DeepEvalBaseLLM):
    """A DeepEval judge model served by Groq instead of OpenAI."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model_name = model or config.EVALUATOR_MODEL
        api_key = api_key or config.require_api_key()
        base_url = base_url if base_url is not None else config.GROQ_BASE_URL

        # Raw client for free-text generation; instructor wrapper for schemas.
        client_kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": _MAX_RETRIES}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._raw_client = Groq(**client_kwargs)
        self._structured_client = instructor.from_groq(
            self._raw_client, mode=instructor.Mode.JSON
        )
        # DeepEvalBaseLLM expects load_model() to provide the underlying model.
        self.model = self._raw_client

    # -- DeepEvalBaseLLM interface -----------------------------------------
    def load_model(self) -> Groq:
        return self._raw_client

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        """Call Groq; return a string, or a ``schema`` instance when requested."""
        if schema is not None:
            # instructor enforces the JSON structure and returns a schema object.
            return self._structured_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_model=schema,
                max_retries=_MAX_RETRIES,
            )

        response = self._raw_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    async def a_generate(
        self, prompt: str, schema: type[BaseModel] | None = None
    ) -> Any:
        """Async variant. Groq's client is sync, so we delegate to ``generate``."""
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"Groq::{self.model_name}"
