"""
Groq-backed evaluator model for DeepEval.

DeepEval's LLM-based metrics (relevancy, faithfulness, hallucination) need a
"judge" model. By default DeepEval reaches for OpenAI, which this project
forbids. This module wraps Groq's OpenAI-compatible Chat Completions endpoint
in a ``DeepEvalBaseLLM`` subclass so every metric is judged by Groq/Llama3.

The class also supports DeepEval's JSON/schema mode: when a metric passes a
Pydantic ``schema``, we ask Groq to return JSON and validate it. This keeps the
metrics deterministic and avoids brittle free-text parsing.
"""

from __future__ import annotations

import json
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from groq import Groq
from pydantic import BaseModel

import config


class GroqEvaluatorLLM(DeepEvalBaseLLM):
    """A DeepEval judge model served by Groq instead of OpenAI."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or config.EVALUATOR_MODEL
        self._api_key = api_key or config.require_api_key()
        self._base_url = base_url or config.GROQ_BASE_URL
        self._client: Groq | None = None
        super().__init__(self.model)

    # -- DeepEvalBaseLLM interface -----------------------------------------
    def load_model(self) -> Groq:
        """Lazily build and cache the Groq client."""
        if self._client is None:
            self._client = Groq(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        """Synchronously call Groq and optionally coerce the reply into ``schema``.

        DeepEval calls ``generate`` with a ``schema`` when it wants structured
        output. We honour that by requesting a JSON object and validating it.
        """
        client = self.load_model()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        if schema is not None:
            # Ask Groq for a raw JSON object so we can parse into the schema.
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""

        if schema is None:
            return content

        data = json.loads(content)
        return schema.model_validate(data)

    async def a_generate(
        self, prompt: str, schema: type[BaseModel] | None = None
    ) -> Any:
        """Async variant. Groq's SDK is sync, so we delegate to ``generate``.

        DeepEval awaits this in its async metric paths; running the sync call
        inline is acceptable for a CI-scale test suite.
        """
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"Groq::{self.model}"
