"""LLM client abstraction using LiteLLM for provider-agnostic access.

Supports: Anthropic Claude, OpenAI GPT, Google Gemini, Ollama, vLLM, AWS Bedrock,
Azure OpenAI, and 100+ other providers through LiteLLM's unified interface.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from synthforge.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client wrapping LiteLLM completion API."""

    def __init__(self, config: LLMConfig):
        self._config = config
        self._call_count = 0
        self._total_tokens = 0

        if not config.enabled:
            return

        try:
            import litellm
            self._litellm = litellm

            # Suppress litellm's verbose logging
            litellm.suppress_debug_info = True

            # Set API key if provided
            if config.api_key:
                if config.provider == "anthropic":
                    os.environ.setdefault("ANTHROPIC_API_KEY", config.api_key)
                elif config.provider == "openai":
                    os.environ.setdefault("OPENAI_API_KEY", config.api_key)

        except ImportError:
            raise ImportError(
                "LLM features require litellm. Install with: pip install 'synthforge[llm]'"
            )

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def stats(self) -> dict[str, int]:
        return {"calls": self._call_count, "total_tokens": self._total_tokens}

    def _get_model_string(self) -> str:
        """Build the LiteLLM model string."""
        provider = self._config.provider or ""
        model = self._config.model or ""

        # LiteLLM uses provider/model format for some providers
        if provider in ("anthropic", "openai", "google", "cohere"):
            return model  # LiteLLM auto-routes these
        if provider == "ollama":
            return f"ollama/{model}"
        if provider == "bedrock":
            return f"bedrock/{model}"
        return model

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        response_format: str = "text",
        max_tokens: int | None = None,
    ) -> str:
        """Send a completion request to the configured LLM.

        Args:
            prompt: User message content.
            system: Optional system message.
            response_format: 'text' or 'json'.
            max_tokens: Override default max_tokens.

        Returns:
            The model's response text.
        """
        if not self.enabled:
            raise RuntimeError("LLM not enabled. Configure LLM settings first.")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self._get_model_string(),
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": max_tokens or self._config.max_tokens,
            "timeout": self._config.timeout,
        }

        if self._config.api_base:
            kwargs["api_base"] = self._config.api_base

        try:
            response = self._litellm.completion(**kwargs)
            self._call_count += 1

            # Track tokens
            if hasattr(response, "usage") and response.usage:
                self._total_tokens += getattr(response.usage, "total_tokens", 0)

            content = response.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            logger.error("LLM call failed: %s", e)
            raise

    def complete_json(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any] | list:
        """Send a completion request expecting JSON output.

        Parses the response as JSON, handling markdown code fences.
        """
        response = self.complete(
            prompt=prompt,
            system=system,
            response_format="json",
            max_tokens=max_tokens,
        )

        # Strip markdown code fences if present
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM JSON response: %s", e)
            logger.debug("Raw response: %s", response[:500])
            raise ValueError(f"LLM returned invalid JSON: {e}") from e
