"""AI client supporting OpenRouter and Ollama providers via OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from .config import loadAISettings

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    provider: str
    model: str
    base_url: str
    api_key: str | None
    timeout: float
    temperature: float
    max_concurrent_calls: int


class AIClient:
    def __init__(self, settings: dict | None = None) -> None:
        self.config = settings if settings is not None else loadAISettings()
        self.primary = self._build_provider_config(self.config)
        fallback_section = self.config.get("fallback")
        self.fallback = (
            self._build_provider_config(fallback_section) if fallback_section else None
        )
        self._clients: dict[str, OpenAI] = {}
        self._validation_retries = self.config.get("validation_retries", 1)

    def _build_provider_config(self, section: dict) -> ProviderConfig:
        provider = section["provider"]
        provider_section = self.config.get(provider, {})
        api_key = None
        if provider == "openrouter":
            api_key = os.environ.get(
                provider_section.get("api_key_env", "OPENROUTER_API_KEY")
            )
        base_url = provider_section.get("base_url", "")
        if provider == "ollama" and not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        return ProviderConfig(
            provider=provider,
            model=section["model"],
            base_url=base_url,
            api_key=api_key,
            timeout=section.get("timeout_seconds", 60),
            temperature=section.get("temperature", 0.2),
            max_concurrent_calls=provider_section.get("max_concurrent_calls", 1),
        )

    def _get_client(self, cfg: ProviderConfig) -> OpenAI:
        if cfg.provider not in self._clients:
            self._clients[cfg.provider] = OpenAI(
                base_url=cfg.base_url,
                api_key=cfg.api_key or "unused",
                timeout=cfg.timeout,
            )
        return self._clients[cfg.provider]

    @staticmethod
    def _supports_response_format(cfg: ProviderConfig) -> bool:
        return cfg.provider == "openrouter"

    def _generate(
        self,
        cfg: ProviderConfig,
        prompt: str,
        *,
        messages: list[dict] | None = None,
        response_format: dict | None = None,
        stream: bool = False,
        **kwargs,
    ) -> str | Generator[str, None, None]:
        client = self._get_client(cfg)
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        kwargs.setdefault("temperature", cfg.temperature)
        kwargs.setdefault("model", cfg.model)
        kwargs["messages"] = messages
        kwargs["stream"] = stream
        if response_format and self._supports_response_format(cfg):
            kwargs["response_format"] = response_format

        logger.debug("Calling %s/%s stream=%s", cfg.provider, cfg.model, stream)
        if stream:
            return self._stream_chat(client, **kwargs)

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    @staticmethod
    def _stream_chat(client: OpenAI, **kwargs) -> Generator[str, None, None]:
        stream = client.chat.completions.create(**kwargs)
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content is not None:
                yield delta.content

    def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            return self._generate(self.primary, prompt, **kwargs)
        except (APIError, RateLimitError, APIConnectionError) as e:
            if self.fallback:
                logger.warning(
                    "Primary provider %s failed (%s), falling back to %s",
                    self.primary.provider,
                    e.__class__.__name__,
                    self.fallback.provider,
                )
                return self._generate(self.fallback, prompt, **kwargs)
            logger.error(
                "Primary provider %s failed with no fallback", self.primary.provider
            )
            raise

    def generate_json(
        self,
        prompt: str,
        schema: dict | None = None,
        **kwargs,
    ) -> dict:
        last_error: Exception | None = None
        for attempt in range(self._validation_retries + 1):
            try:
                text = self.generate_text(
                    prompt, response_format={"type": "json_object"}, **kwargs
                )
                result = self._parse_json(text)
                if schema:
                    self._validate_schema(result, schema)
                return result
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt < self._validation_retries:
                    logger.warning(
                        "JSON generation attempt %d failed (%s), retrying",
                        attempt + 1,
                        e.__class__.__name__,
                    )
                continue
        logger.error("JSON generation failed after %d attempts", attempt + 1)
        raise last_error or RuntimeError("Failed to generate valid JSON")

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        match = re.search(r"```(?:json)?\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1)
        return json.loads(text)

    @staticmethod
    def _validate_schema(data: dict, schema: dict) -> None:
        for key in schema:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

    def generate_stream(
        self,
        prompt: str,
        **kwargs,
    ) -> Generator[str, None, None]:
        try:
            yield from self._generate(self.primary, prompt, stream=True, **kwargs)
        except (APIError, RateLimitError, APIConnectionError):
            if self.fallback:
                logger.warning(
                    "Primary provider %s failed during stream, falling back to %s",
                    self.primary.provider,
                    self.fallback.provider,
                )
                yield from self._generate(self.fallback, prompt, stream=True, **kwargs)
            else:
                raise

    def generate_batch(
        self,
        prompts: list[str],
        *,
        use_fallback: bool = False,
        **kwargs,
    ) -> list[str]:
        cfg = self.fallback if use_fallback and self.fallback else self.primary
        max_workers = min(cfg.max_concurrent_calls, len(prompts))
        results: list[str | None] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._generate, cfg, prompt, **kwargs): i
                for i, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results

    def generate_json_batch(
        self,
        prompts: list[str],
        schema: dict | None = None,
        **kwargs,
    ) -> list[dict]:
        max_workers = min(self.primary.max_concurrent_calls, len(prompts))
        results: list[dict | None] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.generate_json, prompt, schema, **kwargs): i
                for i, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results

    def chat(
        self,
        messages: list[dict],
        **kwargs,
    ) -> str:
        cfg = self.primary
        kwargs.setdefault("temperature", cfg.temperature)
        kwargs.setdefault("model", cfg.model)
        return self._generate(cfg, "", messages=messages, **kwargs)

    def structured_output(
        self,
        prompt: str,
        output_schema: dict,
        **kwargs,
    ) -> dict:
        return self.generate_json(prompt, schema=output_schema, **kwargs)


_client: AIClient | None = None


def get_client(settings: dict | None = None) -> AIClient:
    global _client
    if _client is None:
        _client = AIClient(settings)
    return _client


def generate_text(prompt: str, **kwargs) -> str:
    return get_client().generate_text(prompt, **kwargs)


def generate_json(prompt: str, schema: dict | None = None, **kwargs) -> dict:
    return get_client().generate_json(prompt, schema, **kwargs)


def generate_stream(prompt: str, **kwargs) -> Generator[str, None, None]:
    yield from get_client().generate_stream(prompt, **kwargs)


def generate_batch(prompts: list[str], **kwargs) -> list[str]:
    return get_client().generate_batch(prompts, **kwargs)


def generate_json_batch(
    prompts: list[str], schema: dict | None = None, **kwargs
) -> list[dict]:
    return get_client().generate_json_batch(prompts, schema, **kwargs)


def chat(messages: list[dict], **kwargs) -> str:
    return get_client().chat(messages, **kwargs)


def structured_output(prompt: str, output_schema: dict, **kwargs) -> dict:
    return get_client().structured_output(prompt, output_schema, **kwargs)
