import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests
import yaml
from dotenv import load_dotenv

from src.audit import sha256_json, sha256_text

load_dotenv()
logger = logging.getLogger(__name__)


class AIClientError(RuntimeError):
    pass


class _ProcessProviderLimiter:
    """Coordinate one provider endpoint across every AIClient in this process."""

    def __init__(self, limit: int) -> None:
        self._condition = threading.Condition()
        self._limit = limit
        self._active = 0

    def tighten(self, limit: int) -> int:
        """Apply the most conservative limit seen for this endpoint."""
        with self._condition:
            if limit < self._limit:
                self._limit = limit
                self._condition.notify_all()
            return self._limit

    @property
    def limit(self) -> int:
        with self._condition:
            return self._limit

    def __enter__(self) -> int:
        with self._condition:
            while self._active >= self._limit:
                self._condition.wait()
            self._active += 1
            return self._limit

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        with self._condition:
            self._active -= 1
            self._condition.notify_all()


_PROVIDER_LIMITERS: dict[tuple[str, str], _ProcessProviderLimiter] = {}
_PROVIDER_LIMITERS_LOCK = threading.Lock()


def _provider_endpoint_key(provider: str, endpoint: str) -> tuple[str, str]:
    return provider, endpoint.strip().rstrip("/")


def _process_provider_limiter(
    provider: str,
    endpoint: str,
    configured_limit: int,
) -> _ProcessProviderLimiter:
    key = _provider_endpoint_key(provider, endpoint)
    with _PROVIDER_LIMITERS_LOCK:
        limiter = _PROVIDER_LIMITERS.get(key)
        if limiter is None:
            limiter = _ProcessProviderLimiter(configured_limit)
            _PROVIDER_LIMITERS[key] = limiter
        else:
            limiter.tighten(configured_limit)
        return limiter


@dataclass(frozen=True)
class ProviderSelection:
    provider: str
    model: str
    timeout_seconds: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class AIConfig:
    primary: ProviderSelection
    fallback: ProviderSelection | None
    temperature: float | None
    validation_retries: int
    openrouter_api_key_env: str
    openrouter_base_url: str
    openrouter_max_concurrent_calls: int
    ollama_base_url: str
    ollama_max_concurrent_calls: int


@dataclass(frozen=True)
class ProviderResponse:
    content: str | dict[str, Any]
    metadata: dict[str, Any]


class AIClient:
    def __init__(self, config_path: str = "config.yaml") -> None:
        self._config = self._load_config(config_path)
        self._provider_limiters = {
            "openrouter": _process_provider_limiter(
                "openrouter",
                self._config.openrouter_base_url,
                self._config.openrouter_max_concurrent_calls,
            ),
            "ollama": _process_provider_limiter(
                "ollama",
                self._config.ollama_base_url,
                self._config.ollama_max_concurrent_calls,
            ),
        }

    def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        settings: dict[str, Any] | None = None,
        call_records: list[dict[str, Any]] | None = None,
        call_context: dict[str, Any] | None = None,
    ) -> str:
        content = self._run_with_fallback(
            prompt=prompt,
            system_prompt=system_prompt,
            settings=settings,
            expect_json=False,
            schema=None,
            call_records=call_records,
            call_context=call_context,
        )
        if not isinstance(content, str):
            raise AIClientError("AI provider returned a non-text response.")
        return content

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        settings: dict[str, Any] | None = None,
        call_records: list[dict[str, Any]] | None = None,
        call_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        content = self._run_with_fallback(
            prompt=prompt,
            system_prompt=system_prompt,
            settings=settings,
            expect_json=True,
            schema=schema,
            call_records=call_records,
            call_context=call_context,
        )
        if not isinstance(content, dict):
            raise AIClientError("AI provider returned JSON in an unsupported format.")
        return content

    def _run_with_fallback(
        self,
        prompt: str,
        system_prompt: str | None,
        settings: dict[str, Any] | None,
        expect_json: bool,
        schema: dict[str, Any] | None,
        call_records: list[dict[str, Any]] | None,
        call_context: dict[str, Any] | None,
    ) -> str | dict[str, Any]:
        failures: list[str] = []
        selections = self._resolve_selections(settings)

        for provider_attempt, selection in enumerate(selections, start=1):
            started_at = datetime.now(timezone.utc)
            started_counter = time.perf_counter()
            record: dict[str, Any] = {
                "call_id": uuid.uuid4().hex,
                "provider_attempt": provider_attempt,
                "started_at": started_at.isoformat(timespec="milliseconds").replace(
                    "+00:00",
                    "Z",
                ),
                **selection.as_dict(),
                "expect_json": expect_json,
                "prompt_sha256": sha256_text(prompt),
                "system_prompt_sha256": (
                    sha256_text(system_prompt) if system_prompt else None
                ),
                "schema_sha256": sha256_json(schema) if schema is not None else None,
                "temperature": self._resolve_temperature(settings),
                "configured_max_concurrent_calls": (
                    self._configured_provider_limit(selection.provider)
                ),
                **(call_context or {}),
            }
            try:
                with self._provider_limiters[selection.provider] as effective_limit:
                    record["effective_process_max_concurrent_calls"] = (
                        effective_limit
                    )
                    dispatched = self._dispatch(
                        selection=selection,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        settings=settings,
                        expect_json=expect_json,
                        schema=schema,
                    )
                if isinstance(dispatched, ProviderResponse):
                    response = dispatched.content
                    record["provider_response"] = dispatched.metadata
                else:
                    # Retains a simple seam for tests and custom provider adapters.
                    response = dispatched
                record.update(
                    {
                        "status": "success",
                        "response_sha256": (
                            sha256_json(response)
                            if isinstance(response, dict)
                            else sha256_text(response)
                        ),
                    }
                )
                return response
            except Exception as exc:
                failures.append(f"{selection.provider}/{selection.model}: {exc}")
                record.update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                logger.warning(
                    "AI provider call failed",
                    extra={
                        "event": "model_call_failed",
                        "provider": selection.provider,
                        "model": selection.model,
                        "error_type": type(exc).__name__,
                    },
                )
            finally:
                finished_at = datetime.now(timezone.utc)
                record["finished_at"] = finished_at.isoformat(
                    timespec="milliseconds"
                ).replace("+00:00", "Z")
                record["duration_ms"] = round(
                    (time.perf_counter() - started_counter) * 1000,
                    3,
                )
                if call_records is not None:
                    call_records.append(record)

        raise AIClientError(
            "All AI providers failed. " + " | ".join(failures)
        )

    def _resolve_selections(
        self,
        settings: dict[str, Any] | None,
    ) -> list[ProviderSelection]:
        settings = settings or {}
        provider_override = settings.get("provider")
        model_override = settings.get("model")
        timeout_override = settings.get("timeout_seconds")
        if provider_override is not None or model_override is not None:
            provider = provider_override or self._config.primary.provider
            model = model_override or self._config.primary.model
            matching_selection = next(
                (
                    selection
                    for selection in [
                        self._config.primary,
                        self._config.fallback,
                    ]
                    if selection is not None
                    and selection.provider == provider
                    and selection.model == model
                ),
                self._config.primary,
            )
            timeout_seconds = (
                timeout_override
                if timeout_override is not None
                else matching_selection.timeout_seconds
            )
            return [
                self._parse_selection(
                    {
                        "provider": provider,
                        "model": model,
                        "timeout_seconds": timeout_seconds,
                    },
                    "override",
                )
            ]

        selections = [self._config.primary]
        if self._config.fallback is not None:
            selections.append(self._config.fallback)
        if len(selections) > 1:
            available = [
                selection
                for selection in selections
                if not (
                    selection.provider == "openrouter"
                    and not os.getenv(
                        self._config.openrouter_api_key_env,
                        "",
                    ).strip()
                )
            ]
            if available:
                return available
        return selections

    def describe(self) -> dict[str, Any]:
        """Return non-secret model configuration for manifests and diagnostics."""
        return {
            "primary": self._config.primary.as_dict(),
            "fallback": (
                self._config.fallback.as_dict()
                if self._config.fallback is not None
                else None
            ),
            "temperature": self._config.temperature,
            "validation_retries": self._config.validation_retries,
            "max_concurrent_calls": {
                "openrouter": self._config.openrouter_max_concurrent_calls,
                "ollama": self._config.ollama_max_concurrent_calls,
            },
            "effective_process_max_concurrent_calls": {
                provider: limiter.limit
                for provider, limiter in self._provider_limiters.items()
            },
        }

    @property
    def validation_attempts(self) -> int:
        """Maximum schema/evidence validation attempts for one extraction."""
        return self._config.validation_retries + 1

    def _dispatch(
        self,
        selection: ProviderSelection,
        prompt: str,
        system_prompt: str | None,
        settings: dict[str, Any] | None,
        expect_json: bool,
        schema: dict[str, Any] | None,
    ) -> str | dict[str, Any] | ProviderResponse:
        if selection.provider == "openrouter":
            return self._call_openrouter(
                selection=selection,
                prompt=prompt,
                system_prompt=system_prompt,
                settings=settings,
                expect_json=expect_json,
                schema=schema,
            )

        if selection.provider == "ollama":
            return self._call_ollama(
                selection=selection,
                prompt=prompt,
                system_prompt=system_prompt,
                settings=settings,
                expect_json=expect_json,
                schema=schema,
            )

        raise AIClientError(f"Unsupported AI provider: {selection.provider}")

    def _call_openrouter(
        self,
        selection: ProviderSelection,
        prompt: str,
        system_prompt: str | None,
        settings: dict[str, Any] | None,
        expect_json: bool,
        schema: dict[str, Any] | None,
    ) -> ProviderResponse:
        api_key = os.getenv(self._config.openrouter_api_key_env)
        if not api_key:
            raise AIClientError(
                f"Missing OpenRouter API key in env var {self._config.openrouter_api_key_env}."
            )

        response_format: dict[str, Any] | None = None
        if expect_json:
            response_format = {"type": "json_object"}
            if schema is not None:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "strict": True,
                        "schema": schema,
                    },
                }

        payload: dict[str, Any] = {
            "model": selection.model,
            "messages": self._build_messages(prompt, system_prompt),
        }

        temperature = self._resolve_temperature(settings)
        if temperature is not None:
            payload["temperature"] = temperature
        if response_format is not None:
            payload["response_format"] = response_format

        response = requests.post(
            f"{self._config.openrouter_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=selection.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AIClientError("OpenRouter response did not include message content.") from exc

        normalized = self._normalize_response_content(content, expect_json)
        choice = data.get("choices", [{}])[0]
        return ProviderResponse(
            content=normalized,
            metadata={
                "response_id": data.get("id"),
                "provider": data.get("provider"),
                "returned_model": data.get("model"),
                "created": data.get("created"),
                "finish_reason": (
                    choice.get("finish_reason")
                    if isinstance(choice, dict)
                    else None
                ),
                "system_fingerprint": data.get("system_fingerprint"),
                "usage": data.get("usage"),
            },
        )

    def _call_ollama(
        self,
        selection: ProviderSelection,
        prompt: str,
        system_prompt: str | None,
        settings: dict[str, Any] | None,
        expect_json: bool,
        schema: dict[str, Any] | None,
    ) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": selection.model,
            "prompt": self._build_prompt(prompt, system_prompt),
            "stream": False,
        }

        temperature = self._resolve_temperature(settings)
        if temperature is not None:
            payload["options"] = {"temperature": temperature}
        if expect_json:
            payload["format"] = schema if schema is not None else "json"

        response = requests.post(
            f"{self._config.ollama_base_url.rstrip('/')}/api/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=selection.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()

        content = data.get("response")
        if not isinstance(content, str):
            raise AIClientError("Ollama response did not include text content.")

        normalized = self._normalize_response_content(content, expect_json)
        return ProviderResponse(
            content=normalized,
            metadata={
                "returned_model": data.get("model"),
                "created_at": data.get("created_at"),
                "done_reason": data.get("done_reason"),
                "total_duration": data.get("total_duration"),
                "load_duration": data.get("load_duration"),
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
            },
        )

    @staticmethod
    def _build_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _build_prompt(prompt: str, system_prompt: str | None) -> str:
        if not system_prompt:
            return prompt
        return f"{system_prompt}\n\n{prompt}"

    @staticmethod
    def _normalize_response_content(
        content: Any,
        expect_json: bool,
    ) -> str | dict[str, Any]:
        if not expect_json:
            if isinstance(content, list):
                return "".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict)
                )
            if isinstance(content, str):
                return content
            raise AIClientError("Provider returned text in an unsupported format.")

        if isinstance(content, dict):
            return content

        if isinstance(content, list):
            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            )

        if not isinstance(content, str):
            raise AIClientError("Provider returned JSON in an unsupported format.")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise AIClientError("Provider returned invalid JSON content.") from exc

        if not isinstance(parsed, dict):
            raise AIClientError("Provider JSON response must decode to an object.")

        return parsed

    def _resolve_temperature(self, settings: dict[str, Any] | None) -> float | None:
        temperature = (
            settings["temperature"]
            if settings and "temperature" in settings
            else self._config.temperature
        )
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0 <= temperature <= 2
        ):
            raise AIClientError("temperature must be between 0 and 2.")
        return float(temperature) if temperature is not None else None

    def _configured_provider_limit(self, provider: str) -> int:
        if provider == "openrouter":
            return self._config.openrouter_max_concurrent_calls
        if provider == "ollama":
            return self._config.ollama_max_concurrent_calls
        raise AIClientError(f"Unsupported AI provider: {provider}")

    @staticmethod
    def _load_config(config_path: str) -> AIConfig:
        if not os.path.exists(config_path):
            raise AIClientError(f"Missing config file: {config_path}")

        with open(config_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        ai_config = data.get("ai")
        if not isinstance(ai_config, dict):
            raise AIClientError("Missing required 'ai' configuration section.")

        primary = AIClient._parse_selection(ai_config, "primary")
        fallback_config = ai_config.get("fallback")
        fallback = None
        if fallback_config is not None:
            if not isinstance(fallback_config, dict):
                raise AIClientError("The 'ai.fallback' configuration must be an object.")
            fallback = AIClient._parse_selection(fallback_config, "fallback")

        openrouter_config = ai_config.get("openrouter") or {}
        ollama_config = ai_config.get("ollama") or {}
        if not isinstance(openrouter_config, dict):
            raise AIClientError("The 'ai.openrouter' configuration must be an object.")
        if not isinstance(ollama_config, dict):
            raise AIClientError("The 'ai.ollama' configuration must be an object.")

        temperature = ai_config.get("temperature")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0 <= temperature <= 2
        ):
            raise AIClientError("ai.temperature must be between 0 and 2.")
        validation_retries = ai_config.get("validation_retries", 1)
        if (
            isinstance(validation_retries, bool)
            or not isinstance(validation_retries, int)
            or not 0 <= validation_retries <= 3
        ):
            raise AIClientError("ai.validation_retries must be between 0 and 3.")
        api_key_env = openrouter_config.get(
            "api_key_env",
            "OPENROUTER_API_KEY",
        )
        openrouter_base_url = openrouter_config.get(
            "base_url",
            "https://openrouter.ai/api/v1",
        )
        ollama_base_url = ollama_config.get(
            "base_url",
            "http://localhost:11434",
        )
        openrouter_max_concurrent_calls = openrouter_config.get(
            "max_concurrent_calls",
            8,
        )
        ollama_max_concurrent_calls = ollama_config.get(
            "max_concurrent_calls",
            1,
        )
        for field_name, value in {
            "ai.openrouter.max_concurrent_calls": (
                openrouter_max_concurrent_calls
            ),
            "ai.ollama.max_concurrent_calls": ollama_max_concurrent_calls,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 64
            ):
                raise AIClientError(f"{field_name} must be between 1 and 64.")
        for field_name, value in {
            "ai.openrouter.api_key_env": api_key_env,
            "ai.openrouter.base_url": openrouter_base_url,
            "ai.ollama.base_url": ollama_base_url,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise AIClientError(f"{field_name} must be a non-empty string.")

        return AIConfig(
            primary=primary,
            fallback=fallback,
            temperature=float(temperature) if temperature is not None else None,
            validation_retries=validation_retries,
            openrouter_api_key_env=api_key_env.strip(),
            openrouter_base_url=openrouter_base_url.strip(),
            openrouter_max_concurrent_calls=openrouter_max_concurrent_calls,
            ollama_base_url=ollama_base_url.strip(),
            ollama_max_concurrent_calls=ollama_max_concurrent_calls,
        )

    @staticmethod
    def _parse_selection(config: dict[str, Any], section_name: str) -> ProviderSelection:
        provider = config.get("provider")
        model = config.get("model")
        timeout_seconds = config.get("timeout_seconds", 60)

        if provider not in {"openrouter", "ollama"}:
            raise AIClientError(
                f"Unsupported provider in ai {section_name}: {provider}"
            )
        if not isinstance(model, str) or not model.strip():
            raise AIClientError(f"Missing model in ai {section_name} configuration.")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or timeout_seconds <= 0
        ):
            raise AIClientError(
                f"Invalid timeout_seconds in ai {section_name} configuration."
            )

        return ProviderSelection(
            provider=provider,
            model=model.strip(),
            timeout_seconds=timeout_seconds,
        )
