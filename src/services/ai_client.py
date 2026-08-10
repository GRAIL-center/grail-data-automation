"""OpenRouter/Ollama AI client with tools and web research."""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import time
from collections.abc import Callable, Generator, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from .config import loadAISettings

# Load variables from the nearest .env file without overriding values
# already present in the process environment.
load_dotenv()

logger = logging.getLogger(__name__)

ToolHandler = Callable[..., Any]


class ProviderQuotaError(RuntimeError):
    pass


PROVIDER_ERRORS = (APIError, RateLimitError, APIConnectionError, ProviderQuotaError)

DEFAULT_COMMENT_RESEARCH_SCHEMA: dict[str, Any] = {
    "Submitted By": None,
    "Organization Name": None,
    "Organization Type": None,
    "501c Status": None,
    "NTEE": None,
    "Organization Role": None,
    "Contact Info": {
        "professional_email": None,
        "professional_phone": None,
        "organization_website": None,
        "professional_profile": None,
    },
    "Evidence": {},
    "Sources": [],
    "Confidence": {},
    "Research Notes": None,
}

RESEARCH_SYSTEM_PROMPT = """You are a careful public-records research agent.
Use web search and page fetching before answering. Prefer original filings,
official government records, official organization websites, staff directories,
IRS records, and reputable institutional sources.

Research only public professional information. Never guess or construct an
email, phone number, affiliation, role, nonprofit status, NTEE code, or identity.
Confirm a person with filing context. Distinguish affiliation at filing time
from current affiliation. Return null when a field cannot be verified. Include
source URLs and field-level evidence. Do not use private people-search sites,
leaked data, or personal social media.
"""

JSON_SYSTEM_PROMPT = """Return exactly one valid JSON object using only the
provided research evidence. Preserve uncertainty and use null for unverified
scalar values. Do not include Markdown or text outside the JSON object."""


class ResearchError(RuntimeError):
    pass


class ResearchRateLimitError(ResearchError):
    pass


class ToolExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    base_url: str
    api_key: str | None
    timeout: float
    max_tokens: int | None
    reasoning_effort: str | None
    temperature: float
    max_concurrent_calls: int
    default_headers: dict[str, str]


class AIClient:
    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self.config = settings if settings is not None else loadAISettings()
        self.primary = self._build_provider_config(self.config)

        fallback = self.config.get("fallback")
        self.fallback = (
            self._build_provider_config(fallback)
            if isinstance(fallback, dict)
            else None
        )

        self._clients: dict[tuple[str, str, str], OpenAI] = {}
        self._provider_blocks: dict[str, tuple[float | None, str]] = {}
        self._research_blocks: dict[str, tuple[float | None, str]] = {}
        self._validation_retries = max(0, int(self.config.get("validation_retries", 1)))

        research = self.config.get("research", {})
        research = research if isinstance(research, dict) else {}
        self._research_max_results = min(
            10, max(1, int(research.get("max_results", 5)))
        )
        self._research_max_tool_calls = min(
            30, max(1, int(research.get("max_tool_calls", 6)))
        )
        self._research_max_chars = max(
            2_000, int(research.get("max_result_chars", 16_000))
        )
        self._research_engine = str(research.get("engine", "auto"))
        self._research_context_size = str(research.get("search_context_size", "medium"))
        self._research_timeout = float(research.get("timeout_seconds", 30))
        self._ollama_web_key_env = str(
            research.get("ollama_api_key_env", "OLLAMA_API_KEY")
        )
        self._force_ollama_initial_search = bool(
            research.get("force_ollama_initial_search", True)
        )

    # Provider setup ----------------------------------------------------

    def _build_provider_config(self, section: Mapping[str, Any]) -> ProviderConfig:
        provider = str(section["provider"]).strip().lower()
        if provider not in {"openrouter", "ollama"}:
            raise ValueError(f"Unsupported provider: {provider}")

        provider_settings = self.config.get(provider, {})
        provider_settings = (
            provider_settings if isinstance(provider_settings, dict) else {}
        )

        if provider == "openrouter":
            base_url = str(
                provider_settings.get("base_url", "https://openrouter.ai/api/v1")
            ).rstrip("/")
            key_env = str(provider_settings.get("api_key_env", "OPENROUTER_API_KEY"))
        else:
            base_url = str(
                provider_settings.get("base_url", "http://localhost:11434/v1")
            ).rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            key_env = str(provider_settings.get("api_key_env", "OLLAMA_API_KEY"))

        headers: dict[str, str] = {}
        if provider == "openrouter":
            if provider_settings.get("http_referer"):
                headers["HTTP-Referer"] = str(provider_settings["http_referer"])
            if provider_settings.get("app_title"):
                headers["X-Title"] = str(provider_settings["app_title"])

        return ProviderConfig(
            provider=provider,
            model=str(section["model"]),
            base_url=base_url,
            api_key=os.getenv(key_env),
            timeout=float(section.get("timeout_seconds", 60)),
            max_tokens=(
                int(section["max_tokens"])
                if section.get("max_tokens") is not None
                else None
            ),
            reasoning_effort=(
                str(section["reasoning_effort"])
                if section.get("reasoning_effort") is not None
                else None
            ),
            temperature=float(section.get("temperature", 0.2)),
            max_concurrent_calls=max(
                1, int(provider_settings.get("max_concurrent_calls", 1))
            ),
            default_headers=headers,
        )

    def _get_client(self, cfg: ProviderConfig) -> OpenAI:
        key = (cfg.provider, cfg.base_url, cfg.api_key or "")
        if key not in self._clients:
            self._clients[key] = OpenAI(
                base_url=cfg.base_url,
                api_key=cfg.api_key or "ollama-local",
                timeout=cfg.timeout,
                default_headers=cfg.default_headers or None,
            )
        return self._clients[key]

    def provider_block_reason(self, provider: str | None = None) -> str | None:
        provider_name = self.primary.provider if provider in {None, "primary"} else provider
        blocked = self._provider_blocks.get(str(provider_name).lower())
        if blocked is None:
            return None
        reset_at, reason = blocked
        if reset_at is not None and time.time() >= reset_at:
            self._provider_blocks.pop(str(provider_name).lower(), None)
            return None
        return reason

    def _record_rate_limit(self, cfg: ProviderConfig, error: RateLimitError) -> None:
        body = error.body if isinstance(error.body, dict) else {}
        error_data = body.get("error", body)
        error_data = error_data if isinstance(error_data, dict) else {}
        metadata = error_data.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        message = str(error_data.get("message") or error)
        if (
            metadata.get("limit_source") != "openrouter_free_tier_daily"
            and "free-models-per-day" not in message
        ):
            return

        headers = metadata.get("headers", {})
        headers = headers if isinstance(headers, dict) else {}
        try:
            reset_at = float(headers["X-RateLimit-Reset"]) / 1000
        except (KeyError, TypeError, ValueError):
            reset_at = None

        reason = "OpenRouter free-model daily quota is exhausted"
        if reset_at is not None:
            reset_text = time.strftime(
                "%Y-%m-%d %H:%M:%S %Z",
                time.localtime(reset_at),
            )
            reason += f" until {reset_text}"
        self._provider_blocks[cfg.provider] = (reset_at, reason)
        logger.warning("%s; further %s requests will be skipped", reason, cfg.provider)

    def research_block_reason(self, provider: str | None = None) -> str | None:
        provider_name = self.primary.provider if provider in {None, "primary"} else provider
        blocked = self._research_blocks.get(str(provider_name).lower())
        if blocked is None:
            return None
        reset_at, reason = blocked
        if reset_at is not None and time.time() >= reset_at:
            self._research_blocks.pop(str(provider_name).lower(), None)
            return None
        return reason

    def _record_ollama_research_limit(self, response: requests.Response) -> str:
        retry_after = response.headers.get("Retry-After")
        try:
            reset_at = time.time() + float(retry_after) if retry_after else None
        except ValueError:
            reset_at = None

        reason = "Ollama web research is rate limited"
        if reset_at is not None:
            reset_text = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(reset_at))
            reason += f" until {reset_text}"
        self._research_blocks["ollama"] = (reset_at, reason)
        logger.warning("%s; further Ollama web research requests will be skipped", reason)
        return reason

    def _providers(
        self,
        provider: str | None,
        allow_fallback: bool,
    ) -> list[ProviderConfig]:
        configured = [self.primary]
        if self.fallback:
            configured.append(self.fallback)

        if provider is None or provider == "primary":
            return configured if allow_fallback else [self.primary]
        if provider == "fallback":
            if not self.fallback:
                raise ValueError("No fallback provider configured")
            return [self.fallback]

        matches = [c for c in configured if c.provider == provider.lower()]
        if not matches:
            raise ValueError(f"Provider {provider!r} is not configured")
        return [matches[0]]

    def _create_completion(
        self,
        cfg: ProviderConfig,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        blocked_reason = self.provider_block_reason(cfg.provider)
        if blocked_reason is not None:
            raise ProviderQuotaError(blocked_reason)

        request = dict(kwargs)
        request.setdefault("model", cfg.model)
        request.setdefault("temperature", cfg.temperature)
        if cfg.max_tokens is not None:
            request.setdefault("max_tokens", cfg.max_tokens)
        # Ollama's OpenAI-compatible endpoint rejects this OpenRouter option.
        if cfg.provider == "openrouter" and cfg.reasoning_effort is not None:
            request.setdefault("reasoning_effort", cfg.reasoning_effort)
        request.update(messages=messages, stream=stream)
        if response_format is not None and cfg.provider == "openrouter":
            request["response_format"] = response_format
        if tools is not None:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice

        logger.debug(
            "Calling %s/%s stream=%s tools=%s",
            cfg.provider,
            cfg.model,
            stream,
            bool(tools),
        )
        try:
            return self._get_client(cfg).chat.completions.create(**request)
        except RateLimitError as error:
            self._record_rate_limit(cfg, error)
            raise

    def _generate(
        self,
        cfg: ProviderConfig,
        prompt: str,
        *,
        messages: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | Generator[str, None, None]:
        response = self._create_completion(
            cfg,
            messages or [{"role": "user", "content": prompt}],
            response_format=response_format,
            stream=stream,
            **kwargs,
        )
        if stream:
            return self._stream_chat(response)
        return response.choices[0].message.content or ""

    @staticmethod
    def _stream_chat(stream: Any) -> Generator[str, None, None]:
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content

    # Normal generation ------------------------------------------------

    def generate_text(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        allow_fallback: bool = True,
        **kwargs: Any,
    ) -> str:
        last_error: Exception | None = None
        providers = self._providers(provider, allow_fallback)

        for index, cfg in enumerate(providers):
            try:
                result = self._generate(cfg, prompt, **kwargs)
                if not isinstance(result, str):
                    raise TypeError("Expected a non-streaming response")
                return result
            except PROVIDER_ERRORS as error:
                last_error = error
                if index + 1 < len(providers):
                    logger.warning(
                        "%s failed (%s: %s); falling back to %s",
                        cfg.provider,
                        error.__class__.__name__,
                        error,
                        providers[index + 1].provider,
                    )

        raise last_error or RuntimeError("No provider completed generation")

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        previous_response = ""

        schema_text = (
            json.dumps(schema, ensure_ascii=False, indent=2)
            if schema is not None
            else None
        )

        working_prompt = prompt

        if schema_text:
            working_prompt += f"""

    Return exactly one JSON object matching this template:

    {schema_text}

    Requirements:
    - Include every key from the template.
    - Preserve the exact key names and capitalization.
    - Use null when a scalar value cannot be determined.
    - Use [] when a list value cannot be determined.
    - Use {{}} when an object value cannot be determined.
    - Do not include Markdown or any text outside the JSON object.
    """

        for attempt in range(self._validation_retries + 1):
            try:
                text = self.generate_text(
                    working_prompt,
                    response_format={"type": "json_object"},
                    **kwargs,
                )

                previous_response = text
                result = self._parse_json(text)

                if schema is not None:
                    result = self._coerce_template(result, schema)
                    self._validate_template(result, schema)

                return result

            except (
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as error:
                last_error = error

                if attempt >= self._validation_retries:
                    break

                logger.warning(
                    "JSON generation attempt %d failed: %s",
                    attempt + 1,
                    error,
                )

                working_prompt = f"""Your previous response failed validation.

    Validation error:
    {error}

    Previous response:
    {previous_response[:8000]}

    Original task:
    {prompt}
    """

                if schema_text:
                    working_prompt += f"""

    Return a corrected JSON object matching this exact template:

    {schema_text}

    Include every key exactly as written. Use null, [], or {{}} for fields that
    cannot be determined. Return only the JSON object.
    """

        raise last_error or RuntimeError("Failed to generate valid JSON")

    def generate_stream(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        allow_fallback: bool = True,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        providers = self._providers(provider, allow_fallback)
        for index, cfg in enumerate(providers):
            emitted = False
            try:
                stream = self._generate(cfg, prompt, stream=True, **kwargs)
                for token in stream:  # type: ignore[union-attr]
                    emitted = True
                    yield token
                return
            except PROVIDER_ERRORS:
                if emitted or index + 1 == len(providers):
                    raise

    def generate_batch(
        self,
        prompts: list[str],
        *,
        provider: str | None = None,
        use_fallback: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        if not prompts:
            return []
        cfg = self._providers(
            "fallback" if use_fallback else provider,
            False,
        )[0]
        results: list[str | None] = [None] * len(prompts)
        with ThreadPoolExecutor(
            max_workers=min(cfg.max_concurrent_calls, len(prompts))
        ) as executor:
            futures = {
                executor.submit(
                    self.generate_text,
                    prompt,
                    provider=cfg.provider,
                    allow_fallback=False,
                    **kwargs,
                ): index
                for index, prompt in enumerate(prompts)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [result or "" for result in results]

    def generate_json_batch(
        self,
        prompts: list[str],
        schema: dict[str, Any] | None = None,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        if not prompts:
            return []
        cfg = self._providers(provider, False)[0]
        results: list[dict[str, Any] | None] = [None] * len(prompts)
        with ThreadPoolExecutor(
            max_workers=min(cfg.max_concurrent_calls, len(prompts))
        ) as executor:
            futures = {
                executor.submit(
                    self.generate_json,
                    prompt,
                    schema,
                    provider=cfg.provider,
                    allow_fallback=False,
                    **kwargs,
                ): index
                for index, prompt in enumerate(prompts)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [result or {} for result in results]

    def chat(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        return self.generate_text("", messages=messages, **kwargs)

    def structured_output(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.generate_json(prompt, output_schema, **kwargs)

    # Function tools for OpenRouter and Ollama -------------------------

    def run_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        handlers: Mapping[str, ToolHandler],
        *,
        provider: str | None = None,
        allow_fallback: bool = True,
        max_tool_calls: int | None = None,
        **kwargs: Any,
    ) -> str:
        last_error: Exception | None = None
        providers = self._providers(provider, allow_fallback)

        for index, cfg in enumerate(providers):
            try:
                return self._tool_loop(
                    cfg,
                    messages,
                    tools,
                    handlers,
                    max_tool_calls=max_tool_calls,
                    **kwargs,
                )
            except (*PROVIDER_ERRORS, ToolExecutionError) as error:
                last_error = error
                if index + 1 < len(providers):
                    logger.warning(
                        "%s tool loop failed (%s: %s); falling back to %s",
                        cfg.provider,
                        error.__class__.__name__,
                        error,
                        providers[index + 1].provider,
                    )
        raise last_error or ToolExecutionError("Tool loop failed")

    def _tool_loop(
        self,
        cfg: ProviderConfig,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        handlers: Mapping[str, ToolHandler],
        *,
        max_tool_calls: int | None = None,
        calls_used: int = 0,
        **kwargs: Any,
    ) -> str:
        conversation = deepcopy(messages)
        limit = min(30, max(1, max_tool_calls or self._research_max_tool_calls))

        while calls_used < limit:
            response = self._create_completion(
                cfg,
                conversation,
                tools=tools,
                tool_choice="auto",
                **kwargs,
            )
            message = response.choices[0].message
            tool_calls = list(message.tool_calls or [])

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
            }
            if tool_calls:
                assistant_message["tool_calls"] = []
                for index, call in enumerate(tool_calls):
                    call_id = call.id or f"call_{calls_used}_{index}"
                    arguments = call.function.arguments or "{}"
                    if isinstance(arguments, dict):
                        arguments = json.dumps(arguments)
                    assistant_message["tool_calls"].append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": arguments,
                            },
                        }
                    )
            conversation.append(assistant_message)

            if not tool_calls:
                content = message.content or ""
                if not content:
                    raise ToolExecutionError("Empty tool-loop response")
                return content

            for index, call in enumerate(tool_calls):
                call_id = call.id or f"call_{calls_used}_{index}"
                handler = handlers.get(call.function.name)
                try:
                    arguments = call.function.arguments or "{}"
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be an object")
                    result = (
                        handler(**arguments)
                        if handler
                        else {"error": f"Unknown tool: {call.function.name}"}
                    )
                except Exception as error:
                    result = {"error": f"{error.__class__.__name__}: {error}"}
                calls_used += 1
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result, ensure_ascii=False, default=str)[
                            : self._research_max_chars
                        ],
                    }
                )

        conversation.append(
            {
                "role": "system",
                "content": (
                    "The tool limit was reached. Return the final answer now "
                    "using the gathered evidence."
                ),
            }
        )
        response = self._create_completion(cfg, conversation, **kwargs)
        content = response.choices[0].message.content or ""
        if not content:
            raise ToolExecutionError("Empty final tool-loop response")
        return content

    # Web research for OpenRouter and Ollama ---------------------------

    def research_text(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        allow_fallback: bool = True,
        search_query: str | None = None,
        **kwargs: Any,
    ) -> str:
        messages = [
            {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        last_error: Exception | None = None
        providers = self._providers(provider, allow_fallback)

        for index, cfg in enumerate(providers):
            try:
                return self._research_provider(
                    cfg,
                    messages,
                    search_query,
                    **kwargs,
                )
            except (*PROVIDER_ERRORS, ResearchError, ToolExecutionError) as error:
                last_error = error
                if index + 1 < len(providers):
                    logger.warning(
                        "%s research failed (%s: %s); falling back to %s",
                        cfg.provider,
                        error.__class__.__name__,
                        error,
                        providers[index + 1].provider,
                    )
        raise last_error or ResearchError("Research failed")

    def _research_provider(
        self,
        cfg: ProviderConfig,
        messages: list[dict[str, Any]],
        search_query: str | None,
        **kwargs: Any,
    ) -> str:
        if cfg.provider == "openrouter":
            return self._research_openrouter(cfg, messages, search_query, **kwargs)
        return self._research_ollama(cfg, messages, search_query, **kwargs)

    def _research_openrouter(
        self,
        cfg: ProviderConfig,
        messages: list[dict[str, Any]],
        search_query: str | None,
        **kwargs: Any,
    ) -> str:
        if not cfg.api_key:
            raise ResearchError("OPENROUTER_API_KEY is missing")

        request_messages = deepcopy(messages)
        if search_query:
            request_messages.insert(
                1,
                {
                    "role": "system",
                    "content": f"Begin with this search query: {search_query}",
                },
            )

        tools = [
            {
                "type": "openrouter:web_search",
                "parameters": {
                    "engine": self._research_engine,
                    "max_results": self._research_max_results,
                    "max_total_results": self._research_max_results * 3,
                    "max_uses": self._research_max_tool_calls,
                    "search_context_size": self._research_context_size,
                },
            },
            {
                "type": "openrouter:web_fetch",
                "parameters": {
                    "engine": "auto",
                    "max_uses": self._research_max_tool_calls,
                    "max_content_tokens": 30_000,
                    "blocked_domains": [
                        "facebook.com",
                        "instagram.com",
                        "tiktok.com",
                        "x.com",
                        "twitter.com",
                    ],
                },
            },
        ]
        extra_body = dict(kwargs.pop("extra_body", {}) or {})
        extra_body["max_tool_calls"] = self._research_max_tool_calls

        response = self._create_completion(
            cfg,
            request_messages,
            tools=tools,
            extra_body=extra_body,
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        if not content:
            raise ResearchError("OpenRouter returned empty research")
        return content

    def _research_ollama(
        self,
        cfg: ProviderConfig,
        messages: list[dict[str, Any]],
        search_query: str | None,
        **kwargs: Any,
    ) -> str:
        blocked_reason = self.research_block_reason(cfg.provider)
        if blocked_reason is not None:
            raise ResearchRateLimitError(blocked_reason)
        self._ollama_web_key()
        conversation = deepcopy(messages)
        calls_used = 0

        if self._force_ollama_initial_search:
            query = search_query or self._default_search_query(messages[-1]["content"])
            result = self._ollama_web_search(query)
            conversation.append(
                {
                    "role": "system",
                    "content": (
                        "Initial web results:\n"
                        + json.dumps(result, ensure_ascii=False, default=str)[
                            : self._research_max_chars
                        ]
                        + "\nUse web_fetch or narrower searches to verify them."
                    ),
                }
            )
            calls_used = 1

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the public web.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": "Fetch and read a public source URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            },
        ]

        try:
            return self._tool_loop(
                cfg,
                conversation,
                tools,
                {
                    "web_search": self._ollama_web_search,
                    "web_fetch": self._ollama_web_fetch,
                },
                calls_used=calls_used,
                **kwargs,
            )
        except ToolExecutionError:
            response = self._create_completion(cfg, conversation, **kwargs)
            content = response.choices[0].message.content or ""
            if not content:
                raise ResearchError("Ollama returned empty research")
            return content

    def _ollama_web_key(self) -> str:
        key = os.getenv(self._ollama_web_key_env, "").strip()
        if not key:
            raise ResearchError(
                f"{self._ollama_web_key_env} is required for Ollama web research"
            )
        return key

    def _ollama_web_search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> dict[str, Any]:
        cleaned = " ".join(str(query).split())
        if not cleaned:
            raise ValueError("Search query cannot be empty")
        return self._ollama_web_api(
            "web_search",
            {
                "query": cleaned[:1_000],
                "max_results": min(
                    10,
                    max(1, int(max_results or self._research_max_results)),
                ),
            },
        )

    def _ollama_web_fetch(self, url: str) -> dict[str, Any]:
        self._validate_public_url(url)
        return self._ollama_web_api("web_fetch", {"url": url})

    def _ollama_web_api(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = requests.post(
                f"https://ollama.com/api/{endpoint}",
                headers={
                    "Authorization": f"Bearer {self._ollama_web_key()}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self._research_timeout,
            )
            if response.status_code == 429:
                raise ResearchRateLimitError(
                    self._record_ollama_research_limit(response)
                )
            response.raise_for_status()
            result = response.json()
        except (requests.RequestException, ValueError) as error:
            raise ResearchError(f"Ollama {endpoint} failed: {error}") from error
        if not isinstance(result, dict):
            raise ResearchError(f"Ollama {endpoint} returned invalid JSON")
        return result

    @staticmethod
    def _validate_public_url(url: str) -> None:
        parsed = urlparse(str(url).strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Expected a valid HTTP or HTTPS URL")
        host = parsed.hostname.lower()
        if host == "localhost" or host.endswith(".local"):
            raise ValueError("Local URLs are not allowed")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return
        if not address.is_global:
            raise ValueError("Private and loopback URLs are not allowed")

    @staticmethod
    def _default_search_query(prompt: Any) -> str:
        return " ".join(str(prompt).split())[:500] or "public records research"

    # Researched structured output -------------------------------------

    def research_json(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        *,
        provider: str | None = None,
        allow_fallback: bool = True,
        search_query: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        providers = self._providers(provider, allow_fallback)
        last_error: Exception | None = None

        for index, cfg in enumerate(providers):
            try:
                evidence = self._research_provider(
                    cfg,
                    messages,
                    search_query,
                    **kwargs,
                )
                return self._format_json(cfg, prompt, evidence, schema)
            except (
                *PROVIDER_ERRORS,
                ResearchError,
                ToolExecutionError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as error:
                last_error = error
                if index + 1 < len(providers):
                    logger.warning(
                        "%s researched JSON failed (%s: %s); falling back to %s",
                        cfg.provider,
                        error.__class__.__name__,
                        error,
                        providers[index + 1].provider,
                    )
        raise last_error or ResearchError("Researched JSON failed")

    def _format_json(
        self,
        cfg: ProviderConfig,
        prompt: str,
        evidence: str,
        schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        template = (
            json.dumps(schema, ensure_ascii=False, indent=2)
            if schema is not None
            else "No fixed template"
        )
        formatting_prompt = f"""Original task:
{prompt}

Research evidence:
{evidence[: self._research_max_chars * 2]}

Required JSON template:
{template}
"""
        last_error: Exception | None = None

        for attempt in range(self._validation_retries + 1):
            try:
                text = self._generate(
                    cfg,
                    "",
                    messages=[
                        {"role": "system", "content": JSON_SYSTEM_PROMPT},
                        {"role": "user", "content": formatting_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                if not isinstance(text, str):
                    raise TypeError("Expected non-streaming JSON")
                result = self._parse_json(text)
                if schema is not None:
                    result = self._coerce_template(result, schema)
                    self._validate_template(result, schema)
                return result
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                last_error = error
                if attempt < self._validation_retries:
                    formatting_prompt += (
                        "\nReturn every template key with the same nested "
                        "object/list structure."
                    )
        raise last_error or ValueError("Could not format researched JSON")

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if fenced:
            cleaned = fenced.group(1)
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                # Local models sometimes emit literal newlines inside strings.
                value = json.loads(cleaned, strict=False)
            except json.JSONDecodeError:
                start = cleaned.find("{")
                if start == -1:
                    raise
                value, _ = json.JSONDecoder(strict=False).raw_decode(cleaned[start:])
        if not isinstance(value, dict):
            raise TypeError("Expected a JSON object")
        return value

    @classmethod
    def _coerce_template(cls, value: Any, template: Any) -> Any:
        """Normalize common scalar-for-list model responses before validation."""
        if isinstance(template, dict) and isinstance(value, dict):
            normalized = dict(value)
            for key, child in template.items():
                if key in normalized:
                    normalized[key] = cls._coerce_template(normalized[key], child)
            return normalized
        if isinstance(template, list) and not isinstance(value, list):
            return [] if value is None else [value]
        return value

    @classmethod
    def _validate_template(
        cls,
        value: Any,
        template: Any,
        path: str = "$",
    ) -> None:
        if isinstance(template, dict):
            if not isinstance(value, dict):
                raise ValueError(f"{path} must be an object")
            missing = [key for key in template if key not in value]
            if missing:
                raise ValueError(f"{path} missing keys: {', '.join(missing)}")
            for key, child in template.items():
                cls._validate_template(value[key], child, f"{path}.{key}")
        elif isinstance(template, list) and not isinstance(value, list):
            raise ValueError(f"{path} must be a list")

    # Comment enrichment ------------------------------------------------

    def research_person_contact(
        self,
        person_name: str,
        organization: str | None = None,
        context: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        schema = {
            "name": None,
            "organization": None,
            "title": None,
            "professional_email": None,
            "professional_phone": None,
            "organization_website": None,
            "professional_profile": None,
            "evidence": {},
            "sources": [],
            "confidence": {},
            "notes": None,
        }
        prompt = f"""Research this person's public professional contact details.
Confirm every result refers to the same person.

Person: {person_name}
Organization hint: {organization or "unknown"}
Context: {context or "none"}

Never infer an email from an organization's email pattern. Do not return home
addresses, personal phone numbers, or unrelated personal accounts.
"""
        search_query = " ".join(
            part
            for part in [
                f'"{person_name}"',
                f'"{organization}"' if organization else "",
                "professional contact",
            ]
            if part
        )
        return self.research_json(
            prompt,
            schema,
            search_query=search_query,
            **kwargs,
        )

    def research_comment_fields(
        self,
        comment: dict[str, Any],
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        output_schema = schema or DEFAULT_COMMENT_RESEARCH_SCHEMA
        record = json.dumps(
            comment,
            ensure_ascii=False,
            default=str,
            indent=2,
        )[:20_000]
        prompt = f"""Enrich missing or uncertain metadata for this public
regulatory comment.

Comment record:
{record}

Identify the submitter using filing context. For organization type, nonprofit
status, NTEE, organization role, and professional contact information, include
field-level evidence and source URLs. Prefer evidence true at submission time.
Never guess.
"""
        return self.research_json(
            prompt,
            output_schema,
            search_query=self._comment_search_query(comment),
            **kwargs,
        )

    @staticmethod
    def _comment_search_query(comment: Mapping[str, Any]) -> str:
        keys = (
            "Submitted By",
            "submittedBy",
            "submitterName",
            "firstName",
            "lastName",
            "Organization Name",
            "organizationName",
            "docketId",
            "documentId",
            "commentId",
            "id",
            "title",
        )
        values = [
            str(comment[key]).strip()
            for key in keys
            if comment.get(key) not in (None, "", [], {})
        ]
        return " ".join(dict.fromkeys(values))[:700] or (
            "regulations.gov public comment submitter organization"
        )

    def enrich_comment(
        self,
        comment: dict[str, Any],
        schema: dict[str, Any] | None = None,
        *,
        overwrite_existing: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        researched = self.research_comment_fields(comment, schema, **kwargs)
        if overwrite_existing:
            return self._deep_merge(comment, researched)
        return self._fill_missing(comment, researched)

    @classmethod
    def _fill_missing(cls, original: Any, researched: Any) -> Any:
        if isinstance(original, dict) and isinstance(researched, dict):
            merged = deepcopy(original)
            for key, value in researched.items():
                merged[key] = (
                    cls._fill_missing(merged[key], value)
                    if key in merged
                    else deepcopy(value)
                )
            return merged
        return deepcopy(researched) if cls._is_missing(original) else original

    @classmethod
    def _deep_merge(cls, original: Any, researched: Any) -> Any:
        if isinstance(original, dict) and isinstance(researched, dict):
            merged = deepcopy(original)
            for key, value in researched.items():
                merged[key] = (
                    cls._deep_merge(merged[key], value)
                    if key in merged
                    else deepcopy(value)
                )
            return merged
        return deepcopy(researched)

    @staticmethod
    def _is_missing(value: Any) -> bool:
        return value is None or value == "" or value == [] or value == {}


_client: AIClient | None = None


def get_client(settings: dict[str, Any] | None = None) -> AIClient:
    global _client
    if settings is not None:
        return AIClient(settings)
    if _client is None:
        _client = AIClient()
    return _client


def reset_client() -> None:
    global _client
    _client = None


def generate_text(prompt: str, **kwargs: Any) -> str:
    return get_client().generate_text(prompt, **kwargs)


def generate_json(
    prompt: str,
    schema: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return get_client().generate_json(prompt, schema, **kwargs)


def generate_stream(prompt: str, **kwargs: Any) -> Generator[str, None, None]:
    yield from get_client().generate_stream(prompt, **kwargs)


def generate_batch(prompts: list[str], **kwargs: Any) -> list[str]:
    return get_client().generate_batch(prompts, **kwargs)


def generate_json_batch(
    prompts: list[str],
    schema: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return get_client().generate_json_batch(prompts, schema, **kwargs)


def chat(messages: list[dict[str, Any]], **kwargs: Any) -> str:
    return get_client().chat(messages, **kwargs)


def structured_output(
    prompt: str,
    output_schema: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    return get_client().structured_output(prompt, output_schema, **kwargs)


def run_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    handlers: Mapping[str, ToolHandler],
    **kwargs: Any,
) -> str:
    return get_client().run_tools(messages, tools, handlers, **kwargs)


def research_text(prompt: str, **kwargs: Any) -> str:
    return get_client().research_text(prompt, **kwargs)


def research_json(
    prompt: str,
    schema: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return get_client().research_json(prompt, schema, **kwargs)


def research_person_contact(
    person_name: str,
    organization: str | None = None,
    context: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return get_client().research_person_contact(
        person_name,
        organization,
        context,
        **kwargs,
    )


def research_comment_fields(
    comment: dict[str, Any],
    schema: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return get_client().research_comment_fields(comment, schema, **kwargs)


def enrich_comment(
    comment: dict[str, Any],
    schema: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return get_client().enrich_comment(comment, schema, **kwargs)
