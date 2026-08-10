"""Tests for src.services.ai_client."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIError

from src.services.ai_client import (
    AIClient,
    generate_json,
    generate_text,
)

OPENROUTER_SETTINGS = {
    "provider": "openrouter",
    "model": "openai/gpt-4.1-mini",
    "timeout_seconds": 60,
    "max_tokens": 4096,
    "reasoning_effort": "low",
    "temperature": 0.2,
    "validation_retries": 1,
    "fallback": {
        "provider": "ollama",
        "model": "llama3.1:latest",
        "timeout_seconds": 90,
    },
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "max_concurrent_calls": 8,
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "max_concurrent_calls": 1,
    },
}


def _mock_completion(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _mock_stream_chat(contents: list[str]) -> list[MagicMock]:
    chunks = []
    for text in contents:
        chunk = MagicMock()
        delta = MagicMock()
        delta.content = text
        chunk.choices = [MagicMock(delta=delta)]
        chunks.append(chunk)
    return chunks


_mock_req = httpx.Request("GET", "http://test")


@patch("src.services.ai_client.OpenAI")
def test_build_provider_config_openrouter(mock_openai):
    client = AIClient(OPENROUTER_SETTINGS)
    cfg = client.primary
    assert cfg.provider == "openrouter"
    assert cfg.model == "openai/gpt-4.1-mini"
    assert cfg.base_url == "https://openrouter.ai/api/v1"
    assert cfg.temperature == 0.2
    assert cfg.max_tokens == 4096
    assert cfg.reasoning_effort == "low"
    assert cfg.max_concurrent_calls == 8


@patch("src.services.ai_client.OpenAI")
def test_build_provider_config_ollama(mock_openai):
    client = AIClient(OPENROUTER_SETTINGS)
    cfg = client.fallback
    assert cfg.provider == "ollama"
    assert cfg.model == "llama3.1:latest"
    assert cfg.base_url == "http://localhost:11434/v1"
    assert cfg.timeout == 90


@patch("src.services.ai_client.OpenAI")
def test_generate_text_success(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value = _mock_completion(
        "hello world"
    )
    client = AIClient(OPENROUTER_SETTINGS)
    result = client.generate_text("say hello")
    assert result == "hello world"


@patch("src.services.ai_client.OpenAI")
def test_generate_text_fallback(mock_openai, caplog):
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise APIError("primary failed", _mock_req, body=None)
        return _mock_completion("fallback response")

    mock_openai.return_value.chat.completions.create.side_effect = side_effect
    client = AIClient(OPENROUTER_SETTINGS)
    result = client.generate_text("test")
    assert result == "fallback response"
    assert "APIError: primary failed" in caplog.text


@patch("src.services.ai_client.OpenAI")
def test_generate_text_raises_when_no_fallback(mock_openai):
    mock_openai.return_value.chat.completions.create.side_effect = APIError(
        "failed", _mock_req, body=None
    )
    no_fallback = {
        "provider": "openrouter",
        "model": "openai/gpt-4.1-mini",
        "timeout_seconds": 60,
        "temperature": 0.2,
        "openrouter": {"base_url": "https://openrouter.ai/api/v1"},
    }
    client = AIClient(no_fallback)
    with pytest.raises(APIError):
        client.generate_text("test")


@patch("src.services.ai_client.OpenAI")
def test_generate_json_success(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value = _mock_completion(
        '{"key": "value"}'
    )
    client = AIClient(OPENROUTER_SETTINGS)
    result = client.generate_json("extract json")
    assert result == {"key": "value"}


@patch("src.services.ai_client.OpenAI")
def test_generate_json_with_schema(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value = _mock_completion(
        '{"name": "test", "active": true}'
    )
    client = AIClient(OPENROUTER_SETTINGS)
    schema = {"name": None, "active": None}
    result = client.generate_json("extract json", schema=schema)
    assert result == {"name": "test", "active": True}


@patch("src.services.ai_client.OpenAI")
def test_generate_json_schema_validation_failure(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value = _mock_completion(
        '{"wrong": "data"}'
    )
    client = AIClient(OPENROUTER_SETTINGS)
    schema = {"required_key": None}
    with pytest.raises(ValueError, match=r"\$ missing keys: required_key"):
        client.generate_json("extract json", schema=schema)


@patch("src.services.ai_client.OpenAI")
def test_generate_json_retries_on_decode_error(mock_openai):
    responses = [
        _mock_completion("not json"),
        _mock_completion('{"valid": true}'),
    ]
    mock_openai.return_value.chat.completions.create.side_effect = responses
    client = AIClient(OPENROUTER_SETTINGS)
    result = client.generate_json("extract json")
    assert result == {"valid": True}


@patch("src.services.ai_client.OpenAI")
def test_generate_json_raises_after_retries(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value = _mock_completion(
        "not json at all"
    )
    client = AIClient(OPENROUTER_SETTINGS)
    with pytest.raises(json.JSONDecodeError):
        client.generate_json("extract json")


@patch("src.services.ai_client.OpenAI")
def test_generate_stream(mock_openai):
    chunks = _mock_stream_chat(["hello", " ", "world"])
    mock_openai.return_value.chat.completions.create.return_value = chunks
    client = AIClient(OPENROUTER_SETTINGS)
    result = "".join(client.generate_stream("say hello"))
    assert result == "hello world"


@patch("src.services.ai_client.OpenAI")
def test_generate_stream_fallback(mock_openai):
    call_count = 0

    def fake_generate(cfg, prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise APIError("primary stream failed", _mock_req, body=None)

        def _gen():
            yield "fallback"
            yield " "
            yield "result"

        return _gen()

    client = AIClient(OPENROUTER_SETTINGS)
    with patch.object(client, "_generate", side_effect=fake_generate):
        result = "".join(client.generate_stream("test"))
    assert result == "fallback result"


@patch("src.services.ai_client.OpenAI")
def test_generate_batch(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value = _mock_completion(
        "batch response"
    )
    client = AIClient(OPENROUTER_SETTINGS)
    results = client.generate_batch(["prompt1", "prompt2"])
    assert results == ["batch response", "batch response"]


@patch("src.services.ai_client.OpenAI")
def test_generate_json_batch(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value = _mock_completion(
        '{"id": 42}'
    )
    client = AIClient(OPENROUTER_SETTINGS)
    results = client.generate_json_batch(["q1", "q2"])
    assert results == [{"id": 42}, {"id": 42}]


@patch("src.services.ai_client.OpenAI")
def test_chat(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value = _mock_completion(
        "hello there"
    )
    client = AIClient(OPENROUTER_SETTINGS)
    result = client.chat(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    )
    assert result == "hello there"


@patch("src.services.ai_client.OpenAI")
def test_structured_output(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value = _mock_completion(
        '{"field": "val"}'
    )
    client = AIClient(OPENROUTER_SETTINGS)
    output_schema = {"field": None}
    result = client.structured_output("extract", output_schema)
    assert result == {"field": "val"}


@patch("src.services.ai_client.OpenAI")
def test_ollama_skips_response_format(mock_openai):
    mock_chat = mock_openai.return_value.chat.completions.create
    mock_chat.return_value = _mock_completion('{"ok": true}')
    client = AIClient(OPENROUTER_SETTINGS)
    ollama_cfg = client.fallback
    client._generate(
        ollama_cfg, '{"prompt": "test"}', response_format={"type": "json_object"}
    )
    call_kwargs = mock_chat.call_args.kwargs
    assert "response_format" not in call_kwargs


@patch("src.services.ai_client.OpenAI")
def test_openrouter_passes_response_format(mock_openai):
    mock_chat = mock_openai.return_value.chat.completions.create
    mock_chat.return_value = _mock_completion('{"ok": true}')
    client = AIClient(OPENROUTER_SETTINGS)
    or_cfg = client.primary
    client._generate(
        or_cfg, '{"prompt": "test"}', response_format={"type": "json_object"}
    )
    call_kwargs = mock_chat.call_args.kwargs
    assert "response_format" in call_kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}
    assert call_kwargs["max_tokens"] == 4096
    assert call_kwargs["reasoning_effort"] == "low"


@patch("src.services.ai_client.OpenAI")
def test_convenience_functions(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value = _mock_completion(
        "convenience text"
    )
    result_text = generate_text("test prompt")
    assert result_text == "convenience text"

    mock_openai.return_value.chat.completions.create.return_value = _mock_completion(
        '{"k": "v"}'
    )
    result_json = generate_json("test prompt")
    assert result_json == {"k": "v"}
