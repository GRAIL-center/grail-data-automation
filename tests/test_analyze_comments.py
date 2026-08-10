import logging

import pytest

from src.collect_comments import analyze_comments


def test_analyze_metadata_preserves_comment_when_ai_fails(monkeypatch, caplog):
    comment = analyze_comments.initComment()
    metadata = {"id": "comment-1", "organization": "Source Organization"}
    analyze_comments.fillMetadata(metadata, comment)

    def unavailable(*args, **kwargs):
        raise ValueError("quota exhausted")

    monkeypatch.setattr(analyze_comments, "generate_json", unavailable)

    with caplog.at_level(logging.WARNING):
        result = analyze_comments.analyzeMetadata(metadata, comment)

    assert result is comment
    assert result.organization_name == "Source Organization"
    assert "quota exhausted" in caplog.text


def test_analyze_metadata_skips_ai_while_provider_is_blocked(monkeypatch, caplog):
    comment = analyze_comments.initComment()
    metadata = {"id": "comment-2", "organization": "Source Organization"}
    analyze_comments.fillMetadata(metadata, comment)
    client = type(
        "BlockedClient",
        (),
        {"provider_block_reason": lambda self: "daily quota exhausted"},
    )()

    monkeypatch.setattr(analyze_comments, "get_client", lambda: client)
    monkeypatch.setattr(
        analyze_comments,
        "generate_json",
        lambda *args, **kwargs: pytest.fail("AI should not be called"),
    )

    with caplog.at_level(logging.INFO):
        result = analyze_comments.analyzeMetadata(metadata, comment)

    assert result.organization_name == "Source Organization"
    assert "Skipping AI metadata enrichment" in caplog.text
