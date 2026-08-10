from src.collect_comments import process_comment_text


def test_process_comment_text_skips_ai_while_provider_is_blocked(monkeypatch):
    client = type(
        "BlockedClient",
        (),
        {
            "provider_block_reason": lambda self, provider=None: "daily quota exhausted",
            "research_block_reason": lambda self, provider=None: None,
        },
    )()

    def unexpected_call(*args, **kwargs):
        raise AssertionError("AI should not be called while its provider is blocked")

    monkeypatch.setattr(process_comment_text, "get_client", lambda: client)
    monkeypatch.setattr(process_comment_text, "generate_json", unexpected_call)
    monkeypatch.setattr(process_comment_text, "research_comment_fields", unexpected_call)

    result = process_comment_text.processCommentText(
        "Source comment text",
        {"id": "comment-1", "organization": "Source Organization"},
    )

    assert result["Organization Name"] == "Source Organization"
    assert result["Full Text"] == "Source comment text"
    assert "AI text analysis was unavailable." in result["Research Notes"]
    assert "AI research enrichment was unavailable." in result["Research Notes"]


def test_process_comment_text_skips_rate_limited_research(monkeypatch):
    client = type(
        "ResearchBlockedClient",
        (),
        {
            "provider_block_reason": lambda self, provider=None: None,
            "research_block_reason": lambda self, provider=None: "rate limited",
        },
    )()

    def unexpected_research(*args, **kwargs):
        raise AssertionError("Research should not run while web search is rate limited")

    monkeypatch.setattr(process_comment_text, "get_client", lambda: client)
    monkeypatch.setattr(
        process_comment_text,
        "generate_json",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        process_comment_text,
        "research_comment_fields",
        unexpected_research,
    )

    result = process_comment_text.processCommentText("Source comment text")

    assert "AI research enrichment was unavailable." in result["Research Notes"]
