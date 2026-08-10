from unittest.mock import MagicMock

from src.collect_comments import fetch_comments


def test_get_metadata_uses_retrying_session(monkeypatch):
    session = MagicMock()
    session.__enter__.return_value = session
    session.__exit__.return_value = False
    response = MagicMock()
    response.json.return_value = {
        "data": {
            "id": "comment-1",
            "attributes": {"organization": "Test Organization"},
        }
    }
    session.get.return_value = response
    build_session = MagicMock(return_value=session)

    monkeypatch.setattr(fetch_comments, "buildSession", build_session)
    monkeypatch.setattr(fetch_comments, "loadRegKey", lambda: "test-key")

    result = fetch_comments.getMetadata("comment-1")

    build_session.assert_called_once_with("test-key")
    session.get.assert_called_once()
    response.raise_for_status.assert_called_once()
    assert result["organization"] == "Test Organization"


def test_regulations_session_retries_transient_failures():
    session = fetch_comments.buildSession("test-key")
    retries = session.get_adapter("https://").max_retries

    try:
        assert retries.total == 5
        assert retries.connect == 5
        assert retries.read == 5
        assert retries.status == 5
        assert {502, 504}.issubset(retries.status_forcelist)
    finally:
        session.close()
