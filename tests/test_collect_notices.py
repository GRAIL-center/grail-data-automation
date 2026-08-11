from unittest.mock import MagicMock

from src.collect_notices import collect


def test_count_comments_sums_document_comment_counts(monkeypatch):
    session = MagicMock()
    session.__enter__.return_value = session
    session.__exit__.return_value = False
    monkeypatch.setattr(collect, "buildSession", lambda api_key: session)
    monkeypatch.setattr(collect, "loadRegKey", lambda: "test-key")
    responses = iter(
        [
            {
                "data": [
                    {"attributes": {"objectId": "object-1"}},
                    {"attributes": {"objectId": "object-2"}},
                    {"attributes": {"objectId": "object-1"}},
                ]
            },
            {"meta": {"totalElements": 3}},
            {"meta": {"totalElements": 7}},
        ]
    )
    monkeypatch.setattr(collect, "requestJSON", lambda *args, **kwargs: next(responses))

    assert collect.countComments("2024-22400") == 10


def test_process_notice_appends_comment_count(monkeypatch):
    response = MagicMock()
    response.json.return_value = {"comments_close_on": "2024-11-01"}
    monkeypatch.setattr(collect.req, "get", lambda *args, **kwargs: response)
    monkeypatch.setattr(collect, "getBodyInfo", lambda doc_num: ("Test Agency", "body"))
    monkeypatch.setattr(collect, "generate_text", lambda prompt: "Summary")
    monkeypatch.setattr(collect, "countComments", lambda doc_num: 12)

    row = collect.processNotice(
        {
            "document_number": "2024-22400",
            "publication_date": "2024-10-01",
            "title": "Test notice",
            "type": "NOTICE",
        }
    )

    assert row is not None
    assert len(row) == 10
    assert row[-1] == 12


def test_get_body_info_uses_metadata_agency_when_html_has_no_agency(monkeypatch):
    document_response = MagicMock()
    document_response.json.return_value = {
        "body_html_url": "https://example.test/document.html",
        "agencies": [{"name": "Executive Office of the President"}],
    }
    body_response = MagicMock(status_code=200, text="<html><body>Text</body></html>")
    monkeypatch.setattr(
        collect.req,
        "get",
        MagicMock(side_effect=[document_response, body_response]),
    )

    agency, body_text = collect.getBodyInfo("2021-01646")

    assert agency == "Executive Office of the President"
    assert body_text == "<html><body>Text</body></html>"
