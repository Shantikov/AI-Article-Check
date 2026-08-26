import pytest
from pydantic import ValidationError

from app.models import AnalyzeTextRequest, BatchAnalyzeRequest


def test_batch_request_supports_forced_recheck() -> None:
    normal = BatchAnalyzeRequest(urls=["https://example.com/article"])
    forced = BatchAnalyzeRequest(
        urls=["https://example.com/article"],
        force=True,
    )

    assert normal.force is False
    assert forced.force is True


def test_browser_text_request_strips_fields_and_rejects_blank_text() -> None:
    request = AnalyzeTextRequest(
        url=" https://example.com/story ",
        title=" Example story ",
        text=" rendered article text ",
    )

    assert request.url == "https://example.com/story"
    assert request.title == "Example story"
    assert request.text == "rendered article text"

    with pytest.raises(ValidationError):
        AnalyzeTextRequest(url="https://example.com/story", text="   ")
