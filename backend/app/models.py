from typing import Literal

from pydantic import BaseModel, Field, field_validator


class BatchAnalyzeRequest(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=10)
    force: bool = False

    @field_validator("urls")
    @classmethod
    def deduplicate_urls(cls, urls: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for url in urls:
            value = url.strip()
            if value and value not in seen:
                result.append(value)
                seen.add(value)
        if not result:
            raise ValueError("At least one URL is required")
        return result


class Evidence(BaseModel):
    kind: Literal["strong", "weak", "human", "info"]
    message: str


class AnalysisResult(BaseModel):
    url: str
    final_url: str | None = None
    status: Literal["ok", "error"]
    label: Literal[
        "ai_likely",
        "uncertain",
        "human_likely",
        "unsupported",
        "unavailable",
    ]
    word_count: int = 0
    sampled_word_count: int = Field(default=0, ge=0)
    segments_checked: int = Field(default=0, ge=0)
    ai_segments: int = Field(default=0, ge=0)
    non_ai_segments: int = Field(default=0, ge=0)
    ai_probability: float | None = Field(default=None, ge=0, le=1)
    title: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    error: str | None = None
    error_code: str | None = None
    retryable: bool = False
    content_fingerprint: str | None = Field(default=None, min_length=16, max_length=64)
    content_truncated: bool = False
    analysis_source: Literal["backend_fetch", "browser_page"] = "backend_fetch"
    analyzed_at: str
    cache_hit: bool = False


class BatchAnalyzeResponse(BaseModel):
    results: list[AnalysisResult]


class AnalyzeTextRequest(BaseModel):
    url: str = Field(min_length=1, max_length=4_096)
    canonical_url: str | None = Field(default=None, max_length=4_096)
    title: str | None = Field(default=None, max_length=500)
    text: str = Field(min_length=1, max_length=80_000)
    has_author: bool = False
    has_citations: bool = False

    @field_validator("url", "text")
    @classmethod
    def strip_required_text_fields(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("The field must not be blank")
        return cleaned

    @field_validator("canonical_url", "title")
    @classmethod
    def strip_optional_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class ExtractedArticle(BaseModel):
    title: str | None
    text: str
    word_count: int
    has_author: bool = False
    has_citations: bool = False


class DetectorOutput(BaseModel):
    label: Literal["ai_likely", "uncertain", "human_likely"]
    evidence: list[Evidence]
    sampled_word_count: int = Field(default=0, ge=0)
    segments_checked: int = Field(default=0, ge=0)
    ai_segments: int = Field(default=0, ge=0)
    non_ai_segments: int = Field(default=0, ge=0)
    ai_probability: float | None = Field(default=None, ge=0, le=1)
