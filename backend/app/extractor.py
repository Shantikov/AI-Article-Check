import hashlib
import json
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from .models import ExtractedArticle


_SPACE_RE = re.compile(r"[\t\r\f\v ]+")
_BLANK_RE = re.compile(r"\n{3,}")
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)
_NOISE_HINT_RE = re.compile(
    r"(?:^|[-_\s])(?:nav|menu|footer|header|sidebar|toc|infobox|metadata|"
    r"reference|reflist|citation|caption|breadcrumb|related|share|social|"
    r"comment|advert|cookie|newsletter)(?:$|[-_\s])",
    re.I,
)

_NOISE_SELECTORS = (
    "table",
    "figure",
    "figcaption",
    "sup.reference",
    "[role='navigation']",
    "[aria-hidden='true']",
    ".mw-editsection",
    ".mw-jump-link",
    ".mw-references-wrap",
    ".reflist",
    ".references",
    ".navbox",
    ".vertical-navbox",
    ".infobox",
    ".sidebar",
    ".metadata",
    ".catlinks",
    ".printfooter",
)

_BLOCKED_VISIBLE_RE = re.compile(
    r"access denied|verify (?:that )?you are human|security check|"
    r"unusual traffic|captcha|cloudflare ray id|bot detection",
    re.I,
)
_BLOCKED_MARKER_RE = re.compile(
    r"cf-chl-|challenge-platform|g-recaptcha|hcaptcha",
    re.I,
)
_BLOCKED_HEADING_RE = re.compile(
    r"^(?:access denied|security check|verify (?:that )?you are human|"
    r"just a moment\.{0,3}|attention required!?|unusual traffic)"
    r"(?:\s*[-|–—]\s*.{1,80})?$",
    re.I,
)
_RESTRICTED_RE = re.compile(
    r"subscribe to (?:continue|read)|sign in to (?:continue|read)|"
    r"register to (?:continue|read)|this article is for subscribers|paywall",
    re.I,
)
_RESTRICTED_HEADING_RE = re.compile(
    r"^(?:subscribe|subscription required|sign in to (?:continue|read)|"
    r"register to (?:continue|read))"
    r"(?:\s*[-|–—]\s*.{1,80})?$",
    re.I,
)
_JAVASCRIPT_RE = re.compile(
    r"enable javascript|javascript is required|requires javascript|"
    r"turn on javascript|please enable js",
    re.I,
)
_APP_ROOT_RE = re.compile(
    r"id=[\"'](?:__next|__nuxt|root|app)[\"']|data-reactroot|ng-version=",
    re.I,
)


@dataclass(frozen=True)
class ExtractionFailure:
    code: str
    message: str
    retryable: bool = False


def _clean_text(value: str) -> str:
    lines = []
    for raw_line in value.splitlines():
        line = _SPACE_RE.sub(" ", raw_line).strip()
        if line:
            lines.append(line)
    return _BLANK_RE.sub("\n\n", "\n".join(lines)).strip()


def _has_noise_hint(node) -> bool:
    current = node
    while current is not None:
        identifier = str(current.get("id", "")) if hasattr(current, "get") else ""
        classes = " ".join(current.get("class", [])) if hasattr(current, "get") else ""
        if _NOISE_HINT_RE.search(f"{identifier} {classes}"):
            return True
        current = getattr(current, "parent", None)
    return False


def _is_prose_block(node, text: str) -> bool:
    words = _WORD_RE.findall(text)
    if len(text) < 80 or len(words) < 12 or _has_noise_hint(node):
        return False
    link_text = sum(
        len(_clean_text(link.get_text(" ", strip=True)))
        for link in node.find_all("a")
    )
    return link_text / max(1, len(text)) <= 0.45


def _json_ld_article_bodies(soup: BeautifulSoup) -> list[str]:
    bodies: list[str] = []

    def visit(value) -> None:
        if isinstance(value, dict):
            body = value.get("articleBody")
            if isinstance(body, str):
                cleaned = _clean_text(
                    BeautifulSoup(body, "html.parser").get_text(" ", strip=True)
                )
                if cleaned:
                    bodies.append(cleaned)
            for nested in value.values():
                if isinstance(nested, (dict, list)):
                    visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            visit(json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return bodies


def content_fingerprint(text: str) -> str:
    normalized = " ".join(text.split()).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()[:16]


def article_from_text(
    text: str,
    *,
    title: str | None = None,
    has_author: bool = False,
    has_citations: bool = False,
    max_characters: int = 80_000,
) -> ExtractedArticle:
    cleaned = _clean_text(text)[:max_characters]
    return ExtractedArticle(
        title=_clean_text(title) if title else None,
        text=cleaned,
        word_count=len(_WORD_RE.findall(cleaned)),
        has_author=has_author,
        has_citations=has_citations,
    )


def _visible_page_signals(html: str) -> tuple[list[str], str]:
    """Return visible headings and text without script/configuration contents."""
    soup = BeautifulSoup(html, "html.parser")
    headings = []
    for node in [soup.title, *soup.find_all("h1", limit=2)]:
        if node is None:
            continue
        value = _clean_text(node.get_text(" ", strip=True))
        if value:
            headings.append(value)

    for tag in soup(["script", "style", "noscript", "template", "svg", "canvas"]):
        tag.decompose()
    visible_text = _clean_text(soup.get_text(" ", strip=True))
    return headings, visible_text


def diagnose_extraction_failure(
    html: str,
    *,
    word_count: int,
) -> ExtractionFailure | None:
    headings, visible_text = _visible_page_signals(html)
    blocked_heading = any(_BLOCKED_HEADING_RE.fullmatch(item) for item in headings)
    restricted_heading = any(
        _RESTRICTED_HEADING_RE.fullmatch(item) for item in headings
    )
    short_page = word_count < 80

    if blocked_heading or (
        short_page
        and (
            _BLOCKED_VISIBLE_RE.search(visible_text)
            or _BLOCKED_MARKER_RE.search(html)
        )
    ):
        return ExtractionFailure(
            code="access_blocked",
            message="The website returned an access-check page instead of the article",
        )
    if restricted_heading or (short_page and _RESTRICTED_RE.search(visible_text)):
        return ExtractionFailure(
            code="restricted_content",
            message="The article is behind a sign-in or subscription",
        )
    if not short_page:
        return None
    script_count = len(re.findall(r"<script\b", html, re.I))
    if _JAVASCRIPT_RE.search(visible_text) or (
        script_count >= 2 and _APP_ROOT_RE.search(html)
    ):
        return ExtractionFailure(
            code="javascript_required",
            message="The article requires JavaScript to load",
            retryable=True,
        )
    return ExtractionFailure(
        code="too_little_text",
        message="The page does not contain enough article text",
        retryable=True,
    )


def extract_article(html: str, max_characters: int = 80_000) -> ExtractedArticle:
    soup = BeautifulSoup(html, "html.parser")
    json_ld_bodies = _json_ld_article_bodies(soup)
    title_tag = soup.find("meta", property="og:title")
    title = None
    if title_tag and title_tag.get("content"):
        title = _clean_text(str(title_tag["content"]))
    elif soup.title and soup.title.string:
        title = _clean_text(soup.title.string)

    has_author = bool(
        soup.find("meta", attrs={"name": re.compile(r"^author$", re.I)})
        or soup.find(attrs={"rel": re.compile(r"author", re.I)})
        or soup.find(class_=re.compile(r"author|byline", re.I))
    )
    has_citations = bool(
        soup.select_one("sup.reference, .reflist, .references, [role='doc-bibliography']")
        or (
            soup.find("a", href=re.compile(r"^https?://"))
            and soup.find(string=re.compile(r"sources?|references?|источник", re.I))
        )
    )

    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "canvas",
            "template",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "button",
        ]
    ):
        tag.decompose()

    for selector in _NOISE_SELECTORS:
        for tag in soup.select(selector):
            if tag.parent is not None:
                tag.decompose()

    wikipedia_root = soup.select_one("#mw-content-text .mw-parser-output")
    candidates = soup.find_all(["article", "main"])
    root = wikipedia_root or (
        max(candidates, key=lambda node: len(node.get_text(" ", strip=True)))
        if candidates
        else soup.body or soup
    )

    blocks: list[str] = []
    for node in root.find_all(["p", "blockquote"]):
        text = _clean_text(node.get_text(" ", strip=True))
        if _is_prose_block(node, text):
            blocks.append(text)

    if blocks:
        text = "\n".join(dict.fromkeys(blocks))
    else:
        fallback_blocks = []
        for node in root.find_all(["p", "blockquote"]):
            text = _clean_text(node.get_text(" ", strip=True))
            if len(text) >= 20 and not _has_noise_hint(node):
                fallback_blocks.append(text)
        text = (
            "\n".join(dict.fromkeys(fallback_blocks))
            if fallback_blocks
            else _clean_text(root.get_text("\n", strip=True))
        )

    if len(_WORD_RE.findall(text)) < 80 and json_ld_bodies:
        json_ld_text = max(json_ld_bodies, key=lambda value: len(_WORD_RE.findall(value)))
        if len(_WORD_RE.findall(json_ld_text)) > len(_WORD_RE.findall(text)):
            text = json_ld_text

    text = text[:max_characters]
    words = _WORD_RE.findall(text)
    return ExtractedArticle(
        title=title,
        text=text,
        word_count=len(words),
        has_author=has_author,
        has_citations=has_citations,
    )
