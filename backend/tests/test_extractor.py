import json

from app.extractor import (
    article_from_text,
    content_fingerprint,
    diagnose_extraction_failure,
    extract_article,
)


def test_extracts_article_and_ignores_navigation() -> None:
    html = """
    <html>
      <head><title>Example story</title><meta name="author" content="Ada"></head>
      <body>
        <nav>This navigation text should not be part of the story.</nav>
        <article>
          <h1>Example story</h1>
          <p>This is the first sufficiently long paragraph in the article body.</p>
          <p>This is the second sufficiently long paragraph with useful details.</p>
        </article>
      </body>
    </html>
    """
    result = extract_article(html)
    assert result.title == "Example story"
    assert result.has_author is True
    assert "navigation" not in result.text
    assert "first sufficiently long paragraph" in result.text


def test_wikipedia_layout_keeps_prose_and_removes_interface_noise() -> None:
    html = """
    <html>
      <head><title>Example - Wikipedia</title></head>
      <body>
        <main>
          <div id="mw-content-text">
            <div class="mw-parser-output">
              <table class="infobox"><tr><td>Infobox navigation and labels</td></tr></table>
              <div id="toc"><ul><li>History</li><li>References</li></ul></div>
              <p>This encyclopedia article contains a long factual introduction written
              as ordinary prose with enough words to be selected as article content.</p>
              <p>The second paragraph explains the subject in additional detail and
              contains complete sentences intended for readers of the encyclopedia.</p>
              <div class="mw-references-wrap">
                <p>This references paragraph must not enter the detector input even
                though it contains enough words to look like ordinary article prose.</p>
              </div>
            </div>
          </div>
        </main>
      </body>
    </html>
    """
    result = extract_article(html)
    assert "factual introduction" in result.text
    assert "second paragraph" in result.text
    assert "Infobox navigation" not in result.text
    assert "references paragraph" not in result.text


def test_uses_json_ld_article_body_when_visible_html_is_only_a_shell() -> None:
    body = " ".join(
        f"Sentence {index} contains useful reporting and specific article details."
        for index in range(30)
    )
    body_json = json.dumps(body)
    html = f"""
    <html>
      <head>
        <title>Rendered later</title>
        <script type="application/ld+json">
          {{"@type": "NewsArticle", "articleBody": {body_json}}}
        </script>
      </head>
      <body><div id="root"></div><script src="app.js"></script></body>
    </html>
    """
    result = extract_article(html)

    assert result.word_count >= 80
    assert "specific article details" in result.text


def test_diagnoses_blocked_and_javascript_only_pages() -> None:
    blocked = diagnose_extraction_failure(
        "<html><title>Access denied</title><p>Verify you are human</p></html>",
        word_count=5,
    )
    javascript = diagnose_extraction_failure(
        '<html><div id="root"></div><script src="a.js"></script>'
        '<script src="b.js"></script></html>',
        word_count=0,
    )

    assert blocked.code == "access_blocked"
    assert blocked.retryable is False
    assert javascript.code == "javascript_required"
    assert javascript.retryable is True


def test_diagnoses_long_block_page_but_accepts_long_article() -> None:
    blocked = diagnose_extraction_failure(
        "<html><title>Security check</title><p>Verify you are human</p></html>",
        word_count=200,
    )
    article = diagnose_extraction_failure(
        "<html><article>Ordinary published reporting.</article></html>",
        word_count=200,
    )

    assert blocked is not None
    assert blocked.code == "access_blocked"
    assert article is None


def test_article_is_not_blocked_by_captcha_words_inside_scripts() -> None:
    html = """
    <html>
      <head>
        <title>Donald Trump - Wikipedia</title>
        <script>
          window.siteConfig = {
            captchaProvider: "hcaptcha",
            challengePath: "cf-chl-challenge-platform",
            errorLabel: "Access denied"
          };
        </script>
      </head>
      <body>
        <h1>Donald Trump</h1>
        <article><p>Ordinary encyclopedia article text.</p></article>
      </body>
    </html>
    """

    failure = diagnose_extraction_failure(html, word_count=400)

    assert failure is None


def test_long_article_may_discuss_access_denial_without_being_blocked() -> None:
    html = """
    <html>
      <head><title>History of access control</title></head>
      <body><article><p>
        The report says access denied messages were shown during the incident,
        while captcha systems and bot detection were discussed by researchers.
      </p></article></body>
    </html>
    """

    failure = diagnose_extraction_failure(html, word_count=300)

    assert failure is None


def test_content_fingerprint_ignores_whitespace_but_not_content() -> None:
    assert content_fingerprint("same   article\ntext") == content_fingerprint(
        "same article text"
    )
    assert content_fingerprint("same article text") != content_fingerprint(
        "different article text"
    )


def test_builds_article_from_rendered_browser_text() -> None:
    article = article_from_text(
        "  First rendered paragraph.\n\nSecond rendered paragraph.  ",
        title=" Open article ",
        has_author=True,
        has_citations=True,
    )

    assert article.title == "Open article"
    assert article.text == "First rendered paragraph.\nSecond rendered paragraph."
    assert article.word_count == 6
    assert article.has_author is True
    assert article.has_citations is True
