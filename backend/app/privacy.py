from __future__ import annotations

from html import escape


PRIVACY_UPDATED = "26 August 2026"
DEFAULT_SUPPORT_URL = "https://github.com/Shantikov/AI-Article-Check/issues"


def privacy_html(support_url: str = DEFAULT_SUPPORT_URL) -> str:
    safe_support_url = escape(support_url, quote=True)
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>AI Article Check privacy policy</title>
    <style>
      :root {{ color-scheme: light dark; font-family: Arial, sans-serif; }}
      body {{ margin: 0; }}
      main {{ line-height: 1.6; margin: 0 auto; max-width: 760px; padding: 32px 20px 56px; }}
      h1 {{ line-height: 1.2; margin-bottom: 6px; }}
      h2 {{ font-size: 1.15rem; margin-top: 28px; }}
      .updated {{ color: #64748b; margin-top: 0; }}
      a {{ color: #2563eb; }}
    </style>
  </head>
  <body>
    <main>
      <h1>AI Article Check privacy policy</h1>
      <p class="updated">Last updated: {PRIVACY_UPDATED}</p>
      <p>AI Article Check checks English-language article text for signals
      associated with AI-generated writing. This policy applies to the public
      Chrome extension and its hosted analysis service.</p>

      <h2>Data processed</h2>
      <ul>
        <li>For a Google result, the extension sends the public article URL to
        the analysis service. The service downloads the page and extracts its
        article text.</li>
        <li>When the user selects <strong>Analyze this page</strong>, the
        extension sends the rendered article text, page URL, optional canonical
        URL, title, and simple author or citation indicators.</li>
        <li>The extension stores recent results locally in Chrome and stores the
        selected automatic-check count in Chrome sync storage.</li>
        <li>Results can include short excerpts from the highest-scoring and
        lowest-scoring article samples so the user can inspect the text behind
        those comparative signals.</li>
        <li>The hosting provider may process IP addresses and standard request
        metadata for delivery, security, rate limiting, and operational logs.</li>
      </ul>

      <h2>Purpose and retention</h2>
      <p>Data is used only to provide article analysis, explain the result,
      cache recent results, prevent abuse, and maintain the service. Complete
      raw article text is not stored in an application database and is
      not used to train a model. Results from automatic URL checks, including
      short evidence excerpts, may remain in service memory for up to 12 hours.
      Results created from the user-submitted text of an open page are not put
      in this shared cache. Browser-side cached results expire after 24 hours.
      Hosting-provider logs follow the provider's configured retention
      settings.</p>

      <h2>Sharing</h2>
      <p>Requests are processed by the service operator and its hosting
      infrastructure. Article text is analyzed by the pinned detector on that
      backend and is not sold, used for advertising, or sent to a paid
      third-party detector in the default public configuration. Data may be
      disclosed when required by law or to protect the service from abuse.</p>

      <h2>User choices</h2>
      <p>Users can reduce automatic checks, avoid manual page analysis, clear
      the extension's stored data in Chrome, or uninstall the extension.</p>

      <h2>Contact</h2>
      <p>For privacy or support questions, use the
      <a href="{safe_support_url}">AI Article Check support page</a>.</p>
    </main>
  </body>
</html>
"""
