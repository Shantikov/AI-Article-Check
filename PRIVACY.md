# AI Article Check privacy policy

Last updated: 26 August 2026

> Publication template: replace `[operator name]`, `[contact email]`, and
> `[privacy policy URL]` before submitting the extension to the Chrome Web Store.

AI Article Check checks English-language article text for signals associated
with AI-generated writing. This policy describes the public hosted version
operated by `[operator name]` at `[privacy policy URL]`.

## Data processed

- When a Google result is checked, the extension sends the article URL to the
  AI Article Check API. The API downloads the public page and extracts article
  text for analysis.
- When the user chooses **Analyze this page**, the extension extracts the
  rendered article text, page URL, optional canonical URL, title, and simple
  author/citation flags and sends them to the API.
- The extension stores analysis results and their expiration times in Chrome's
  local extension storage. It stores the selected auto-check count in Chrome
  sync storage.
- The API host may process an IP address and standard request metadata for
  delivery, security, rate limiting, and operational logs.

## Purpose and retention

The data is used only to provide article analysis, return an explanation, cache
recent results, prevent abuse, and maintain the service. The application does
not persist raw article text in its own database. Analysis results are cached in
server memory for up to 12 hours by default and disappear when that process is
restarted. Browser-side cached results expire after 24 hours.

Hosting-provider logs and their retention must be configured and disclosed by
the operator before publication. The extension does not sell personal data,
use article content for advertising, or use submitted content to train a model.

## Sharing

Requests are processed by the operator's hosting provider and the infrastructure
needed to run the service. Article text is analyzed by the pinned detector on
that backend; it is not sent to a paid third-party detector in the default
configuration. Data may be disclosed when required by law or to protect the
service from abuse.

## User choices

Users can reduce automatic checks by choosing the smallest available count,
avoid manual page analysis, clear the extension's site data in Chrome, or
uninstall the extension. The local development build sends requests only to the
user's own computer and is not the Chrome Web Store build described above.

For privacy questions or deletion requests concerning retained service logs,
contact `[contact email]`.
