# Chrome Web Store listing draft

The complete copy-and-paste dashboard fields, privacy declarations, reviewer
instructions, and screenshot plan are in [CWS_SUBMISSION.md](CWS_SUBMISSION.md).

## Name

AI Article Check

## Short description

Checks English articles for AI-writing signals directly from Google Search.

## Public links

- Privacy policy: <https://ai-article-check-production.up.railway.app/privacy>
- Support: <https://github.com/Shantikov/AI-Article-Check/issues>

## Single purpose

AI Article Check adds an AI-match estimate beside Google Search article results
and lets the user analyze the article currently open in Chrome. Results include
short evidence and an optional detailed explanation. The estimate is not proof
of authorship.

## Permission justifications

| Permission | Why it is required |
| --- | --- |
| `storage` | Saves the auto-check preference and recent analysis results. |
| `activeTab` | Gives temporary access only after the user chooses to analyze the open article. |
| `scripting` | Extracts readable text from that user-selected open article. |
| Google Search matches | Adds result badges and manual check buttons to supported Google Search pages. |
| Production API host | Sends URLs or user-requested rendered article text to the AI Article Check analysis service. |

The store build must not request `http://*/*`, `https://*/*`, localhost, or any
host other than the exact production API origin and the declared Google Search
content-script matches.

## Reviewer notes

The extension automatically checks only the first 3, 6, or 8 ordinary results
on the first Google results page. Later results and later Google pages require a
manual click. Manual open-page analysis is disabled on browser-internal pages
and Google Search itself. The current model supports English articles only.

The owned 128x128 icon and 440x280 promotional tile are in `store-assets`.
Before submission, capture at least one real 1280x800 screenshot from the final
installed build, choose the category, and confirm the support repository is
publicly accessible.
