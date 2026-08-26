# Chrome Web Store listing draft

## Name

AI Article Check

## Short description

Checks English articles for AI-writing signals directly from Google Search.

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

Before submission, add the public privacy-policy URL, support contact, owned
icons, screenshots, category, and the tested production API URL.
