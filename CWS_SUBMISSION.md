# Chrome Web Store submission — AI Article Check 0.9.6

Use `AI-Article-Check-0.9.6-store.zip` for the store draft. Do not upload the
source, development, backend, or GitHub-update archives.

## Store listing

**Name**

AI Article Check

**Summary**

Checks English articles for AI-writing signals directly from Google Search.

**Category**

Productivity

**Language**

English

**Detailed description**

AI Article Check adds a cautious AI-match estimate beside article results in
Google Search. The first selected results on page one can be checked
automatically, while later results remain available through a manual Check for
AI button.

Click a completed badge to see the number of article samples checked, the
signals supporting and opposing an AI classification, and short excerpts from
the strongest AI-like and human-like passages. The extension can also analyze
the article currently open in Chrome when a website blocks direct downloading
or renders its content with JavaScript.

The current detector supports English-language articles. Its result is an
estimate based on writing patterns and direct page indicators; it is not proof
of who wrote a text. Cached results make repeated checks faster.

Article URLs, and rendered article text only when the user explicitly chooses
Analyze this page, are sent to the AI Article Check analysis service. The
extension does not sell submitted data, use it for advertising, or use it to
train a model. See the privacy policy for retention and hosting details.

## Public links

- Privacy policy: <https://ai-article-check-production.up.railway.app/privacy>
- Support: <https://github.com/Shantikov/AI-Article-Check/issues>

## Privacy practices

**Single purpose**

AI Article Check estimates whether English-language articles linked from Google
Search contain AI-writing signals and explains the article-specific signals
behind each estimate.

**Permission justifications**

| Permission or access | Justification |
| --- | --- |
| `storage` | Saves the selected automatic-check count and recent analysis results so the same article is not reprocessed unnecessarily. |
| `activeTab` | Grants temporary access to the current article only after the user opens the extension and chooses Analyze this page. |
| `scripting` | Runs the packaged article-text extraction function in that user-selected tab. |
| Google Search content-script matches | Adds result badges and Check for AI buttons on the four supported Google Search domains. |
| Production API host | Sends checked article URLs or user-requested rendered article text to the fixed AI Article Check analysis service and receives results. |

**Remote code**

Select **No, I am not using remote code**. All extension JavaScript and CSS is
packaged in the submitted ZIP. The HTTPS backend returns analysis data, not
executable code. The ONNX model runs on the backend and is not downloaded or
executed by the extension.

**Data types to disclose**

- **Web history:** URLs of articles selected automatically or manually for
  checking.
- **Website content:** article text and basic page metadata used for analysis.

Do not select personal communications, authentication information, financial
information, health information, precise location, or general user activity;
the extension does not request or use those categories.

Certify that the disclosed data is used only for the extension's single
analysis purpose, is not sold, is not used for advertising or creditworthiness,
and is handled according to the linked privacy policy.

## Distribution

- Visibility: Public
- Regions: All regions
- In-app purchases: No
- Pricing: Free

The model currently supports English articles even though users may install the
extension in any region.

## Reviewer instructions

No account or paid credentials are required.

1. Open a supported Google Search page, for example
   `https://www.google.com/search?q=english+technology+news`.
2. Wait for the first configured article results to show AI match badges.
3. Click a completed badge to open the detailed evidence card.
4. Open a normal English article, click the AI Article Check toolbar icon, and
   choose Analyze this page to test the user-initiated `activeTab` flow.
5. Page-one auto-checking can be set to 3, 6, or 8 results in the popup. Later
   pages use manual Check for AI buttons.

The production API is
`https://ai-article-check-production.up.railway.app`. The result is explicitly
presented as an estimate rather than proof of authorship.

## Required graphic assets

- Store icon: `store-assets/icon-128.png` (128×128)
- Small promo tile: `store-assets/small-promo-tile.png` (440×280)
- Screenshot 1: real Google results with several completed badges (1280×800)
- Screenshot 2: an open detailed evidence card with article-specific excerpts
  (1280×800)
- Screenshot 3: toolbar popup showing Analyze this page and the auto-check
  selector (1280×800)

Screenshots must come from the final installed build and must not contain
private tabs, account names, bookmarks, or unrelated browser extensions.

## Final dashboard sequence

1. Upload the store ZIP as a draft.
2. Verify `/health`, URL analysis, and Analyze this page with the uploaded build.
3. Complete Store listing, Privacy practices, Distribution, and publisher
   contact information.
4. Upload the real screenshots and existing store assets.
5. Submit the draft for review.
