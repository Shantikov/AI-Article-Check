const BADGE_CLASS = "acs-badge";
const RESULT_SELECTOR = "h3";
const badgeByAnchor = new WeakMap();
let scanTimer = null;
let scanRunning = false;
let activeDetailsPanel = null;
let activeDetailsBadge = null;

const LABELS = {
  ai_likely: "Strong AI signal",
  uncertain: "Mixed signals",
  human_likely: "No strong AI signal",
  unsupported: "English only",
  unavailable: "Unavailable",
};

const ERROR_LABELS = {
  access_blocked: "Access blocked",
  backend_unavailable: "Server offline",
  detector_unavailable: "Detector offline",
  dns_error: "Domain unavailable",
  extension_error: "Extension error",
  http_error: "Site error",
  http_access_blocked: "Access blocked",
  internal_error: "Analysis failed",
  invalid_response: "Invalid response",
  invalid_url: "Invalid address",
  javascript_required: "Needs JavaScript",
  network_error: "Connection failed",
  non_html: "Not an article",
  page_not_found: "Page not found",
  private_address: "Blocked address",
  rate_limited: "Try later",
  restricted_content: "Restricted article",
  server_error: "Site error",
  timeout: "Timed out",
  too_many_redirects: "Redirect loop",
  too_little_text: "Not enough text",
};

scheduleScan(150);

const observer = new MutationObserver(() => scheduleScan(500));
observer.observe(document.documentElement, { childList: true, subtree: true });

window.addEventListener("popstate", () => {
  closeDetailsPanel();
  scheduleScan(250);
});
window.addEventListener("pageshow", () => scheduleScan(150));
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") scheduleScan(150);
});
window.addEventListener("resize", closeDetailsPanel);
window.addEventListener("scroll", (event) => {
  if (
    activeDetailsPanel
    && (
      event.target === activeDetailsPanel
      || activeDetailsPanel.contains(event.target)
    )
  ) {
    return;
  }

  closeDetailsPanel();
}, true);
document.addEventListener("click", (event) => {
  if (
    activeDetailsPanel
    && !activeDetailsPanel.contains(event.target)
    && !activeDetailsBadge?.contains(event.target)
  ) {
    closeDetailsPanel();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeDetailsPanel();
});

function scheduleScan(delay) {
  clearTimeout(scanTimer);
  scanTimer = setTimeout(scanResults, delay);
}

async function scanResults() {
  if (scanRunning) return;
  scanRunning = true;
  let pending = [];
  try {
    const settings = await chrome.runtime.sendMessage({ type: "GET_SETTINGS" });
    const targets = collectTargets(20);
    const autoResults = isFirstSearchPage() ? (settings.autoResults || 6) : 0;

    const cachedResponse = await chrome.runtime.sendMessage({
      type: "GET_CACHED_RESULTS",
      urls: targets.map((target) => target.url),
    });
    const cachedMap = new Map(
      (cachedResponse?.results || []).map((result) => [result.url, result]),
    );
    for (const target of targets) {
      const cached = cachedMap.get(target.url);
      const current = target.badge._acsResult;
      if (
        cached
        && target.badge.dataset.acsState !== "loading"
        && (
          ["new", "manual"].includes(target.badge.dataset.acsState)
          || current?.status === "error"
        )
      ) {
        renderResult(target.badge, cached);
      }
    }

    targets.slice(autoResults).forEach((target) => {
      if (target.badge.dataset.acsState === "new") {
        setManual(target.badge, target.url);
      }
    });

    pending = targets
      .slice(0, autoResults)
      .filter((target) => !["loading", "complete"].includes(target.badge.dataset.acsState));
    if (!pending.length) return;

    for (const target of pending) setLoading(target.badge);
    await Promise.all(pending.map(analyzeSearchTarget));
  } catch (error) {
    for (const target of pending) {
      if (target.badge.dataset.acsState === "loading") {
        renderResult(target.badge, {
          status: "error",
          label: "unavailable",
          error: error?.message || "Extension error",
          error_code: "extension_error",
          retryable: true,
        });
      }
    }
  } finally {
    scanRunning = false;
  }
}

async function analyzeSearchTarget(target) {
  try {
    const response = await chrome.runtime.sendMessage({
      type: "ANALYZE_URLS",
      urls: [target.url],
    });
    renderResult(target.badge, response?.results?.[0]);
  } catch (error) {
    renderResult(target.badge, {
      status: "error",
      label: "unavailable",
      error: error?.message || "Extension error",
      error_code: "extension_error",
      retryable: true,
    });
  }
}

function collectTargets(limit) {
  const targets = [];
  const seen = new Set();
  for (const heading of document.querySelectorAll(RESULT_SELECTOR)) {
    const anchor = heading.closest("a");
    if (!anchor) continue;

    const url = normalizeResultUrl(anchor.href);
    if (!url || seen.has(url)) continue;
    seen.add(url);

    let badge = badgeByAnchor.get(anchor);
    if (!badge || !badge.isConnected) {
      badge = document.createElement("span");
      badge.className = BADGE_CLASS;
      badge.dataset.acsState = "new";
      badge.dir = "ltr";
      badge.setAttribute("aria-live", "polite");
      // Keep the badge inside the heading's own rendering subtree. Google can
      // counter-transform the heading while mirroring its surrounding result.
      heading.append(badge);
      bindBadgeInteractions(badge);
      badgeByAnchor.set(anchor, badge);
    }
    if (badge.dataset.acsUrl !== url) {
      badge.dataset.acsUrl = url;
      badge.dataset.acsState = "new";
    }
    syncBadgeOrientation(badge);
    targets.push({ url, badge });
    if (targets.length >= limit) break;
  }
  return targets;
}

function bindBadgeInteractions(badge) {
  if (badge.dataset.acsBound === "true") return;
  badge.dataset.acsBound = "true";

  badge.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (badge.dataset.acsState === "manual") {
      checkSingleResult(badge);
    } else if (badge.dataset.acsState === "complete") {
      toggleDetailsPanel(badge);
    }
  });
  badge.addEventListener("keydown", (event) => {
    if (!["Enter", " "].includes(event.key)) return;
    event.preventDefault();
    event.stopPropagation();
    if (badge.dataset.acsState === "manual") {
      checkSingleResult(badge);
    } else if (badge.dataset.acsState === "complete") {
      toggleDetailsPanel(badge);
    }
  });
}

function isFirstSearchPage() {
  const searchUrl = new URL(location.href);
  const start = Number(searchUrl.searchParams.get("start") || 0);
  const page = Number(searchUrl.searchParams.get("page") || 1);
  return start <= 0 && page <= 1;
}

function syncBadgeOrientation(badge) {
  // Google applies presentation transforms inside some result containers.
  // Keep this custom host neutral. Do not infer or counter-transform ancestors:
  // Google already balances those transforms for the visible result subtree.
  badge.style.setProperty("transform", "none", "important");
  badge.style.setProperty("scale", "1 1", "important");
  badge.style.setProperty("rotate", "none", "important");
  badge.style.setProperty("transform-origin", "center", "important");
  badge.style.setProperty("direction", "ltr", "important");
  badge.style.setProperty("unicode-bidi", "isolate", "important");
  badge.style.setProperty("writing-mode", "horizontal-tb", "important");
}

function normalizeResultUrl(rawUrl) {
  try {
    const url = new URL(rawUrl, location.href);
    if (!["http:", "https:"].includes(url.protocol)) return null;
    if (/(^|\.)google\.[a-z.]+$/i.test(url.hostname)) return null;
    url.hash = "";
    return url.href;
  } catch {
    return null;
  }
}

function setLoading(badge) {
  if (activeDetailsBadge === badge) closeDetailsPanel();
  badge.className = `${BADGE_CLASS} acs-loading`;
  syncBadgeOrientation(badge);
  badge.dataset.acsState = "loading";
  badge.textContent = "Checking…";
  badge.title = "The article is being analyzed";
  badge.setAttribute("role", "status");
  badge.removeAttribute("tabindex");
  badge._acsResult = null;
}

function setManual(badge, url) {
  if (activeDetailsBadge === badge) closeDetailsPanel();
  badge.className = `${BADGE_CLASS} acs-manual`;
  syncBadgeOrientation(badge);
  badge.dataset.acsState = "manual";
  badge.dataset.acsUrl = url;
  badge.textContent = "Check for AI";
  badge.title = "Analyze this result";
  badge.setAttribute("role", "button");
  badge.setAttribute("tabindex", "0");
  badge._acsResult = null;
}

async function checkSingleResult(badge, force = false) {
  if (!["manual", "complete"].includes(badge.dataset.acsState)) return;
  const url = badge.dataset.acsUrl;
  setLoading(badge);
  try {
    const response = await chrome.runtime.sendMessage({
      type: "ANALYZE_URLS",
      urls: [url],
      force,
    });
    renderResult(badge, response?.results?.[0]);
    openDetailsPanel(badge);
  } catch (error) {
    renderResult(badge, {
      status: "error",
      label: "unavailable",
      error: error?.message || "Extension error",
      error_code: "extension_error",
      retryable: true,
    });
  }
}

function renderResult(badge, result) {
  const safeResult = result || {
    label: "unavailable",
    error: "The server returned no result",
  };
  const label = safeResult.label || "unavailable";
  const segments = resultSegmentCounts(safeResult);
  const hasDirectEvidence = resultHasDirectEvidence(safeResult);
  const needsMoreText = resultNeedsMoreSamples(safeResult);
  const isError = safeResult.status === "error" || label === "unavailable";
  const badgeText = isError
    ? ERROR_LABELS[safeResult.error_code] || LABELS.unavailable
    : hasDirectEvidence
      ? "AI evidence found"
      : needsMoreText
        ? `Need more text · ${aiMatchPercentage(safeResult, segments)}%`
        : segments.checked
          ? `AI match · ${aiMatchPercentage(safeResult, segments)}%`
          : LABELS[label] || LABELS.unavailable;

  badge.className = `${BADGE_CLASS} acs-${needsMoreText ? "insufficient" : label}`;
  syncBadgeOrientation(badge);
  badge.textContent = badgeText;
  badge.dataset.acsState = "complete";
  badge._acsResult = safeResult;
  badge.title = buildTooltip(safeResult);
  badge.setAttribute("aria-label", badge.title);
  badge.setAttribute("role", "button");
  badge.setAttribute("tabindex", "0");
}

function canRecheckResult(result) {
  return result.retryable === true
    || ["access_blocked", "http_access_blocked", "restricted_content"]
      .includes(result.error_code);
}

function buildTooltip(result) {
  if (result.status === "error" || result.label === "unavailable") {
    return [
      result.error || "The page could not be analyzed.",
      canRecheckResult(result) ? "Click for details and retry." : "Click for details.",
    ].join("\n");
  }
  if (result.label === "unsupported") {
    return "Only English-language articles are supported in this version.";
  }

  const segments = resultSegmentCounts(result);
  const groups = collectEvidenceGroups(result);
  const aiLines = formatReasonSection(groups.ai, 2);
  const humanLines = formatReasonSection(groups.human, 2);
  const summaryLines = resultHasDirectEvidence(result)
    ? ["Direct AI evidence found."]
    : resultNeedsMoreSamples(result)
      ? [
          `AI match: ${aiMatchPercentage(result, segments)}%`,
          `Only ${segments.checked} samples were available; more text is required.`,
        ]
      : [
          `AI match: ${aiMatchPercentage(result, segments)}%`,
          `AI-like samples: ${segments.ai} of ${segments.checked}`,
        ];
  return [
    ...summaryLines,
    "",
    "Evidence for AI:",
    ...aiLines,
    "",
    "Evidence against AI:",
    ...humanLines,
    "",
    "Click for details.",
  ]
    .join("\n");
}

function resultSegmentCounts(result) {
  const checked = Math.max(0, Number(result.segments_checked) || 0);
  const ai = Math.min(checked, Math.max(0, Number(result.ai_segments) || 0));
  const nonAi = Math.min(
    checked,
    Math.max(0, Number(result.non_ai_segments) || checked - ai),
  );
  return { checked, ai, nonAi };
}

function aiMatchPercentage(result, segments) {
  const rawValue = result?.ai_probability;
  const calibrated = Number(rawValue);
  if (
    rawValue !== null
    && rawValue !== undefined
    && Number.isFinite(calibrated)
    && calibrated >= 0
    && calibrated <= 1
  ) {
    return Math.round(calibrated * 100);
  }
  if (!segments.checked) return 0;
  return Math.round((segments.ai / segments.checked) * 100);
}

function hasCalibratedScore(result) {
  const rawValue = result?.ai_probability;
  const value = Number(rawValue);
  return rawValue !== null
    && rawValue !== undefined
    && Number.isFinite(value)
    && value >= 0
    && value <= 1;
}

function resultHasDirectEvidence(result) {
  return (result.evidence || []).some((item) => item.kind === "strong");
}

function resultNeedsMoreSamples(result) {
  return result?.label === "uncertain" && (result.evidence || []).some((item) =>
    /at least \d+ independently checked samples are required/i.test(item.message || "")
  );
}

function collectEvidenceGroups(result) {
  const legacyProbability = Math.round(Number(result.ai_probability || 0) * 100);
  const evidence = result.evidence || [];
  const ai = evidence
    .filter((item) => ["strong", "weak"].includes(item.kind))
    .map(normalizeEvidenceItem)
    .filter((item) => item.message);
  const human = evidence
    .filter((item) => item.kind === "human")
    .map(normalizeEvidenceItem)
    .filter((item) => item.message);
  const notes = evidence
    .filter((item) => item.kind === "info")
    .map(normalizeEvidenceItem)
    .filter((item) => item.message);

  const oldModelNote = evidence.some((item) =>
    /local.*(?:onnx|model).*analy/i.test(item.message || "")
  );
  if (oldModelNote && legacyProbability >= 72) {
    ai.unshift({ message: "AI-like writing patterns.", detail: "", excerpt: "" });
  } else if (oldModelNote && legacyProbability <= 28) {
    human.unshift({ message: "Human-like writing patterns.", detail: "", excerpt: "" });
  }

  return {
    ai: uniqueEvidenceItems(ai),
    human: uniqueEvidenceItems(human),
    notes: uniqueEvidenceItems(notes),
  };
}

function formatReasonSection(reasons, limit = 3) {
  const unique = uniqueEvidenceItems(reasons).slice(0, limit);
  return unique.length
    ? unique.map((reason) => `• ${reason.message}`)
    : ["• None found"];
}

function normalizeEvidenceItem(item) {
  return {
    message: conciseEvidenceMessage(item?.message),
    detail: String(item?.detail || "").trim(),
    excerpt: String(item?.excerpt || "").trim(),
  };
}

function uniqueEvidenceItems(items) {
  const seen = new Set();
  return items.filter((item) => {
    const key = `${item.message}\n${item.excerpt}`;
    if (!item.message || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function conciseEvidenceMessage(message) {
  const text = String(message || "").trim();
  const rewrites = [
    [/explicitly discloses.*AI/i, "Page discloses AI use."],
    [/phrase typical of an unedited chatbot/i, "Unedited chatbot phrase found."],
    [/sentence lengths.*uniform/i, "Very uniform sentence lengths."],
    [/paragraph lengths.*uniform/i, "Very uniform paragraph lengths."],
    [/formulaic.*phrases/i, "Many formulaic phrases."],
    [/repeated (?:constructions|wording)/i, "Repeated wording patterns."],
    [/names an author|named author/i, "Named author or editorial team."],
    [/sources section|sources or references/i, "Sources or references found."],
    [/no clear writing-style indicators/i, "No clear style indicators."],
    [/local model found AI-like patterns/i, "AI-like writing patterns."],
    [/local model found human-like patterns/i, "Human-like writing patterns."],
    [/external model found AI-like patterns/i, "AI-like writing patterns."],
    [/external model found human-like patterns/i, "Human-like writing patterns."],
    [/(?:local|external) model was inconclusive/i, "No clear writing-pattern signal."],
    [/local.*(?:ONNX|model).*analy/i, ""],
    [/external classifier was unavailable/i, "Additional classifier unavailable."],
    [/at least 3 agreeing samples/i, "Fewer than 3 samples were available."],
    [/at least \d+ independently checked samples.*required/i, "Fewer than 3 samples were available."],
    [/independently checked text samples disagreed/i, "Text samples produced mixed results."],
    [/calibrated score stayed between.*threshold/i, "Score stayed between the decision limits."],
    [/additional classifier marked.*AI-like/i, "Additional check was AI-like."],
    [/additional classifier did not mark.*AI-like/i, "Additional check was not AI-like."],
    [/page was large.*(?:portion|part).*analy/i, "Large page: only part was analyzed."],
  ];
  for (const [pattern, replacement] of rewrites) {
    if (pattern.test(text)) return replacement;
  }
  return text.length <= 55 ? text : `${text.slice(0, 52).trimEnd()}…`;
}

function toggleDetailsPanel(badge) {
  if (activeDetailsBadge === badge) {
    closeDetailsPanel();
  } else {
    openDetailsPanel(badge);
  }
}

function openDetailsPanel(badge) {
  const result = badge._acsResult;
  if (!result) return;
  closeDetailsPanel();

  const panel = document.createElement("section");
  panel.className = "acs-details";
  panel.dir = "ltr";
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", "AI content analysis details");

  const header = document.createElement("div");
  header.className = "acs-details-header";
  const heading = document.createElement("strong");
  heading.textContent = result.title || "Analysis details";
  const closeButton = document.createElement("button");
  closeButton.className = "acs-details-close";
  closeButton.type = "button";
  closeButton.textContent = "×";
  closeButton.setAttribute("aria-label", "Close details");
  closeButton.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    closeDetailsPanel();
  });
  header.append(heading, closeButton);
  panel.append(header);

  if (result.status === "error" || result.label === "unavailable") {
    appendDetailsText(panel, result.error || "Page unavailable.", "acs-details-error");
    appendDetailsText(
      panel,
      errorHelpMessage(result.error_code),
      "acs-details-summary",
    );
    if (canRecheckResult(result)) appendRecheckButton(panel, badge, "Try again");
  } else if (result.label === "unsupported") {
    appendDetailsText(panel, "Only English-language articles are supported.");
    appendRecheckButton(panel, badge, "Recheck page");
  } else {
    const segments = resultSegmentCounts(result);
    appendDetailsText(
      panel,
      resultHasDirectEvidence(result)
        ? "Direct AI evidence found"
        : `AI match: ${aiMatchPercentage(result, segments)}%`,
      "acs-details-result",
    );
    appendDetailsText(panel, detailedResultSummary(result), "acs-details-summary");

    const metrics = document.createElement("div");
    metrics.className = "acs-details-metrics";
    appendMetric(metrics, "Samples checked", String(segments.checked));
    appendMetric(metrics, "AI-like samples", String(segments.ai));
    appendMetric(metrics, "Not AI-like", String(segments.nonAi));
    appendMetric(metrics, "Words analyzed", analyzedWordsSummary(result));
    panel.append(metrics);

    const groups = collectEvidenceGroups(result);
    appendDetailsSection(panel, "Evidence for AI", groups.ai);
    appendDetailsSection(panel, "Evidence against AI", groups.human);
    if (groups.notes.length) appendDetailsSection(panel, "Analysis notes", groups.notes);
    appendDetailsText(
      panel,
      hasCalibratedScore(result)
        ? "AI match is calibrated against labelled human and AI texts. The final label uses the benchmark-fitted probability curve and decision limits for the amount of text available."
        : "AI match is the percentage of checked samples classified as AI-like. A strong result requires direct page evidence or at least 3 separately checked samples with unanimous AI-like results.",
      "acs-details-rule",
    );

    const source = resultSource(result);
    if (source) appendDetailsText(panel, `Source: ${source}`, "acs-details-source");
    appendDetailsText(
      panel,
      "Recheck page downloads the article again and replaces the saved result.",
      "acs-details-source",
    );
    appendRecheckButton(panel, badge, "Recheck page");
  }

  panel.style.visibility = "hidden";
  document.documentElement.append(panel);
  activeDetailsPanel = panel;
  activeDetailsBadge = badge;
  positionDetailsPanel(panel, badge);
  panel.style.visibility = "visible";
}

function appendDetailsText(panel, text, className = "") {
  const paragraph = document.createElement("p");
  paragraph.className = className;
  paragraph.textContent = text;
  panel.append(paragraph);
}

function appendMetric(container, label, value) {
  const metric = document.createElement("div");
  const name = document.createElement("span");
  const data = document.createElement("strong");
  name.textContent = label;
  data.textContent = value;
  metric.append(name, data);
  container.append(metric);
}

function analyzedWordsSummary(result) {
  const analyzed = Math.max(0, Number(result.sampled_word_count) || 0);
  const article = Math.max(0, Number(result.word_count) || 0);
  if (analyzed && article && analyzed < article) {
    return `${analyzed.toLocaleString("en-US")} of ${article.toLocaleString("en-US")}`;
  }
  return (analyzed || article).toLocaleString("en-US");
}

function appendRecheckButton(panel, badge, label) {
  const button = document.createElement("button");
  button.className = "acs-details-recheck";
  button.type = "button";
  button.textContent = label;
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    closeDetailsPanel();
    checkSingleResult(badge, true);
  });
  panel.append(button);
}

function errorHelpMessage(errorCode) {
  const messages = {
    access_blocked:
      "The site refused the backend request. Opening the page in Chrome may still work.",
    backend_unavailable:
      "Start the local backend and keep its window open, then try again.",
    detector_unavailable:
      "Run setup.cmd to verify the model files, then restart the backend.",
    dns_error:
      "The domain could not be resolved. This may be temporary.",
    extension_error:
      "The browser extension could not complete this request. Reload it and try again.",
    http_error:
      "The website returned an HTTP response that the backend could not analyze.",
    http_access_blocked:
      "The site explicitly refused the backend request with HTTP 401 or 403.",
    internal_error:
      "The backend hit an unexpected analysis error. Restart it before retrying.",
    invalid_response:
      "The website returned a malformed or incomplete response.",
    invalid_url:
      "The result did not contain a valid public HTTP or HTTPS address.",
    javascript_required:
      "The downloaded HTML contains only an app shell; the article appears after browser scripts run.",
    network_error:
      "The backend could not establish a connection to the website.",
    non_html:
      "The link returned a file or another unsupported content type instead of an HTML article.",
    page_not_found:
      "The link returned a permanent not-found response.",
    private_address:
      "Private and local network addresses are blocked for backend safety.",
    rate_limited:
      "The site is temporarily limiting requests. Wait briefly before retrying.",
    restricted_content:
      "The article text is hidden behind a login or subscription page.",
    server_error:
      "The website returned a temporary server-side error.",
    timeout:
      "The website did not respond within the backend time limit.",
    too_many_redirects:
      "The website redirected repeatedly without reaching an article page.",
    too_little_text:
      "The page loaded, but it did not expose enough article text for a reliable check.",
  };
  return messages[errorCode]
    || "The page could not be checked with the current backend response.";
}

function appendDetailsSection(panel, title, reasons) {
  const section = document.createElement("div");
  section.className = "acs-details-section";
  const heading = document.createElement("strong");
  heading.textContent = title;
  const list = document.createElement("div");
  list.className = "acs-details-reasons";

  if (!reasons.length) {
    const empty = document.createElement("p");
    empty.className = "acs-details-empty";
    empty.textContent = "No supporting signals were found.";
    list.append(empty);
  } else {
    for (const reason of reasons) {
      const item = document.createElement("div");
      item.className = "acs-details-reason";
      const reasonTitle = document.createElement("strong");
      reasonTitle.textContent = reason.message;
      const explanation = document.createElement("p");
      explanation.textContent = reason.detail || detailedEvidenceMessage(reason.message);
      item.append(reasonTitle, explanation);
      if (reason.excerpt) {
        const excerpt = document.createElement("blockquote");
        excerpt.className = "acs-details-excerpt";
        excerpt.textContent = `“${reason.excerpt}”`;
        item.append(excerpt);
      }
      list.append(item);
    }
  }
  section.append(heading, list);
  panel.append(section);
}

function detailedResultSummary(result) {
  const segments = resultSegmentCounts(result);
  if (resultHasDirectEvidence(result)) {
    return "The page contains a direct AI-use disclosure or an unedited chatbot phrase.";
  }
  if (hasCalibratedScore(result)) {
    const percentage = aiMatchPercentage(result, segments);
    if (resultNeedsMoreSamples(result)) {
      return `The AI match is ${percentage}%, but only ${segments.checked} article sample${segments.checked === 1 ? " was" : "s were"} available. More text is required before issuing a strong AI result.`;
    }
    if (result.label === "ai_likely") {
      return `The calibrated ${percentage}% AI match crossed the strong-AI decision limit using ${segments.checked} article samples.`;
    }
    if (result.label === "human_likely") {
      return `The calibrated ${percentage}% AI match stayed below the human-text decision limit.`;
    }
    return `The calibrated ${percentage}% AI match stayed between the two decision limits, so the result remains uncertain.`;
  }
  if (result.label === "ai_likely") {
    return `All ${segments.checked} independently checked text samples were AI-like.`;
  }
  if (result.label === "human_likely") {
    return `None of the ${segments.checked} checked text samples were AI-like.`;
  }
  if (!segments.checked) {
    return "No model samples were available; only direct page indicators were checked.";
  }
  if (segments.checked && segments.ai === segments.checked && segments.checked < 3) {
    return `Only ${segments.checked} usable sample${segments.checked === 1 ? " was" : "s were"} available; at least 3 agreeing samples are required.`;
  }
  return `${segments.ai} of ${segments.checked} checked samples were AI-like, so the result remains mixed.`;
}

function detailedEvidenceMessage(reason) {
  const aiSampleMatch = reason.match(/^(\d+) of (\d+) text samples were AI-like\.$/i);
  if (aiSampleMatch) {
    return `${aiSampleMatch[1]} independently checked article samples crossed the model's published decision boundary.`;
  }
  const nonAiSampleMatch = reason.match(
    /^(\d+) of (\d+) text samples were not AI-like\.$/i,
  );
  if (nonAiSampleMatch) {
    return `${nonAiSampleMatch[1]} independently checked article samples stayed below the model's published decision boundary.`;
  }
  const explanations = {
    "Page discloses AI use.":
      "The page contains an explicit statement that the article was written, generated, or prepared with AI.",
    "Unedited chatbot phrase found.":
      "The text contains wording normally produced when a chatbot describes its own identity or limitations.",
    "Very uniform sentence lengths.":
      "Sentence lengths vary unusually little across the analyzed article.",
    "Very uniform paragraph lengths.":
      "Paragraph lengths follow an unusually regular pattern across the article.",
    "Many formulaic phrases.":
      "Stock phrases such as “in conclusion” or “it is important to note” appear unusually often.",
    "Repeated wording patterns.":
      "The same four-word sequences recur unusually often in the analyzed text.",
    "AI-like writing patterns.":
      "Word choices and sequence patterns are closer to AI-generated English examples than to human-written examples.",
    "Human-like writing patterns.":
      "Word choices and sequence patterns are closer to human-written English examples than to AI-generated examples.",
    "Named author or editorial team.":
      "The extracted page identifies a named author or an editorial team.",
    "Sources or references found.":
      "The article includes citations, source links, or a references section.",
    "No clear style indicators.":
      "No strong sentence, paragraph, phrase, author, or citation signal was detected.",
    "No clear writing-pattern signal.":
      "The writing-pattern analysis did not strongly match either side.",
    "Large page: only part was analyzed.":
      "The page exceeded the download limit, so the score uses only the portion that was downloaded.",
    "Additional classifier unavailable.":
      "The optional secondary check did not run and contributed no evidence to this result.",
    "Fewer than 3 samples were available.":
      "A strong AI label requires at least three separately checked samples from the article.",
    "Text samples produced mixed results.":
      "Different parts of the article landed on different sides of the decision boundary.",
    "Score stayed between the decision limits.":
      "The benchmark-calibrated score was not strong enough for either a human or AI result.",
    "Additional check was AI-like.":
      "The optional independent classifier also marked the article text as AI-like.",
    "Additional check was not AI-like.":
      "The optional independent classifier did not mark the article text as AI-like.",
  };
  return explanations[reason]
    || "This signal was found in the extracted article text or page metadata.";
}

function resultSource(result) {
  try {
    return new URL(result.final_url || result.url).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

function positionDetailsPanel(panel, badge) {
  const margin = 12;
  const gap = 8;
  const badgeRect = badge.getBoundingClientRect();
  const panelRect = panel.getBoundingClientRect();
  const maxLeft = Math.max(margin, window.innerWidth - panelRect.width - margin);
  let left = Math.min(Math.max(margin, badgeRect.left), maxLeft);
  let top = badgeRect.bottom + gap;
  if (top + panelRect.height > window.innerHeight - margin) {
    top = badgeRect.top - panelRect.height - gap;
  }
  top = Math.max(margin, top);
  panel.style.left = `${Math.round(left)}px`;
  panel.style.top = `${Math.round(top)}px`;
}

function closeDetailsPanel() {
  activeDetailsPanel?.remove();
  activeDetailsPanel = null;
  activeDetailsBadge = null;
}
