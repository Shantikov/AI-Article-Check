const autoResultsSelect = document.querySelector("#auto-results");
const autoMessage = document.querySelector("#auto-message");
const analyzeCurrentPageButton = document.querySelector("#analyze-current-page");
const analyzeHelper = document.querySelector("#analyze-helper");
const currentPageResult = document.querySelector("#current-page-result");
const currentPageSummary = document.querySelector("#current-page-summary");
const currentPageDetails = document.querySelector("#current-page-details");
const analysisDetailsToggle = document.querySelector("#analysis-details-toggle");
const analysisDetails = document.querySelector("#analysis-details");

let autoMessageTimer;

initialize();

async function initialize() {
  const [settings, tabContext] = await Promise.all([
    chrome.runtime.sendMessage({ type: "GET_SETTINGS" }),
    chrome.runtime.sendMessage({ type: "GET_CURRENT_TAB_CONTEXT" }),
  ]);
  autoResultsSelect.value = String(settings.autoResults);
  setCurrentPageAvailability(tabContext);
}

autoResultsSelect.addEventListener("change", async () => {
  try {
    await chrome.storage.sync.set({
      autoResults: Number(autoResultsSelect.value),
    });
    showAutoMessage("Saved");
  } catch (error) {
    showAutoMessage(error?.message || "Could not save");
  }
});

analyzeCurrentPageButton.addEventListener("click", analyzeCurrentPage);
analysisDetailsToggle.addEventListener("click", toggleAnalysisDetails);

function showAutoMessage(message) {
  window.clearTimeout(autoMessageTimer);
  autoMessage.textContent = message;
  autoMessageTimer = window.setTimeout(() => {
    autoMessage.textContent = "";
  }, 1800);
}

function setCurrentPageAvailability(tabContext) {
  const canAnalyze = tabContext?.canAnalyze === true;
  analyzeCurrentPageButton.disabled = !canAnalyze;
  analyzeCurrentPageButton.title = canAnalyze
    ? ""
    : "Open an article page to use manual analysis";
  analyzeHelper.textContent = canAnalyze
    ? "Use when a Google result could not be checked."
    : "Open an article page to analyze it.";
}

async function analyzeCurrentPage() {
  resetAnalysisDetails();
  analyzeCurrentPageButton.disabled = true;
  analyzeCurrentPageButton.setAttribute("aria-busy", "true");
  analyzeCurrentPageButton.textContent = "Analyzing…";
  currentPageResult.hidden = false;
  currentPageResult.className = "page-result page-result-loading";
  currentPageSummary.textContent = "Reading this page…";
  currentPageDetails.textContent = "Keep this popup open until the check finishes.";

  try {
    const response = await chrome.runtime.sendMessage({
      type: "ANALYZE_CURRENT_TAB",
    });
    renderCurrentPageResult(response);
  } catch (error) {
    renderCurrentPageResult({
      ok: false,
      error: error?.message || "Could not analyze the current page",
    });
  } finally {
    analyzeCurrentPageButton.disabled = false;
    analyzeCurrentPageButton.removeAttribute("aria-busy");
    analyzeCurrentPageButton.textContent = "Analyze this page";
  }
}

function renderCurrentPageResult(response) {
  const result = response?.result;
  if (!response?.ok || !result) {
    resetAnalysisDetails();
    currentPageResult.className = "page-result page-result-error";
    const failure = userFacingFailure(response);
    currentPageSummary.textContent = failure.summary;
    currentPageDetails.textContent = failure.details;
    return;
  }

  const percentage = calibratedPercentage(result.ai_probability);
  const isUnsupported = result.label === "unsupported";
  const isAi = result.label === "ai_likely";
  const isHuman = result.label === "human_likely";
  currentPageResult.className = `page-result ${
    isUnsupported
      ? "page-result-neutral"
      : isAi
        ? "page-result-ai"
        : isHuman
          ? "page-result-human"
          : "page-result-uncertain"
  }`;
  currentPageSummary.textContent = isUnsupported
    ? "English articles only"
    : percentage === null
      ? "Analysis complete"
      : `AI match · ${percentage}%`;
  const title = result.title || "Current article";
  currentPageDetails.textContent = `${title} · ${result.word_count || 0} words. Saved for Google results.`;
  if (isUnsupported) {
    resetAnalysisDetails();
  } else {
    prepareAnalysisDetails(result);
  }
}

function calibratedPercentage(value) {
  if (value === null || value === undefined) return null;
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 0 || numeric > 1) return null;
  return Math.round(numeric * 100);
}

function resetAnalysisDetails() {
  analysisDetailsToggle.hidden = true;
  analysisDetailsToggle.textContent = "View details";
  analysisDetailsToggle.setAttribute("aria-expanded", "false");
  analysisDetails.hidden = true;
  analysisDetails.replaceChildren();
}

function toggleAnalysisDetails() {
  const willOpen = analysisDetailsToggle.getAttribute("aria-expanded") !== "true";
  analysisDetailsToggle.setAttribute("aria-expanded", String(willOpen));
  analysisDetailsToggle.textContent = willOpen ? "Hide details" : "View details";
  analysisDetails.hidden = !willOpen;
}

function prepareAnalysisDetails(result) {
  resetAnalysisDetails();

  const summary = document.createElement("p");
  summary.className = "analysis-details-summary";
  summary.textContent = detailedResultSummary(result);
  analysisDetails.append(summary);

  const segments = resultSegmentCounts(result);
  const metrics = document.createElement("div");
  metrics.className = "analysis-metrics";
  appendAnalysisMetric(metrics, "Samples checked", segments.checked);
  appendAnalysisMetric(metrics, "AI-like samples", segments.ai);
  appendAnalysisMetric(metrics, "Not AI-like", segments.nonAi);
  appendAnalysisMetric(
    metrics,
    "Words analyzed",
    Number(result.sampled_word_count) || Number(result.word_count) || 0,
  );
  analysisDetails.append(metrics);

  const groups = collectEvidenceGroups(result);
  appendEvidenceSection("Evidence for AI", groups.ai);
  appendEvidenceSection("Evidence against AI", groups.human);
  if (groups.notes.length) appendEvidenceSection("Analysis notes", groups.notes);

  analysisDetailsToggle.hidden = false;
}

function appendAnalysisMetric(container, label, value) {
  const metric = document.createElement("div");
  metric.className = "analysis-metric";
  const name = document.createElement("span");
  name.textContent = label;
  const data = document.createElement("strong");
  data.textContent = String(value);
  metric.append(name, data);
  container.append(metric);
}

function appendEvidenceSection(title, reasons) {
  const section = document.createElement("section");
  section.className = "evidence-section";
  const heading = document.createElement("h3");
  heading.textContent = title;
  section.append(heading);

  if (!reasons.length) {
    const empty = document.createElement("p");
    empty.className = "evidence-empty";
    empty.textContent = "None found.";
    section.append(empty);
  } else {
    const list = document.createElement("ul");
    list.className = "evidence-list";
    for (const reason of reasons) {
      const item = document.createElement("li");
      item.className = "evidence-item";
      const reasonTitle = document.createElement("strong");
      reasonTitle.textContent = reason;
      const explanation = document.createElement("p");
      explanation.textContent = detailedEvidenceMessage(reason);
      item.append(reasonTitle, explanation);
      list.append(item);
    }
    section.append(list);
  }
  analysisDetails.append(section);
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

function collectEvidenceGroups(result) {
  const evidence = Array.isArray(result.evidence) ? result.evidence : [];
  const group = (kinds) => [...new Set(evidence
    .filter((item) => kinds.includes(item?.kind))
    .map((item) => conciseEvidenceMessage(item?.message))
    .filter(Boolean))];
  return {
    ai: group(["strong", "weak"]),
    human: group(["human"]),
    notes: group(["info"]),
  };
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

function resultHasDirectEvidence(result) {
  return (result.evidence || []).some((item) => item.kind === "strong");
}

function resultNeedsMoreSamples(result) {
  return result?.label === "uncertain" && (result.evidence || []).some((item) =>
    /at least \d+ independently checked samples are required/i.test(item.message || "")
  );
}

function detailedResultSummary(result) {
  const segments = resultSegmentCounts(result);
  if (resultHasDirectEvidence(result)) {
    return "The page contains a direct AI-use disclosure or an unedited chatbot phrase.";
  }
  const percentage = calibratedPercentage(result.ai_probability);
  if (percentage !== null) {
    if (resultNeedsMoreSamples(result)) {
      return `The AI match is ${percentage}%, but only ${segments.checked} article sample${segments.checked === 1 ? " was" : "s were"} available. More text is required for a strong result.`;
    }
    if (result.label === "ai_likely") {
      return `The ${percentage}% AI match crossed the strong-AI decision limit using ${segments.checked} article samples.`;
    }
    if (result.label === "human_likely") {
      return `The ${percentage}% AI match stayed below the human-text decision limit.`;
    }
    return `The ${percentage}% AI match stayed between the two decision limits, so the result remains uncertain.`;
  }
  if (result.label === "ai_likely") {
    return `All ${segments.checked} independently checked text samples were AI-like.`;
  }
  if (result.label === "human_likely") {
    return `None of the ${segments.checked} checked text samples were AI-like.`;
  }
  return `${segments.ai} of ${segments.checked} checked samples were AI-like, so the result remains mixed.`;
}

function detailedEvidenceMessage(reason) {
  const aiSampleMatch = reason.match(/^(\d+) of (\d+) text samples were AI-like\.$/i);
  if (aiSampleMatch) {
    return `${aiSampleMatch[1]} independently checked article samples crossed the model's decision boundary.`;
  }
  const nonAiSampleMatch = reason.match(
    /^(\d+) of (\d+) text samples were not AI-like\.$/i,
  );
  if (nonAiSampleMatch) {
    return `${nonAiSampleMatch[1]} independently checked article samples stayed below the model's decision boundary.`;
  }
  const explanations = {
    "Page discloses AI use.": "The page explicitly says the article was written, generated, or prepared with AI.",
    "Unedited chatbot phrase found.": "The text contains wording normally produced when a chatbot describes its identity or limitations.",
    "Very uniform sentence lengths.": "Sentence lengths vary unusually little across the analyzed article.",
    "Very uniform paragraph lengths.": "Paragraph lengths follow an unusually regular pattern across the article.",
    "Many formulaic phrases.": "Stock phrases such as “in conclusion” appear unusually often.",
    "Repeated wording patterns.": "The same four-word sequences recur unusually often.",
    "AI-like writing patterns.": "Word choices and sequence patterns are closer to AI-generated English examples.",
    "Human-like writing patterns.": "Word choices and sequence patterns are closer to human-written English examples.",
    "Named author or editorial team.": "The page identifies a named author or editorial team.",
    "Sources or references found.": "The article includes citations, source links, or a references section.",
    "No clear style indicators.": "No strong sentence, paragraph, phrase, author, or citation signal was detected.",
    "No clear writing-pattern signal.": "The writing-pattern analysis did not strongly match either side.",
    "Large page: only part was analyzed.": "The score uses only the part of the page that could be processed.",
    "Additional classifier unavailable.": "The optional secondary check did not contribute evidence.",
    "Fewer than 3 samples were available.": "A strong AI result requires at least three separately checked samples.",
    "Text samples produced mixed results.": "Different parts of the article landed on different sides of the decision boundary.",
    "Score stayed between the decision limits.": "The score was not strong enough for either a human or AI result.",
    "Additional check was AI-like.": "An independent secondary check also marked the text as AI-like.",
    "Additional check was not AI-like.": "An independent secondary check did not mark the text as AI-like.",
  };
  return explanations[reason]
    || "This signal was found in the analyzed article text or page metadata.";
}

function userFacingFailure(response) {
  const result = response?.result;
  const rawMessage = String(result?.error || response?.error || "");
  const errorCode = String(result?.error_code || "");
  if (/open a normal article/i.test(rawMessage)) {
    return {
      summary: "Open an article first",
      details: "This button works on normal article pages.",
    };
  }
  if (/extract article text|cannot access|chrome:\/\//i.test(rawMessage)) {
    return {
      summary: "This page cannot be analyzed",
      details: "Open a normal article page and try again.",
    };
  }
  const serviceFailure = [
    "backend_unavailable",
    "detector_unavailable",
    "internal_error",
  ].includes(errorCode) || /connect|server|http \d{3}|timed? out|timeout|network|fetch/i.test(rawMessage);
  if (serviceFailure) {
    return {
      summary: "Service temporarily unavailable",
      details: "Try again in a moment.",
    };
  }
  return {
    summary: "Could not analyze this page",
    details: rawMessage || "Try another article.",
  };
}
