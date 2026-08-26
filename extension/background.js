const BUILD_MODE = "local";
const DEFAULT_SETTINGS = Object.freeze({
  apiBase: "http://127.0.0.1:8787",
  autoResults: 6,
});

const CACHE_TTL_MS = 24 * 60 * 60 * 1000;
const MAX_CACHE_ITEMS = 200;

chrome.runtime.onInstalled.addListener(async () => {
  const stored = await chrome.storage.sync.get({
    apiBase: DEFAULT_SETTINGS.apiBase,
    autoResults: null,
    maxResults: null,
  });
  await chrome.storage.sync.set({
    apiBase: selectedApiBase(stored.apiBase),
    autoResults: normalizeAutoResults(stored.autoResults, stored.maxResults),
  });
  await chrome.storage.sync.remove("maxResults");
  // Do not keep stale classifications across extension updates.
  await chrome.storage.local.remove("analysisCache");
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "GET_SETTINGS") {
    getSettings().then(sendResponse);
    return true;
  }

  if (message?.type === "GET_CURRENT_TAB_CONTEXT") {
    getCurrentTabContext().then(sendResponse);
    return true;
  }

  if (message?.type === "ANALYZE_URLS") {
    analyzeUrls(message.urls, message.force === true).then(sendResponse);
    return true;
  }

  if (message?.type === "GET_CACHED_RESULTS") {
    getCachedResults(message.urls).then(sendResponse);
    return true;
  }

  if (message?.type === "ANALYZE_CURRENT_TAB") {
    analyzeCurrentTab().then(sendResponse);
    return true;
  }

  if (message?.type === "PING_BACKEND") {
    pingBackend().then(sendResponse);
    return true;
  }

  return false;
});

async function getSettings() {
  const stored = await chrome.storage.sync.get({
    apiBase: DEFAULT_SETTINGS.apiBase,
    autoResults: null,
    maxResults: null,
  });
  return {
    apiBase: selectedApiBase(stored.apiBase),
    autoResults: normalizeAutoResults(stored.autoResults, stored.maxResults),
  };
}

function selectedApiBase(storedValue) {
  if (BUILD_MODE === "public") return DEFAULT_SETTINGS.apiBase;
  return normalizeApiBase(storedValue || DEFAULT_SETTINGS.apiBase);
}

async function getCurrentTabContext() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    return {
      ok: true,
      canAnalyze: Boolean(tab?.id && isPublicHttpUrl(tab.url)),
    };
  } catch {
    return { ok: false, canAnalyze: false };
  }
}

function normalizeAutoResults(value, legacyValue) {
  const numeric = Number(value);
  if ([3, 6, 8].includes(numeric)) return numeric;
  const legacy = Number(legacyValue);
  if (legacy === 3) return 3;
  if (legacy === 10) return 8;
  return DEFAULT_SETTINGS.autoResults;
}

function normalizeApiBase(value) {
  return String(value || "").trim().replace(/\/+$/, "");
}

async function pingBackend() {
  const { apiBase } = await getSettings();
  try {
    const response = await fetch(`${apiBase}/health`, {
      method: "GET",
      signal: AbortSignal.timeout(6000),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    return {
      ok: payload.status === "ok",
      detector: payload.detector || "unknown",
      version: payload.version || "unknown",
      modelConfigured: payload.model_configured === true,
      modelLoaded: payload.model_loaded === true,
      calibrated: payload.calibrated === true,
      calibrationDataset: payload.calibration_dataset || "unknown",
    };
  } catch (error) {
    return { ok: false, error: readableError(error) };
  }
}

async function analyzeUrls(inputUrls, force = false) {
  const urls = [...new Set((inputUrls || []).filter(isPublicHttpUrl))].slice(0, 10);
  if (!urls.length) return { ok: true, results: [] };

  const cache = await loadCache();
  const now = Date.now();
  const resultsByUrl = new Map();
  const missing = [];

  for (const url of urls) {
    const key = cacheKey(url);
    const item = cache[key];
    if (force) {
      delete cache[key];
      missing.push(url);
    } else if (item && item.expiresAt > now) {
      resultsByUrl.set(url, { ...item.result, cache_hit: true });
    } else {
      missing.push(url);
    }
  }

  if (missing.length) {
    const { apiBase } = await getSettings();
    try {
      const response = await fetch(`${apiBase}/api/v1/analyze/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls: missing, force }),
        signal: AbortSignal.timeout(120000),
      });
      await requireSuccessfulResponse(response);
      const payload = await response.json();
      for (const result of payload.results || []) {
        resultsByUrl.set(result.url, result);
        if (result.status === "ok") {
          storeCachedResult(cache, result, [result.url, result.final_url], now);
        }
      }
      await saveCache(cache);
    } catch (error) {
      for (const url of missing) {
        resultsByUrl.set(url, {
          url,
          status: "error",
          label: "unavailable",
          error: readableError(error),
          error_code: error?.apiCode || "backend_unavailable",
          retryable: true,
        });
      }
    }
  }

  return { ok: true, results: urls.map((url) => resultsByUrl.get(url)) };
}

async function getCachedResults(inputUrls) {
  const urls = [...new Set((inputUrls || []).filter(isPublicHttpUrl))].slice(0, 30);
  const cache = await loadCache();
  const now = Date.now();
  const results = [];
  for (const url of urls) {
    const item = cache[cacheKey(url)];
    if (item?.expiresAt > now) {
      results.push({ ...item.result, url, cache_hit: true });
    }
  }
  return { ok: true, results };
}

async function analyzeCurrentTab() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id || !isPublicHttpUrl(tab.url)) {
      throw new Error("Open a normal article page first");
    }

    const [injection] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractRenderedArticle,
    });
    const page = injection?.result;
    if (!page?.text) {
      throw new Error("Could not extract article text from this page");
    }

    const { apiBase } = await getSettings();
    const response = await fetch(`${apiBase}/api/v1/analyze/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: page.url,
        canonical_url: page.canonicalUrl,
        title: page.title,
        text: page.text,
        has_author: page.hasAuthor,
        has_citations: page.hasCitations,
      }),
      signal: AbortSignal.timeout(120000),
    });
    await requireSuccessfulResponse(response);
    const result = await response.json();
    if (result.status === "ok") {
      const cache = await loadCache();
      storeCachedResult(
        cache,
        result,
        [page.url, page.canonicalUrl, result.url, result.final_url],
      );
      await saveCache(cache);
    }
    return {
      ok: result.status === "ok",
      result,
      error: result.error || null,
      extractedWords: page.wordCount,
    };
  } catch (error) {
    return { ok: false, error: readableError(error) };
  }
}

async function requireSuccessfulResponse(response) {
  if (response.ok) return;
  const error = new Error(
    response.status === 429
      ? "Too many checks were requested. Try again shortly"
      : `Analysis service returned HTTP ${response.status}`,
  );
  error.apiCode = response.status === 429 ? "rate_limited" : "backend_unavailable";
  throw error;
}

function extractRenderedArticle() {
  const normalizeText = (value) => String(value || "")
    .replace(/[\t\r\f\v ]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  const wordsIn = (value) => (
    String(value || "").match(/[A-Za-z]+(?:['’\-][A-Za-z]+)*/g) || []
  );

  const candidates = [...document.querySelectorAll(
    "article, main, [role='main'], [itemprop='articleBody'], .mw-parser-output",
  )];
  const root = candidates.length
    ? candidates.reduce((best, node) => (
        normalizeText(node.textContent).length > normalizeText(best.textContent).length
          ? node
          : best
      ))
    : document.body;
  if (!root) return null;

  const hasAuthor = Boolean(document.querySelector(
    "meta[name='author'], [rel~='author'], [itemprop='author'], .author, .byline, "
    + "[class*='author-' i], [class*='-author' i], [class*='byline-' i]",
  ));
  const rootText = normalizeText(root.textContent);
  const hasCitations = Boolean(root.querySelector(
    "sup.reference, .reflist, .references, [role='doc-bibliography']",
  )) || (
    /\b(?:sources|references|bibliography)\b/i.test(rootText)
    && Boolean(root.querySelector("a[href^='http']"))
  );
  const clone = root.cloneNode(true);
  clone.querySelectorAll(
    "script, style, noscript, template, svg, canvas, nav, header, footer, aside, "
    + "form, button, figure, figcaption, table, [hidden], [aria-hidden='true'], "
    + ".reflist, .references, .navbox, .infobox, .sidebar, .metadata",
  ).forEach((node) => node.remove());

  const blocks = [];
  const seen = new Set();
  for (const node of clone.querySelectorAll("p, blockquote")) {
    const text = normalizeText(node.textContent);
    const wordCount = wordsIn(text).length;
    const linkText = [...node.querySelectorAll("a")]
      .reduce((sum, link) => sum + normalizeText(link.textContent).length, 0);
    if (
      text.length >= 80
      && wordCount >= 12
      && linkText / Math.max(1, text.length) <= 0.45
      && !seen.has(text)
    ) {
      seen.add(text);
      blocks.push(text);
    }
  }

  let text = blocks.length
    ? blocks.join("\n")
    : normalizeText(clone.textContent);
  text = text.slice(0, 80000);
  const canonicalCandidate = document.querySelector("link[rel='canonical']")?.href;
  let canonicalUrl = null;
  try {
    const parsed = new URL(canonicalCandidate || "", location.href);
    if (["http:", "https:"].includes(parsed.protocol)) {
      parsed.hash = "";
      canonicalUrl = parsed.href;
    }
  } catch {
    canonicalUrl = null;
  }

  const currentUrl = new URL(location.href);
  currentUrl.hash = "";
  return {
    url: currentUrl.href,
    canonicalUrl,
    title: normalizeText(document.title).slice(0, 500) || null,
    text,
    wordCount: wordsIn(text).length,
    hasAuthor,
    hasCitations,
  };
}

function isPublicHttpUrl(value) {
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) && !isGoogleHost(url.hostname);
  } catch {
    return false;
  }
}

function isGoogleHost(hostname) {
  return /(^|\.)google\.[a-z.]+$/i.test(hostname);
}

async function loadCache() {
  const stored = await chrome.storage.local.get({ analysisCache: {} });
  return stored.analysisCache || {};
}

function cacheKey(value) {
  try {
    const url = new URL(value);
    url.hash = "";
    return url.href;
  } catch {
    return String(value || "");
  }
}

function storeCachedResult(cache, result, aliases = [], now = Date.now()) {
  const item = {
    expiresAt: now + CACHE_TTL_MS,
    result: { ...result, cache_hit: false },
  };
  for (const alias of aliases) {
    if (alias && isPublicHttpUrl(alias)) cache[cacheKey(alias)] = item;
  }
}

async function saveCache(cache) {
  const entries = Object.entries(cache)
    .filter(([, item]) => item?.expiresAt > Date.now())
    .sort((a, b) => b[1].expiresAt - a[1].expiresAt)
    .slice(0, MAX_CACHE_ITEMS);
  await chrome.storage.local.set({ analysisCache: Object.fromEntries(entries) });
}

function readableError(error) {
  if (error?.name === "TimeoutError") return "The server did not respond in time";
  if (error instanceof TypeError) return "Could not connect to the server";
  return error?.message || "Unknown error";
}
