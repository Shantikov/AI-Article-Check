import re
import statistics
from collections import Counter

from .models import DetectorOutput, Evidence, ExtractedArticle


_WORD_RE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)
_SENTENCE_RE = re.compile(r"[^.!?。！？]+[.!?。！？]", re.UNICODE)

_DISCLOSURE_PATTERNS = (
    r"(?:written|generated|created|produced)\s+(?:by|with|using)\s+(?:an?\s+)?(?:ai|artificial intelligence)",
    r"ai[- ]generated (?:article|content|text)",
    r"(?:создан|написан|сгенерирован)[а-яё ]{0,35}(?:ии|искусственн(?:ым|ого) интеллект)",
    r"(?:создано|подготовлено)\s+при\s+помощи\s+ии",
    r"(?:인공지능|ai)(?:으로|를 활용해?|가)\s*(?:생성|작성)",
)

_MODEL_LEAK_PATTERNS = (
    r"as an ai language model",
    r"i(?:'m| am) unable to browse the internet",
    r"как (?:языковая )?модель искусственного интеллекта",
    r"я не могу просматривать интернет в реальном времени",
    r"regenerate response",
)

_FORMULAIC_PHRASES = (
    "in conclusion",
    "it is important to note",
    "in today's rapidly evolving",
    "delve into",
    "a testament to",
    "в заключение",
    "важно отметить",
    "в современном быстро меняющемся мире",
    "следует подчеркнуть",
)


def _coefficient_of_variation(values: list[int]) -> float:
    if len(values) < 4:
        return 1.0
    mean = statistics.fmean(values)
    if mean == 0:
        return 1.0
    return statistics.pstdev(values) / mean


def _repeated_fourgram_ratio(words: list[str]) -> float:
    if len(words) < 80:
        return 0.0
    grams = list(zip(words, words[1:], words[2:], words[3:]))
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(grams))


def analyze_heuristically(article: ExtractedArticle) -> DetectorOutput:
    text = article.text
    lowered = text.casefold()
    words = [word.casefold() for word in _WORD_RE.findall(text)]
    evidence: list[Evidence] = []

    disclosure_hits = sum(
        1 for pattern in _DISCLOSURE_PATTERNS if re.search(pattern, lowered, re.I)
    )
    model_leak_hits = sum(
        1 for pattern in _MODEL_LEAK_PATTERNS if re.search(pattern, lowered, re.I)
    )

    if disclosure_hits:
        evidence.append(
            Evidence(
                kind="strong",
                message="Page discloses AI use.",
            )
        )
        return DetectorOutput(
            label="ai_likely",
            evidence=evidence,
        )

    if model_leak_hits:
        evidence.append(
            Evidence(
                kind="strong",
                message="Unedited chatbot phrase found.",
            )
        )
        return DetectorOutput(
            label="ai_likely",
            evidence=evidence,
        )

    sentences = [
        _WORD_RE.findall(sentence)
        for sentence in _SENTENCE_RE.findall(text)
    ]
    sentence_lengths = [len(sentence) for sentence in sentences if len(sentence) >= 3]
    sentence_cv = _coefficient_of_variation(sentence_lengths)
    if len(sentence_lengths) >= 8 and sentence_cv < 0.36:
        evidence.append(
            Evidence(
                kind="weak",
                message="Very uniform sentence lengths.",
            )
        )

    paragraphs = [
        _WORD_RE.findall(line) for line in text.splitlines() if len(line) >= 40
    ]
    paragraph_lengths = [len(paragraph) for paragraph in paragraphs]
    paragraph_cv = _coefficient_of_variation(paragraph_lengths)
    if len(paragraph_lengths) >= 6 and paragraph_cv < 0.42:
        evidence.append(
            Evidence(
                kind="weak",
                message="Very uniform paragraph lengths.",
            )
        )

    formulaic_count = sum(lowered.count(phrase) for phrase in _FORMULAIC_PHRASES)
    normalized_formulaic = formulaic_count / max(1, len(words) / 500)
    if normalized_formulaic >= 2:
        evidence.append(
            Evidence(
                kind="weak",
                message="Many formulaic phrases.",
            )
        )

    repetition = _repeated_fourgram_ratio(words)
    if repetition > 0.035:
        evidence.append(
            Evidence(
                kind="weak",
                message="Repeated wording patterns.",
            )
        )

    if article.has_author:
        evidence.append(
            Evidence(kind="info", message="Named author or editorial team found.")
        )
    if article.has_citations:
        evidence.append(
            Evidence(kind="info", message="Sources or references found.")
        )

    if not evidence:
        evidence.append(
            Evidence(
                kind="info",
                message="No clear writing-style indicators.",
            )
        )

    return DetectorOutput(
        label="uncertain",
        evidence=evidence,
    )


def combine_with_local_model(
    heuristic: DetectorOutput,
    model: DetectorOutput,
) -> DetectorOutput:
    has_direct_ai_evidence = any(item.kind == "strong" for item in heuristic.evidence)
    label = "ai_likely" if has_direct_ai_evidence else model.label
    evidence = [*model.evidence, *heuristic.evidence]
    if has_direct_ai_evidence:
        evidence = [
            *[item for item in heuristic.evidence if item.kind == "strong"],
            *model.evidence,
            *[item for item in heuristic.evidence if item.kind != "strong"],
        ]

    return DetectorOutput(
        label=label,
        evidence=evidence,
        sampled_word_count=model.sampled_word_count,
        segments_checked=model.segments_checked,
        ai_segments=model.ai_segments,
        non_ai_segments=model.non_ai_segments,
        ai_probability=model.ai_probability,
    )


def combine_with_external(
    local: DetectorOutput,
    external_probability: float,
    _external_confidence: float | None,
) -> DetectorOutput:
    external_probability = min(1.0, max(0.0, external_probability))
    external_is_ai = external_probability > 0.5
    has_direct_ai_evidence = any(item.kind == "strong" for item in local.evidence)
    if has_direct_ai_evidence:
        label = "ai_likely"
    elif local.label == "ai_likely" and external_is_ai:
        label = "ai_likely"
    elif local.label == "human_likely" and not external_is_ai:
        label = "human_likely"
    else:
        label = "uncertain"

    if external_is_ai:
        external_evidence = Evidence(
            kind="weak",
            message="The additional classifier marked the text as AI-like.",
        )
    else:
        external_evidence = Evidence(
            kind="human",
            message="The additional classifier did not mark the text as AI-like.",
        )
    evidence = [*local.evidence, external_evidence]
    return DetectorOutput(
        label=label,
        evidence=evidence,
        sampled_word_count=local.sampled_word_count,
        segments_checked=local.segments_checked,
        ai_segments=local.ai_segments,
        non_ai_segments=local.non_ai_segments,
        ai_probability=local.ai_probability,
    )
