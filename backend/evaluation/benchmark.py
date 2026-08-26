import hashlib
import json
from pathlib import Path
from typing import Iterable


SOURCE_REPO = "Hello-SimpleAI/HC3"
SOURCE_REVISION = "fc599255a8541d147f644131d6401a267b8f8b7b"
SOURCE_LICENSE = "CC-BY-SA-4.0"
DEFAULT_DOMAINS = ("wiki_csai", "open_qa", "finance", "medicine")


def word_count(text: str) -> int:
    return len(text.split())


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _select_answer(answers: object, key: str, min_words: int) -> str | None:
    if not isinstance(answers, list):
        return None
    eligible = [
        str(answer).strip()
        for answer in answers
        if isinstance(answer, str) and word_count(answer) >= min_words
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda answer: stable_hash(f"{key}\0{answer}"))


def _join_parts(parts: list[str], max_words: int) -> str:
    words = "\n\n".join(parts).split()
    return " ".join(words[:max_words])


def _compose_documents(
    candidates: list[tuple[str, str, str, str]],
    *,
    min_document_words: int,
    max_document_words: int,
) -> list[tuple[str, str, str, str]]:
    """Combine paired answers into article-sized paired documents."""
    documents: list[tuple[str, str, str, str]] = []
    ranks: list[str] = []
    source_ids: list[str] = []
    human_parts: list[str] = []
    ai_parts: list[str] = []
    human_words = 0
    ai_words = 0

    for rank, source_id, human, ai in candidates:
        ranks.append(rank)
        source_ids.append(source_id)
        human_parts.append(human)
        ai_parts.append(ai)
        human_words += word_count(human)
        ai_words += word_count(ai)
        if human_words < min_document_words or ai_words < min_document_words:
            continue

        composite_key = "|".join(ranks)
        documents.append((
            stable_hash(composite_key),
            stable_hash("|".join(source_ids))[:16],
            _join_parts(human_parts, max_document_words),
            _join_parts(ai_parts, max_document_words),
        ))
        ranks = []
        source_ids = []
        human_parts = []
        ai_parts = []
        human_words = 0
        ai_words = 0

    return documents


def build_records(
    rows_by_domain: dict[str, Iterable[dict]],
    *,
    pairs_per_domain: int = 50,
    calibration_fraction: float = 0.75,
    min_words: int = 40,
    min_document_words: int = 660,
    max_document_words: int = 1_100,
) -> list[dict]:
    """Create a balanced, paired and deterministic HC3 subset.

    ``pairs_per_domain`` defines the target average. If one domain contains too
    few long pairs, its shortfall is distributed across the remaining domains
    while the requested total size stays unchanged.
    """
    if pairs_per_domain < 2:
        raise ValueError("pairs_per_domain must be at least 2")
    if not 0.5 <= calibration_fraction <= 0.9:
        raise ValueError("calibration_fraction must be between 0.5 and 0.9")
    if min_document_words < min_words:
        raise ValueError("min_document_words cannot be smaller than min_words")
    if max_document_words < min_document_words:
        raise ValueError("max_document_words cannot be smaller than min_document_words")

    candidates_by_domain: dict[str, list[tuple[str, str, str, str]]] = {}
    seen_texts: set[str] = set()
    for domain in sorted(rows_by_domain):
        candidates: list[tuple[str, str, str, str]] = []
        for row_index, row in enumerate(rows_by_domain[domain]):
            source_id = str(row.get("id", row_index))
            pair_key = f"{domain}:{source_id}:{row.get('question', '')}"
            human = _select_answer(row.get("human_answers"), f"{pair_key}:human", min_words)
            ai = _select_answer(row.get("chatgpt_answers"), f"{pair_key}:ai", min_words)
            if not human or not ai or human == ai:
                continue
            human_hash = stable_hash(human)
            ai_hash = stable_hash(ai)
            if human_hash in seen_texts or ai_hash in seen_texts:
                continue
            candidates.append((stable_hash(pair_key), source_id, human, ai))
            seen_texts.update((human_hash, ai_hash))

        candidates_by_domain[domain] = _compose_documents(
            sorted(candidates, key=lambda item: item[0]),
            min_document_words=min_document_words,
            max_document_words=max_document_words,
        )

    target_pairs = pairs_per_domain * len(candidates_by_domain)
    available_pairs = sum(len(items) for items in candidates_by_domain.values())
    if available_pairs < target_pairs:
        availability = ", ".join(
            f"{domain}={len(items)}"
            for domain, items in sorted(candidates_by_domain.items())
        )
        raise ValueError(
            f"Only {available_pairs} eligible pairs are available; "
            f"{target_pairs} required ({availability})"
        )

    quotas = {
        domain: min(pairs_per_domain, len(items))
        for domain, items in candidates_by_domain.items()
    }
    deficit = target_pairs - sum(quotas.values())
    while deficit:
        progress = False
        for domain in sorted(candidates_by_domain):
            if quotas[domain] >= len(candidates_by_domain[domain]):
                continue
            quotas[domain] += 1
            deficit -= 1
            progress = True
            if not deficit:
                break
        if not progress:
            raise ValueError("Could not distribute the requested benchmark pairs")

    records: list[dict] = []
    for domain in sorted(candidates_by_domain):
        selected = candidates_by_domain[domain][:quotas[domain]]
        calibration_pairs = round(len(selected) * calibration_fraction)
        calibration_pairs = min(len(selected) - 1, max(1, calibration_pairs))
        for index, (_rank, source_id, human, ai) in enumerate(selected):
            split = "calibration" if index < calibration_pairs else "test"
            pair_id = f"hc3:{domain}:{source_id}"
            records.extend((
                {
                    "id": f"{pair_id}:human",
                    "pair_id": pair_id,
                    "split": split,
                    "label": "human",
                    "domain": domain,
                    "source": SOURCE_REPO,
                    "text": human,
                },
                {
                    "id": f"{pair_id}:ai",
                    "pair_id": pair_id,
                    "split": split,
                    "label": "ai",
                    "domain": domain,
                    "source": SOURCE_REPO,
                    "text": ai,
                },
            ))
    return records


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dataset_digest(records: Iterable[dict]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
