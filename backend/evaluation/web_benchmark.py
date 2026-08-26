import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


SOURCE_REPO = "yaful/MAGE"
SOURCE_REVISION = "342663f0a2b775455c023f5d36a1341ff0ec5402"
SOURCE_FILENAME = "test.csv"
SOURCE_LICENSE = "Apache-2.0"
DEFAULT_DOMAINS = (
    "cmv",
    "eli5",
    "sci_gen",
    "squad",
    "tldr",
    "wp",
    "xsum",
    "yelp",
)

_GENERATION_MODES = ("continuation", "specified", "topical")
_GENERATOR_NAMES = {
    "7B": "llama_7b",
    "13B": "llama_13b",
    "30B": "llama_30b",
    "65B": "llama_65b",
    "gpt-3.5-trubo": "gpt-3.5-turbo",
}


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").split())


def parse_source(source: str, label: str) -> tuple[str, str, str]:
    """Return MAGE domain, generator and generation mode."""
    source = source.strip()
    if label == "human":
        if not source.endswith("_human"):
            raise ValueError(f"Unexpected human source: {source}")
        return source.removesuffix("_human"), "human", "human"

    marker = "_machine_"
    if marker not in source:
        raise ValueError(f"Unexpected AI source: {source}")
    domain, details = source.split(marker, 1)
    mode = "unknown"
    generator = details
    for candidate in _GENERATION_MODES:
        prefix = candidate + "_"
        if details.startswith(prefix):
            mode = candidate
            generator = details.removeprefix(prefix)
            break
    generator = _GENERATOR_NAMES.get(generator, generator).lower()
    return domain, generator, mode


def load_csv_rows(path: str | Path) -> Iterable[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _balanced_take(
    groups: dict[str, list[dict]],
    total: int,
) -> list[dict]:
    """Take records round-robin across groups with deterministic ordering."""
    ordered = {
        key: sorted(items, key=lambda item: item["rank"])
        for key, items in sorted(groups.items())
        if items
    }
    selected: list[dict] = []
    offset = 0
    while len(selected) < total:
        progress = False
        for key in ordered:
            items = ordered[key]
            if offset >= len(items):
                continue
            selected.append(items[offset])
            progress = True
            if len(selected) == total:
                break
        if not progress:
            break
        offset += 1
    return selected


def build_web_records(
    rows: Iterable[dict[str, str]],
    *,
    domains: tuple[str, ...] = DEFAULT_DOMAINS,
    records_per_domain: int = 100,
    min_words: int = 80,
    split_name: str = "external_test",
) -> list[dict]:
    """Build an evaluation-only, balanced and deterministic MAGE subset.

    Every selected domain contains the same number of human and machine texts.
    Machine records are selected round-robin across generators so one model
    family cannot dominate the result.
    """
    if records_per_domain < 4 or records_per_domain % 2:
        raise ValueError("records_per_domain must be an even number of at least 4")
    if min_words < 40:
        raise ValueError("min_words must be at least 40")
    if not domains or len(set(domains)) != len(domains):
        raise ValueError("domains must be non-empty and unique")
    if not split_name.strip():
        raise ValueError("split_name must not be empty")

    wanted_domains = set(domains)
    candidates: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen_texts: set[str] = set()
    for row_index, row in enumerate(rows):
        raw_label = str(row.get("label", "")).strip()
        if raw_label not in {"0", "1"}:
            continue
        label = "human" if raw_label == "1" else "ai"
        try:
            domain, generator, generation_mode = parse_source(
                str(row.get("src", "")), label
            )
        except ValueError:
            continue
        if domain not in wanted_domains:
            continue

        text = normalize_text(row.get("text"))
        words = len(text.split())
        if words < min_words:
            continue
        text_hash = stable_hash(text)
        if text_hash in seen_texts:
            continue
        seen_texts.add(text_hash)
        candidates[(domain, label)].append({
            "rank": stable_hash(
                f"{domain}\0{label}\0{generator}\0{generation_mode}\0{text_hash}"
            ),
            "row_index": row_index,
            "text_hash": text_hash,
            "text": text,
            "word_count": words,
            "domain": domain,
            "label": label,
            "generator": generator,
            "generation_mode": generation_mode,
            "source_detail": str(row.get("src", "")).strip(),
        })

    per_label = records_per_domain // 2
    selected: list[dict] = []
    for domain in domains:
        humans = sorted(
            candidates[(domain, "human")], key=lambda item: item["rank"]
        )[:per_label]
        ai_groups: dict[str, list[dict]] = defaultdict(list)
        for item in candidates[(domain, "ai")]:
            # Most generators provide continuations, while a few also provide
            # topical and instruction-specified generations. Balance both axes
            # so those less common generation modes remain represented.
            group_key = f"{item['generator']}:{item['generation_mode']}"
            ai_groups[group_key].append(item)
        machines = _balanced_take(ai_groups, per_label)
        if len(humans) < per_label or len(machines) < per_label:
            raise ValueError(
                f"Domain '{domain}' has {len(humans)} eligible human and "
                f"{len(machines)} eligible AI texts; {per_label} of each required"
            )
        selected.extend(humans)
        selected.extend(machines)

    records: list[dict] = []
    for item in selected:
        record_id = (
            f"mage:{item['domain']}:{item['label']}:"
            f"{item['text_hash'][:16]}"
        )
        records.append({
            "id": record_id,
            "split": split_name,
            "label": item["label"],
            "domain": item["domain"],
            "generator": item["generator"],
            "generation_mode": item["generation_mode"],
            "source": SOURCE_REPO,
            "source_detail": item["source_detail"],
            "word_count": item["word_count"],
            "text": item["text"],
        })
    return sorted(records, key=lambda item: item["id"])


def summarize_web_records(records: Iterable[dict]) -> dict:
    items = list(records)
    labels = Counter(item["label"] for item in items)
    domains = Counter(item["domain"] for item in items)
    domain_labels = Counter(
        (item["domain"], item["label"]) for item in items
    )
    generators = Counter(
        item["generator"] for item in items if item["label"] == "ai"
    )
    modes = Counter(
        item["generation_mode"] for item in items if item["label"] == "ai"
    )
    words = [int(item["word_count"]) for item in items]
    return {
        "records": len(items),
        "labels": dict(sorted(labels.items())),
        "domains": dict(sorted(domains.items())),
        "domain_labels": {
            f"{domain}_{label}": count
            for (domain, label), count in sorted(domain_labels.items())
        },
        "ai_generators": dict(sorted(generators.items())),
        "ai_generation_modes": dict(sorted(modes.items())),
        "word_counts": {
            "minimum": min(words) if words else 0,
            "maximum": max(words) if words else 0,
            "mean": sum(words) / len(words) if words else 0.0,
        },
    }
