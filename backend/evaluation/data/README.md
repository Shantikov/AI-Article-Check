# Benchmark data

Run `python -m evaluation.build_benchmark` from `backend` to create the
deterministic 400-document HC3 benchmark here. Source answers are combined into
paired, article-sized documents of 660 to 1,100 words so every evaluated text
uses at least three detector samples. Generated text data is excluded
from the repository archive; the builder records the pinned source revision,
license, class balance, split counts, and SHA-256 digest in a metadata file.

Human and ChatGPT answers from the same HC3 question always remain in the same
split. The default split is 75% for calibration and 25% for held-out testing.
If one domain has fewer than 50 eligible composed pairs, the builder automatically
redistributes the shortfall across the other domains while keeping 400 texts.

Run `python -m evaluation.build_web_benchmark` to additionally create
`web_benchmark.jsonl`. This independent evaluation set contains 800 natural
MAGE test records, balanced 400/400 by label and 50/50 within each of 8 domains.
It is never consumed by the calibration command. Its metadata pins the source
revision, license, selection settings, class/domain counts, generator counts,
word-count range, and digest.

Run `python -m evaluation.build_validation_benchmark` to create a separate
1,600-text subset from MAGE `valid.csv`. It is balanced 800/800 and is the only
MAGE file consumed by `evaluation.calibrate_length_aware`. The calibration
command refuses `external_test` records, while the external evaluator refuses
`calibration` records, preventing accidental test leakage.

`python -m evaluation.evaluate_web` stores compatible raw model outputs in
`web_scores.jsonl`. These scores contain no source text and can be reused when
only the calibration profile changes. Pass `--rescore` after changing the ONNX
model. Generated benchmarks and score caches remain excluded from the archive.
Length-aware calibration similarly caches validation outputs in
`web_validation_scores.jsonl`.

`python -m evaluation.compare_models` keeps revision-aware validation and test
score caches under `model_comparison/`. The existing TMR caches are imported
when repository, filename, benchmark digest, IDs, and score shape are
compatible. Challenger caches always include the pinned model revision.
