# Calibration reports

`python -m evaluation.calibrate` writes the latest held-out evaluation report
to this directory. Reports include the confusion counts, false-positive rate,
false-negative rate, coverage, decided accuracy, Brier score, log loss, and
per-domain metrics. They also include the sample-count and displayed-probability
distributions plus an `accepted` field. A failed quality gate writes a separate
rejected candidate instead of activating it as `backend/calibration.json`.

`python -m evaluation.evaluate_web` writes `web_latest.json`. This is a separate
evaluation-only report over 800 natural MAGE test records. It includes overall,
per-domain, per-generator, per-generation-mode, and per-length metrics, plus
probability/segment distributions and compact references to the strongest
errors. Creating this report never changes calibration parameters or thresholds.

`python -m evaluation.calibrate_length_aware` writes
`length_calibration.json` with fit-set diagnostics for the 1-, 2-, and
3+-sample curves. Those diagnostics verify the requested fit constraints but
are not final accuracy; `web_latest.json` remains the untouched test result.

`python -m evaluation.compare_models` writes `model_comparison_latest.json` and
candidate details under `model_comparison/`. The report records the deterministic
validation fit/selection split, pinned candidate metadata, selection failures,
quality gates, scoring time, full calibration profiles, final external reports,
and a keep/replace recommendation. It never activates the recommendation.
