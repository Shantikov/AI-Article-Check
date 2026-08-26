import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MIN_DISPLAY_PROBABILITY = 0.01


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def display_percentage(probability: float) -> int:
    """Return an honest display value without implying absolute certainty."""
    percentage = round(min(1.0, max(0.0, probability)) * 100)
    return min(99, max(1, percentage))


@dataclass(frozen=True)
class CalibrationBand:
    """Calibration curve and decision limits for one text-length band."""

    slope: float
    intercept: float
    feature_mean: float
    feature_scale: float
    human_max: float
    ai_min: float
    probability_floor: float = MIN_DISPLAY_PROBABILITY

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationBand":
        calibrator = payload.get("calibrator", {})
        if calibrator.get("kind") != "regularized_platt":
            raise ValueError("Unsupported calibration method")
        values = {
            "slope": float(calibrator["slope"]),
            "intercept": float(calibrator["intercept"]),
            "feature_mean": float(calibrator["feature_mean"]),
            "feature_scale": float(calibrator["feature_scale"]),
            "probability_floor": float(
                calibrator.get("probability_floor", MIN_DISPLAY_PROBABILITY)
            ),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Calibration parameters must be finite")
        if values["slope"] <= 0:
            raise ValueError("Calibration slope must be positive")
        if values["feature_scale"] <= 0:
            raise ValueError("Calibration feature scale must be positive")
        if not 0 < values["probability_floor"] < 0.5:
            raise ValueError("Calibration probability floor must be between 0 and 0.5")

        thresholds = payload.get("thresholds", {})
        human_max = float(thresholds["human_max"])
        ai_min = float(thresholds["ai_min"])
        if not 0 <= human_max < ai_min <= 1:
            raise ValueError("Calibration thresholds must satisfy 0 <= human < AI <= 1")
        return cls(**values, human_max=human_max, ai_min=ai_min)

    def predict(self, raw_score: float) -> float:
        value = min(1.0 - 1e-6, max(1e-6, float(raw_score)))
        feature = math.log(value / (1.0 - value))
        standardized = (feature - self.feature_mean) / self.feature_scale
        probability = _sigmoid(self.slope * standardized + self.intercept)
        return min(
            1.0 - self.probability_floor,
            max(self.probability_floor, probability),
        )

    @property
    def human_limit_percentage(self) -> int:
        return max(1, math.floor(self.human_max * 100 + 1e-9))

    @property
    def ai_limit_percentage(self) -> int:
        return min(99, math.ceil(self.ai_min * 100 - 1e-9))


def segment_band(segments_checked: int) -> str:
    if segments_checked <= 1:
        return "1_segment"
    if segments_checked == 2:
        return "2_segments"
    return "3_or_more_segments"


@dataclass(frozen=True)
class CalibrationProfile:
    """A regularized Platt mapping from article score to AI probability."""

    slope: float
    intercept: float
    feature_mean: float
    feature_scale: float
    human_max: float
    ai_min: float
    min_ai_segments: int = 3
    probability_floor: float = MIN_DISPLAY_PROBABILITY
    dataset: str = "unknown"
    bands: dict[str, CalibrationBand] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationProfile":
        schema_version = int(payload.get("schema_version", 0))
        if schema_version == 4:
            band_payloads = payload.get("bands", {})
            required = {"1_segment", "2_segments", "3_or_more_segments"}
            if set(band_payloads) != required:
                raise ValueError("Length-aware calibration must contain all three bands")
            bands = {
                key: CalibrationBand.from_dict(band_payloads[key])
                for key in sorted(required)
            }
            default_band = bands["3_or_more_segments"]
            profile = cls(
                slope=default_band.slope,
                intercept=default_band.intercept,
                feature_mean=default_band.feature_mean,
                feature_scale=default_band.feature_scale,
                human_max=default_band.human_max,
                ai_min=default_band.ai_min,
                min_ai_segments=1,
                probability_floor=default_band.probability_floor,
                dataset=str(payload.get("dataset", "unknown")),
                bands=bands,
            )
            for band in bands.values():
                if band.human_limit_percentage >= band.ai_limit_percentage:
                    raise ValueError("Displayed calibration thresholds must not overlap")
            return profile

        # Schema 3 is retained for existing single-curve calibration files.
        if schema_version != 3:
            raise ValueError("Unsupported calibration profile version")

        legacy_band = CalibrationBand.from_dict({
            "calibrator": payload.get("calibrator", {}),
            "thresholds": payload.get("thresholds", {}),
        })
        thresholds = payload.get("thresholds", {})

        profile = cls(
            slope=legacy_band.slope,
            intercept=legacy_band.intercept,
            feature_mean=legacy_band.feature_mean,
            feature_scale=legacy_band.feature_scale,
            human_max=legacy_band.human_max,
            ai_min=legacy_band.ai_min,
            min_ai_segments=max(1, int(thresholds.get("min_ai_segments", 3))),
            probability_floor=legacy_band.probability_floor,
            dataset=str(payload.get("dataset", "unknown")),
        )
        if profile.human_limit_percentage >= profile.ai_limit_percentage:
            raise ValueError("Displayed calibration thresholds must not overlap")
        return profile

    @classmethod
    def from_file(cls, path: str | Path) -> "CalibrationProfile":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def band_for(self, segments_checked: int) -> CalibrationBand | None:
        if not self.bands:
            return None
        return self.bands[segment_band(segments_checked)]

    def predict(self, raw_score: float, segments_checked: int = 3) -> float:
        """Transform an article score into a smooth, regularized probability."""
        band = self.band_for(segments_checked)
        if band:
            return band.predict(raw_score)
        value = min(1.0 - 1e-6, max(1e-6, float(raw_score)))
        feature = math.log(value / (1.0 - value))
        standardized = (feature - self.feature_mean) / self.feature_scale
        probability = _sigmoid(self.slope * standardized + self.intercept)
        return min(
            1.0 - self.probability_floor,
            max(self.probability_floor, probability),
        )

    @staticmethod
    def display_percentage(probability: float) -> int:
        return display_percentage(probability)

    @property
    def human_limit_percentage(self) -> int:
        return max(1, math.floor(self.human_max * 100 + 1e-9))

    @property
    def ai_limit_percentage(self) -> int:
        return min(99, math.ceil(self.ai_min * 100 - 1e-9))

    def classify(self, probability: float, segments_checked: int) -> str:
        band = self.band_for(segments_checked)
        human_limit = (
            band.human_limit_percentage if band else self.human_limit_percentage
        )
        ai_limit = band.ai_limit_percentage if band else self.ai_limit_percentage
        percentage = self.display_percentage(probability)
        if percentage <= human_limit:
            return "human_likely"
        if percentage >= ai_limit and (
            band is not None or segments_checked >= self.min_ai_segments
        ):
            return "ai_likely"
        return "uncertain"

    def needs_more_samples(self, probability: float, segments_checked: int) -> bool:
        if self.bands:
            return False
        return (
            self.display_percentage(probability) >= self.ai_limit_percentage
            and segments_checked < self.min_ai_segments
        )


def load_calibration(path: str | Path | None) -> CalibrationProfile | None:
    if not path:
        return None
    calibration_path = Path(path)
    if not calibration_path.is_file():
        return None
    try:
        return CalibrationProfile.from_file(calibration_path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
