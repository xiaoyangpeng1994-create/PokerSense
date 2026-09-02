"""Monotonic empirical-bin confidence calibration (MVP, no sklearn).

Maps a raw detector score to a calibrated confidence in [0,1] via piecewise-
constant monotonic bins learned from a calibration set.

The calibrator is NOT aware of Task 5 Frozen confidence thresholds. A detector
may additionally carry an ``abstain_floor`` (per-detector, versioned, learned
only from the calibration set) below which the raw score is considered
perceptually unreadable / out-of-domain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .protocols import CalibratedConfidence, _check_raw_score


@dataclass(frozen=True)
class CalibrationBins:
    """A monotonic piecewise-constant mapping: raw score -> calibrated confidence.

    ``edges`` are strictly increasing raw-score boundaries (length N+1),
    ``confidence`` are N calibrated values for the N bins. The mapping is
    monotonic non-decreasing by construction (validated at build time).
    """

    edges: tuple[float, ...]          # N+1 strictly increasing
    confidence: tuple[float, ...]     # N values in [0,1]

    def __post_init__(self) -> None:
        edges = tuple(self.edges)
        conf = tuple(self.confidence)
        if len(edges) < 2:
            raise ValueError("need at least two edges (one bin)")
        if len(conf) != len(edges) - 1:
            raise ValueError("confidence must have len(edges)-1 entries")
        for e in edges:
            # each edge must be a finite value in [0,1] (raw-score domain)
            _check_raw_score(e, "edge")
        # strictly increasing edges (check after the finite/range check)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                raise ValueError("edges must be strictly increasing")
        for c in conf:
            if isinstance(c, bool) or not isinstance(c, (int, float)):
                raise TypeError("confidence must be numeric")
            if not (0.0 <= float(c) <= 1.0):
                raise ValueError("confidence must be in [0,1]")
        # monotonic non-decreasing
        for i in range(1, len(conf)):
            if conf[i] < conf[i - 1]:
                raise ValueError("confidence must be monotonic non-decreasing")
        object.__setattr__(self, "edges", tuple(float(e) for e in edges))
        object.__setattr__(self, "confidence", tuple(float(c) for c in conf))

    def map(self, raw_score: float) -> float:
        """Return calibrated confidence for a raw score (piecewise-constant)."""
        _check_raw_score(raw_score, "raw_score")
        if raw_score <= self.edges[0]:
            return self.confidence[0]
        if raw_score >= self.edges[-1]:
            return self.confidence[-1]
        for i in range(len(self.confidence)):
            if self.edges[i] <= raw_score < self.edges[i + 1]:
                return self.confidence[i]
        return self.confidence[-1]  # unreachable


@dataclass(frozen=True)
class ConfidenceCalibrator:
    """A named, versioned calibrator with an optional abstain floor."""

    name: str
    version: int
    bins: CalibrationBins
    abstain_floor: float | None = None   # per-detector, not a Task 5 threshold

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty str")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("version must be an int")
        if self.abstain_floor is not None:
            f = self.abstain_floor
            if isinstance(f, bool) or not isinstance(f, (int, float)):
                raise TypeError("abstain_floor must be a float or None")
            if not (0.0 <= float(f) <= 1.0):
                raise ValueError("abstain_floor must be in [0,1]")
            object.__setattr__(self, "abstain_floor", float(f))

    def calibrate(self, raw_score: float) -> CalibratedConfidence:
        _check_raw_score(raw_score, "raw_score")
        return CalibratedConfidence(confidence=self.bins.map(raw_score))

    def should_abstain(self, raw_score: float) -> bool:
        _check_raw_score(raw_score, "raw_score")
        return self.abstain_floor is not None and raw_score < self.abstain_floor


def wilson_lower_bound_95(correct: int, total: int) -> float:
    """One-sided 95% confidence lower bound on accuracy (Wilson interval).

    Answers "given ``correct`` of ``total`` observed correct, what accuracy
    can we actually claim?" — which is *not* the observed ratio. 62/62
    correct supports a claim of ~95.8%, not 100%: a small sample cannot
    demonstrate a high rate no matter how clean it looks.

    This is what keeps a calibrated confidence honest. A recognizer that
    scores 100% on 62 samples must not report 0.999 confidence, because the
    evidence for that number does not exist yet.
    """
    if not isinstance(correct, int) or isinstance(correct, bool):
        raise TypeError("correct must be an int")
    if not isinstance(total, int) or isinstance(total, bool):
        raise TypeError("total must be an int")
    if total < 0 or correct < 0 or correct > total:
        raise ValueError("require 0 <= correct <= total")
    if total == 0:
        return 0.0
    z = 1.6448536269514722  # one-sided 95%
    phat = correct / total
    denom = 1.0 + z * z / total
    center = phat + z * z / (2 * total)
    margin = z * math.sqrt(
        (phat * (1.0 - phat) + z * z / (4 * total)) / total
    )
    return max(0.0, (center - margin) / denom)


@dataclass(frozen=True)
class MeasuredCalibration:
    """A calibration derived from a recorded accuracy measurement.

    Rather than hand-picking confidence numbers, this records what was
    actually measured — how many samples, how many correct, and the raw
    score below which input is not a readable card — and derives both the
    calibrated confidence and the abstain floor from it.

    ``readable_score_floor`` comes from the observed separation between
    readable and unreadable input, not from tuning: it belongs strictly
    between the highest score any non-card produced and the lowest score any
    correctly-read card produced.
    """

    samples: int
    correct: int
    readable_score_floor: float
    unreadable_score_ceiling: float
    source: str

    def __post_init__(self) -> None:
        ceiling, floor = self.unreadable_score_ceiling, self.readable_score_floor
        if not (0.0 <= ceiling < floor <= 1.0):
            raise ValueError(
                "require 0 <= unreadable_score_ceiling < readable_score_floor <= 1; "
                "an overlap means readable and unreadable input are not separable"
            )
        if not isinstance(self.source, str) or not self.source:
            raise ValueError(
                "source must be a non-empty str describing the measurement"
            )

    @property
    def justified_confidence(self) -> float:
        """The highest confidence this measurement supports."""
        return wilson_lower_bound_95(self.correct, self.samples)

    def to_calibrator(self, name: str, version: int = 1) -> "ConfidenceCalibrator":
        """Build a calibrator that reports only what the evidence supports."""
        justified = self.justified_confidence
        return ConfidenceCalibrator(
            name=name,
            version=version,
            bins=CalibrationBins(
                edges=(0.0, self.readable_score_floor, 1.0),
                confidence=(0.0, justified),
            ),
            abstain_floor=self.readable_score_floor,
        )


__all__ = [
    "CalibrationBins",
    "ConfidenceCalibrator",
    "MeasuredCalibration",
    "wilson_lower_bound_95",
]
