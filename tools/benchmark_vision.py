"""Vision benchmark harness — end-to-end field-level evaluation.

Pipeline: Golden sample -> Frame + TableMap -> VisionEngine -> RawObservation
-> ground-truth comparison -> FieldMetric -> report.

Reports per-field accuracy / coverage / unknown-reject rate / negative
false-positive rate / 95% one-sided confidence lower bound / target / pass-fail.
Synthetic and Real datasets are reported separately; only Real counts toward
Frozen FINAL acceptance.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

sys.path.insert(0, "src")

from poker_engine.perceptual.vision.calibration import (  # noqa: E402
    wilson_lower_bound_95,
)

# Frozen acceptance thresholds (field name -> target accuracy). Read-only.
FROZEN_TARGETS = MappingProxyType({
    "hero_cards": 0.995,
    "board_cards": 0.995,
    "street": 0.999,
    "pot": 0.99,
    "bet_size": 0.99,
    "stack": 0.99,
    "action": 0.99,
})


@dataclass(frozen=True)
class FieldMetric:
    correct: int        # VALID and value == truth (among readable)
    total: int          # readable samples (truth non-None)
    unknown: int        # readable predicted UNKNOWN (reject)
    conflict: int       # readable predicted CONFLICT
    negative_total: int  # negative samples (truth None / expected absent)
    negative_fp: int    # negative samples wrongly predicted as present

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def coverage(self) -> float:
        # coverage = fraction that produced a confident (VALID) answer
        return (self.total - self.unknown - self.conflict) / self.total \
            if self.total else 0.0

    @property
    def unknown_rate(self) -> float:
        return self.unknown / self.total if self.total else 0.0

    @property
    def conflict_rate(self) -> float:
        return self.conflict / self.total if self.total else 0.0

    @property
    def negative_fp_rate(self) -> float | None:
        """Negative false-positive rate, or None (N/A) when there are no
        negative samples. 0 negative samples does NOT mean 0% FP rate."""
        if self.negative_total == 0:
            return None
        return self.negative_fp / self.negative_total

    def lower_bound_95(self) -> float:
        """One-sided 95% confidence lower bound over the readable samples.

        Shared with the production confidence calibrator so a benchmark and a
        live run can never disagree about what a sample count supports.
        """
        return wilson_lower_bound_95(self.correct, self.total)


def _is_hallucination(pred) -> bool:
    """A negative sample is 'hallucinated' only if a non-null AND non-empty
    value was produced. An empty tuple/list counts as a correct rejection."""
    if pred is None:
        return False
    if isinstance(pred, (tuple, list, set, frozenset)) and len(pred) == 0:
        return False
    return True


# Allowed validation_status values in an entry (schema-enforced).
_ALLOWED_STATUS = ("valid", "unknown", "conflict")
# Allowed sample source declarations (decides acceptance eligibility).
_ALLOWED_SOURCE = ("synthetic-render", "real-platform")


@dataclass(frozen=True)
class DatasetVerdict:
    """Whether a dataset is eligible for statistical acceptance, derived from
    the dataset's own content (per-sample source + sample_id/image uniqueness),
    NOT from a caller-supplied boolean switch."""
    eligible: bool
    reason: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """Atomic, indivisible evaluation outcome.

    Produced only by ``evaluate()`` — validation, metric computation, and
    report building run together and cannot be split or bypassed. A caller
    cannot obtain a report without the dataset having been validated.

    ``metrics`` and ``report`` are deeply immutable (read-only mappings);
    serialization to plain dicts happens only via ``to_dict()``.
    """
    verdict: DatasetVerdict
    metrics: Mapping[str, FieldMetric]
    report: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "report", _freeze_report(self.report))

    def to_dict(self) -> dict:
        """Serialize to a plain (mutable) dict for JSON output."""
        return _report_to_plain(self.report)


def _freeze_report(report: Mapping[str, object]) -> Mapping:
    """Deep-freeze a report mapping (top-level + each field's dict)."""
    frozen_fields = {}
    for field, fdata in report["fields"].items():
        frozen_fields[field] = MappingProxyType(dict(fdata))
    return MappingProxyType({
        "dataset": report["dataset"],
        "acceptance_eligible": report["acceptance_eligible"],
        "eligibility_reason": report["eligibility_reason"],
        "fields": MappingProxyType(frozen_fields),
    })


def _report_to_plain(report: Mapping) -> dict:
    """Convert a frozen report back to plain dicts (for JSON)."""
    return {
        "dataset": report["dataset"],
        "acceptance_eligible": report["acceptance_eligible"],
        "eligibility_reason": report["eligibility_reason"],
        "fields": {f: dict(d) for f, d in report["fields"].items()},
    }


def _validate_dataset(entries: list[dict]) -> DatasetVerdict:
    """Derive acceptance eligibility from the entries themselves.

    A dataset is NOT eligible if:
      - sources are MIXED (every entry must share the same declared source;
        real and synthetic samples cannot be interleaved), OR
      - the source is synthetic-render (only real-platform counts), OR
      - any sample_id is duplicated, OR
      - any image hash is duplicated (a distinct sample_id reusing the same
        image content is still a duplicate).
    """
    sources: set[str] = set()
    sample_ids: list[str] = []
    image_hashes: list[str] = []
    for entry in entries:
        source = entry.get("_source")
        if source not in _ALLOWED_SOURCE:
            raise ValueError(
                f"invalid sample source {source!r}; allowed: {_ALLOWED_SOURCE}"
            )
        sources.add(source)
        sid = entry.get("_sample_id")
        if not isinstance(sid, str) or not sid:
            raise ValueError("every entry must carry a non-empty str _sample_id")
        sample_ids.append(sid)
        imgh = entry.get("_image_sha")
        if not isinstance(imgh, str) or not imgh:
            raise ValueError("every entry must carry a non-empty str _image_sha")
        image_hashes.append(imgh)

    # mixed sources are rejected outright (no partial eligibility)
    if len(sources) > 1:
        return DatasetVerdict(eligible=False, reason="mixed sample sources")

    if "real-platform" not in sources:
        return DatasetVerdict(eligible=False,
                              reason="dataset not eligible for acceptance")
    if len(sample_ids) != len(set(sample_ids)):
        return DatasetVerdict(eligible=False, reason="duplicate sample_id")
    if len(image_hashes) != len(set(image_hashes)):
        return DatasetVerdict(eligible=False, reason="duplicate image content")
    return DatasetVerdict(eligible=True)


def _compute_metrics(entries: list[dict]) -> Mapping[str, FieldMetric]:
    """Score entries by explicit validation_status (NOT inferred from pred)."""
    counts = {
        f: {
            "correct": 0, "total": 0, "unknown": 0, "conflict": 0,
            "neg_total": 0, "neg_fp": 0,
        }
        for f in FROZEN_TARGETS
    }
    for entry in entries:
        for field, v in entry.items():
            if field.startswith("_"):
                continue  # dataset metadata (sample_id/source/image_sha)
            if field not in FROZEN_TARGETS:
                raise ValueError(f"unknown benchmark field {field!r}")
            c = counts[field]
            truth = v["truth"]
            pred = v["pred"]
            status = v.get("status")
            if status not in _ALLOWED_STATUS:
                raise ValueError(
                    f"invalid status {status!r} for field {field!r}; "
                    f"allowed: {_ALLOWED_STATUS}"
                )
            if truth is None:
                c["neg_total"] += 1
                if status == "valid" and _is_hallucination(pred):
                    c["neg_fp"] += 1
            else:
                c["total"] += 1
                if status == "unknown":
                    c["unknown"] += 1
                elif status == "conflict":
                    c["conflict"] += 1
                elif status == "valid":
                    if pred == truth:
                        c["correct"] += 1
    return {
        f: FieldMetric(
            correct=c["correct"],
            total=c["total"],
            unknown=c["unknown"],
            conflict=c["conflict"],
            negative_total=c["neg_total"],
            negative_fp=c["neg_fp"],
        )
        for f, c in counts.items()
    }


def _build_report(
    metrics: Mapping[str, FieldMetric],
    label: str,
    verdict: DatasetVerdict,
) -> dict:
    """Build a per-field report."""
    out = {
        "dataset": label,
        "acceptance_eligible": bool(verdict.eligible),
        "eligibility_reason": verdict.reason,
        "fields": {},
    }
    for field, target in FROZEN_TARGETS.items():
        m = metrics[field]
        lb = m.lower_bound_95()
        nfp = m.negative_fp_rate

        smoke_pass = bool(m.total > 0 and m.accuracy >= target)
        acceptance_met = bool(verdict.eligible and lb >= target)
        if not verdict.eligible:
            reason = verdict.reason or "dataset not eligible for acceptance"
        elif m.total == 0:
            reason = "no readable samples"
        elif acceptance_met:
            reason = None
        elif m.accuracy >= target:
            reason = "insufficient sample size for statistical acceptance"
        else:
            reason = "accuracy below target"

        out["fields"][field] = {
            "accuracy": round(m.accuracy, 4),
            "coverage": round(m.coverage, 4),
            "unknown_rate": round(m.unknown_rate, 4),
            "conflict_rate": round(m.conflict_rate, 4),
            "negative_false_positive_rate": (
                round(nfp, 4) if nfp is not None else None
            ),
            "lower_bound_95": round(lb, 4),
            "target": target,
            "smoke_pass": smoke_pass,
            "acceptance_target_met": acceptance_met,
            "acceptance_reason": reason,
            "n": m.total,
            "n_unknown": m.unknown,
            "n_conflict": m.conflict,
            "n_negative": m.negative_total,
        }
    return out


def evaluate(entries: list[dict], label: str = "synthetic") -> EvaluationResult:
    """Atomic evaluation: validate dataset -> compute metrics -> build report.

    These three steps run together and cannot be split or bypassed. Returns an
    ``EvaluationResult`` carrying the verdict, metrics, and report.
    """
    verdict = _validate_dataset(entries)
    metrics = _compute_metrics(entries)
    result_report = _build_report(metrics, label, verdict)
    return EvaluationResult(verdict=verdict, metrics=metrics,
                            report=result_report)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("golden_json", help="Golden test set JSON")
    ap.add_argument("--dataset", default="synthetic",
                    help="dataset label ('real' or 'synthetic')")
    args = ap.parse_args()

    with open(args.golden_json, encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", [])
    result = evaluate(entries, args.dataset)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
