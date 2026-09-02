"""Stage H hand/session-isolated data splits (guide section 11).

Rules enforced here:

- Frames are grouped by ``session_id + hand_id``; a group is never split.
  Because pre/post-action frames and perceptual near-duplicates live inside
  one hand, this also satisfies rule 7 (adjacent frames must share a split)
  without needing a separate similarity pass.
- Groups are assigned by cycling through a ratio-derived pattern, so
  consecutive hands of the same session land in different splits. A naive
  "first 60% of sorted groups" cut would put whole sessions into one split
  and quietly destroy the session diversity section 5 paid for.
- Every split must contain both stable positives and real hard negatives;
  :func:`validate_split_plan` reports splits that are one-sided instead of
  pretending a degenerate split is acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from .coverage import ANOMALY_SCENES
from .schema import FrameLabel, LabelStatus, Scene

TRAIN = "train"
CALIBRATION = "calibration"
VALIDATION = "validation"
NEGATIVE = "negative"
TEMPORAL = "temporal"

DEFAULT_RATIOS: Mapping[str, float] = {
    TRAIN: 0.6,
    CALIBRATION: 0.2,
    VALIDATION: 0.2,
}
PRIMARY_SPLITS: tuple[str, ...] = (TRAIN, CALIBRATION, VALIDATION)
AUXILIARY_SPLITS: tuple[str, ...] = (NEGATIVE, TEMPORAL)

TEMPORAL_SCENES = frozenset(
    {
        Scene.DEAL_TRANSITION,
        Scene.ACTION_TRANSITION,
        Scene.SIGNAL_LOSS,
        Scene.RECONNECT,
    }
)


@dataclass(frozen=True)
class SplitAssignment:
    """One split: the groups it owns and the frames they contain."""

    name: str
    groups: tuple[str, ...]
    frames: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group_count": len(self.groups),
            "frame_count": len(self.frames),
            "groups": list(self.groups),
            "frames": list(self.frames),
        }


@dataclass(frozen=True)
class SplitPlan:
    """Complete split assignment for a labelled dataset."""

    assignments: tuple[SplitAssignment, ...]
    auxiliary: tuple[SplitAssignment, ...]
    ratios: Mapping[str, float]

    def by_name(self, name: str) -> SplitAssignment | None:
        for assignment in self.assignments:
            if assignment.name == name:
                return assignment
        for assignment in self.auxiliary:
            if assignment.name == name:
                return assignment
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ratios": dict(self.ratios),
            "splits": [item.to_dict() for item in self.assignments],
            "auxiliary": [item.to_dict() for item in self.auxiliary],
        }

    def write(self, splits_dir: Path | str) -> dict[str, Path]:
        """Write ``splits/<name>.txt`` for every split."""
        target = Path(splits_dir)
        target.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for assignment in (*self.assignments, *self.auxiliary):
            path = target / f"{assignment.name}.txt"
            path.write_text(
                "\n".join(assignment.frames) + ("\n" if assignment.frames else ""),
                encoding="utf-8",
            )
            written[assignment.name] = path
        return written


def _cycle(ratios: Mapping[str, float]) -> list[str]:
    """Expand ratios into a repeating assignment pattern.

    0.6 / 0.2 / 0.2 becomes ``[train, train, train, calibration,
    validation]``, which yields exact proportions while spreading
    consecutive groups across splits.
    """
    fractions = [
        Fraction(value).limit_denominator(1000) for value in ratios.values()
    ]
    if any(fraction <= 0 for fraction in fractions):
        raise ValueError("split ratios must be positive")
    total = sum(fractions, Fraction(0))
    if total != 1:
        raise ValueError(f"split ratios must sum to 1, got {float(total)!r}")
    denominator = 1
    for fraction in fractions:
        denominator = denominator * fraction.denominator // _gcd(
            denominator, fraction.denominator
        )
    pattern: list[str] = []
    for name, fraction in zip(ratios.keys(), fractions):
        pattern.extend([name] * int(fraction * denominator))
    return pattern


def _gcd(first: int, second: int) -> int:
    while second:
        first, second = second, first % second
    return first


def build_split_plan(
    labels: Sequence[FrameLabel],
    *,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
) -> SplitPlan:
    """Assign every labelled frame to a primary split, grouped by hand."""
    if not labels:
        raise ValueError("cannot build splits from an empty label set")
    unknown = set(ratios) - set(PRIMARY_SPLITS)
    if unknown or set(ratios) != set(PRIMARY_SPLITS):
        raise ValueError(
            f"ratios must cover exactly {PRIMARY_SPLITS}; got {sorted(ratios)}"
        )
    pattern = _cycle(ratios)

    order: dict[str, list[str]] = {}
    for label in labels:
        order.setdefault(label.group_id, [])
        if label.frame not in order[label.group_id]:
            order[label.group_id].append(label.frame)

    buckets: dict[str, list[str]] = {name: [] for name in ratios}
    for index, group in enumerate(sorted(order)):
        buckets[pattern[index % len(pattern)]].append(group)

    assignments = []
    for name in ratios:
        groups = tuple(buckets[name])
        frames = tuple(frame for group in groups for frame in order[group])
        assignments.append(SplitAssignment(name=name, groups=groups, frames=frames))

    negative_frames: list[str] = []
    temporal_frames: list[str] = []
    for label in labels:
        if label.scene in ANOMALY_SCENES or _has_unknown_field(label):
            negative_frames.append(label.frame)
        if label.scene in TEMPORAL_SCENES:
            temporal_frames.append(label.frame)
    auxiliary = (
        SplitAssignment(name=NEGATIVE, groups=(), frames=tuple(negative_frames)),
        SplitAssignment(name=TEMPORAL, groups=(), frames=tuple(temporal_frames)),
    )

    return SplitPlan(
        assignments=tuple(assignments), auxiliary=auxiliary, ratios=dict(ratios)
    )


def _has_unknown_field(label: FrameLabel) -> bool:
    fields = (
        label.hero_cards,
        label.board_cards,
        label.street,
        label.pot,
    )
    if any(field.status is not LabelStatus.VALID for field in fields):
        return True
    return any(
        slot.stack.status is not LabelStatus.VALID
        or slot.dealer.status is not LabelStatus.VALID
        or slot.occupancy.status is not LabelStatus.VALID
        for slot in label.slots
    )


def detect_leakage(plan: SplitPlan) -> list[str]:
    """Return every group or frame that appears in more than one split."""
    problems: list[str] = []
    group_owner: dict[str, str] = {}
    frame_owner: dict[str, str] = {}
    for assignment in plan.assignments:
        for group in assignment.groups:
            previous = group_owner.setdefault(group, assignment.name)
            if previous != assignment.name:
                problems.append(
                    f"group {group} appears in {previous} and {assignment.name}"
                )
        for frame in assignment.frames:
            previous = frame_owner.setdefault(frame, assignment.name)
            if previous != assignment.name:
                problems.append(
                    f"frame {frame} appears in {previous} and {assignment.name}"
                )
    return problems


def validate_split_plan(
    plan: SplitPlan, labels: Sequence[FrameLabel]
) -> list[str]:
    """Return human-readable problems; empty means the plan is usable."""
    problems = detect_leakage(plan)
    by_frame = {label.frame: label for label in labels}
    for assignment in plan.assignments:
        if not assignment.frames:
            problems.append(f"split {assignment.name} is empty")
            continue
        positives = 0
        negatives = 0
        for name in assignment.frames:
            label = by_frame.get(name)
            if label is None:
                problems.append(
                    f"split {assignment.name} references unknown frame {name}"
                )
                continue
            if label.stable and label.scene is Scene.TABLE and not _has_unknown_field(
                label
            ):
                positives += 1
            if label.scene in ANOMALY_SCENES or _has_unknown_field(label):
                negatives += 1
        if positives == 0:
            problems.append(
                f"split {assignment.name} has no stable positive frames"
            )
        if negatives == 0:
            problems.append(
                f"split {assignment.name} has no real hard negatives"
            )
    known = {label.frame for label in labels}
    assigned = {name for item in plan.assignments for name in item.frames}
    missing = sorted(known - assigned)
    if missing:
        problems.append(
            f"{len(missing)} labelled frames are not in any split, e.g. "
            + ", ".join(missing[:5])
        )
    return problems


def read_split_file(path: Path | str) -> tuple[str, ...]:
    """Read a ``splits/<name>.txt`` file back."""
    text = Path(path).read_text(encoding="utf-8")
    return tuple(line.strip() for line in text.splitlines() if line.strip())


__all__ = [
    "AUXILIARY_SPLITS",
    "CALIBRATION",
    "DEFAULT_RATIOS",
    "NEGATIVE",
    "PRIMARY_SPLITS",
    "SplitAssignment",
    "SplitPlan",
    "TEMPORAL",
    "TEMPORAL_SCENES",
    "TRAIN",
    "VALIDATION",
    "build_split_plan",
    "detect_leakage",
    "read_split_file",
    "validate_split_plan",
]
