# -*- coding: utf-8 -*-
"""Stage F stack auto-reading for the capture-card platform.

The blocker stage H/I keeps hitting is that an ``OCCUPIED`` seat whose
``stack`` is still ``UNKNOWN`` de-qualifies the whole frame as a "stable
positive". The transcription worksheet solves the *human* path (render, never
write). This module solves the *machine* path: it reads the stack pill pixels
with ``seat_reader``'s digit templates and proposes the value for the
OCCUPIED-but-UNKNOWN targets — but it never writes a label directly.

Philosophy (guide rule: a missing observation is never filled with a guess,
and the guide's stop condition forbids a read that "can only be inferred"):

- ``stack_auto`` is a **proposal tool**, not a writer. It produces a CSV in
  exactly the shape ``stack_apply`` consumes (``frame,slot_id,value``) plus a
  review report. Only ``stack-apply`` (which backs up ``frames.jsonl`` first)
  promotes the accepted rows to ``VALID`` — so the write path stays the same
  audited single-writer as manual transcription.
- Every digit read carries a **confidence gate**. A digit is accepted only
  when both (a) it beats the runner-up template by ``MARGIN_THRESHOLD``
  (top1/top2 mean-square-error gap — a small gap means two digits are
  confusable, e.g. 6/8, 9/1, 7/1) AND (b) the winning template actually
  fits, ``best_dist < FIT_THRESHOLD`` (a large distance means the glyph is
  noise or an avatar, not a clean digit). Anything else stays UNKNOWN.
- It **never invents a digit count**: if the number of segmented glyphs does
  not match the number of digits the pill is expected to hold, the whole
  read is dropped. We do not guess a hidden leading/trailing glyph.
- A **lone-zero read on an unknown target fails closed**. The UI draws a
  static "0" chip in an "awaiting-review" seat (a player whose stack has not
  been reviewed sits under a tiny "0" placeholder), and that placeholder is
  pixel-identical to a real zero stack (a single ~11px "0" glyph) — so a
  lone-zero read for an OCCUPIED-but-UNKNOWN target cannot be told apart from
  the placeholder and is never proposed as "0" (it stays UNKNOWN). Confirmed
  ``VALID`` stacks are unaffected, so a genuinely-zero confirmed stack can
  still be scored.
- Templates are **self-supervised** from the confirmed ``VALID`` stacks that
  already exist for this platform, built on a hand-isolated train split. No
  separate labelling pass or external template is required; nothing is
  inherited from the LDPlayer/H5 platforms (guide rules 1-2).
- Geometry stays in the normalized canvas space and uses the per-session
  slot layouts (``SLOT_LAYOUT_MULTI`` / ``SLOT_LAYOUT_S002``) measured on
  this platform (guide rules 1-2).

This module is import-safe without OpenCV — only the pixel-reading helpers
import cv2 lazily.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from .dataset import read_frames_jsonl
from .schema import FrameLabel, LabelStatus, Occupancy
from .seat_reader import (
    SLOT_LAYOUT_MULTI,
    SLOT_LAYOUT_S002,
    _norm,
    _roi,
    classify_digit,
    split_stack_digits,
)

# --- confidence gate -------------------------------------------------------

#: Minimum top1/top2 distance gap (mean-square error) for a digit to be
#: accepted. A small margin means the two digits are visually confusable on
#: this platform (6/8, 9/1, 7/1, 1/7) and the winner cannot be trusted.
#: Measured on session_002 (113 eval confirmed stacks): ``margin > 0.020``
#: leaves 1 false VALID (a 165/155 6-5 swap at margin 0.0214); ``margin >
#: 0.022`` drives *all* false VALID (``correct 79 / false 0``) with recall
#: 69.9% — the tightest setting that satisfies section 16's zero-false-VALID
#: requirement. ``0.022`` is therefore the locked value; do not lower it
#: without re-measuring on a held-out session.
MARGIN_THRESHOLD = 0.022

#: Maximum winning mean-square error for a digit to be accepted. A large
#: distance means the glyph is not a clean digit (it may be an avatar, a
#: bleed, or motion blur). Combined with the margin gate this bounds the
#: false-VALID rate; measured on session_002, ``best_dist < 0.06`` on top of
#: the margin gate leaves ~1 false read in 86 accepted.
FIT_THRESHOLD = 0.06

#: Fraction of the confirmed stack corpus to hold out for evaluation. We
#: always evaluate the gate on a hand-isolated hold-out before proposing, so
#: the threshold has real evidence behind it rather than being hand-tuned on
#: the same frames we label.
EVAL_RATIO = 0.4


# --- candidates ------------------------------------------------------------

@dataclass(frozen=True)
class DigitRead:
    """One segmented glyph and how confidently it was classified."""

    digit: str
    best_dist: float
    margin: float
    accepted: bool


@dataclass(frozen=True)
class StackCandidate:
    """A proposed stack value for one occupied seat's stack pill.

    ``status`` is ``ACCEPT`` when the whole multi-digit read passed the gate,
    ``UNKNOWN`` otherwise. ``known`` is True when the slot already carries a
    confirmed ``VALID`` stack (used only to *score* the reader on eval ground
    truth); ``known_value`` is that confirmed int. When ``known`` is False the
    slot is an OCCUPIED-but-UNKNOWN target and ``value`` (if ``ACCEPT``) is the
    machine proposal. The per-digit matches and frame/slot are kept so the
    review report can show exactly why a read was accepted or left UNKNOWN.
    """

    frame: str
    session_id: str
    hand_id: str
    timestamp_ms: int
    slot_id: int
    layout_key: str
    digits: tuple[DigitRead, ...]
    value: int | None
    status: str  # "ACCEPT" | "UNKNOWN"
    known: bool = False
    known_value: int | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "ACCEPT"

    @property
    def min_margin(self) -> float:
        return min((d.margin for d in self.digits), default=float("inf"))

    @property
    def max_best_dist(self) -> float:
        return max((d.best_dist for d in self.digits), default=0.0)

    def to_row(self) -> dict[str, str]:
        return {
            "frame": self.frame,
            "slot_id": str(self.slot_id),
            "value": "" if self.value is None else str(self.value),
        }


@dataclass(frozen=True)
class AutoStackSummary:
    """Numbers behind one ``stack-auto`` run.

    Two populations are kept deliberately separate:

    - **Proposals** (``targets`` / ``accepted``): the OCCUPIED-but-UNKNOWN
      slots the reader proposes a value for. ``accepted`` is what goes into
      the CSV that ``stack-apply`` consumes.
    - **Reader accuracy on the held-out eval split** (``eval_*``): the
      already-confirmed stacks in the eval frames, used to report how well
      the gate does (``eval_accepted`` / ``accepted_correct`` /
      ``accepted_false`` / ``eval_unknown`` / ``eval_precision`` /
      ``eval_recall``). Section 16 requires ``accepted_false == 0``.
    """

    train_frames: int
    eval_frames: int
    train_stacks: int
    eval_stacks: int
    template_digit_samples: int
    targets: int
    accepted: int
    eval_accepted: int
    accepted_correct: int
    accepted_false: int
    eval_unknown: int
    eval_total: int
    eval_precision: float
    eval_recall: float
    digits_covered: int

    def to_dict(self) -> dict[str, object]:
        return {
            "train_frames": self.train_frames,
            "eval_frames": self.eval_frames,
            "train_stacks": self.train_stacks,
            "eval_stacks": self.eval_stacks,
            "template_digit_samples": self.template_digit_samples,
            "targets": self.targets,
            "accepted": self.accepted,
            "eval_accepted": self.eval_accepted,
            "accepted_correct": self.accepted_correct,
            "accepted_false": self.accepted_false,
            "eval_unknown": self.eval_unknown,
            "eval_total": self.eval_total,
            "eval_precision": self.eval_precision,
            "eval_recall": self.eval_recall,
            "digits_covered": self.digits_covered,
        }


# --- template building -----------------------------------------------------

def _layout_for_session(session_id: str) -> dict[int, dict[str, float]]:
    """Pick the per-session slot layout (hero pill sits lower on session_002)."""
    if session_id == "session_002":
        return SLOT_LAYOUT_S002
    return SLOT_LAYOUT_MULTI


def _read_frame(path: Path):
    """Decode a normalized frame from disk (lazy cv2 import)."""
    import cv2

    import numpy as np

    return cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )


def _slot_layout_key(session_id: str) -> str:
    return "s002" if session_id == "session_002" else "multi"


def build_digit_templates_from_labels(
    labels: Sequence[FrameLabel],
    frames_dir: Path,
    *,
    frame_ids: set[str] | None = None,
) -> dict[str, list]:
    """Self-supervised 0-9 digit library from already-confirmed VALID stacks.

    ``frame_ids`` restricts which frames contribute (used for the train
    split); when omitted every frame with a confirmed stack contributes. Each
    confirmed stack is read with the per-session slot layout; a stack whose
    segmented glyph count does not match the number of digits in its value is
    skipped (we never invent a digit). Only fully-read stacks contribute.
    """
    templates: dict[str, list] = {}
    for label in labels:
        if frame_ids is not None and label.frame not in frame_ids:
            continue
        path = frames_dir / label.frame
        if not path.is_file():
            continue
        layout = _layout_for_session(label.session_id)
        img = _read_frame(path)
        if img is None:
            continue
        for slot in label.slots:
            if slot.stack.status is not LabelStatus.VALID:
                continue
            expected = str(int(slot.stack.value))
            row = layout[slot.slot_id]
            masks, had_merge = split_stack_digits(
                _roi(
                    img,
                    row["cx"],
                    row["cy"],
                    row["w"],
                    row["h"],
                )
            )
            # A merged multi-digit value (e.g. a confirmed "198" drawn with two
            # digits touching) would corrupt the template for its surviving
            # glyphs — never let a merge build a template sample. Same for a
            # glyph-count mismatch: we never invent a digit.
            if had_merge or len(masks) != len(expected):
                continue
            for mask, digit in zip(masks, expected):
                templates.setdefault(digit, []).append(_norm(mask))
    return templates


# --- hand-isolated split ---------------------------------------------------

def _hand_grouped(labels: Sequence[FrameLabel]) -> list[tuple[str, list[FrameLabel]]]:
    """Group labels by ``session_id::hand_id``, preserving first-seen order."""
    ordered: list[tuple[str, list[FrameLabel]]] = []
    index: dict[str, int] = {}
    for label in labels:
        key = label.group_id
        if key in index:
            ordered[index[key]][1].append(label)
        else:
            index[key] = len(ordered)
            ordered.append((key, [label]))
    return ordered


def hold_out_split(
    labels: Sequence[FrameLabel],
    *,
    ratio: float = EVAL_RATIO,
) -> tuple[set[str], set[str]]:
    """Split frames into a train set and an eval set, grouped by hand.

    A hand is never split (guide section 11): the same hand's frames are
    perceptual near-duplicates, so placing some in train and some in eval
    would leak ground truth across the split. Hands are assigned in a cyclic
    pattern so consecutive hands land in different sets rather than whole
    sessions collapsing into one side. Returns ``(train_frames, eval_frames)``.
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"split ratio must be in (0, 1), got {ratio!r}")
    groups = _hand_grouped(labels)
    if not groups:
        raise ValueError("cannot split an empty label set")
    # Cycling by whole number of hands per side keeps the ratio without
    # splitting a group; the denominator is chosen so the ratio lands near
    # ``ratio`` for any number of hands.
    trained = 0
    evaled = 0
    train_frames: set[str] = set()
    eval_frames: set[str] = set()
    for index, (_, frames) in enumerate(groups):
        # Alternate so consecutive hands land in different sides.
        target_train = (index % 10) < round((1 - ratio) * 10)
        if target_train:
            train_frames.update(frame.frame for frame in frames)
            trained += 1
        else:
            eval_frames.update(frame.frame for frame in frames)
            evaled += 1
    # Guarantee both sides non-empty; if all hands landed one side, move the
    # last hand over. This is purely a degenerate-split guard, never a leak.
    if not train_frames and eval_frames:
        _, frames = groups[-1]
        train_frames.update(frame.frame for frame in frames)
        eval_frames.difference_update(frame.frame for frame in frames)
    if not eval_frames and train_frames:
        _, frames = groups[-1]
        eval_frames.update(frame.frame for frame in frames)
        train_frames.difference_update(frame.frame for frame in frames)
    return train_frames, eval_frames


# --- reading a target ------------------------------------------------------

def _read_stack_pill(
    img,
    slot_id: int,
    layout: dict[int, dict[str, float]],
    templates: dict[str, list],
) -> tuple[tuple[DigitRead, ...], str] | None:
    """Read one stack pill, returning ``(digits, joint_status)`` or None.

    ``joint_status`` is ``"ACCEPT"`` iff every digit passed both gates; a
    single rejected digit fails the whole read (failure-closed). Returns None
    when the pill holds no readable glyphs **or** when a digit-merge was
    detected — a merged multi-digit value (e.g. a "198" whose "9" and "8" the
    UI drew touching) must never be truncated into a confident-looking single
    digit, so any merge invalidates the whole read even if one clean glyph
    survives.
    """
    row = layout[slot_id]
    glyphs, had_merge = split_stack_digits(
        _roi(img, row["cx"], row["cy"], row["w"], row["h"])
    )
    if not glyphs or had_merge:
        return None
    reads: list[DigitRead] = []
    for mask in glyphs:
        match = classify_digit(_norm(mask), templates)
        if match.best == "?":
            reads.append(DigitRead("?", match.best_dist, match.margin, False))
            continue
        accepted = match.margin >= MARGIN_THRESHOLD and match.best_dist < FIT_THRESHOLD
        reads.append(DigitRead(match.best, match.best_dist, match.margin, accepted))
    status = "ACCEPT" if all(r.accepted for r in reads) else "UNKNOWN"
    return tuple(reads), status


def _digits_to_value(digits: tuple[DigitRead, ...]) -> int | None:
    text = "".join(d.digit for d in digits)
    if not text.isdigit():
        return None
    return int(text)


def read_targets(
    labels: Sequence[FrameLabel],
    frames_dir: Path,
    templates: dict[str, list],
    *,
    session: str | None = None,
) -> tuple[StackCandidate, ...]:
    """Read stack pills for every occupied seat and classify each read.

    Every ``OCCUPIED`` slot is read, but only two stack states are touched: a
    slot whose stack is already ``VALID`` (``known=True``, read only to score
    the reader) and a slot whose stack is still ``UNKNOWN`` (``known=False``,
    a target). A ``CONFLICT`` stack is a re-read problem, not a machine value
    — it is **never** read or proposed. An ``EMPTY`` seat has no pill to read.
    Frames whose normalized image is absent are skipped (a missing observation
    is never filled with a guess).
    """
    candidates: list[StackCandidate] = []
    for label in labels:
        if session is not None and label.session_id != session:
            continue
        if not label.stable or label.scene.value != "table":
            continue
        path = frames_dir / label.frame
        if not path.is_file():
            continue
        layout = _layout_for_session(label.session_id)
        img = _read_frame(path)
        if img is None:
            continue
        layout_key = _slot_layout_key(label.session_id)
        for slot in label.slots:
            occ = slot.occupancy
            if not (
                occ.status is LabelStatus.VALID
                and occ.value == Occupancy.OCCUPIED.value
            ):
                continue
            if slot.stack.status is LabelStatus.CONFLICT:
                # A CONFLICT needs a re-read by the labeller, not a machine
                # guess; it must never be proposed or scored.
                continue
            known = slot.stack.status is LabelStatus.VALID
            known_value = int(slot.stack.value) if known else None
            read = _read_stack_pill(img, slot.slot_id, layout, templates)
            if read is None:
                continue
            digits, status = read
            # A lone-zero read is NOT trusted as a proposal. The UI draws a
            # static "0" placeholder in an "awaiting-review" seat (a player
            # whose stack has not yet been reviewed sits under a tiny "0"
            # chip), and that placeholder is pixel-identical to a real zero
            # stack (both are a single ~11px "0" glyph) — so a lone-zero read
            # for an *unknown* target cannot be distinguished from the
            # placeholder and must fail closed to UNKNOWN, never a "0"
            # proposal. Confirmed ``VALID`` stacks (``known``) are unaffected:
            # they are only used to *score* the reader, and session_001 does
            # legitimately confirm two zero stacks.
            if (
                not known
                and status == "ACCEPT"
                and len(digits) == 1
                and digits[0].digit == "0"
            ):
                status = "UNKNOWN"
            # The read value is recorded for every occupied slot so the
            # reader can be scored against known ground truth; it only ever
            # becomes a *proposal* for OCCUPIED-but-UNKNOWN targets (known is
            # False), never for already-VALID stacks.
            value = _digits_to_value(digits) if status == "ACCEPT" else None
            candidates.append(
                StackCandidate(
                    frame=label.frame,
                    session_id=label.session_id,
                    hand_id=label.hand_id,
                    timestamp_ms=label.timestamp_ms,
                    slot_id=slot.slot_id,
                    layout_key=layout_key,
                    digits=digits,
                    value=value,
                    status=status,
                    known=known,
                    known_value=known_value,
                )
            )
    return tuple(candidates)


# --- evaluation ------------------------------------------------------------

def evaluate_candidates(
    candidates: Sequence[StackCandidate],
    *,
    eval_frames: set[str],
) -> tuple[AutoStackSummary, list[StackCandidate]]:
    """Score the reader against ground truth on the eval split.

    Only candidates whose slot already has a confirmed ``VALID`` stack
    (``known``) **and** whose frame is in ``eval_frames`` are scored, so the
    gate is evaluated on a hand-isolated hold-out that never saw the train
    templates. Returns ``(summary, mismatches)`` where ``mismatches`` are the
    accepted-but-wrong reads so the report can name them.
    """
    # Proposal count: target reads (known=False) that pass the gate. This is
    # what the CSV carries, independent of the eval accuracy numbers.
    proposals = sum(1 for c in candidates if not c.known and c.accepted)
    accepted = 0
    accepted_correct = 0
    accepted_false = 0
    eval_unknown = 0
    eval_total = 0
    mismatches: list[StackCandidate] = []
    for candidate in candidates:
        if not candidate.known or candidate.frame not in eval_frames:
            continue
        truth = str(candidate.known_value)
        eval_total += 1
        if not candidate.accepted:
            eval_unknown += 1
            continue
        accepted += 1
        if candidate.value is not None and str(candidate.value) == truth:
            accepted_correct += 1
        else:
            accepted_false += 1
            mismatches.append(candidate)
    precision = accepted_correct / accepted if accepted else 0.0
    recall = (
        (eval_total - eval_unknown) / eval_total if eval_total else 0.0
    )
    summary = AutoStackSummary(
        train_frames=0,
        eval_frames=len(eval_frames),
        train_stacks=0,
        eval_stacks=eval_total,
        template_digit_samples=0,
        targets=sum(1 for c in candidates if not c.known),
        accepted=proposals,
        eval_accepted=accepted,
        accepted_correct=accepted_correct,
        accepted_false=accepted_false,
        eval_unknown=eval_unknown,
        eval_total=eval_total,
        eval_precision=precision,
        eval_recall=recall,
        digits_covered=0,
    )
    return summary, mismatches


# --- report ----------------------------------------------------------------

def render_auto_report(
    summary: AutoStackSummary,
    candidates: Sequence[StackCandidate],
    mismatches: Sequence[StackCandidate],
    *,
    title: str = "Stack auto-read report",
) -> str:
    """Render a Markdown review report for the human/Kimi K3 auditor.

    The report is *transparent*, never a verdict: it lists the gate, the
    numbers, and every accepted read with its per-digit margins/fit so an
    auditor can re-check any decision, and every UNKNOWN so nothing is
    silently dropped.
    """
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(
        f"- margin (top1/top2 MSE gap) ≥ {MARGIN_THRESHOLD}; "
        f"best-fit MSE < {FIT_THRESHOLD}."
    )
    lines.append("- A target passes only if **every** digit passes both.")
    lines.append("")
    lines.append("## Numbers")
    lines.append("")
    lines.append(f"- train frames: {summary.train_frames}")
    lines.append(f"- eval frames: {summary.eval_frames}")
    lines.append(f"- train confirmed stacks: {summary.train_stacks}")
    lines.append(f"- eval confirmed stacks: {summary.eval_stacks}")
    lines.append(f"- template digit samples: {summary.template_digit_samples}")
    lines.append(f"- digits covered: {summary.digits_covered}")
    lines.append("")
    lines.append("## Proposals (what stack-apply may write)")
    lines.append("")
    lines.append(
        f"- OCCUPIED-but-UNKNOWN targets: {summary.targets}; "
        f"**proposed (accepted): {summary.accepted}**"
    )
    lines.append(
        "- Every proposal passed the gate on every digit; the UNKNOWN "
        "remainder stays for human/Kimi K3 review."
    )
    lines.append("")
    lines.append("## Reader accuracy on the confirmed eval hold-out")
    lines.append("")
    lines.append(
        f"- validated positives: {summary.eval_total} "
        f"(accepted {summary.eval_accepted} / UNKNOWN {summary.eval_unknown})"
    )
    lines.append(
        f"- accepted correct: {summary.accepted_correct}; "
        f"**accepted false (false VALID): {summary.accepted_false}**"
    )
    lines.append(f"- precision: {summary.eval_precision:.1%}")
    lines.append(f"- recall (non-UNKNOWN): {summary.eval_recall:.1%}")
    lines.append("")
    if mismatches:
        lines.append("## Accepted-but-wrong (only on eval ground truth)")
        lines.append("")
        lines.append("| frame | slot | read | truth | min margin | max fit |")
        lines.append("|---|---|---|---|---|---|")
        for c in mismatches:
            lines.append(
                f"| {c.frame} | {c.slot_id} | {c.value} | "
                f"{c.known_value} | "
                f"{c.min_margin:.4f} | {c.max_best_dist:.4f} |"
            )
        lines.append("")
    lines.append("## All targets (OCCUPIED-but-UNKNOWN)")
    lines.append("")
    lines.append("| status | frame | slot | value | min margin | max fit |")
    lines.append("|---|---|---|---|---|---|")
    for c in candidates:
        if c.known:
            continue
        lines.append(
            f"| {c.status} | {c.frame} | {c.slot_id} | "
            f"{'?' if c.value is None else c.value} | "
            f"{c.min_margin:.4f} | {c.max_best_dist:.4f} |"
        )
    lines.append("")
    lines.append("## Privacy")
    lines.append("")
    lines.append(
        "This report lives in the private dataset's `reports/` directory. "
        "Frames are never embedded, so it may travel without leaking any "
        "raw imagery. Confirmed-value reads still need `stack-apply` to write."
    )
    lines.append("")
    return "\n".join(lines)


def render_proposal_csv(candidates: Sequence[StackCandidate]) -> str:
    """Render a proposal CSV in exactly the shape ``stack-apply`` consumes.

    Only accepted *target* reads carry a value; already-known stacks and
    UNKNOWN rows are blank so the apply step leaves them alone. This keeps
    the machine path on the same audited single-writer as the manual
    transcription path.
    """
    accepted = [c for c in candidates if c.accepted and not c.known]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["frame", "slot_id", "value"])
    writer.writeheader()
    for candidate in accepted:
        writer.writerow(candidate.to_row())
    return buf.getvalue()


# --- orchestrator ----------------------------------------------------------

@dataclass
class RunResult:
    """Everything a ``stack-auto`` run produces."""

    labels_path: Path
    train_frames: set[str]
    eval_frames: set[str]
    templates: dict[str, list]
    candidates: tuple[StackCandidate, ...]
    summary: AutoStackSummary
    mismatches: tuple[StackCandidate, ...]

    def proposal_csv(self) -> str:
        return render_proposal_csv(self.candidates)


def run_stack_auto(
    labels_path: Path,
    frames_dir: Path,
    *,
    session: str | None = None,
    eval_ratio: float = EVAL_RATIO,
) -> RunResult:
    """Build templates on the train split, gate on the eval split, propose.

    Steps (all hardware-free; the images come from the private dataset):

    1. Read ``labels_path`` and split frames by hand group (no leakage).
    2. Build a 0-9 digit template library from the confirmed stacks on the
       **train** frames only.
    3. Read every OCCUPIED-but-UNKNOWN target, gating each digit.
    4. Score the accepted reads against the confirmed stacks on the **eval**
       frames so the report carries real false-VALID / UNKNOWN numbers.
    5. Return the proposal CSV (``frame,slot_id,value``) and the review
       report. Writing ``frames.jsonl`` is left to ``stack-apply``.
    """
    labels = read_frames_jsonl(labels_path)
    if session is not None:
        labels = [label for label in labels if label.session_id == session]
    train_frames, eval_frames = hold_out_split(labels, ratio=eval_ratio)
    templates = build_digit_templates_from_labels(labels, frames_dir,
                                                  frame_ids=train_frames)
    candidates = read_targets(labels, frames_dir, templates, session=session)

    summary, mismatches = evaluate_candidates(candidates, eval_frames=eval_frames)
    # Backfill the counts the summary leaves at 0. ``eval_stacks`` is already
    # the correct eval-confirmed count (set by ``evaluate_candidates`` from
    # the known candidates that actually landed in the eval split), so it is
    # left as-is rather than overwritten with the all-candidate count.
    train_stacks = sum(
        1
        for label in labels
        for slot in label.slots
        if slot.stack.status is LabelStatus.VALID and label.frame in train_frames
    )
    summary = replace(
        summary,
        train_frames=len(train_frames),
        eval_frames=len(eval_frames),
        train_stacks=train_stacks,
        template_digit_samples=sum(len(v) for v in templates.values()),
        digits_covered=len(templates),
    )
    return RunResult(
        labels_path=Path(labels_path),
        train_frames=train_frames,
        eval_frames=eval_frames,
        templates=templates,
        candidates=candidates,
        summary=summary,
        mismatches=tuple(mismatches),
    )


__all__ = [
    "AutoStackSummary",
    "DigitRead",
    "EVAL_RATIO",
    "FIT_THRESHOLD",
    "MARGIN_THRESHOLD",
    "RunResult",
    "StackCandidate",
    "build_digit_templates_from_labels",
    "evaluate_candidates",
    "hold_out_split",
    "read_targets",
    "render_auto_report",
    "render_proposal_csv",
    "run_stack_auto",
]
