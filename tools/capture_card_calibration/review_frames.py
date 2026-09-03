# -*- coding: utf-8 -*-
"""Stage F label top-up review: render a labeller-facing HTML contact sheet.

Stage F ground truth is labelled frame by frame from pixels that are actually
visible. The heavy lifting for stage F (and its successor stage G/H) is the
human looking at a normalized frame and promoting ``UNKNOWN`` fields to
``VALID``. A terminal ``coverage``/``splits`` run tells the owner *how many*
samples are missing but never *which pixel to look at* — so this module turns
an audit report plus the label set into a per-frame review page.

Each card answers, for one frame, the three questions a labeller actually
has:

1. **What does the frame look like?**  (the normalized PNG, embedded)
2. **What is still missing or suspicious?**  (UNKNOWN fields, and any audit
   WARN/ERROR that points at this frame or one of its slots)
3. **What is already labelled?**  (a slot-by-slot read-out)

The output is a self-contained HTML document (images inlined as base64) so it
can be opened on any machine without a data directory — which matters because
the normalized frames are a **private** dataset that must never enter Git.
The review lives in the private dataset's ``reports/`` directory.

This module is import-safe without OpenCV; it only manipulates labels, audit
issues and bytes. It never rewrites a label and never guesses a value — it
just surfaces what is left, honouring the guide's failure-closed philosophy.
"""

from __future__ import annotations

import base64
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .audit import AuditReport, Issue
from .schema import FieldValue, FrameLabel, LabelStatus

#: Fields reviewed at the frame level (not per slot).
_FRAME_FIELDS = ("hero_cards", "board_cards", "street", "pot")
#: Per-slot fields that drive stage I calibration.
_SLOT_FIELDS = ("occupancy", "stack", "dealer", "completed_action", "current_actor")

#: Human labels for the frame-level meta table.
_FRAME_META = (
    ("frame", "frame"),
    ("session_id", "session"),
    ("hand_id", "hand"),
    ("scene", "scene"),
    ("street", "street"),
    ("timestamp_ms", "timestamp"),
)

#: CSS injected into the page. Kept deliberately small and system-font driven;
#: the review is a working document, not a marketing page.
_CSS = """
:root {
  --bg: #f5f6f8;
  --card-bg: #ffffff;
  --ink: #1c1e21;
  --muted: #6a7280;
  --line: #e2e5ea;
  --gap: #c2410c;
  --gap-bg: #fff5ec;
  --err: #b91c1c;
  --err-bg: #fef2f2;
  --ok: #15803d;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 24px;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.5 -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei",
    sans-serif;
}
h1 { font-size: 20px; margin: 0 0 6px; }
.sub { color: var(--muted); margin: 0 0 20px; }
.stat {
  display: inline-block;
  margin-right: 16px;
  padding: 6px 12px;
  background: var(--card-bg);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.stat b { color: var(--ink); }
.grid { display: grid; gap: 20px; }
.card {
  background: var(--card-bg);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 16px;
}
.card-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}
.card-title { font-weight: 600; font-size: 15px; }
.badge {
  font-size: 12px;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid var(--line);
  color: var(--muted);
  white-space: nowrap;
}
.badge.gap { color: var(--gap); border-color: var(--gap); background: var(--gap-bg); }
.card-body { display: flex; gap: 16px; flex-wrap: wrap; }
.frame-img {
  flex: 0 0 auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #000;
  max-height: 560px;
  max-width: 100%;
}
.panel { flex: 1 1 360px; min-width: 300px; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
th, td {
  text-align: left;
  padding: 5px 8px;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
}
th { color: var(--muted); font-weight: 500; white-space: nowrap; }
td .v { font-variant-numeric: tabular-nums; }
.miss { color: var(--gap); font-weight: 600; }
.err { color: var(--err); }
.warn { color: var(--gap); }
.ok { color: var(--ok); }
.gap-note {
  margin: 0 0 10px;
  padding: 8px 10px;
  font-size: 12px;
  border-radius: 6px;
}
.gap-note.has-gap { background: var(--gap-bg); color: var(--gap); }
.issue {
  margin: 4px 0;
  padding: 6px 10px;
  font-size: 12px;
  border-radius: 6px;
}
.issue.ERROR { background: var(--err-bg); color: var(--err); }
.issue.WARN { background: var(--gap-bg); color: var(--gap); }
.foot { margin-top: 24px; color: var(--muted); font-size: 12px; }
"""


@dataclass(frozen=True)
class Gap:
    """One missing/suspicious field on one frame."""

    kind: str  # "frame" | "slot"
    field: str
    slot_id: int | None = None

    @property
    def label(self) -> str:
        if self.slot_id is None:
            return self.field
        return f"slot{self.slot_id}.{self.field}"

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "field": self.field,
            "slot_id": self.slot_id,
        }


@dataclass(frozen=True)
class SlotView:
    """Rendered, labeller-friendly view of one slot's fields."""

    slot_id: int
    occupancy: str
    stack: str
    dealer: str
    action: str
    actor: str
    missing: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "occupancy": self.occupancy,
            "stack": self.stack,
            "dealer": self.dealer,
            "action": self.action,
            "actor": self.actor,
            "missing": list(self.missing),
        }


@dataclass(frozen=True)
class FrameCard:
    """Everything needed to render one review card."""

    frame: str
    session_id: str
    hand_id: str
    scene: str
    street: str
    timestamp_ms: int
    image: str  # base64 data URI (no prefix)
    image_path: str  # absolute path, for diagnostics / optional ref mode
    has_image: bool
    frame_fields: Mapping[str, str]  # hero_cards/board_cards/street/pot text
    slots: tuple[SlotView, ...]
    gaps: tuple[Gap, ...]
    issues: tuple[Issue, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "session_id": self.session_id,
            "hand_id": self.hand_id,
            "scene": self.scene,
            "street": self.street,
            "timestamp_ms": self.timestamp_ms,
            "has_image": self.has_image,
            "image_path": self.image_path,
            "frame_fields": dict(self.frame_fields),
            "slots": [slot.to_dict() for slot in self.slots],
            "gaps": [gap.to_dict() for gap in self.gaps],
            "issues": [issue.to_dict() for issue in self.issues],
        }


# --- field rendering -------------------------------------------------------


def render_field(field: FieldValue) -> str:
    """Render a ``FieldValue`` to a short human-readable text.

    ``VALID`` prints its value; ``UNKNOWN`` prints an em-dash; ``CONFLICT``
    prints a tilde. This is presentation only — it never invents a value.
    """
    if field.status is LabelStatus.VALID:
        value = field.value
        if isinstance(value, (list, tuple)):
            return " ".join(str(item) for item in value)
        if isinstance(value, bool):
            return "dealer" if value else "—"
        return str(value)
    if field.status is LabelStatus.CONFLICT:
        return "✗"
    return "—"


def _field(item: FieldValue) -> str:
    """Render a FieldValue holding an enum value (occupancy etc.)."""
    return render_field(item)


def _missing(field: FieldValue) -> bool:
    """True when a field is not VALID (i.e. UNKNOWN or CONFLICT)."""
    return field.status is not LabelStatus.VALID


def _slot_missing(slot) -> tuple[str, ...]:
    """Names of the per-slot fields that are still UNKNOWN/CONFLICT."""
    return tuple(name for name in _SLOT_FIELDS if _missing(getattr(slot, name)))


def collect_gaps(label: FrameLabel) -> tuple[Gap, ...]:
    """Return the frame-level and per-slot fields that are not yet VALID.

    This is the mechanical definition of "what still needs labelling" on a
    frame: any field whose status is not VALID. Slot-level fields do not have
    a "whole-slot empty" concept — each field is judged on its own.
    """
    frame_gaps = tuple(
        Gap(kind="frame", field=name)
        for name in _FRAME_FIELDS
        if _missing(getattr(label, name))
    )
    slot_gaps: list[Gap] = []
    for slot in label.slots:
        for name in _slot_missing(slot):
            slot_gaps.append(Gap(kind="slot", field=name, slot_id=slot.slot_id))
    return frame_gaps + tuple(slot_gaps)


def slot_views(label: FrameLabel) -> tuple[SlotView, ...]:
    """Build a labeller-friendly read-out for every slot on a frame."""
    views: list[SlotView] = []
    for slot in label.slots:
        views.append(
            SlotView(
                slot_id=slot.slot_id,
                occupancy=_field(slot.occupancy),
                stack=render_field(slot.stack),
                dealer=render_field(slot.dealer),
                action=render_field(slot.completed_action),
                actor=render_field(slot.current_actor),
                missing=_slot_missing(slot),
            )
        )
    return tuple(views)


def frame_field_text(label: FrameLabel) -> dict[str, str]:
    """Frame-level fields (hero/board/street/pot) rendered as text."""
    return {name: render_field(getattr(label, name)) for name in _FRAME_FIELDS}


def index_issues(report: AuditReport) -> dict[str, list[Issue]]:
    """Bucket audit issues by frame name (preserving severity order)."""
    index: dict[str, list[Issue]] = {}
    for issue in report.issues:
        index.setdefault(issue.frame, []).append(issue)
    return index


def has_gaps(label: FrameLabel) -> bool:
    """True when any frame-level or slot-level field is not VALID."""
    return bool(collect_gaps(label))


def encode_image_bytes(data: bytes) -> str:
    """Base64-encode raw image bytes for a ``data:`` URI (no prefix)."""
    return base64.b64encode(data).decode("ascii")


def encode_frame_image(path: Path) -> str:
    """Read a PNG/JPEG frame and return its base64 payload.

    Missing or unreadable files yield an empty string so the caller can render
    a placeholder instead of crashing the whole review.
    """
    try:
        return encode_image_bytes(path.read_bytes())
    except OSError:
        return ""


# --- HTML rendering --------------------------------------------------------


def _esc(text: str) -> str:
    return html.escape(str(text))


def _issue_html(issue: Issue) -> str:
    slot = f" slot{issue.slot_id}" if issue.slot_id is not None else ""
    return (
        f'<div class="issue {_esc(issue.severity)}">'
        f"<b>[{_esc(issue.severity)}]</b> {_esc(issue.rule)}{slot} — "
        f"{_esc(issue.message)}</div>"
    )


def _card_html(card: FrameCard) -> str:
    if card.has_image and card.image:
        image_html = (
            f'<img class="frame-img" alt="{_esc(card.frame)}" '
            f'src="data:image/png;base64,{card.image}">'
        )
    else:
        image_html = (
            '<div class="frame-img" style="display:grid;place-items:center;'
            'color:#666;padding:40px 24px;">frame image missing</div>'
        )

    meta_rows = []
    for key, header in _FRAME_META:
        value = getattr(card, key, "")
        meta_rows.append(f"<tr><th>{_esc(header)}</th><td>{_esc(value)}</td></tr>")
    meta_table = (
        "<table>" + "".join(meta_rows)
        + "</table>"
    )

    frame_rows = "".join(
        f"<tr><th>{_esc(k)}</th><td class='v'>{_esc(v)}</td></tr>"
        for k, v in card.frame_fields.items()
    )
    frame_table = "<table>" + frame_rows + "</table>"

    slot_rows = []
    for slot in card.slots:
        cells = "".join(
            f"<td class='v'>{_esc(slot.occupancy)}</td>"
            f"<td class='v'>{_esc(slot.stack)}</td>"
            f"<td class='v'>{_esc(slot.dealer)}</td>"
            f"<td class='v'>{_esc(slot.action)}</td>"
            f"<td class='v'>{_esc(slot.actor)}</td>"
        )
        missing = "".join(
            f"<span class='miss'>{_esc(name)}</span> " for name in slot.missing
        )
        slot_rows.append(
            f"<tr><th>slot{slot.slot_id}</th>{cells}"
            f"<td class='miss'>{missing}</td></tr>"
        )
    slot_table = (
        "<table>"
        "<thead><tr><th></th><th>occ</th><th>stack</th><th>dealer</th>"
        "<th>action</th><th>actor</th><th>missing</th></tr></thead>"
        + "".join(slot_rows)
        + "</table>"
    )

    issue_html = "".join(_issue_html(issue) for issue in card.issues)
    badge = (
        '<span class="badge gap">needs label</span>'
        if card.gaps
        else '<span class="badge ok">complete</span>'
    )

    return (
        '<div class="card">'
        f'<div class="card-head"><span class="card-title">{_esc(card.frame)}</span>'
        f"{badge}</div>"
        '<div class="card-body">'
        f"{image_html}"
        "<div class=\"panel\">"
        f"{meta_table}"
        "<h3>frame fields</h3>"
        f"{frame_table}"
        "<h3>slots</h3>"
        f"{slot_table}"
        f"{issue_html}"
        "</div></div></div>"
    )


def render_review_html(
    cards: Sequence[FrameCard],
    *,
    title: str = "Capture-card label review",
    summary: str = "",
) -> str:
    """Render cards into a self-contained HTML document."""
    total = len(cards)
    n_gap = sum(1 for card in cards if card.gaps)
    n_issue = sum(1 for card in cards if card.issues)

    stats = (
        f'<span class="stat">frames <b>{total}</b></span>'
        f'<span class="stat">need label <b>{n_gap}</b></span>'
        f'<span class="stat">audit findings <b>{n_issue}</b></span>'
    )
    grid = "".join(_card_html(card) for card in cards)

    body = (
        f"<!DOCTYPE html>\n"
        "<html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{_esc(title)}</title>"
        f"<style>{_CSS}</style></head><body>"
        f"<h1>{_esc(title)}</h1>"
        f"<p class=\"sub\">{_esc(summary)}</p>"
        f"{stats}"
        f'<div class="grid">{grid}</div>'
        f'<p class="foot">Private dataset — full frames must never enter Git. '
        "This page is generated from labelled ground truth; it never edits a "
        "label or guesses a value.</p>"
        "</body></html>"
    )
    return body


def build_card(
    label: FrameLabel,
    *,
    image_path: Path | None = None,
    image_bytes: bytes | None = None,
    issue_bucket: Mapping[str, Sequence[Issue]] | None = None,
    include_image: bool = True,
) -> FrameCard:
    """Assemble a :class:`FrameCard` for one label.

    ``image_path`` points at the normalized frame in a private dataset; when
    ``image_bytes`` is supplied it is used verbatim (for tests, where we build
    synthetic frames without touching the disk). ``issue_bucket`` is the audit
    issue index (see :func:`index_issues`); a frame with no bucket entry has no
    issues. ``include_image=False`` renders a placeholder — useful for a
    label-only pass or when the frames must not be copied out.
    """
    gaps = collect_gaps(label)
    if include_image and image_bytes is None and image_path is not None:
        image = encode_frame_image(image_path)
    elif include_image and image_bytes is not None:
        image = encode_image_bytes(image_bytes)
    else:
        image = ""
    issues = tuple(issue_bucket.get(label.frame, ())) if issue_bucket else ()
    card = FrameCard(
        frame=label.frame,
        session_id=label.session_id,
        hand_id=label.hand_id,
        scene=label.scene.value,
        street=render_field(label.street),
        timestamp_ms=label.timestamp_ms,
        image=image,
        image_path=str(image_path) if image_path else "",
        has_image=bool(include_image),
        frame_fields=frame_field_text(label),
        slots=slot_views(label),
        gaps=gaps,
        issues=issues,
    )
    return card


def render_card_json(cards: Sequence[FrameCard]) -> str:
    """Dump the review cards as JSON (for ``--json-out`` or paired tooling).

    Images are included as base64 payloads so the JSON is self-contained; use
    ``--json-out`` to inspect the machine-readable side of the review.
    """
    import json

    return (
        json.dumps(
            [card.to_dict() for card in cards],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


__all__ = [
    "FrameCard",
    "Gap",
    "SlotView",
    "build_card",
    "collect_gaps",
    "encode_frame_image",
    "encode_image_bytes",
    "frame_field_text",
    "has_gaps",
    "index_issues",
    "render_card_json",
    "render_field",
    "render_review_html",
    "slot_views",
]
