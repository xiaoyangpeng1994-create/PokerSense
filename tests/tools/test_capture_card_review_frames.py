# -*- coding: utf-8 -*-
"""Tests for the stage-F label review page (review_frames.py).

The review surfaces what still needs labelling; it must never edit a label or
invent a value. These tests pin the gap detection (UNKNOWN/CONFLICT are gaps,
VALID is not), the field rendering (VALID prints its value, UNKNOWN and
CONFLICT never do), the slot view, the per-frame issue indexing, and the
self-contained HTML/JSON output. They use synthetic labels and a tiny valid
PNG byte string — no private dataset and no OpenCV.
"""

from __future__ import annotations

import re

from tools.capture_card_calibration.audit import audit_labels
from tools.capture_card_calibration.review_frames import (
    build_card,
    collect_gaps,
    encode_image_bytes,
    frame_field_text,
    has_gaps,
    index_issues,
    render_card_json,
    render_field,
    render_review_html,
    slot_views,
)
from tools.capture_card_calibration.schema import (
    CompletedAction,
    FieldValue,
    FrameLabel,
    Occupancy,
    Review,
    Scene,
    SlotLabel,
    Street,
)

# A minimal valid PNG (1x1 transparent pixel) — just enough pixels to prove
# embedding works, and small enough to keep the test suite fast.
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360606060000000050001a5f0e2ff0000000049454e44ae426082"
)


def _slot(slot_id: int, **overrides: object) -> SlotLabel:
    data: dict[str, object] = {
        "slot_id": slot_id,
        "occupancy": FieldValue.valid(Occupancy.OCCUPIED.value).to_dict(),
        "stack": FieldValue.valid(100 + slot_id).to_dict(),
        "dealer": FieldValue.valid(False).to_dict(),
        "completed_action": FieldValue.valid(CompletedAction.CHECK.value).to_dict(),
        "current_actor": FieldValue.valid("HERO").to_dict(),
    }
    data.update(overrides)
    return SlotLabel.from_dict(data)


def _label(
    frame: str = "a.png",
    *,
    hand_id: str = "h1",
    timestamp_ms: int = 0,
    board: object = FieldValue.valid(["4H", "KC", "TS"]),
    street: object = FieldValue.valid(Street.FLOP.value),
    pot: object = FieldValue.valid(120),
    **overrides: object,
) -> FrameLabel:
    kwargs: dict[str, object] = {
        "frame": frame,
        "sha256": "a" * 64,
        "session_id": "session_001",
        "hand_id": hand_id,
        "timestamp_ms": timestamp_ms,
        "stable": True,
        "scene": Scene.TABLE,
        "hero_cards": FieldValue.valid(["TS", "JD"]),
        "board_cards": board,
        "street": street,
        "pot": pot,
        "slots": tuple(
            _slot(i, dealer=FieldValue.valid(i == 3).to_dict())
            for i in range(8)
        ),
        "review": Review(reviewer="tester"),
    }
    kwargs.update(overrides)
    return FrameLabel(**kwargs)


# --- field rendering -------------------------------------------------------


def test_render_field_valid_scalar() -> None:
    assert render_field(FieldValue.valid(120)) == "120"


def test_render_field_valid_card_list() -> None:
    assert render_field(FieldValue.valid(["TS", "JD"])) == "TS JD"


def test_render_field_valid_bool_dealer() -> None:
    assert render_field(FieldValue.valid(True)) == "dealer"
    assert render_field(FieldValue.valid(False)) == "—"


def test_render_field_unknown_is_em_dash() -> None:
    assert render_field(FieldValue.unknown()) == "—"


def test_render_field_unknown_never_invents_value() -> None:
    # UNKNOWN must not render as a plausible-looking default.
    rendered = render_field(FieldValue.unknown())
    assert rendered in {"—", ""}


def test_render_field_conflict_is_cross() -> None:
    assert render_field(FieldValue.conflict()) == "✗"


# --- gap detection ---------------------------------------------------------


def test_full_label_has_no_gaps() -> None:
    # All frame fields VALID, all slots OCCUPIED with a stack + dealer.
    label = _label()
    assert not has_gaps(label)
    assert collect_gaps(label) == ()


def test_unknown_frame_field_is_a_gap() -> None:
    label = _label(hero_cards=FieldValue.unknown())
    gaps = collect_gaps(label)
    kinds = {(gap.kind, gap.field) for gap in gaps}
    assert ("frame", "hero_cards") in kinds


def test_unknown_slot_field_is_a_slot_gap() -> None:
    # Make one slot's stack UNKNOWN (default is VALID) and its action UNKNOWN.
    slots = list(_label().slots)
    slots[2] = SlotLabel.from_dict(
        {**slots[2].to_dict(), "stack": FieldValue.unknown().to_dict()}
    )
    label = _label(slots=tuple(slots))
    gaps = collect_gaps(label)
    kind_slots = [(g.kind, g.field, g.slot_id) for g in gaps]
    assert ("slot", "stack", 2) in kind_slots


# --- slot views ------------------------------------------------------------


def test_slot_views_lists_all_slots() -> None:
    views = slot_views(_label())
    assert len(views) == 8
    assert [view.slot_id for view in views] == list(range(8))


def test_slot_view_marks_missing_fields() -> None:
    # Construct a slot whose per-slot fields are partly UNKNOWN: this is the
    # state the labeller actually has to resolve.
    slots = list(_label().slots)
    slots[1] = SlotLabel.from_dict(
        {
            **slots[1].to_dict(),
            "occupancy": FieldValue.unknown().to_dict(),
            "stack": FieldValue.unknown().to_dict(),
            "dealer": FieldValue.unknown().to_dict(),
            "completed_action": FieldValue.unknown().to_dict(),
            "current_actor": FieldValue.unknown().to_dict(),
        }
    )
    label = _label(slots=tuple(slots))
    views = slot_views(label)
    for view in views:
        if view.slot_id == 1:
            assert set(view.missing) == {
                "occupancy",
                "stack",
                "dealer",
                "completed_action",
                "current_actor",
            }
        else:
            assert view.missing == ()
            # Slot 3 is the dealer (True); all others are non-dealer (False).
            assert view.dealer == ("dealer" if view.slot_id == 3 else "—")
            assert view.action == "CHECK"


def test_frame_field_text_snapshot() -> None:
    text = frame_field_text(_label())
    assert text["hero_cards"] == "TS JD"
    assert text["board_cards"] == "4H KC TS"
    assert text["street"] == "FLOP"
    assert text["pot"] == "120"


# --- issue indexing --------------------------------------------------------


def test_index_issues_buckets_by_frame() -> None:
    labels = [_label(frame="a.png"), _label(frame="b.png", hand_id="h2")]
    report = audit_labels(labels)
    index = index_issues(report)
    # No issues are expected for these fully-consistent labels.
    assert all(value == [] for value in index.values()) or not index


def test_index_issues_contains_violating_frame() -> None:
    # A street/board mismatch must be bucketed under its frame.
    label = _label(board=FieldValue.valid(["4H", "KC"]))  # FLOP expects 3
    report = audit_labels([label])
    index = index_issues(report)
    assert label.frame in index
    assert any(issue.rule == "street-board-count" for issue in index[label.frame])


# --- build_card ------------------------------------------------------------


def test_build_card_reads_image_bytes() -> None:
    label = _label()
    card = build_card(label, image_bytes=_PNG, include_image=True)
    assert card.has_image is True
    assert card.image  # base64 payload present
    assert card.frame == label.frame


def test_build_card_omits_image_when_disabled() -> None:
    label = _label()
    card = build_card(label, include_image=False)
    assert card.has_image is False
    assert card.image == ""


def test_build_card_attaches_issues() -> None:
    label = _label(board=FieldValue.valid(["4H", "KC"]))
    report = audit_labels([label])
    index = index_issues(report)
    card = build_card(label, image_bytes=_PNG, issue_bucket=index)
    assert card.issues
    assert card.issues[0].rule == "street-board-count"


def test_encode_image_bytes_roundtrips() -> None:
    payload = encode_image_bytes(_PNG)
    assert payload
    assert payload == encode_image_bytes(_PNG)


# --- HTML / JSON rendering -------------------------------------------------


def test_render_review_html_is_self_contained() -> None:
    label = _label()
    card = build_card(label, image_bytes=_PNG)
    page = render_review_html([card], title="t", summary="s")
    assert "<!DOCTYPE html>" in page
    assert "data:image/png;base64," in page
    assert "slot0" in page
    assert "needs label" not in page  # fully labelled


def test_render_review_html_marks_gap_card() -> None:
    label = _label(hero_cards=FieldValue.unknown())
    card = build_card(label, image_bytes=_PNG)
    page = render_review_html([card], title="t", summary="s")
    assert "needs label" in page
    assert "hero_cards" in page


def test_render_review_html_escapes_frame_name() -> None:
    label = _label(frame="<script>alert(1)</script>.png")
    card = build_card(label, image_bytes=_PNG)
    page = render_review_html([card], title="t", summary="s")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_render_review_html_placeholder_when_no_image() -> None:
    label = _label()
    card = build_card(label, include_image=False)
    page = render_review_html([card], title="t", summary="s")
    assert "frame image missing" in page


def test_render_card_json_contains_cards() -> None:
    label = _label()
    card = build_card(label, image_bytes=_PNG)
    text = render_card_json([card])
    assert '"frame"' in text
    assert '"gaps"' in text
    assert '"slots"' in text


def test_render_review_html_multiple_cards_and_stats() -> None:
    labels = [_label(frame="a.png"), _label(frame="b.png", hand_id="h2")]
    cards = [build_card(label, image_bytes=_PNG) for label in labels]
    page = render_review_html(cards, title="t", summary="s")
    assert "frames" in page
    assert "a.png" in page
    assert "b.png" in page


# --- PCI / contract --------------------------------------------------------


def test_html_has_no_external_refs() -> None:
    # The page must be self-contained: no <link>, <script src=...>, or http src.
    label = _label()
    card = build_card(label, image_bytes=_PNG)
    page = render_review_html([card], title="t", summary="s")
    assert "<link" not in page
    assert "<script" not in page
    assert "http://" not in page and "https://" not in page
    assert not re.search(r'src="(?!data:)', page)
