"""Boundary regressions for the P0 selected #19 migration.

Everything here uses SYNTHETIC metadata and temporary files only: no real
recording is opened, no frame is decoded, no historical label file is imported,
no model is invoked and no training is executed. Every value below is an
invented fixture, never a real observation.
"""

import hashlib
import json
from pathlib import Path

import pytest

from tools.aa_annotation_records import (
    EVIDENCE_ACTUALLY_CONSUMED, EVIDENCE_RELEASED, EVIDENCE_RESERVED,
    FIELD_POT_DISPLAY, FIELD_VISIBLE_ACTION_GLYPH,
    AnnotationStore, FieldSemanticsMismatch, ForbiddenTruthFieldError,
    Label, LabelValidationError, ObservationContext, PURPOSE_TEMPLATE_SELECTION,
    PURPOSE_TRAINING, RevisionConflictError, SourceIdentityError,
    STATE_ACTUALLY_CONSUMED, STATE_NONE_UNDER_COVERAGE, STATE_POSSIBLY_USED,
    STATE_UNKNOWN, STATUS_KNOWN, STATUS_NOT_APPLICABLE, STATUS_UNKNOWN,
    UnsupportedFieldError, VERDICT_ALLOWED_FOR_PURPOSE, VERDICT_BLOCKED,
    append_annotation, assess_use, exposure_projection, get_annotations,
    record_exposure,
)

ACTOR = "synth-annotator"
POLICY_REF = "synth-policy-aa-annotations-p0"


def digest(seed):
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def source_ref(
    *,
    audit="manifest-synth-A",
    media="media-synth-A",
    frame=100,
    slot_count=8,
    mapping=None,
    parent=None,
    implementation="vision-pipeline-synth-1",
    digest_seed=None,
):
    """Synthetic, fully bound source identity (no paths, names or UI rows)."""
    seed = digest_seed or f"{media}|{frame}|v1"
    return {
        "source_audit_ref": audit,
        "source_audit_sha256": digest("audit|" + audit),
        "media_id": media,
        "media_sha256": digest("media|" + media),
        "frame_index": frame,
        "frame_sha256": digest("frame|" + seed),
        "layout_id": "aa-synth-layout-8seat",
        "slot_count": slot_count,
        "slot_mapping": mapping if mapping is not None else {
            index: index for index in range(slot_count)
        },
        "observation_snapshot_digest": digest("observation|" + seed),
        "implementation_revision": implementation,
        "parent_frame_ref": parent,
    }


def load_reservations():
    path = Path(__file__).resolve().parents[2] / (
        "configs/reproduction/aa_holdout_reservations_v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def policy(store_dir, **overrides):
    bundle = {
        "policy_ref": POLICY_REF,
        "role": "development",
        "reservations": load_reservations(),
        "development_evidence_ref": "synth:dev-frame-evidence-000001",
        "declared_store_paths": [str(store_dir)],
        "required_fields": [FIELD_POT_DISPLAY],
    }
    bundle.update(overrides)
    return bundle


def annotate_pot(store, ref, *, seat=0, chips=120, expected=None):
    return append_annotation(
        store,
        ref,
        FIELD_POT_DISPLAY,
        seat,
        Label(status=STATUS_KNOWN, value=chips, unit="chips", reason="synth"),
        expected,
        ACTOR,
    )


def current_rows(payload, field_id=None):
    rows = list(payload["current"].values())
    if field_id is not None:
        rows = [row for row in rows if row["field_id"] == field_id]
    return rows


def run_ref(name="synth-run-0001"):
    return {"kind": "synthetic-run", "run_id": name}


@pytest.fixture()
def store(tmp_path):
    return AnnotationStore(tmp_path / "annotation-store")


# 1 ------------------------------------------------------------------------
def test_two_revisions_keep_the_old_revision(store):
    ref = source_ref()
    first = annotate_pot(store, ref, chips=120)
    second = annotate_pot(store, ref, chips=150, expected=first)

    payload = get_annotations(store, ref, include_history=True)
    history = payload["history"]
    assert [row["revision_id"] for row in history] == [first, second]
    assert history[0]["value"] == 120 and history[1]["value"] == 150
    assert history[1]["supersedes"] == first
    rows = current_rows(payload, FIELD_POT_DISPLAY)
    assert len(rows) == 1 and rows[0]["value"] == 150


# 2 ------------------------------------------------------------------------
def test_expected_revision_conflict_is_refused(store):
    ref = source_ref()
    first = annotate_pot(store, ref, chips=120)
    annotate_pot(store, ref, chips=150, expected=first)

    with pytest.raises(RevisionConflictError, match="revision_conflict"):
        annotate_pot(store, ref, chips=180, expected=first)
    with pytest.raises(RevisionConflictError, match="expected_revision_required"):
        annotate_pot(store, ref, chips=180)
    rows = current_rows(get_annotations(store, ref), FIELD_POT_DISPLAY)
    assert rows[0]["value"] == 150


# 3 ------------------------------------------------------------------------
def test_source_frame_and_seat_do_not_bleed(store):
    first = source_ref(media="media-synth-A", frame=100)
    second = source_ref(media="media-synth-B", frame=100)
    annotate_pot(store, first, seat=0, chips=120)
    annotate_pot(store, second, seat=0, chips=990)
    annotate_pot(store, first, seat=1, chips=7)

    assert current_rows(get_annotations(store, first), FIELD_POT_DISPLAY)[0][
        "value"
    ] == 120
    assert current_rows(get_annotations(store, second), FIELD_POT_DISPLAY)[0][
        "value"
    ] == 990
    seats = {
        row["seat"]: row["value"]
        for row in get_annotations(store, first)["current"].values()
    }
    assert seats == {0: 120, 1: 7}


# 4 ------------------------------------------------------------------------
def test_history_survives_restart(tmp_path):
    ref = source_ref()
    annotate_pot(AnnotationStore(tmp_path / "store"), ref, chips=120)
    reloaded = AnnotationStore(tmp_path / "store")
    rows = current_rows(get_annotations(reloaded, ref), FIELD_POT_DISPLAY)
    assert rows and rows[0]["value"] == 120
    assert (tmp_path / "store" / "annotations.jsonl").exists()


# 5 ------------------------------------------------------------------------
def test_known_unknown_and_not_applicable_are_distinct(store):
    ref = source_ref(frame=110)
    append_annotation(
        store, ref, FIELD_POT_DISPLAY, 0,
        Label(status=STATUS_UNKNOWN, reason="synth unreadable"), None, ACTOR,
    )
    append_annotation(
        store, ref, FIELD_POT_DISPLAY, 1,
        Label(status=STATUS_NOT_APPLICABLE, reason="synth no pot widget"),
        None, ACTOR,
    )
    annotate_pot(store, ref, seat=2, chips=0)

    by_seat = {
        row["seat"]: row["status"]
        for row in get_annotations(store, ref)["current"].values()
    }
    assert by_seat == {
        0: STATUS_UNKNOWN,
        1: STATUS_NOT_APPLICABLE,
        2: STATUS_KNOWN,
    }
    with pytest.raises(LabelValidationError, match="reason_required_for"):
        append_annotation(
            store, ref, FIELD_POT_DISPLAY, 3,
            Label(status=STATUS_UNKNOWN), None, ACTOR,
        )
    with pytest.raises(LabelValidationError, match="known_label_requires_value"):
        append_annotation(
            store, ref, FIELD_POT_DISPLAY, 4,
            Label(status=STATUS_KNOWN, unit="chips"), None, ACTOR,
        )


# 6 ------------------------------------------------------------------------
def test_zero_is_not_unknown(store):
    ref = source_ref(frame=120)
    annotate_pot(store, ref, chips=0)
    rows = current_rows(get_annotations(store, ref), FIELD_POT_DISPLAY)
    assert rows[0]["status"] == STATUS_KNOWN
    assert rows[0]["value"] == 0
    assert rows[0]["value"] is not None


# 7 ------------------------------------------------------------------------
def test_visible_none_is_not_unknown(store):
    ref = source_ref(frame=130)
    append_annotation(
        store, ref, FIELD_VISIBLE_ACTION_GLYPH, 0,
        Label(status=STATUS_KNOWN, value="none", reason="synth no glyph"),
        None, ACTOR,
    )
    append_annotation(
        store, ref, FIELD_VISIBLE_ACTION_GLYPH, 1,
        Label(status=STATUS_UNKNOWN, reason="synth glyph area cropped"),
        None, ACTOR,
    )
    rows = {
        row["seat"]: row
        for row in get_annotations(store, ref)["current"].values()
    }
    assert rows[0]["status"] == STATUS_KNOWN and rows[0]["value"] == "none"
    assert rows[1]["status"] == STATUS_UNKNOWN and rows[1]["value"] is None


# 8 ------------------------------------------------------------------------
@pytest.mark.parametrize(
    "role", sorted({"current_street_wager", "call_price", "total_commitment"})
)
def test_pot_display_vs_wager_or_call_price_rejected(store, role):
    ref = source_ref(frame=140)
    with pytest.raises(FieldSemanticsMismatch, match="pot_display_vs_wager"):
        append_annotation(
            store, ref, FIELD_POT_DISPLAY, 0,
            Label(status=STATUS_KNOWN, value=120, unit="chips"),
            None, ACTOR, amount_role=role,
        )


# 9 ------------------------------------------------------------------------
def test_visible_glyph_vs_full_actions_rejected(store):
    ref = source_ref(frame=150)
    with pytest.raises(
        FieldSemanticsMismatch, match="visible_action_glyph_vs_full_actions"
    ):
        append_annotation(
            store, ref, FIELD_VISIBLE_ACTION_GLYPH, 0,
            Label(status=STATUS_KNOWN, value="call"),
            None, ACTOR, full_actions=[{"street": "flop", "action": "call"}],
        )


# 10 -----------------------------------------------------------------------
def test_legacy_nine_slot_mapping_is_not_guessed(store):
    legacy = source_ref(frame=160, slot_count=9)
    with pytest.raises(SourceIdentityError, match="seat_not_mappable"):
        annotate_pot(store, legacy, seat=3)
    decision = assess_use(store, legacy, PURPOSE_TRAINING, policy(store.path))
    assert decision["verdict"] == VERDICT_BLOCKED
    assert any("slot_count_not_supported:9" in reason
               for reason in decision["reasons"])

    partial = source_ref(frame=170, mapping={index: index for index in range(7)})
    identity = get_annotations(store, partial)
    assert identity["identity_class"] != "COMPLETE"
    assert any("slot_mapping" in gap for gap in identity["identity_gaps"])


# 11 -----------------------------------------------------------------------
def test_reserved_and_development_conflict_blocked(store):
    ref = source_ref(frame=3000)
    annotate_pot(store, ref)
    decision = assess_use(store, ref, PURPOSE_TRAINING, policy(store.path))
    assert decision["verdict"] == VERDICT_BLOCKED
    assert "frame_reserved_or_invalid" in decision["reasons"]
    assert "role_conflicts_reservation" in decision["reasons"]


# 12 -----------------------------------------------------------------------
def test_unknown_exposure_history_blocked(store):
    ref = source_ref(frame=100)
    annotate_pot(store, ref)
    bundle = policy(store.path)
    bundle["declared_store_paths"] = [str(store.path / "missing-store")]
    decision = assess_use(store, ref, PURPOSE_TRAINING, bundle)
    assert decision["verdict"] == VERDICT_BLOCKED
    assert decision["exposure_state"] == STATE_UNKNOWN
    assert any("declared_store_missing" in reason
               for reason in decision["reasons"])


# 13 -----------------------------------------------------------------------
def test_reserved_only_check_does_not_release(store):
    from tools.aa_data_separation import assert_training_frames

    ref = source_ref(frame=100)
    annotate_pot(store, ref)
    reservations = load_reservations()
    assert_training_frames([ref["frame_index"]], reservations)

    bundle = policy(store.path)
    bundle.pop("declared_store_paths")
    bundle.pop("reservations")
    decision = assess_use(store, ref, PURPOSE_TRAINING, bundle)
    assert decision["verdict"] == VERDICT_BLOCKED
    assert "reservations_missing" in decision["reasons"]
    assert "exposure_history_unknown" in decision["reasons"]

    full = assess_use(store, ref, PURPOSE_TRAINING, policy(store.path))
    assert full["verdict"] == VERDICT_ALLOWED_FOR_PURPOSE
    assert full["usable_scope"] == "PURPOSE_BOUND"
    assert full["exposure_state"] == STATE_NONE_UNDER_COVERAGE


# 14 -----------------------------------------------------------------------
def test_exposure_survives_annotation_change(store):
    ref = source_ref(frame=100)
    first = annotate_pot(store, ref, chips=120)
    record_exposure(
        store, ref, [first], PURPOSE_TRAINING, run_ref("run-A"),
        EVIDENCE_ACTUALLY_CONSUMED, policy_ref=POLICY_REF,
        role="development", actor=ACTOR,
        artifact_ref={"sha256": digest("artifact-A")},
    )
    second = annotate_pot(store, ref, chips=150, expected=first)
    assert second != first
    rows = current_rows(get_annotations(store, ref), FIELD_POT_DISPLAY)
    assert rows[0]["value"] == 150

    decision = assess_use(store, ref, PURPOSE_TRAINING, policy(store.path))
    assert decision["exposure_state"] == STATE_ACTUALLY_CONSUMED
    assert "exposed_development_reuse_not_declared" in decision["reasons"]
    projection = exposure_projection(
        store,
        {"sources": [ref], "declared_store_paths": [str(store.path)]},
    )
    assert projection["rows"][0]["exposure_state"] == STATE_ACTUALLY_CONSUMED


# 15 -----------------------------------------------------------------------
def test_identity_is_location_and_ui_independent(store):
    ref = source_ref(frame=100)
    for key in ("path", "replay_name", "ui_index"):
        broken = dict(ref)
        broken[key] = "C:/synth/recording.mp4" if key == "path" else "synth"
        with pytest.raises(SourceIdentityError, match="not_an_identity"):
            get_annotations(store, broken)

    first = annotate_pot(store, ref, chips=120)
    record_exposure(
        store, ref, [first], PURPOSE_TRAINING, run_ref("run-B"),
        EVIDENCE_ACTUALLY_CONSUMED, policy_ref=POLICY_REF,
        role="development", actor=ACTOR,
    )
    # Identity ignores mutable location names and model revisions; only the
    # bound source / media / frame triple counts.
    moved = source_ref(frame=100, implementation="vision-pipeline-synth-2")
    after = assess_use(store, moved, PURPOSE_TRAINING, policy(store.path))
    assert after["source_key"] == assess_use(
        store, ref, PURPOSE_TRAINING, policy(store.path)
    )["source_key"]
    assert after["exposure_state"] == STATE_ACTUALLY_CONSUMED


# 16 -----------------------------------------------------------------------
def test_derived_crop_tracks_parent_exposure(store):
    parent = source_ref(frame=200, media="media-synth-parent")
    child = source_ref(
        frame=200, media="media-synth-parent-crop",
        digest_seed="crop-7-seat3", parent=parent,
    )
    parent_rev = annotate_pot(store, parent, chips=80)
    record_exposure(
        store, parent, [parent_rev], PURPOSE_TRAINING, run_ref("run-crop-1"),
        EVIDENCE_ACTUALLY_CONSUMED, policy_ref=POLICY_REF,
        role="development", actor=ACTOR,
    )
    annotate_pot(store, child, chips=80)
    bundle = policy(store.path)
    child_decision = assess_use(store, child, PURPOSE_TRAINING, bundle)
    assert child_decision["exposure_state"] == STATE_ACTUALLY_CONSUMED

    second_parent = source_ref(frame=210, media="media-synth-parent-2")
    second_child = source_ref(
        frame=210, media="media-synth-parent-2-crop",
        digest_seed="crop-2-seat1", parent=second_parent,
    )
    child_rev = annotate_pot(store, second_child, chips=40)
    record_exposure(
        store, second_child, [child_rev], PURPOSE_TRAINING,
        run_ref("run-crop-2"), EVIDENCE_ACTUALLY_CONSUMED,
        policy_ref=POLICY_REF, role="development", actor=ACTOR,
    )
    annotate_pot(store, second_parent, chips=40)
    parent_decision = assess_use(store, second_parent, PURPOSE_TRAINING, bundle)
    assert parent_decision["exposure_state"] == STATE_ACTUALLY_CONSUMED


# 17 -----------------------------------------------------------------------
def test_fresh_store_never_claims_untouched_history(tmp_path):
    fresh = AnnotationStore(tmp_path / "fresh-store")
    ref = source_ref(frame=100)
    annotate_pot(fresh, ref)
    undecided = assess_use(
        fresh, ref, PURPOSE_TRAINING,
        {"policy_ref": POLICY_REF, "role": "development",
         "reservations": load_reservations()},
    )
    assert undecided["exposure_state"] == STATE_UNKNOWN
    assert undecided["verdict"] == VERDICT_BLOCKED

    declared = assess_use(fresh, ref, PURPOSE_TRAINING, policy(fresh.path))
    assert declared["exposure_state"] == STATE_NONE_UNDER_COVERAGE
    blob = json.dumps(declared, ensure_ascii=False).lower()
    assert "clean" not in blob and "never_trained" not in blob


# 18 -----------------------------------------------------------------------
def test_persistence_failure_blocks_the_consumer(store, monkeypatch):
    ref = source_ref(frame=100)
    annotate_pot(store, ref)

    def boom(self, row):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(AnnotationStore, "append_exposure", boom)
    decision = assess_use(
        store, ref, PURPOSE_TRAINING, policy(store.path), reserve=True,
        actor=ACTOR,
    )
    assert decision["verdict"] == VERDICT_BLOCKED
    assert "consumption_record_not_durable" in decision["reasons"]
    assert decision["reservation_receipt"] is None

    monkeypatch.undo()
    allowed = assess_use(
        store, ref, PURPOSE_TRAINING, policy(store.path), reserve=True,
        actor=ACTOR,
    )
    assert allowed["verdict"] == VERDICT_ALLOWED_FOR_PURPOSE
    assert allowed["reservation_receipt"]["evidence_kind"] == EVIDENCE_RESERVED


# 19 -----------------------------------------------------------------------
def test_possibly_used_and_actually_consumed_are_distinct(store):
    ref = source_ref(frame=100)
    revision = annotate_pot(store, ref)
    common = {
        "policy_ref": POLICY_REF, "role": "development", "actor": ACTOR,
    }
    record_exposure(
        store, ref, [revision], PURPOSE_TRAINING, run_ref("run-1"),
        EVIDENCE_RESERVED, **common,
    )
    state_after_reservation = exposure_projection(
        store,
        {"sources": [ref], "declared_store_paths": [str(store.path)]},
    )["rows"][0]["exposure_state"]
    assert state_after_reservation == STATE_POSSIBLY_USED

    record_exposure(
        store, ref, [revision], PURPOSE_TRAINING, run_ref("run-1"),
        EVIDENCE_RELEASED, **common,
    )
    after_release = exposure_projection(
        store,
        {"sources": [ref], "declared_store_paths": [str(store.path)]},
    )["rows"][0]["exposure_state"]
    assert after_release == STATE_POSSIBLY_USED
    kinds = sorted(
        row["evidence_kind"]
        for row in AnnotationStore(store.path).read_exposures()
    )
    assert kinds == sorted([EVIDENCE_RESERVED, EVIDENCE_RELEASED])

    record_exposure(
        store, ref, [revision], PURPOSE_TRAINING, run_ref("run-2"),
        EVIDENCE_ACTUALLY_CONSUMED, **common,
    )
    consumed = exposure_projection(
        store,
        {"sources": [ref], "declared_store_paths": [str(store.path)]},
    )["rows"][0]["exposure_state"]
    assert consumed == STATE_ACTUALLY_CONSUMED
    assert consumed != STATE_POSSIBLY_USED


# 20 -----------------------------------------------------------------------
def test_policy_violation_is_kept_and_blocks(store):
    ref = source_ref(frame=100)
    revision = annotate_pot(store, ref)
    event = record_exposure(
        store, ref, [revision], PURPOSE_TRAINING, run_ref("run-violation"),
        EVIDENCE_ACTUALLY_CONSUMED, policy_ref=POLICY_REF, role="holdout",
        actor=ACTOR, policy_violation=True,
    )
    stored = [
        row for row in AnnotationStore(store.path).read_exposures()
        if row["event_id"] == event["event_id"]
    ]
    assert stored and stored[0]["policy_violation"] is True

    bundle = policy(store.path)
    bundle["allow_reuse_of_exposed_development"] = True
    decision = assess_use(store, ref, PURPOSE_TRAINING, bundle)
    assert decision["verdict"] == VERDICT_BLOCKED
    assert "known_policy_violation_exposure" in decision["reasons"]


# 21 -----------------------------------------------------------------------
def test_projection_never_forges_clean_history(store):
    ref = source_ref(frame=100)
    revision = annotate_pot(store, ref)
    undecided = exposure_projection(store, {"sources": [ref]})
    assert undecided["history_coverage"] == "INCOMPLETE"
    assert undecided["rows"]
    assert all(not row["freeze_bindable"] for row in undecided["rows"])
    assert undecided["unresolved"]

    record_exposure(
        store, ref, [revision], PURPOSE_TEMPLATE_SELECTION, run_ref("run-3"),
        EVIDENCE_ACTUALLY_CONSUMED, policy_ref=POLICY_REF,
        role="development", actor=ACTOR,
    )
    missing_artifact = exposure_projection(
        store, {"sources": [ref], "declared_store_paths": [str(store.path)]},
    )
    assert missing_artifact["history_coverage"] == "COMPLETE"
    row = missing_artifact["rows"][0]
    assert not row["freeze_bindable"]
    assert "artifact_sha256_missing" in row["unresolved"]
    assert row["used_for"] == PURPOSE_TEMPLATE_SELECTION
    blob = json.dumps(missing_artifact, ensure_ascii=False).lower()
    assert "clean" not in blob and "never_trained" not in blob


# 22 -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "field_id",
    [
        "full_actions", "acceptance", "full_visual_acceptance",
        "strategy_eligible", "strategy_ready", "canonical_verified",
        "confirmed_facts", "confirmed_poker_facts", "holdout_gold",
        "phh_confirmed_actions",
    ],
)
def test_annotation_cannot_own_truth_fields(store, field_id):
    ref = source_ref(frame=100)
    with pytest.raises(ForbiddenTruthFieldError):
        append_annotation(
            store, ref, field_id, 0,
            Label(status=STATUS_KNOWN, value=True, reason="synth"),
            None, ACTOR,
        )


@pytest.mark.parametrize(
    "field_id",
    ["seat_presence", "current_bet", "street_wager", "actor", "special_mode"],
)
def test_p1_candidate_fields_fail_closed(store, field_id):
    ref = source_ref(frame=100)
    with pytest.raises(UnsupportedFieldError, match="field_not_supported_in_p0"):
        append_annotation(
            store, ref, field_id, 0,
            Label(status=STATUS_KNOWN, value=1, reason="synth"),
            None, ACTOR,
        )


def test_current_view_has_no_truth_shortcuts(store):
    ref = source_ref(frame=100)
    annotate_pot(store, ref)
    rows = current_rows(get_annotations(store, ref))
    forbidden = {"acceptance", "strategy_eligible", "canonical_verified"}
    assert rows and not (set(rows[0]) & forbidden)


# 23 -----------------------------------------------------------------------
def test_model_output_does_not_become_the_current_value(store):
    ref = source_ref(frame=100)
    context = ObservationContext(
        model_output=250, model_reason="synth model guess",
        model_output_seen="YES", implementation_revision="model-synth-9",
    )
    revision = append_annotation(
        store, ref, FIELD_POT_DISPLAY, 0,
        Label(status=STATUS_KNOWN, value=200, unit="chips", reason="synth"),
        None, ACTOR, observation=context,
    )
    payload = get_annotations(store, ref, include_history=True)
    assert current_rows(payload)[0]["value"] == 200
    stored = [
        row for row in payload["history"]
        if row["revision_id"] == revision
    ][0]
    assert stored["model_output"] == 250
    assert stored["model_output_seen"] == "YES"
    assert stored["implementation_revision"] == "model-synth-9"

    assisted = assess_use(store, ref, PURPOSE_TRAINING, policy(store.path))
    assert assisted["verdict"] == VERDICT_BLOCKED
    assert "model_assisted_label_without_policy" in assisted["reasons"]
    allowed = assess_use(
        store, ref, PURPOSE_TRAINING,
        policy(store.path, allow_model_assisted_labels=True),
    )
    assert allowed["verdict"] == VERDICT_ALLOWED_FOR_PURPOSE
