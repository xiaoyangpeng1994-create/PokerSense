"""Boundary regressions for the P0 selected #19 migration.

Everything here uses SYNTHETIC metadata and temporary files only: no real
recording is opened, no frame is decoded, no historical label file is imported,
no model is invoked and no training is executed. Every value below is an
invented fixture, never a real observation.

Sections 1-24 keep the original contract regressions. Sections 30+ are the
adversarial counter-examples for blockers B1-B6: they assert that a caller can
never be the authority for development role, reservations, consumption
permission, identity aliasing, parent lineage, history coverage, revision graph
integrity or consumer / revision binding.
"""

import hashlib
import json
from pathlib import Path

import pytest

import tools.aa_annotation_records as module
from tools.aa_annotation_records import (
    COVERAGE_UNKNOWN, EVIDENCE_ACTUALLY_CONSUMED, EVIDENCE_RELEASED,
    EVIDENCE_RESERVED, FIELD_POT_DISPLAY, FIELD_VISIBLE_ACTION_GLYPH,
    AnnotationStore, AnnotationError, ExposureError, FieldSemanticsMismatch,
    ForbiddenTruthFieldError, Label, LabelValidationError, ObservationContext,
    PURPOSE_TEMPLATE_SELECTION, PURPOSE_TRAINING, RevisionConflictError,
    SCHEMA_VERSION, STATE_ACTUALLY_CONSUMED, STATE_POSSIBLY_USED, STATE_UNKNOWN,
    STATUS_KNOWN, STATUS_NOT_APPLICABLE, STATUS_UNKNOWN,
    StoreIntegrityError, UnsupportedFieldError, VERDICT_BLOCKED,
    _canonical_source_ref, _group_key, append_annotation, assess_use,
    exposure_projection, get_annotations, record_exposure,
    validate_annotation_rows,
)

ACTOR = "synth-annotator"
POLICY_REF = "synth-policy-aa-annotations-p0"

# A development frame that is neither reserved nor a known exploration frame.
# Development intervals in the reviewed artifact are [0, 1800]; exploration
# frames are the multiples of 900.
DEV_FRAME = 1000


def digest(seed):
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def load_reservations():
    path = Path(__file__).resolve().parents[2] / (
        "configs/reproduction/aa_holdout_reservations_v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


TRUSTED_SOURCE_SHA256 = load_reservations()["source_sha256"]


def source_ref(
    *,
    audit="manifest-synth-A",
    media="media-synth-A",
    frame=DEV_FRAME,
    media_sha256=None,
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
        "media_sha256": media_sha256 or TRUSTED_SOURCE_SHA256,
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


def consumer(targets=None, cid="synth-consumer-1"):
    if targets is None:
        targets = [{"field_id": FIELD_POT_DISPLAY, "seat": 0}]
    return {
        "consumer_id": cid,
        "digest": digest("consumer|" + cid),
        "required_targets": targets,
    }


def policy(store_dir, **overrides):
    bundle = {
        "policy_ref": POLICY_REF,
        "role": "development",
        "reservations": load_reservations(),
        "development_evidence_ref": "synth:dev-frame-evidence-000001",
        "declared_store_paths": [str(store_dir)],
        "required_fields": [FIELD_POT_DISPLAY],
        "consumer": consumer(),
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
    return {
        "kind": "synthetic-run",
        "run_id": name,
        "digest": digest("run|" + name),
    }


def consume(
    store,
    ref,
    revisions,
    kind=EVIDENCE_ACTUALLY_CONSUMED,
    *,
    run="run-1",
    targets=None,
    role="development",
    purpose=PURPOSE_TRAINING,
    **kwargs,
):
    return record_exposure(
        store, ref, revisions, purpose, run_ref(run), kind,
        consumer=consumer(targets), policy_ref=POLICY_REF, role=role,
        actor=ACTOR, **kwargs,
    )


def raw_row(ref, field_id, seat, revision_id, supersedes=None, **overrides):
    """A syntactically valid annotation row, used to forge broken graphs."""
    identity = _canonical_source_ref(ref, allow_incomplete=True)
    row = {
        "schema_version": SCHEMA_VERSION,
        "event_id": revision_id,
        "event_type": "annotation",
        "revision_id": revision_id,
        "group_key": _group_key(identity["source_key"], field_id, seat),
        "recorded_at": "2026-01-01T00:00:00Z",
        "actor": ACTOR,
        "source_key": identity["source_key"],
        "source_ref": identity["normalised"],
        "field_id": field_id,
        "seat": seat,
        "status": STATUS_KNOWN,
        "value": 120,
        "unit": "chips",
        "reason": "synth",
        "supersedes": supersedes,
        "expected_revision": supersedes,
        "metadata": {"amount_role": None, "full_actions_declared": False},
    }
    row.update(overrides)
    return row


def write_rows(store, rows):
    for row in rows:
        store.append_annotation(row)


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
    second = source_ref(
        media="media-synth-B", frame=100,
        media_sha256=digest("media|media-synth-B"),
    )
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
    with pytest.raises(AnnotationError, match="seat_not_mappable"):
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
    assert "frame_reserved_or_invalid:child" in decision["reasons"]
    assert any(
        "frame_not_in_trusted_development_intervals" in reason
        for reason in decision["reasons"]
    )


# 12 -----------------------------------------------------------------------
def test_unknown_exposure_history_blocked(store):
    ref = source_ref()
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
    """A passing reservation check is not a development proof."""
    from tools.aa_data_separation import assert_training_frames

    ref = source_ref()
    annotate_pot(store, ref)
    reservations = load_reservations()
    assert_training_frames([ref["frame_index"]], reservations)

    bundle = policy(store.path)
    bundle.pop("declared_store_paths")
    bundle.pop("reservations")
    decision = assess_use(store, ref, PURPOSE_TRAINING, bundle)
    assert decision["verdict"] == VERDICT_BLOCKED
    assert "exposure_history_unknown" in decision["reasons"]
    assert "global_history_inventory_unavailable" in decision["reasons"]

    full = assess_use(store, ref, PURPOSE_TRAINING, policy(store.path))
    assert full["verdict"] == VERDICT_BLOCKED
    assert "trusted_consumer_registry_unavailable" in full["reasons"]
    assert full["usable_scope"] == "ANNOTATION_ONLY"


# 14 -----------------------------------------------------------------------
def test_exposure_survives_annotation_change(store):
    ref = source_ref()
    first = annotate_pot(store, ref, chips=120)
    consume(store, ref, [first], run="run-A",
            artifact_ref={"sha256": digest("artifact-A")})
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
    ref = source_ref()
    for key in ("path", "replay_name", "ui_index"):
        broken = dict(ref)
        broken[key] = "C:/synth/recording.mp4" if key == "path" else "synth"
        with pytest.raises(AnnotationError, match="not_an_identity"):
            get_annotations(store, broken)

    first = annotate_pot(store, ref, chips=120)
    consume(store, ref, [first], run="run-B")
    # Identity ignores mutable location names and model revisions; only the
    # bound source / media / frame triple counts.
    moved = source_ref(implementation="vision-pipeline-synth-2")
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
    consume(store, parent, [parent_rev], run="run-crop-1")
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
    consume(store, second_child, [child_rev], run="run-crop-2")
    annotate_pot(store, second_parent, chips=40)
    parent_decision = assess_use(store, second_parent, PURPOSE_TRAINING, bundle)
    assert parent_decision["exposure_state"] == STATE_ACTUALLY_CONSUMED


# 17 -----------------------------------------------------------------------
def test_fresh_store_never_claims_untouched_history(tmp_path):
    fresh = AnnotationStore(tmp_path / "fresh-store")
    ref = source_ref()
    annotate_pot(fresh, ref)
    undecided = assess_use(
        fresh, ref, PURPOSE_TRAINING,
        {"policy_ref": POLICY_REF, "role": "development",
         "reservations": load_reservations()},
    )
    assert undecided["exposure_state"] == STATE_UNKNOWN
    assert undecided["verdict"] == VERDICT_BLOCKED

    declared = assess_use(fresh, ref, PURPOSE_TRAINING, policy(fresh.path))
    assert declared["exposure_state"] == STATE_UNKNOWN
    assert declared["history_coverage"] == COVERAGE_UNKNOWN
    blob = json.dumps(declared, ensure_ascii=False).lower()
    assert "clean" not in blob and "never_trained" not in blob


# 18 -----------------------------------------------------------------------
def test_persistence_failure_never_produces_a_receipt(store, monkeypatch):
    ref = source_ref()
    annotate_pot(store, ref)

    def boom(self, row):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(AnnotationStore, "append_exposure", boom)
    decision = assess_use(
        store, ref, PURPOSE_TRAINING, policy(store.path), reserve=True,
        actor=ACTOR,
    )
    assert decision["verdict"] == VERDICT_BLOCKED
    assert decision["reservation_receipt"] is None

    monkeypatch.undo()
    blocked = assess_use(
        store, ref, PURPOSE_TRAINING, policy(store.path), reserve=True,
        actor=ACTOR,
    )
    assert blocked["verdict"] == VERDICT_BLOCKED
    assert blocked["reservation_receipt"] is None
    assert AnnotationStore(store.path).read_exposures() == []


# 19 -----------------------------------------------------------------------
def test_possibly_used_and_actually_consumed_are_distinct(store):
    ref = source_ref()
    revision = annotate_pot(store, ref)
    consume(store, ref, [revision], EVIDENCE_RESERVED, run="run-1")
    state_after_reservation = exposure_projection(
        store,
        {"sources": [ref], "declared_store_paths": [str(store.path)]},
    )["rows"][0]["exposure_state"]
    assert state_after_reservation == STATE_POSSIBLY_USED

    consume(store, ref, [revision], EVIDENCE_RELEASED, run="run-1")
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

    consume(store, ref, [revision], EVIDENCE_ACTUALLY_CONSUMED, run="run-2")
    consumed = exposure_projection(
        store,
        {"sources": [ref], "declared_store_paths": [str(store.path)]},
    )["rows"][0]["exposure_state"]
    assert consumed == STATE_ACTUALLY_CONSUMED
    assert consumed != STATE_POSSIBLY_USED


# 20 -----------------------------------------------------------------------
def test_policy_violation_is_kept_and_blocks(store):
    ref = source_ref()
    revision = annotate_pot(store, ref)
    event = consume(
        store, ref, [revision], run="run-violation", role="holdout",
        policy_violation=True,
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
    ref = source_ref()
    revision = annotate_pot(store, ref)
    undecided = exposure_projection(store, {"sources": [ref]})
    assert undecided["history_coverage"] == COVERAGE_UNKNOWN
    assert undecided["rows"]
    assert all(not row["freeze_bindable"] for row in undecided["rows"])
    assert undecided["unresolved"]

    consume(
        store, ref, [revision], run="run-3", purpose=PURPOSE_TEMPLATE_SELECTION,
    )
    missing_artifact = exposure_projection(
        store, {"sources": [ref], "declared_store_paths": [str(store.path)]},
    )
    assert missing_artifact["history_coverage"] == COVERAGE_UNKNOWN
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
    ref = source_ref()
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
    ref = source_ref()
    with pytest.raises(UnsupportedFieldError, match="field_not_supported_in_p0"):
        append_annotation(
            store, ref, field_id, 0,
            Label(status=STATUS_KNOWN, value=1, reason="synth"),
            None, ACTOR,
        )


def test_current_view_has_no_truth_shortcuts(store):
    ref = source_ref()
    annotate_pot(store, ref)
    rows = current_rows(get_annotations(store, ref))
    forbidden = {"acceptance", "strategy_eligible", "canonical_verified"}
    assert rows and not (set(rows[0]) & forbidden)


# 23 -----------------------------------------------------------------------
def test_model_output_does_not_become_the_current_value(store):
    ref = source_ref()
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
    assert allowed["verdict"] == VERDICT_BLOCKED


# 24 -----------------------------------------------------------------------
def test_positive_consumption_permission_is_closed(store):
    """No trusted consumer registry exists, so no ALLOWED can be derived."""
    ref = source_ref()
    annotate_pot(store, ref)
    assert module.TRUSTED_CONSUMER_REGISTRY_AVAILABLE is False
    assert module.TRUSTED_HISTORY_INVENTORY_AVAILABLE is False
    decision = assess_use(store, ref, PURPOSE_TRAINING, policy(store.path))
    assert decision["verdict"] == VERDICT_BLOCKED
    assert "trusted_consumer_registry_unavailable" in decision["reasons"]
    assert "global_history_inventory_unavailable" in decision["reasons"]
    assert decision["usable_scope"] == "ANNOTATION_ONLY"


# 30 B1 -- client cannot self-certify development use -----------------------
def test_unregistered_frame_is_blocked(store):
    ref = source_ref(frame=4000)
    annotate_pot(store, ref)
    decision = assess_use(store, ref, PURPOSE_TRAINING, policy(store.path))
    assert decision["verdict"] == VERDICT_BLOCKED
    assert any(
        "frame_not_in_trusted_development_intervals" in reason
        for reason in decision["reasons"]
    )


def test_fake_development_evidence_grants_nothing(store):
    ref = source_ref(frame=2700)
    annotate_pot(store, ref)
    for evidence in (None, "anything", "synth:totally-made-up", ""):
        bundle = policy(store.path, development_evidence_ref=evidence)
        decision = assess_use(store, ref, PURPOSE_TRAINING, bundle)
        assert decision["verdict"] == VERDICT_BLOCKED
        assert any(
            "exploration_frame_is_not_development_authority" in reason
            for reason in decision["reasons"]
        )


def test_caller_cannot_overwrite_reservations(store):
    ref = source_ref(frame=3000)
    annotate_pot(store, ref)
    mutated = load_reservations()
    mutated["reserved_inclusive_intervals"] = []
    decision = assess_use(
        store, ref, PURPOSE_TRAINING, policy(store.path, reservations=mutated),
    )
    assert decision["verdict"] == VERDICT_BLOCKED
    assert "caller_reservations_do_not_match_canonical_artifact" in (
        decision["reasons"]
    )
    assert any("frame_reserved_or_invalid" in reason
               for reason in decision["reasons"])


def test_missing_annotations_fields_and_consumer_all_block(store):
    ref = source_ref()
    decision = assess_use(store, ref, PURPOSE_TRAINING, policy(store.path))
    assert decision["verdict"] == VERDICT_BLOCKED
    assert "required_annotation_missing:" + FIELD_POT_DISPLAY in decision["reasons"]

    annotate_pot(store, ref)
    no_consumer = policy(store.path)
    no_consumer.pop("consumer")
    decision = assess_use(store, ref, PURPOSE_TRAINING, no_consumer)
    assert decision["verdict"] == VERDICT_BLOCKED
    assert "consumer_identity_missing" in decision["reasons"]

    no_role = policy(store.path, role="exploration")
    decision = assess_use(store, ref, PURPOSE_TRAINING, no_role)
    assert decision["verdict"] == VERDICT_BLOCKED
    assert "role_unknown:exploration" in decision["reasons"]


def test_holdout_training_history_cannot_be_redeveloped(store):
    ref = source_ref()
    revision = annotate_pot(store, ref)
    consume(store, ref, [revision], run="run-holdout", role="holdout")
    bundle = policy(store.path)
    bundle["allow_reuse_of_exposed_development"] = True
    decision = assess_use(store, ref, PURPOSE_TRAINING, bundle)
    assert decision["verdict"] == VERDICT_BLOCKED
    assert "holdout_training_history_cannot_be_redeveloped" in decision["reasons"]
    # The historical fact is preserved, not deleted.
    assert AnnotationStore(store.path).read_exposures()


def test_calibration_role_has_no_trusted_intervals(store):
    ref = source_ref()
    annotate_pot(store, ref)
    decision = assess_use(
        store, ref, PURPOSE_TRAINING, policy(store.path, role="calibration"),
    )
    assert decision["verdict"] == VERDICT_BLOCKED
    assert "calibration_role_has_no_trusted_interval_authority" in (
        decision["reasons"]
    )


def test_reservation_artifact_sha_mismatch_is_reported(store):
    ref = source_ref()
    annotate_pot(store, ref)
    decision = assess_use(
        store, ref, PURPOSE_TRAINING,
        policy(store.path, reservation_artifact_sha256=digest("wrong")),
    )
    assert decision["verdict"] == VERDICT_BLOCKED
    assert "reservation_artifact_sha_mismatch" in decision["reasons"]


# 31 B2 -- canonical identity and long-term exposure root -------------------
def test_sha_case_alias_does_not_split_identity(store):
    ref = source_ref()
    revision = annotate_pot(store, ref)
    consume(store, ref, [revision], run="run-case")

    alias = source_ref()
    alias["media_sha256"] = alias["media_sha256"].upper()
    alias["frame_sha256"] = alias["frame_sha256"].upper()
    alias["source_audit_sha256"] = alias["source_audit_sha256"].upper()

    first = _canonical_source_ref(ref, allow_incomplete=True)
    second = _canonical_source_ref(alias, allow_incomplete=True)
    assert first["source_key"] == second["source_key"]
    assert first["exposure_root_key"] == second["exposure_root_key"]

    decision = assess_use(store, alias, PURPOSE_TRAINING, policy(store.path))
    assert decision["exposure_state"] == STATE_ACTUALLY_CONSUMED


def test_reaudit_and_media_alias_keep_consumed_history(store):
    ref = source_ref(audit="manifest-synth-A", media="media-synth-A")
    revision = annotate_pot(store, ref)
    consume(store, ref, [revision], run="run-reaudit")

    # Same bytes, same frame digest: only the audit manifest and the media id
    # changed, which must not reset long-term exposure.
    alias = source_ref(
        audit="manifest-synth-B-reaudit", media="media-alias-A2",
        digest_seed="media-synth-A|" + str(DEV_FRAME) + "|v1",
    )
    first = _canonical_source_ref(ref, allow_incomplete=True)
    second = _canonical_source_ref(alias, allow_incomplete=True)
    assert first["source_key"] != second["source_key"]
    assert first["exposure_root_key"] == second["exposure_root_key"]

    decision = assess_use(store, alias, PURPOSE_TRAINING, policy(store.path))
    assert decision["exposure_state"] == STATE_ACTUALLY_CONSUMED
    projection = exposure_projection(
        store,
        {"sources": [alias], "declared_store_paths": [str(store.path)]},
    )
    assert projection["rows"][0]["exposure_state"] == STATE_ACTUALLY_CONSUMED


def test_different_real_media_never_shares_exposure(store):
    ref = source_ref(media="media-synth-A")
    other = source_ref(
        media="media-synth-OTHER", media_sha256=digest("media|other"),
        digest_seed="other-content",
    )
    first = _canonical_source_ref(ref, allow_incomplete=True)
    second = _canonical_source_ref(other, allow_incomplete=True)
    assert first["exposure_root_key"] != second["exposure_root_key"]

    revision = annotate_pot(store, ref)
    consume(store, ref, [revision], run="run-isolation")
    decision = assess_use(store, other, PURPOSE_TRAINING, policy(store.path))
    assert decision["exposure_state"] == STATE_UNKNOWN


def test_incomplete_identity_is_not_bindable(store):
    ref = source_ref()
    del ref["layout_id"]
    identity = _canonical_source_ref(ref, allow_incomplete=True)
    assert identity["identity_class"] != "COMPLETE"
    decision = assess_use(store, ref, PURPOSE_TRAINING, policy(store.path))
    assert decision["verdict"] == VERDICT_BLOCKED
    assert any("source_identity_incomplete" in reason
               for reason in decision["reasons"])
    projection = exposure_projection(store, {"sources": [ref]})
    assert all(not row["freeze_bindable"] for row in projection["rows"])


# 32 B3 -- parent lineage --------------------------------------------------
def test_empty_parent_is_blocked(store):
    child = source_ref(frame=200, media="media-child", parent={})
    annotate_pot(store, child)
    decision = assess_use(store, child, PURPOSE_TRAINING, policy(store.path))
    assert decision["verdict"] == VERDICT_BLOCKED
    assert any("parent_frame_ref_empty_or_not_mapping" in reason
               for reason in decision["reasons"])
    identity = _canonical_source_ref(child, allow_incomplete=True)
    assert identity["identity_class"] != "COMPLETE"
    assert identity["parent_source_key"] is None


def test_bad_parent_sha_is_blocked(store):
    parent = source_ref(frame=200, media="media-parent")
    parent["media_sha256"] = "not-a-hash"
    child = source_ref(frame=200, media="media-child", parent=parent)
    decision = assess_use(store, child, PURPOSE_TRAINING, policy(store.path))
    assert decision["verdict"] == VERDICT_BLOCKED
    assert any("parent_bad_sha256:media_sha256" in reason
               for reason in decision["reasons"])
    assert _canonical_source_ref(child, allow_incomplete=True)[
        "identity_class"
    ] != "COMPLETE"


def test_reserved_parent_blocks_the_child(store):
    parent = source_ref(frame=3000, media="media-reserved-parent")
    child = source_ref(
        frame=200, media="media-child", digest_seed="child-of-reserved",
        parent=parent,
    )
    annotate_pot(store, child)
    decision = assess_use(store, child, PURPOSE_TRAINING, policy(store.path))
    assert decision["verdict"] == VERDICT_BLOCKED
    assert "frame_reserved_or_invalid:parent" in decision["reasons"]


def test_nested_parent_is_refused(store):
    grandparent = source_ref(frame=200, media="media-grand")
    parent = source_ref(
        frame=200, media="media-parent", digest_seed="parent-seed",
        parent=grandparent,
    )
    child = source_ref(
        frame=200, media="media-child", digest_seed="child-seed", parent=parent,
    )
    identity = _canonical_source_ref(child, allow_incomplete=True)
    assert "nested_parent_not_supported" in identity["identity_gaps"]
    assert identity["identity_class"] != "COMPLETE"
    decision = assess_use(store, child, PURPOSE_TRAINING, policy(store.path))
    assert decision["verdict"] == VERDICT_BLOCKED
    assert any("nested_parent_not_supported" in reason
               for reason in decision["reasons"])


def test_broken_parent_reference_is_blocked(store):
    child = source_ref(frame=200, media="media-child", parent={"frame_index": 1})
    decision = assess_use(store, child, PURPOSE_TRAINING, policy(store.path))
    assert decision["verdict"] == VERDICT_BLOCKED
    assert any("parent_missing:" in reason for reason in decision["reasons"])


def test_one_level_crop_keeps_highest_contamination(store):
    parent = source_ref(frame=200, media="media-parent-one")
    child = source_ref(
        frame=200, media="media-child-one", digest_seed="crop-one",
        parent=parent,
    )
    parent_revision = annotate_pot(store, parent, chips=80)
    consume(store, parent, [parent_revision], run="run-one")
    child_revision = annotate_pot(store, child, chips=80)
    consume(store, child, [child_revision], EVIDENCE_RESERVED, run="run-one-2")

    for target in (parent, child):
        decision = assess_use(store, target, PURPOSE_TRAINING, policy(store.path))
        assert decision["exposure_state"] == STATE_ACTUALLY_CONSUMED


# 33 B4 -- multi-store history ----------------------------------------------
def test_consumed_in_a_is_visible_from_b(tmp_path):
    store_a = AnnotationStore(tmp_path / "storeA")
    store_b = AnnotationStore(tmp_path / "storeB")
    ref = source_ref()
    revision = annotate_pot(store_a, ref)
    consume(store_a, ref, [revision], run="run-A")
    declared = [str(store_a.path), str(store_b.path)]

    from_b = assess_use(
        store_b, ref, PURPOSE_TRAINING,
        policy(store_b.path, declared_store_paths=declared),
    )
    assert from_b["exposure_state"] == STATE_ACTUALLY_CONSUMED

    projection = exposure_projection(
        store_b, {"sources": [ref], "declared_store_paths": declared},
    )
    assert projection["rows"]
    assert all(
        row["exposure_state"] == STATE_ACTUALLY_CONSUMED
        for row in projection["rows"]
    )


def test_released_in_b_never_lowers_consumed_in_a(tmp_path):
    store_a = AnnotationStore(tmp_path / "storeA")
    store_b = AnnotationStore(tmp_path / "storeB")
    ref = source_ref()
    revision_a = annotate_pot(store_a, ref)
    consume(store_a, ref, [revision_a], run="run-A")
    revision_b = annotate_pot(store_b, ref)
    consume(store_b, ref, [revision_b], EVIDENCE_RELEASED, run="run-B")
    declared = [str(store_a.path), str(store_b.path)]

    decision = assess_use(
        store_b, ref, PURPOSE_TRAINING,
        policy(store_b.path, declared_store_paths=declared),
    )
    assert decision["exposure_state"] == STATE_ACTUALLY_CONSUMED
    projection = exposure_projection(
        store_b, {"sources": [ref], "declared_store_paths": declared},
    )
    assert all(
        row["exposure_state"] == STATE_ACTUALLY_CONSUMED
        for row in projection["rows"]
    )
    assert STATE_POSSIBLY_USED not in {
        row["exposure_state"] for row in projection["rows"]
    }


def test_new_empty_store_never_claims_no_exposure(tmp_path):
    fresh = AnnotationStore(tmp_path / "brand-new")
    ref = source_ref()
    annotate_pot(fresh, ref)
    decision = assess_use(fresh, ref, PURPOSE_TRAINING, policy(fresh.path))
    assert decision["verdict"] == VERDICT_BLOCKED
    assert decision["exposure_state"] == STATE_UNKNOWN
    assert decision["history_coverage"] == COVERAGE_UNKNOWN
    assert "no_exposure_recorded" not in decision["exposure_state"].lower()


def test_declared_store_without_history_file_is_not_coverage(tmp_path):
    old = tmp_path / "old-store"
    old.mkdir()
    store = AnnotationStore(tmp_path / "current")
    ref = source_ref()
    annotate_pot(store, ref)
    decision = assess_use(
        store, ref, PURPOSE_TRAINING,
        policy(store.path, declared_store_paths=[str(old), str(store.path)]),
    )
    assert decision["verdict"] == VERDICT_BLOCKED
    assert decision["history_coverage"] == COVERAGE_UNKNOWN
    assert decision["exposure_state"] == STATE_UNKNOWN


def test_missing_old_store_blocks(tmp_path):
    store = AnnotationStore(tmp_path / "current")
    ref = source_ref()
    annotate_pot(store, ref)
    decision = assess_use(
        store, ref, PURPOSE_TRAINING,
        policy(
            store.path,
            declared_store_paths=[
                str(tmp_path / "old-store-that-vanished"), str(store.path),
            ],
        ),
    )
    assert decision["verdict"] == VERDICT_BLOCKED
    assert any("declared_store_missing" in reason
               for reason in decision["reasons"])


def test_corrupt_declared_store_blocks(tmp_path):
    corrupt = AnnotationStore(tmp_path / "corrupt")
    (corrupt.path / "exposures.jsonl").write_text("{not json", encoding="utf-8")
    store = AnnotationStore(tmp_path / "current")
    ref = source_ref()
    annotate_pot(store, ref)
    decision = assess_use(
        store, ref, PURPOSE_TRAINING,
        policy(
            store.path,
            declared_store_paths=[str(corrupt.path), str(store.path)],
        ),
    )
    assert decision["verdict"] == VERDICT_BLOCKED
    assert any("declared_store_unreadable" in reason
               for reason in decision["reasons"])
    assert decision["exposure_state"] == STATE_UNKNOWN


def test_cross_store_duplicate_event_id_is_deduplicated(tmp_path):
    store_a = AnnotationStore(tmp_path / "storeA")
    store_b = AnnotationStore(tmp_path / "storeB")
    ref = source_ref()
    revision = annotate_pot(store_a, ref)
    event = consume(store_a, ref, [revision], run="run-dup")
    store_b.append_exposure(dict(event))
    declared = [str(store_a.path), str(store_b.path)]

    decision = assess_use(
        store_b, ref, PURPOSE_TRAINING,
        policy(store_b.path, declared_store_paths=declared),
    )
    assert not any("cross_store_event_id_conflict" in reason
                   for reason in decision["reasons"])
    assert decision["exposure_state"] == STATE_ACTUALLY_CONSUMED


def test_cross_store_conflicting_event_id_blocks(tmp_path):
    store_a = AnnotationStore(tmp_path / "storeA")
    store_b = AnnotationStore(tmp_path / "storeB")
    ref = source_ref()
    revision = annotate_pot(store_a, ref)
    event = consume(store_a, ref, [revision], run="run-conflict")
    forged = dict(event)
    forged["evidence_kind"] = EVIDENCE_RELEASED
    store_b.append_exposure(forged)
    declared = [str(store_a.path), str(store_b.path)]

    decision = assess_use(
        store_b, ref, PURPOSE_TRAINING,
        policy(store_b.path, declared_store_paths=declared),
    )
    assert decision["verdict"] == VERDICT_BLOCKED
    assert any("cross_store_event_id_conflict" in reason
               for reason in decision["reasons"])


def test_unrelated_source_does_not_contaminate(store):
    ref = source_ref(media="media-target")
    other = source_ref(
        media="media-unrelated", media_sha256=digest("media|unrelated"),
        digest_seed="unrelated-content",
    )
    other_revision = annotate_pot(store, other, chips=5)
    consume(store, other, [other_revision], run="run-unrelated")
    annotate_pot(store, ref, chips=120)

    decision = assess_use(store, ref, PURPOSE_TRAINING, policy(store.path))
    assert decision["exposure_state"] == STATE_UNKNOWN
    assert decision["verdict"] == VERDICT_BLOCKED


# 34 B5 -- revision graph validation on recovery ----------------------------
def test_orphan_supersedes_is_refused(store):
    ref = source_ref()
    write_rows(store, [
        raw_row(ref, FIELD_POT_DISPLAY, 0, "ann-1", supersedes="ann-missing"),
    ])
    with pytest.raises(StoreIntegrityError, match="orphan_supersedes"):
        store.read_annotations()
    decision = assess_use(store, ref, PURPOSE_TRAINING, policy(store.path))
    assert decision["verdict"] == VERDICT_BLOCKED
    assert any("annotation_store_integrity_error" in reason
               for reason in decision["reasons"])


def test_cross_source_supersedes_is_refused(store):
    ref = source_ref(media="media-a")
    other = source_ref(media="media-b", digest_seed="b-content")
    first = raw_row(ref, FIELD_POT_DISPLAY, 0, "ann-1")
    second = raw_row(other, FIELD_POT_DISPLAY, 0, "ann-2", supersedes="ann-1")
    write_rows(store, [first, second])
    with pytest.raises(StoreIntegrityError, match="cross_group_supersedes"):
        store.read_annotations()


def test_cross_field_supersedes_is_refused(store):
    ref = source_ref()
    first = raw_row(ref, FIELD_POT_DISPLAY, 0, "ann-1")
    second = raw_row(ref, FIELD_VISIBLE_ACTION_GLYPH, 0, "ann-2",
                     supersedes="ann-1", value="call", unit=None)
    write_rows(store, [first, second])
    with pytest.raises(StoreIntegrityError, match="cross_group_supersedes"):
        store.read_annotations()


def test_cross_seat_supersedes_is_refused(store):
    ref = source_ref()
    first = raw_row(ref, FIELD_POT_DISPLAY, 0, "ann-1")
    second = raw_row(ref, FIELD_POT_DISPLAY, 1, "ann-2", supersedes="ann-1")
    write_rows(store, [first, second])
    with pytest.raises(StoreIntegrityError, match="cross_group_supersedes"):
        store.read_annotations()


def test_duplicate_revision_id_is_refused(store):
    ref = source_ref()
    write_rows(store, [
        raw_row(ref, FIELD_POT_DISPLAY, 0, "ann-1", event_id="evt-1"),
        raw_row(ref, FIELD_POT_DISPLAY, 1, "ann-1", event_id="evt-2"),
    ])
    with pytest.raises(StoreIntegrityError, match="duplicate_revision_id"):
        store.read_annotations()


def test_revision_cycle_is_refused_and_cannot_be_resurrected(store):
    ref = source_ref()
    write_rows(store, [
        raw_row(ref, FIELD_POT_DISPLAY, 0, "ann-1", supersedes="ann-2"),
        raw_row(ref, FIELD_POT_DISPLAY, 0, "ann-2", supersedes="ann-1"),
    ])
    with pytest.raises(StoreIntegrityError, match="revision_cycle"):
        store.read_annotations()

    with pytest.raises(AnnotationError):
        annotate_pot(store, ref, chips=999)
    with pytest.raises(StoreIntegrityError, match="revision_cycle"):
        store.read_annotations()
    decision = assess_use(store, ref, PURPOSE_TRAINING, policy(store.path))
    assert decision["verdict"] == VERDICT_BLOCKED


def test_known_label_with_null_value_is_refused(store):
    ref = source_ref()
    write_rows(store, [
        raw_row(ref, FIELD_POT_DISPLAY, 0, "ann-1", value=None),
    ])
    with pytest.raises(StoreIntegrityError, match="known_label_requires_value"):
        store.read_annotations()
    with pytest.raises(StoreIntegrityError, match="label_violation"):
        get_annotations(store, ref)


def test_forged_group_key_is_refused(store):
    ref = source_ref()
    row = raw_row(ref, FIELD_POT_DISPLAY, 0, "ann-1")
    row["group_key"] = "forged|" + FIELD_POT_DISPLAY + "|0"
    write_rows(store, [row])
    with pytest.raises(StoreIntegrityError, match="forged_group_key"):
        store.read_annotations()


def test_forged_source_key_is_refused(store):
    ref = source_ref()
    row = raw_row(ref, FIELD_POT_DISPLAY, 0, "ann-1")
    row["source_key"] = digest("attacker-chosen-identity")
    write_rows(store, [row])
    with pytest.raises(StoreIntegrityError, match="source_key_mismatch"):
        store.read_annotations()


def test_healthy_graph_survives_validation(store):
    ref = source_ref()
    first = annotate_pot(store, ref, chips=120)
    annotate_pot(store, ref, chips=150, expected=first)
    rows = store.read_annotations()
    validate_annotation_rows(rows)
    assert len(rows) == 2


# 35 B6 -- consumer / revision binding --------------------------------------
def test_wrong_source_revision_cannot_be_consumed(store):
    ref = source_ref(media="media-target")
    other = source_ref(
        media="media-other", media_sha256=digest("media|other-b"),
        digest_seed="other-b-content",
    )
    annotate_pot(store, ref)
    foreign = annotate_pot(store, other, chips=5)
    with pytest.raises(ExposureError, match="consumed_revision_source_mismatch"):
        consume(store, ref, [foreign], run="run-wrong-source")


def test_wrong_field_revision_cannot_be_consumed(store):
    ref = source_ref()
    annotate_pot(store, ref)
    glyph = append_annotation(
        store, ref, FIELD_VISIBLE_ACTION_GLYPH, 0,
        Label(status=STATUS_KNOWN, value="call", reason="synth"), None, ACTOR,
    )
    with pytest.raises(ExposureError, match="not_a_consumer_target"):
        consume(store, ref, [glyph], run="run-wrong-field")


def test_wrong_seat_revision_cannot_be_consumed(store):
    ref = source_ref()
    annotate_pot(store, ref, seat=0)
    other_seat = annotate_pot(store, ref, seat=1, chips=7)
    with pytest.raises(ExposureError, match="not_a_consumer_target"):
        consume(store, ref, [other_seat], run="run-wrong-seat")


def test_unknown_and_missing_revisions_cannot_be_consumed(store):
    ref = source_ref()
    annotate_pot(store, ref)
    with pytest.raises(ExposureError, match="consumed_revision_not_found"):
        consume(store, ref, ["ann-does-not-exist"], run="run-missing")
    with pytest.raises(ExposureError, match="bad_annotation_revision_id"):
        consume(store, ref, [""], run="run-bad-id")


def test_unknown_label_cannot_be_consumed(store):
    ref = source_ref()
    unknown = append_annotation(
        store, ref, FIELD_POT_DISPLAY, 0,
        Label(status=STATUS_UNKNOWN, reason="synth unreadable"), None, ACTOR,
    )
    with pytest.raises(ExposureError, match="unknown_revision_cannot_be_consumed"):
        consume(store, ref, [unknown], run="run-unknown-label")


def test_empty_or_wrong_consumer_and_manifest_are_refused(store):
    ref = source_ref()
    revision = annotate_pot(store, ref)
    with pytest.raises(ExposureError, match="consumer_identity_required"):
        record_exposure(
            store, ref, [revision], PURPOSE_TRAINING, run_ref("run-x"),
            EVIDENCE_ACTUALLY_CONSUMED, policy_ref=POLICY_REF, actor=ACTOR,
        )
    with pytest.raises(ExposureError, match="consumer_id_required"):
        record_exposure(
            store, ref, [revision], PURPOSE_TRAINING, run_ref("run-x"),
            EVIDENCE_ACTUALLY_CONSUMED, consumer={}, policy_ref=POLICY_REF,
            actor=ACTOR,
        )
    with pytest.raises(ExposureError, match="consumer_required_targets_required"):
        record_exposure(
            store, ref, [revision], PURPOSE_TRAINING, run_ref("run-x"),
            EVIDENCE_ACTUALLY_CONSUMED,
            consumer={"consumer_id": "c", "digest": digest("c")},
            policy_ref=POLICY_REF, actor=ACTOR,
        )
    with pytest.raises(ExposureError, match="consumer_digest_required"):
        record_exposure(
            store, ref, [revision], PURPOSE_TRAINING, run_ref("run-x"),
            EVIDENCE_ACTUALLY_CONSUMED,
            consumer={"consumer_id": "c", "required_targets": [
                {"field_id": FIELD_POT_DISPLAY, "seat": 0}]},
            policy_ref=POLICY_REF, actor=ACTOR,
        )
    with pytest.raises(ExposureError, match="run_or_manifest_ref_must_not_be_empty"):
        record_exposure(
            store, ref, [revision], PURPOSE_TRAINING, {},
            EVIDENCE_ACTUALLY_CONSUMED, consumer=consumer(),
            policy_ref=POLICY_REF, actor=ACTOR,
        )
    with pytest.raises(ExposureError, match="run_or_manifest_ref_digest_required"):
        record_exposure(
            store, ref, [revision], PURPOSE_TRAINING, {"kind": "synthetic-run"},
            EVIDENCE_ACTUALLY_CONSUMED, consumer=consumer(),
            policy_ref=POLICY_REF, actor=ACTOR,
        )


def test_consumption_receipt_covers_exactly_the_declared_targets(store):
    ref = source_ref()
    pot = annotate_pot(store, ref, seat=0, chips=120)
    append_annotation(
        store, ref, FIELD_VISIBLE_ACTION_GLYPH, 1,
        Label(status=STATUS_UNKNOWN, reason="synth glyph cropped"), None, ACTOR,
    )
    event = consume(store, ref, [pot], run="run-exact")
    assert event["binding_status"] == "BOUND"
    assert event["annotation_revision_refs"] == [{
        "revision_id": pot,
        "field_id": FIELD_POT_DISPLAY,
        "seat": 0,
        "status": STATUS_KNOWN,
    }]
    # Another seat's UNKNOWN revision is never swept into the receipt.
    stored = [
        row for row in AnnotationStore(store.path).read_exposures()
        if row["event_id"] == event["event_id"]
    ][0]
    assert len(stored["annotation_revision_ids"]) == 1

    two_targets = [
        {"field_id": FIELD_POT_DISPLAY, "seat": 0},
        {"field_id": FIELD_POT_DISPLAY, "seat": 1},
    ]
    with pytest.raises(ExposureError, match="consumer_target_not_consumed"):
        consume(store, ref, [pot], run="run-partial", targets=two_targets)


def test_stored_cross_source_event_stays_untrusted(tmp_path):
    """A historical mis-bound event is kept but never becomes a trusted fact."""
    store = AnnotationStore(tmp_path / "store")
    ref = source_ref(media="media-target")
    other = source_ref(
        media="media-other", media_sha256=digest("media|other-c"),
        digest_seed="other-c-content",
    )
    foreign = annotate_pot(store, other, chips=5)
    event = consume(store, other, [foreign], run="run-forged")
    identity = _canonical_source_ref(ref, allow_incomplete=True)
    forged = dict(event)
    forged["event_id"] = "exp-synth-misbound"
    forged["source_key"] = identity["source_key"]
    forged["exposure_root_key"] = identity["exposure_root_key"]
    forged["source_ref"] = identity["normalised"]
    forged["binding_status"] = "UNRESOLVED"
    store.append_exposure(forged)

    decision = assess_use(store, ref, PURPOSE_TRAINING, policy(store.path))
    assert decision["verdict"] == VERDICT_BLOCKED
    assert any("exposure_event_not_bound" in reason
               for reason in decision["reasons"])
    projection = exposure_projection(
        store, {"sources": [ref], "declared_store_paths": [str(store.path)]},
    )
    assert any("exposure_event_not_bound" in row["unresolved"]
               for row in projection["rows"])
    # The fact itself is preserved, never deleted.
    assert AnnotationStore(store.path).read_exposures()


# 36 freeze projection wording ----------------------------------------------
def test_projection_does_not_claim_a_live_freeze_adapter(store):
    ref = source_ref()
    revision = annotate_pot(store, ref)
    consume(
        store, ref, [revision], run="run-freeze",
        artifact_ref={"sha256": digest("artifact-freeze")},
    )
    projection = exposure_projection(
        store, {"sources": [ref], "declared_store_paths": [str(store.path)]},
    )
    assert projection["freeze_adapter_attached"] is False
    assert all(row["freeze_adapter_attached"] is False
               for row in projection["rows"])
    assert all(not row["freeze_bindable"] for row in projection["rows"])
    assert any("coverage_unproven" in item for item in projection["unresolved"])

    doc = (exposure_projection.__doc__ or "").lower()
    module_doc = (module.__doc__ or "").lower()
    assert "not attached" in doc
    assert "not attached" in module_doc
    assert "validate_freeze" in doc
