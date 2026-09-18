"""Source-bound development field annotations + irreversible exposure history.

P0 of the selected #19 migration. Reference: Issue #27 comment
``PR19_FINAL_ARCHITECTURE_DECISION_READY``.

This module is a **data governance record store**, nothing else:

* Development annotations are recorded per (stable source identity, field, seat)
  as append-only revisions. Old revisions are never overwritten.
* ``UNKNOWN`` is a first-class label state; it never collapses to 0 / false /
  none, and ``NOT_APPLICABLE`` always carries context.
* Training / tuning / template-selection exposure is recorded at source level
  and only ever grows. No rename, replay move, crop rebuild, revision,
  retraction or store rebuild can turn exposed development data back into an
  untouched holdout.
* ``assess_use`` derives eligibility read-only and fails closed. Clients cannot
  submit ``training_eligible=true``.

Explicit non-goals of P0 (see ``docs/AA-FIELD-ANNOTATIONS-V1.zh-CN.md``):

* No per-event hash chain (``previous_sha256`` / ``entry_sha256`` linkage) was
  migrated. Source / frame / artifact / snapshot hashes are retained as
  identity anchors, never as a chain.
* No UI, no HTTP endpoint, no export pipeline, no training execution, no crop
  generation, no model invocation.
* No promotion bridge into gold, confirmed poker facts, PHH actions, acceptance
  or strategy eligibility. This store is not a second truth source.
"""

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tools.aa_data_separation import assert_training_frames

SCHEMA_VERSION = "aa-field-annotations-v1"

SUPPORTED_SLOT_COUNT = 8

FIELD_POT_DISPLAY = "pot_display"
FIELD_VISIBLE_ACTION_GLYPH = "visible_action_glyph"

STATUS_KNOWN = "KNOWN"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"
STATUSES = frozenset({STATUS_KNOWN, STATUS_UNKNOWN, STATUS_NOT_APPLICABLE})

MODEL_OUTPUT_SEEN_YES = "YES"
MODEL_OUTPUT_SEEN_NO = "NO"
MODEL_OUTPUT_SEEN_UNKNOWN = "UNKNOWN"
MODEL_OUTPUT_SEEN = frozenset({
    MODEL_OUTPUT_SEEN_YES,
    MODEL_OUTPUT_SEEN_NO,
    MODEL_OUTPUT_SEEN_UNKNOWN,
})

PURPOSE_TRAINING = "training"
PURPOSE_TUNING = "tuning"
PURPOSE_TEMPLATE_SELECTION = "template_selection"
PURPOSES = frozenset({PURPOSE_TRAINING, PURPOSE_TUNING, PURPOSE_TEMPLATE_SELECTION})

EVIDENCE_RESERVED = "RESERVED"
EVIDENCE_RELEASED = "RELEASED"
EVIDENCE_POSSIBLY_USED = "POSSIBLY_USED"
EVIDENCE_ACTUALLY_CONSUMED = "ACTUALLY_CONSUMED"
EVIDENCE_KINDS = frozenset({
    EVIDENCE_RESERVED,
    EVIDENCE_RELEASED,
    EVIDENCE_POSSIBLY_USED,
    EVIDENCE_ACTUALLY_CONSUMED,
})

VERDICT_BLOCKED = "BLOCKED"
VERDICT_ALLOWED_FOR_PURPOSE = "ALLOWED_FOR_PURPOSE"

SCOPE_ANNOTATION_ONLY = "ANNOTATION_ONLY"
SCOPE_PURPOSE_BOUND = "PURPOSE_BOUND"

COVERAGE_COMPLETE = "COMPLETE"
COVERAGE_INCOMPLETE = "INCOMPLETE"
COVERAGE_UNKNOWN = "UNKNOWN"

STATE_ACTUALLY_CONSUMED = "EXPOSED_ACTUALLY_CONSUMED"
STATE_POSSIBLY_USED = "EXPOSED_POSSIBLY_USED"
STATE_UNKNOWN = "EXPOSURE_UNKNOWN"
STATE_NONE_UNDER_COVERAGE = "NO_EXPOSURE_RECORDED_UNDER_DECLARED_COVERAGE"

IDENTITY_COMPLETE = "COMPLETE"
IDENTITY_INCOMPLETE = "INCOMPLETE"
IDENTITY_UNBOUND = "UNBOUND"

AMOUNT_ROLE_POT_DISPLAY_TOTAL = "pot_display_total"
WAGER_AMOUNT_ROLES = frozenset({
    "current_street_wager",
    "street_wager",
    "call_price",
    "cumulative_commitment",
    "total_commitment",
})

GLYPH_VALUES = frozenset({
    "fold", "check", "call", "bet", "raise", "all_in", "muck", "none",
})

FIELD_CONTRACTS = {
    FIELD_POT_DISPLAY: {
        "value_types": (int,),
        "unit_allowed": frozenset({"chips"}),
        "unit_required": True,
        "allowed_values": None,
        "forbidden_amount_roles": WAGER_AMOUNT_ROLES,
        "sequence_bound": False,
    },
    FIELD_VISIBLE_ACTION_GLYPH: {
        "value_types": (str,),
        "unit_allowed": frozenset(),
        "unit_required": False,
        "allowed_values": GLYPH_VALUES,
        "forbidden_amount_roles": frozenset(),
        "sequence_bound": True,
    },
}

# Fields whose meaning belongs to another truth source. Even after additional
# field vocabularies are added (P1) these stay rejected: annotations can never
# be the authority for acceptance, strategy eligibility or verified facts.
FORBIDDEN_TRUTH_FIELDS = frozenset({
    "full_actions",
    "acceptance",
    "full_visual_acceptance",
    "strategy_eligible",
    "strategy_ready",
    "canonical_verified",
    "confirmed_facts",
    "confirmed_poker_facts",
    "holdout_gold",
    "phh_confirmed_actions",
})

# Historic #19 field vocabulary deliberately NOT ported in P0.
P1_FIELD_CANDIDATES = frozenset({
    "seat_presence",
    "current_bet",
    "street_wager",
    "actor",
    "special_mode",
})

TEXT_IDENTITY_FIELDS = (
    "source_audit_ref",
    "source_audit_sha256",
    "media_id",
    "media_sha256",
    "frame_sha256",
)
CORE_IDENTITY_FIELDS = TEXT_IDENTITY_FIELDS + ("frame_index",)
REQUIRED_IDENTITY_FIELDS = CORE_IDENTITY_FIELDS + (
    "layout_id",
    "slot_count",
    "slot_mapping",
    "observation_snapshot_digest",
)
OPTIONAL_IDENTITY_FIELDS = ("implementation_revision", "parent_frame_ref")
IDENTITY_FIELDS = frozenset(REQUIRED_IDENTITY_FIELDS + OPTIONAL_IDENTITY_FIELDS)

# Long-term identity may never be expressed as a mutable location or a UI row.
FORBIDDEN_IDENTITY_KEYS = frozenset({
    "path",
    "file_path",
    "current_path",
    "media_path",
    "replay_path",
    "replay_name",
    "replay_file",
    "ui_index",
    "ui_row",
    "display_index",
})

ANNOTATION_FILE = "annotations.jsonl"
EXPOSURE_FILE = "exposures.jsonl"
LOCK_FILE = ".aa-annotation-store.lock"
LOCK_TIMEOUT_SECONDS = 5.0


class AnnotationError(Exception):
    """Base class for every fail-closed refusal of this module."""


class SourceIdentityError(AnnotationError):
    """Source reference is structurally unusable as a long-term identity."""


class UnsupportedFieldError(AnnotationError):
    """field_id is outside the vocabulary supported by this module."""


class ForbiddenTruthFieldError(AnnotationError):
    """field_id belongs to another truth source; annotations may not hold it."""


class FieldSemanticsMismatch(AnnotationError):
    """Caller mixed up two different amount / action semantics."""


class RevisionConflictError(AnnotationError):
    """expected_revision did not match the durable head revision."""


class LabelValidationError(AnnotationError):
    """Label violates the per-field contract or the UNKNOWN rules."""


class ExposureError(AnnotationError):
    """Exposure event is malformed for a historical fact record."""


class StoreIntegrityError(AnnotationError):
    """Store cannot be trusted for history coverage; never silently repaired."""


class StoreLockError(AnnotationError):
    """Store lock could not be acquired."""


@dataclass(frozen=True)
class Label:
    """A developer annotation value with an explicit epistemic status."""

    status: str
    value: object = None
    unit: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ObservationContext:
    """Comparison material bound to the annotation, never overwritten later."""

    model_output: object = None
    model_reason: str | None = None
    model_output_seen: str = MODEL_OUTPUT_SEEN_UNKNOWN
    implementation_revision: str | None = None


def utc_now():
    """UTC second-precision timestamp in ISO-8601 form."""
    stamp = datetime.now(timezone.utc).replace(microsecond=0)
    return stamp.isoformat().replace("+00:00", "Z")


def _is_int(value):
    return type(value) is int


def _is_non_empty_str(value):
    return isinstance(value, str) and value.strip() != ""


def _looks_like_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value.lower()
    )


class _StoreLock:
    """Best-effort exclusive lock; enough for single-user local tooling."""

    def __init__(self, lock_path):
        self._lock_path = Path(lock_path)
        self._acquired = False

    def __enter__(self):
        deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
        while True:
            try:
                handle = os.open(
                    str(self._lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                os.close(handle)
                self._acquired = True
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise StoreLockError(
                        "store_lock_unavailable:" + str(self._lock_path)
                    )
                time.sleep(0.02)

    def __exit__(self, exc_type, exc, traceback):
        if self._acquired:
            try:
                os.remove(str(self._lock_path))
            except FileNotFoundError:
                pass
            self._acquired = False
        return False


class AnnotationStore:
    """Append-only JSONL record store for one annotation namespace."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._annotations = self.path / ANNOTATION_FILE
        self._exposures = self.path / EXPOSURE_FILE
        self._lock_path = self.path / LOCK_FILE

    def lock(self):
        return _StoreLock(self._lock_path)

    def read_annotations(self):
        return self._read(self._annotations, "annotation")

    def read_exposures(self):
        return self._read(self._exposures, "exposure")

    def append_annotation(self, row):
        self._append(self._annotations, row)

    def append_exposure(self, row):
        self._append(self._exposures, row)

    def _read(self, target, event_type):
        if not target.exists():
            return []
        rows = []
        seen = set()
        raw_text = target.read_text(encoding="utf-8")
        for lineno, raw in enumerate(raw_text.splitlines(), start=1):
            if raw.strip() == "":
                raise StoreIntegrityError(f"{event_type}:blank_line:{lineno}")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise StoreIntegrityError(
                    f"{event_type}:malformed_json:{lineno}"
                ) from exc
            if not isinstance(row, dict):
                raise StoreIntegrityError(f"{event_type}:not_object:{lineno}")
            if row.get("schema_version") != SCHEMA_VERSION:
                raise StoreIntegrityError(
                    f"{event_type}:unknown_schema_version:{lineno}"
                )
            if row.get("event_type") != event_type:
                raise StoreIntegrityError(f"{event_type}:wrong_event_type:{lineno}")
            event_id = row.get("event_id")
            if not _is_non_empty_str(event_id):
                raise StoreIntegrityError(f"{event_type}:missing_event_id:{lineno}")
            if event_id in seen:
                raise StoreIntegrityError(f"{event_type}:duplicate_event_id:{lineno}")
            seen.add(event_id)
            rows.append(row)
        return rows

    def _append(self, target, row):
        line = json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def _coerce_store(store):
    if isinstance(store, AnnotationStore):
        return store
    return AnnotationStore(store)


def _coerce_label(label):
    if isinstance(label, Label):
        return label
    if isinstance(label, dict):
        known = {
            "status", "value", "unit", "reason",
        }
        unknown = set(label) - known
        if unknown:
            names = ",".join(sorted(unknown))
            raise LabelValidationError("unknown_label_keys:" + names)
        return Label(
            status=label.get("status"),
            value=label.get("value"),
            unit=label.get("unit"),
            reason=label.get("reason"),
        )
    raise LabelValidationError("label_must_be_label_or_mapping")


def _canonical_source_ref(source_ref, *, allow_incomplete):
    """Validate a source reference and derive its stable identity key.

    Missing identity fields never raise when ``allow_incomplete`` is set; they
    are reported as gaps so that historical exposure facts stay recordable.
    Structural misuse (forbidden path/name/index keys, wrong key set) always
    raises, because those can never be a long-term identity.
    """
    if source_ref is None or not isinstance(source_ref, dict):
        raise SourceIdentityError("source_ref_must_be_mapping")
    forbidden = sorted(set(source_ref) & FORBIDDEN_IDENTITY_KEYS)
    if forbidden:
        raise SourceIdentityError(
            "mutable_location_or_ui_index_not_an_identity:" + ",".join(forbidden)
        )
    unknown = sorted(set(source_ref) - set(IDENTITY_FIELDS))
    if unknown:
        raise SourceIdentityError("unknown_source_ref_keys:" + ",".join(unknown))

    gaps = []
    for name in TEXT_IDENTITY_FIELDS:
        if not _is_non_empty_str(source_ref.get(name)):
            gaps.append("missing:" + name)
    for name in ("source_audit_sha256", "media_sha256", "frame_sha256"):
        value = source_ref.get(name)
        if value is not None and not _looks_like_sha256(value):
            gaps.append("bad_sha256:" + name)
    frame = source_ref.get("frame_index")
    if not (_is_int(frame) and frame >= 0):
        gaps.append("bad_frame_index")
    for name in ("layout_id", "observation_snapshot_digest"):
        if not _is_non_empty_str(source_ref.get(name)):
            gaps.append("missing:" + name)

    slot_count = source_ref.get("slot_count")
    slot_mapping = source_ref.get("slot_mapping")
    if slot_count is None:
        gaps.append("missing:slot_count")
    elif not _is_int(slot_count):
        gaps.append("bad_slot_count")
    elif slot_count != SUPPORTED_SLOT_COUNT:
        # No slot-count guessing: legacy 9-slot layouts are not remapped here.
        gaps.append("slot_count_not_supported:" + str(slot_count))
    if slot_mapping is None:
        gaps.append("missing:slot_mapping")
    elif not isinstance(slot_mapping, dict):
        gaps.append("bad_slot_mapping")
    else:
        if _is_int(slot_count) and slot_count == SUPPORTED_SLOT_COUNT:
            if [str(index) for index in sorted(slot_mapping, key=str)] != [
                str(index) for index in range(SUPPORTED_SLOT_COUNT)
            ] or len(slot_mapping) != SUPPORTED_SLOT_COUNT:
                gaps.append("slot_mapping_incomplete")
            elif sorted(
                str(value) for value in slot_mapping.values()
            ) != [str(index) for index in range(SUPPORTED_SLOT_COUNT)]:
                gaps.append("slot_mapping_not_bijective")

    parent = source_ref.get("parent_frame_ref")
    if parent is not None and not isinstance(parent, dict):
        gaps.append("bad_parent_frame_ref")
        parent = None

    core_missing = [
        name for name in TEXT_IDENTITY_FIELDS
        if not _is_non_empty_str(source_ref.get(name))
    ]
    if not (_is_int(frame) and frame >= 0):
        core_missing.append("frame_index")
    if core_missing:
        identity_class = IDENTITY_UNBOUND
    elif gaps:
        identity_class = IDENTITY_INCOMPLETE
    else:
        identity_class = IDENTITY_COMPLETE

    if not allow_incomplete and identity_class != IDENTITY_COMPLETE:
        raise SourceIdentityError(
            "source_identity_not_complete:" + ",".join(sorted(set(gaps)))
        )

    key_material = {
        name: source_ref.get(name) for name in CORE_IDENTITY_FIELDS
    }
    source_key = None
    if not core_missing:
        source_key = _fingerprint(key_material)

    parent_key = None
    if isinstance(parent, dict):
        parent_key_missing = [
            name for name in TEXT_IDENTITY_FIELDS
            if not _is_non_empty_str(parent.get(name))
        ]
        if not (_is_int(parent.get("frame_index"))
                and parent["frame_index"] >= 0):
            parent_key_missing.append("frame_index")
        if not parent_key_missing:
            parent_key = _fingerprint(
                {name: parent.get(name) for name in CORE_IDENTITY_FIELDS}
            )
        else:
            gaps.append("parent_frame_identity_incomplete")

    normalised = {
        name: source_ref.get(name) for name in REQUIRED_IDENTITY_FIELDS
    }
    normalised["implementation_revision"] = source_ref.get("implementation_revision")
    normalised["parent_frame_ref"] = source_ref.get("parent_frame_ref")
    return {
        "source_key": source_key,
        "parent_source_key": parent_key,
        "identity_class": identity_class,
        "identity_gaps": sorted(set(gaps)),
        "frame_index": source_ref.get("frame_index"),
        "slot_count": slot_count,
        "slot_mapping": slot_mapping if isinstance(slot_mapping, dict) else {},
        "normalised": normalised,
    }


def _fingerprint(payload):
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _sha256_hex(blob.encode("utf-8"))


def _sha256_hex(blob):
    return hashlib.sha256(blob).hexdigest()


def _validate_field(field_id):
    if not _is_non_empty_str(field_id):
        raise UnsupportedFieldError("field_id_required")
    if field_id in FORBIDDEN_TRUTH_FIELDS:
        raise ForbiddenTruthFieldError(
            "annotations_are_not_a_truth_source_for:" + field_id
        )
    if field_id not in FIELD_CONTRACTS:
        if field_id in P1_FIELD_CANDIDATES:
            raise UnsupportedFieldError("field_not_supported_in_p0:" + field_id)
        raise UnsupportedFieldError("unsupported_field_id:" + field_id)


def _validate_label_for_field(field_id, label, *, amount_role, full_actions):
    contract = FIELD_CONTRACTS[field_id]
    if label.status not in STATUSES:
        raise LabelValidationError("unknown_label_status:" + str(label.status))

    if label.status == STATUS_KNOWN:
        if label.value is None:
            raise LabelValidationError("known_label_requires_value")
        if label.unit is not None and label.unit not in contract["unit_allowed"]:
            raise LabelValidationError("unit_not_allowed_for_field:" + field_id)
        if contract["unit_required"] and not _is_non_empty_str(label.unit):
            raise LabelValidationError("unit_required_for_field:" + field_id)
        if type(label.value) is bool or not isinstance(
            label.value, contract["value_types"]
        ):
            raise LabelValidationError("value_type_mismatch:" + field_id)
        allowed = contract["allowed_values"]
        if allowed is not None and label.value not in allowed:
            raise LabelValidationError("value_not_allowed:" + str(label.value))
        if isinstance(label.value, int) and label.value < 0:
            raise LabelValidationError("negative_amount_not_allowed")
    else:
        if label.value is not None:
            raise LabelValidationError("non_known_label_must_not_carry_value")
        if not _is_non_empty_str(label.reason):
            raise LabelValidationError(
                "reason_required_for:" + label.status
            )

    if contract["sequence_bound"] and full_actions is not None:
        raise FieldSemanticsMismatch("visible_action_glyph_vs_full_actions")
    if amount_role is not None:
        if amount_role in contract["forbidden_amount_roles"]:
            raise FieldSemanticsMismatch("pot_display_vs_wager_or_call_price")
        if field_id == FIELD_POT_DISPLAY and (
            amount_role != AMOUNT_ROLE_POT_DISPLAY_TOTAL
        ):
            raise FieldSemanticsMismatch("pot_display_amount_role_mismatch")


def _validate_seat(seat, identity):
    if seat is None:
        return
    if not _is_int(seat) or seat < 0:
        raise SourceIdentityError("seat_must_be_non_negative_int_or_null")
    slot_count = identity["slot_count"]
    if not _is_int(slot_count) or slot_count != SUPPORTED_SLOT_COUNT:
        raise SourceIdentityError("seat_not_mappable_without_supported_layout")
    mapping = identity["slot_mapping"]
    if str(seat) not in {str(value) for value in mapping.values()}:
        raise SourceIdentityError("seat_not_mapped_in_layout:" + str(seat))


def _group_key(source_key, field_id, seat):
    return source_key + "|" + field_id + "|" + ("null" if seat is None else str(seat))


def _fold_annotations(records):
    """Return {group_key: [records in append order]} plus per-group head."""
    groups = {}
    for row in records:
        key = row.get("group_key")
        groups.setdefault(key, []).append(row)
    heads = {}
    for key, rows in groups.items():
        superseded = {
            row.get("supersedes")
            for row in rows
            if row.get("supersedes") is not None
        }
        live = [row for row in rows if row.get("revision_id") not in superseded]
        heads[key] = live
    return groups, heads


def _current_view(row, *, include_context):
    view = {
        "revision_id": row.get("revision_id"),
        "field_id": row.get("field_id"),
        "seat": row.get("seat"),
        "status": row.get("status"),
        "value": row.get("value"),
        "unit": row.get("unit"),
        "reason": row.get("reason"),
        "supersedes": row.get("supersedes"),
        "recorded_at": row.get("recorded_at"),
        "actor": row.get("actor"),
        "identity_class": row.get("identity_class"),
        "model_output": row.get("model_output"),
        "model_reason": row.get("model_reason"),
        "model_output_seen": row.get("model_output_seen"),
        "implementation_revision": row.get("implementation_revision"),
    }
    if include_context:
        view["source_ref"] = row.get("source_ref")
    return view


def get_annotations(store, source_ref=None, include_history=False, **filters):
    """Read current annotation values (and optionally the full revision chain).

    The winning value is folded through ``supersedes`` / revision identity,
    never guessed from a timestamp.
    """
    resolved = _coerce_store(store)
    source_identity = None
    wanted_key = None
    if source_ref is not None:
        source_identity = _canonical_source_ref(source_ref, allow_incomplete=True)
        wanted_key = source_identity["source_key"]
    rows = resolved.read_annotations()
    if wanted_key is not None:
        rows = [row for row in rows if row.get("source_key") == wanted_key]
    field_filter = filters.get("field_id")
    if field_filter is not None:
        rows = [row for row in rows if row.get("field_id") == field_filter]
    if "seat" in filters and filters["seat"] is not None:
        rows = [row for row in rows if row.get("seat") == filters["seat"]]
    groups, heads = _fold_annotations(rows)
    current = {}
    conflicts = []
    for key, live in heads.items():
        if len(live) == 1:
            current[key] = _current_view(
                live[0], include_context=include_history
            )
        else:
            conflicts.append(
                {
                    "group_key": key,
                    "reason": "multiple_heads",
                    "revision_ids": sorted(row["revision_id"] for row in live),
                }
            )
    identity_class = source_identity["identity_class"] if source_identity else None
    identity_gaps = source_identity["identity_gaps"] if source_identity else []
    payload = {
        "source_key": wanted_key,
        "identity_class": identity_class,
        "identity_gaps": identity_gaps,
        "current": current,
        "conflicts": conflicts,
    }
    if include_history:
        payload["history"] = rows
    return payload


def append_annotation(
    store,
    source_ref,
    field_id,
    seat,
    label,
    expected_revision,
    actor,
    *,
    observation=None,
    amount_role=None,
    full_actions=None,
    recorded_at=None,
):
    """Append one annotation revision. Returns the new durable revision id.

    Old revisions are never overwritten; ``expected_revision`` must match the
    durable head for the same (source, field, seat) or the append is refused.
    """
    resolved = _coerce_store(store)
    _validate_field(field_id)
    identity = _canonical_source_ref(source_ref, allow_incomplete=True)
    _validate_seat(seat, identity)
    resolved_label = _coerce_label(label)
    _validate_label_for_field(
        field_id,
        resolved_label,
        amount_role=amount_role,
        full_actions=full_actions,
    )
    context = observation or ObservationContext()
    if context.model_output_seen not in MODEL_OUTPUT_SEEN:
        raise LabelValidationError(
            "bad_model_output_seen:" + str(context.model_output_seen)
        )
    if not _is_non_empty_str(actor):
        raise LabelValidationError("annotator_actor_required")

    group = _group_key(identity["source_key"], field_id, seat)
    with resolved.lock():
        existing = [
            row for row in resolved.read_annotations()
            if row.get("group_key") == group
        ]
        _, heads = _fold_annotations(existing)
        live = heads.get(group, [])
        if len(live) > 1:
            raise RevisionConflictError("multiple_heads_for_group")
        head = live[0] if live else None
        head_revision = head.get("revision_id") if head else None
        if expected_revision is None and head_revision is not None:
            raise RevisionConflictError("expected_revision_required")
        if expected_revision is not None and head_revision is None:
            raise RevisionConflictError("expected_revision_not_found")
        if expected_revision is not None and expected_revision != head_revision:
            raise RevisionConflictError("revision_conflict")
        revision_id = "ann-" + uuid.uuid4().hex
        row = {
            "schema_version": SCHEMA_VERSION,
            "event_id": revision_id,
            "event_type": "annotation",
            "revision_id": revision_id,
            "group_key": group,
            "recorded_at": recorded_at or utc_now(),
            "actor": actor,
            "source_key": identity["source_key"],
            "parent_source_key": identity["parent_source_key"],
            "identity_class": identity["identity_class"],
            "identity_gaps": identity["identity_gaps"],
            "source_ref": identity["normalised"],
            "field_id": field_id,
            "seat": seat,
            "status": resolved_label.status,
            "value": resolved_label.value,
            "unit": resolved_label.unit,
            "reason": resolved_label.reason,
            "supersedes": head_revision,
            "expected_revision": expected_revision,
            "model_output": context.model_output,
            "model_reason": context.model_reason,
            "model_output_seen": context.model_output_seen,
            "implementation_revision": context.implementation_revision,
            "observation_snapshot_digest": identity["normalised"].get(
                "observation_snapshot_digest"
            ),
            "metadata": {
                "amount_role": amount_role,
                "full_actions_declared": full_actions is not None,
            },
        }
        try:
            resolved.append_annotation(row)
        except OSError as exc:
            raise StoreIntegrityError("annotation_persistence_failed") from exc
    return revision_id


def _query_keys(identity):
    keys = set()
    if identity["source_key"]:
        keys.add(identity["source_key"])
    if identity["parent_source_key"]:
        keys.add(identity["parent_source_key"])
    return keys


def record_exposure(
    store,
    source_ref,
    annotation_revision_ids,
    purpose,
    run_or_manifest_ref,
    evidence_kind,
    *,
    policy_ref=None,
    artifact_ref=None,
    role=None,
    actor=None,
    policy_violation=False,
    recorded_at=None,
):
    """Append an irreversible source-level use / release / consumption fact.

    Historical facts are always recorded, including ones that violated policy:
    refusing to store them would let past contamination disappear from history.
    """
    resolved = _coerce_store(store)
    if purpose not in PURPOSES:
        raise ExposureError("unsupported_purpose:" + str(purpose))
    if evidence_kind not in EVIDENCE_KINDS:
        raise ExposureError("unsupported_evidence_kind:" + str(evidence_kind))
    identity = _canonical_source_ref(source_ref, allow_incomplete=True)
    if identity["source_key"] is None:
        raise SourceIdentityError("exposure_requires_core_source_identity")
    if not isinstance(annotation_revision_ids, (list, tuple)):
        raise ExposureError("annotation_revision_ids_must_be_sequence")
    for revision in annotation_revision_ids:
        if not _is_non_empty_str(revision):
            raise ExposureError("bad_annotation_revision_id")
    if run_or_manifest_ref is None or not isinstance(run_or_manifest_ref, dict):
        raise ExposureError("run_or_manifest_ref_required")
    if artifact_ref is not None and not isinstance(artifact_ref, dict):
        raise ExposureError("artifact_ref_must_be_mapping")
    if role is not None and role not in ("development", "calibration", "holdout"):
        raise ExposureError("unsupported_exposure_role:" + str(role))
    if policy_violation not in (True, False):
        raise ExposureError("policy_violation_must_be_bool")

    event_id = "exp-" + uuid.uuid4().hex
    row = {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "event_type": "exposure",
        "recorded_at": recorded_at or utc_now(),
        "actor": actor,
        "source_key": identity["source_key"],
        "parent_source_key": identity["parent_source_key"],
        "source_ref": identity["normalised"],
        "purpose": purpose,
        "evidence_kind": evidence_kind,
        "annotation_revision_ids": list(annotation_revision_ids),
        "run_or_manifest_ref": dict(run_or_manifest_ref),
        "artifact_ref": dict(artifact_ref) if artifact_ref else None,
        "role": role,
        "policy_ref": policy_ref,
        "policy_violation": policy_violation,
        "identity_class": identity["identity_class"],
    }
    try:
        with resolved.lock():
            resolved.append_exposure(row)
    except OSError as exc:
        raise StoreIntegrityError("exposure_persistence_failed") from exc
    return row


def _fold_exposure_state(rows, *, coverage_complete=False):
    """Monotonic union: releases and later revisions never reduce exposure.

    An empty event set is ``EXPOSURE_UNKNOWN`` unless every known store was
    declared and read: absence of records is never silently read as clean.
    """
    if not rows:
        return (
            STATE_NONE_UNDER_COVERAGE if coverage_complete else STATE_UNKNOWN
        )
    kinds = {row.get("evidence_kind") for row in rows}
    if EVIDENCE_ACTUALLY_CONSUMED in kinds:
        return STATE_ACTUALLY_CONSUMED
    intent_kinds = {EVIDENCE_RESERVED, EVIDENCE_RELEASED, EVIDENCE_POSSIBLY_USED}
    if kinds & intent_kinds:
        return STATE_POSSIBLY_USED
    return STATE_UNKNOWN


def _policy_bundle_view(policy_bundle):
    if policy_bundle is None or not isinstance(policy_bundle, dict):
        return None
    return policy_bundle


def _read_declared_stores(policy_bundle, own_path):
    reasons = []
    rows = []
    declared = policy_bundle.get("declared_store_paths")
    if declared is None:
        return rows, ["exposure_history_coverage_undeclared"]
    if not isinstance(declared, (list, tuple)) or not declared:
        return rows, ["exposure_history_coverage_undeclared"]
    own = str(Path(own_path).resolve()).casefold()
    paths = []
    for entry in declared:
        target = Path(entry)
        if not target.exists():
            reasons.append("declared_store_missing:" + str(target))
            continue
        paths.append(str(target.resolve()).casefold())
        try:
            rows.extend(AnnotationStore(target).read_exposures())
        except (StoreIntegrityError, OSError):
            reasons.append("declared_store_unreadable:" + str(target))
    if own not in paths:
        reasons.append("own_store_not_declared_in_history_coverage")
    return rows, reasons


def assess_use(
    store,
    source_ref,
    purpose,
    policy_bundle,
    *,
    actor=None,
    reserve=False,
    required_fields=None,
):
    """Derive purpose eligibility read-only. Clients cannot supply the verdict.

    Absence of a reservation hit is never sufficient: unknown role, unknown
    history coverage, incomplete identity, missing policy binding and
    consumption without a durable record all resolve to ``BLOCKED``.
    """
    resolved = _coerce_store(store)
    bundle = _policy_bundle_view(policy_bundle)
    reasons = []

    if bundle is None:
        reasons.append("policy_bundle_missing")
        bundle = {}
    policy_ref = bundle.get("policy_ref")
    if not _is_non_empty_str(policy_ref):
        reasons.append("policy_ref_missing")
    if purpose not in PURPOSES:
        reasons.append("purpose_unsupported:" + str(purpose))

    identity = _canonical_source_ref(source_ref, allow_incomplete=True)
    if identity["identity_class"] != IDENTITY_COMPLETE:
        reasons.append(
            "source_identity_incomplete:" + ",".join(identity["identity_gaps"])
        )

    reservations = bundle.get("reservations")
    frame = identity["frame_index"]
    if reservations is None:
        reasons.append("reservations_missing")
    else:
        try:
            assert_training_frames([frame], reservations)
        except Exception:
            reasons.append("frame_reserved_or_invalid")
        declared_role = bundle.get("role")
        if declared_role not in ("development", "calibration"):
            reasons.append("role_unknown:" + str(declared_role))
        else:
            reserved = reservations.get("reserved_inclusive_intervals", [])
            if any(
                _is_int(frame) and start <= frame <= end
                for start, end in reserved
            ):
                reasons.append("role_conflicts_reservation")
        exposed_frames = reservations.get("known_exploration_frames", [])
        if _is_int(frame) and frame in exposed_frames:
            if not _is_non_empty_str(bundle.get("development_evidence_ref")):
                reasons.append("exploration_not_development")

    annotations = get_annotations(resolved, source_ref, include_history=False)
    wanted = set(required_fields or bundle.get("required_fields") or [])
    for field_id in sorted(wanted):
        try:
            _validate_field(field_id)
        except AnnotationError:
            reasons.append("required_field_unsupported:" + field_id)
            continue
        view = None
        for key, row in annotations["current"].items():
            if row.get("field_id") == field_id:
                view = row
                break
        if view is None:
            reasons.append("required_annotation_missing:" + field_id)
        elif view.get("status") != STATUS_KNOWN:
            reasons.append("required_annotation_not_known:" + field_id)
    if annotations["conflicts"]:
        reasons.append("annotation_history_conflict")

    allow_assisted = bundle.get("allow_model_assisted_labels") is True
    for row in annotations["current"].values():
        if row.get("model_output_seen") == MODEL_OUTPUT_SEEN_YES and not allow_assisted:
            reasons.append("model_assisted_label_without_policy")
            break

    coverage_rows, coverage_reasons = _read_declared_stores(bundle, resolved.path)
    reasons.extend(coverage_reasons)
    keys = _query_keys(identity)
    own_rows = [
        row for row in resolved.read_exposures()
        if row.get("source_key") in keys or row.get("parent_source_key") in keys
    ]
    merged = coverage_rows + own_rows
    coverage_ok = not coverage_reasons
    state = _fold_exposure_state(merged, coverage_complete=coverage_ok)
    if state == STATE_ACTUALLY_CONSUMED:
        if bundle.get("allow_reuse_of_exposed_development") is not True:
            reasons.append("exposed_development_reuse_not_declared")
    elif state == STATE_POSSIBLY_USED:
        if bundle.get("allow_reuse_of_exposed_development") is not True:
            reasons.append("possibly_used_reuse_not_declared")
    elif state == STATE_UNKNOWN:
        reasons.append("exposure_history_unknown")
    if any(row.get("policy_violation") for row in merged):
        reasons.append("known_policy_violation_exposure")

    allowed = not reasons
    receipt = None
    if allowed and reserve:
        try:
            event = record_exposure(
                resolved,
                source_ref,
                sorted(row["revision_id"] for row in annotations["current"].values()),
                purpose,
                {
                    "kind": "consumption_reservation",
                    "policy_ref": policy_ref,
                    "actor": actor,
                },
                EVIDENCE_RESERVED,
                policy_ref=policy_ref,
                role=bundle.get("role"),
                actor=actor,
            )
            verify = [row for row in resolved.read_exposures()
                      if row.get("event_id") == event["event_id"]]
            if not verify:
                raise StoreIntegrityError("reservation_not_visible_after_write")
            receipt = {
                "event_id": event["event_id"],
                "evidence_kind": EVIDENCE_RESERVED,
                "purpose": purpose,
            }
        except AnnotationError as exc:
            if "persistence" in str(exc) or "not_visible" in str(exc):
                reasons.append("consumption_record_not_durable")
            else:
                reasons.append("consumption_reservation_refused:" + str(exc))
        except OSError:
            reasons.append("consumption_record_not_durable")

    verdict = VERDICT_ALLOWED_FOR_PURPOSE if not reasons else VERDICT_BLOCKED
    coverage = COVERAGE_COMPLETE if not coverage_reasons else COVERAGE_INCOMPLETE
    return {
        "schema_version": SCHEMA_VERSION,
        "source_key": identity["source_key"],
        "identity_class": identity["identity_class"],
        "purpose": purpose,
        "verdict": verdict,
        "usable_scope": (
            SCOPE_PURPOSE_BOUND if verdict == VERDICT_ALLOWED_FOR_PURPOSE
            else SCOPE_ANNOTATION_ONLY
        ),
        "reasons": sorted(set(reasons)) if reasons else [],
        "policy_ref": policy_ref,
        "exposure_state": state,
        "history_coverage": coverage,
        "reservation_receipt": receipt,
    }


def exposure_projection(store, scope):
    """Read-only projection bindable by the existing freeze.exposures audit.

    Missing history is never projected as clean: unresolved gaps are carried on
    every row and mark it non-bindable.
    """
    resolved = _coerce_store(store)
    scope = scope or {}
    identity_targets = []
    for ref in scope.get("sources") or []:
        identity_targets.append(_canonical_source_ref(ref, allow_incomplete=True))
    declared = scope.get("declared_store_paths")
    _, coverage_reasons = _read_declared_stores(
        {"declared_store_paths": declared}, resolved.path
    )
    coverage = (
        COVERAGE_COMPLETE if not coverage_reasons else COVERAGE_INCOMPLETE
    )
    coverage_ok = not coverage_reasons
    purposes = set(scope.get("purposes") or PURPOSES)

    rows = []
    unresolved_global = list(coverage_reasons)
    matched_events = set()
    for identity in identity_targets:
        keys = _query_keys(identity)
        events = [
            row for row in resolved.read_exposures()
            if (row.get("source_key") in keys or row.get("parent_source_key") in keys)
            and row.get("purpose") in purposes
        ]
        state = _fold_exposure_state(events, coverage_complete=coverage_ok)
        if not events:
            unresolved_global.append(
                "no_exposure_history_evidence:" + str(identity["source_key"])
            )
            rows.append(
                {
                    "source_key": identity["source_key"],
                    "parent_source_key": identity["parent_source_key"],
                    "used_for": None,
                    "role": "unknown",
                    "artifact_sha256": None,
                    "annotation_revision_ids": [],
                    "policy_ref": None,
                    "exposure_state": STATE_UNKNOWN,
                    "evidence_kind": None,
                    "policy_violation": False,
                    "freeze_bindable": False,
                    "unresolved": ["no_exposure_history_evidence"],
                }
            )
            continue
        for event in events:
            matched_events.add(event.get("event_id"))
            artifact = event.get("artifact_ref") or {}
            artifact_sha = artifact.get("sha256")
            row_unresolved = list(unresolved_global)
            if not _is_non_empty_str(artifact_sha):
                row_unresolved.append("artifact_sha256_missing")
            if event.get("role") not in ("development", "calibration"):
                row_unresolved.append("exposure_role_unknown")
            if event.get("policy_violation"):
                row_unresolved.append("policy_violation_recorded")
            if state == STATE_UNKNOWN:
                row_unresolved.append("exposure_history_unknown")
            rows.append(
                {
                    "event_id": event.get("event_id"),
                    "source_key": event.get("source_key"),
                    "parent_source_key": event.get("parent_source_key"),
                    "used_for": event.get("purpose"),
                    "role": event.get("role") or "unknown",
                    "artifact_sha256": artifact_sha,
                    "annotation_revision_ids": list(
                        event.get("annotation_revision_ids") or []
                    ),
                    "policy_ref": event.get("policy_ref"),
                    "evidence_kind": event.get("evidence_kind"),
                    "policy_violation": bool(event.get("policy_violation")),
                    "exposure_state": state,
                    "freeze_bindable": not row_unresolved,
                    "unresolved": sorted(set(row_unresolved)),
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "history_coverage": coverage,
        "rows": rows,
        "unresolved": sorted(set(unresolved_global)),
    }
