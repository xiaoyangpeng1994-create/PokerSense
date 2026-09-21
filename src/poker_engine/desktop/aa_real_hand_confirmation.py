"""Evidence-bound TARGET-S real-hand confirmation: the ONLY acceptance authority.

This module exists because a real hand can only be accepted on facts a human
actually confirmed against bound evidence, and because three other things in
this repository look like confirmation but must never become it:

* ``aa_semantics`` machine observations. They stay ``UNKNOWN`` /
  ``VISIBLE_LABEL_CANDIDATE`` / ``PRICE_DERIVED_CANDIDATE`` forever. This module
  never mutates them and never promotes one by rule
  (``DO_NOT_MUTATE_MACHINE_CANDIDATE``).
* ``tools.aa_annotation_records`` (#31). That is development-label and exposure
  governance. An annotation row is never evidence here.
* ``aa_hand_input`` / ``aa_analysis_records`` manual hypothesis entry. Its scope
  is ``MANUAL_HYPOTHESIS_OFFLINE_NOT_LIVE_ADVICE``: a hand typed into the form
  with ``provenance == "human_confirmed"`` is an offline what-if, not an
  accepted real hand. It is preserved unchanged and is NOT an acceptance path.

What this module is:

* TARGET-S only: the threeway river solver-ready river-start fact contract.
* TARGET-P / PHH full-hand readiness is explicitly DEFERRED. PHH needs opening
  stacks, forced bets and the whole action sequence from preflop; a stack-delta
  commitment is an aggregate and must never masquerade as PHH forced bets or
  actions. ``TARGET_S_READY`` never implies ``TARGET_P_PHH_READY``.

Scope decision for P0: the first controlled capture deliberately targets a hand
where Hero is the river first-to-act, so the river public history is legally
empty. The empty history is not a gap: it is only accepted when
``hero_is_first_river_actor`` is itself a confirmed fact. When Hero is not
first-to-act this module returns ``TARGET_S_NOT_READY`` rather than silently
falling back to per-action confirmation (that is a later phase).

Persistence is append-only JSONL plus a best-effort lock plus a per-record
digest chain, following the infrastructure discipline of #31 and deliberately
reusing none of its annotation truth semantics. No database, no workflow engine,
no schema migration.

Every confirmed fact binds: confirmation id, schema version, hand/epoch
identity, capture session, marker, source recording identity, exact frame,
evidence digest, structured value, reviewer, timestamp, revision, supersedes and
``provenance = "human_confirmed"``. ``audit_note`` is stored for audit only and
can never become a fact value. A fact with no evidence digest is refused.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

SCHEMA_VERSION = "aa-real-hand-confirmation-v2"
EVIDENCE_SCHEMA_VERSION = "aa-real-hand-evidence-v2"
IMPLEMENTATION_VERSION = "aa-real-hand-confirmation-p0-target-s-v2"

#: The only acceptance target implemented here.
TARGET = "TARGET_S"
SCOPE = "REAL_HAND_ACCEPTANCE_TARGET_S_RIVER_THREEWAY"

#: The manual-hypothesis scope this module must never be confused with. Keeping
#: it here lets a test assert the two scopes are distinct strings.
MANUAL_HYPOTHESIS_SCOPE = "MANUAL_HYPOTHESIS_OFFLINE_NOT_LIVE_ADVICE"

CONFIRMATION_FILE = "confirmations.jsonl"
LOCK_FILE = ".aa-real-hand-confirmation.lock"
LOCK_TIMEOUT_SECONDS = 5.0

STATUS_CONFIRMED = "CONFIRMED"
STATUS_UNCONFIRMED = "UNCONFIRMED"
STATUS_CONFLICT = "CONFLICT"

#: The one status that means "this real hand is factually accepted for TARGET-S"
#: and the one status everything else falls back to.
ACCEPTED = "TARGET_S_REAL_HAND_CONFIRMED"
NOT_READY = "TARGET_S_NOT_READY"

PROV_HUMAN = "human_confirmed"
PROV_DERIVED = "derived_from_confirmed_observations"

#: A derived value must report itself as derived. Calling a computed number
#: ``human_confirmed`` would put a derivation where a human statement belongs.
DERIVATION_STACK_DELTA = "stack_delta_v1"

MARKER_HAND_START = "HAND_START"
MARKER_RIVER_START = "RIVER_START"
MARKER_PAYOUT = "PAYOUT"
MARKERS = (MARKER_HAND_START, MARKER_RIVER_START, MARKER_PAYOUT)

#: TARGET-P is out of scope for this module; these are stated so a caller cannot
#: read a TARGET-S receipt as a PHH or a strategy result.
PHH_READY = False
STRATEGY_ASSESSED = False

CARD = re.compile(r"\A[2-9TJQKA][shdc]\Z")
HEX64 = re.compile(r"\A[a-f0-9]{64}\Z")

RULE_KEYS = ("table_size", "small_blind", "big_blind", "ante", "ante_mode",
             "straddle_amount", "straddle_mode", "revision")
ANTE_MODES = ("none", "per_dealt_player")
STRADDLE_MODES = ("none", "mandatory_utg", "optional_explicit_utg")

#: Every condition behind ``hand_committed = opening - river_start``. Each one
#: is an independent assertion; a blanket "nothing happened" is not accepted.
STACK_DELTA_ASSERTIONS = (
    "hand_start_marker_valid",
    "river_start_marker_valid",
    "seat_identity_continuous",
    "no_rebuy_or_topup",
    "no_chip_return",
    "no_payout_before_river",
    "no_jackpot_or_cashout",
    "no_insurance",
    "no_chip_side_fee_or_rake",
    "same_integer_precision",
)
STACK_DELTA_BINDINGS = ("hand_start_evidence_digest",
                        "river_start_evidence_digest")

FACT_KEYS = (
    "ended_hand_confirmed", "hero_seat", "hero_cards", "board_cards",
    "seat_set", "active_seats", "all_in_seats", "folded_seats", "action_order",
    "dealer_button_seat", "river_start_stacks", "opening_stacks",
    "stack_delta_assertions", "street_wagers_zero", "pot_display",
    "hero_is_first_river_actor", "table_rules", "other_fees",
    "straddle_posted_this_hand", "no_side_pot", "no_pending_action",
)

#: The TARGET-S contract. ``hand_committed`` and ``history`` are derived, not
#: confirmed directly, so they are absent here on purpose.
TARGET_S_REQUIRED_FACTS = (
    "ended_hand_confirmed", "hero_seat", "hero_cards", "board_cards",
    "seat_set", "active_seats", "all_in_seats", "folded_seats", "action_order",
    "dealer_button_seat", "river_start_stacks", "opening_stacks",
    "stack_delta_assertions", "street_wagers_zero", "pot_display",
    "hero_is_first_river_actor", "table_rules", "other_fees",
    "straddle_posted_this_hand", "no_side_pot", "no_pending_action",
)

HAND_START_FACTS = frozenset(("opening_stacks", "seat_set",
                              "dealer_button_seat", "table_rules"))
PAYOUT_FACTS = frozenset(("ended_hand_confirmed", "other_fees"))
FACT_MARKERS = {
    key: (MARKER_HAND_START if key in HAND_START_FACTS else
          MARKER_PAYOUT if key in PAYOUT_FACTS else MARKER_RIVER_START)
    for key in FACT_KEYS
}
EVIDENCE_KEYS = frozenset((
    "schema_version", "hand_id", "observed_epoch", "capture_session_id",
    "source_ref", "marker", "window_start_frame", "window_end_frame",
    "frames", "stable", "boundary_continuity",
))
RECORD_KEYS = frozenset((
    "schema_version", "confirmation_id", "hand_id", "observed_epoch",
    "capture_session_id", "source_ref", "source_digest", "marker",
    "source_frame", "evidence_descriptor", "evidence_digest",
    "snapshot_digest", "fact_key", "value", "reviewer", "recorded_at",
    "revision", "supersedes", "provenance", "audit_note",
))


class RealHandConfirmationError(ValueError):
    """Base error for the TARGET-S confirmation contract."""


class EvidenceBindingError(RealHandConfirmationError):
    """A confirmation was offered without evidence it can be bound to."""


class IdentityMismatchError(RealHandConfirmationError):
    """A confirmation disagreed with the hand identity already on record."""


class RevisionError(RealHandConfirmationError):
    """A revision would overwrite history instead of superseding it."""


class FactValueError(RealHandConfirmationError):
    """A fact value is not the structured shape its key requires."""


class StoreIntegrityError(RealHandConfirmationError):
    """The append-only store no longer matches its own digests."""


class StoreLockError(RealHandConfirmationError):
    """The store lock could not be taken."""


def canonical(value):
    """Stable serialisation; the digest of a value must not depend on order."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def _is_non_empty_str(value):
    return isinstance(value, str) and value.strip() != ""


def _is_int(value):
    return type(value) is int


def _is_seat(value):
    return _is_int(value) and 0 <= value <= 7


def _source_identity(ref):
    """Canonical source identity; a recording must be nameable to be evidence."""
    if not isinstance(ref, dict) or not ref:
        raise EvidenceBindingError("source_ref_must_be_a_non_empty_object")
    if set(ref) - {"media_sha256", "recording_id"}:
        raise EvidenceBindingError("source_ref_unknown_fields")
    media = ref.get("media_sha256")
    recording = ref.get("recording_id")
    if not (_is_non_empty_str(media) or _is_non_empty_str(recording)):
        raise EvidenceBindingError(
            "source_ref_needs_media_sha256_or_recording_id")
    if media is not None and not (isinstance(media, str) and HEX64.match(media)):
        raise EvidenceBindingError("source_ref_media_sha256_invalid")
    if "recording_id" in ref and not _is_non_empty_str(recording):
        raise EvidenceBindingError("source_ref_recording_id_invalid")
    if "media_sha256" in ref and media is None:
        raise EvidenceBindingError("source_ref_media_sha256_invalid")
    cleaned = dict(ref)
    return cleaned, digest(cleaned)


def _is_digest(value):
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _validate_evidence(descriptor):
    """Validate a human-attested source/window binding, not image truth.

    Content hashes identify the reviewed local evidence. They are not proof
    that a reviewer read it, or authentication against a malicious operator.
    An inter-hand opening window explicitly belongs to the upcoming hand.
    """
    if not isinstance(descriptor, dict) or set(descriptor) != EVIDENCE_KEYS:
        raise EvidenceBindingError("evidence_descriptor_exact_schema_required")
    if descriptor["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise EvidenceBindingError("unsupported_evidence_schema")
    for key in ("hand_id", "observed_epoch", "capture_session_id"):
        if not _is_non_empty_str(descriptor[key]):
            raise EvidenceBindingError("evidence_identity_required:" + key)
    _source_identity(descriptor["source_ref"])
    if descriptor["marker"] not in MARKERS:
        raise EvidenceBindingError("evidence_marker_invalid")
    start, end = (descriptor["window_start_frame"],
                  descriptor["window_end_frame"])
    if not (_is_int(start) and _is_int(end) and 0 <= start <= end):
        raise EvidenceBindingError("evidence_window_invalid")
    if (descriptor["stable"] is not True
            or descriptor["boundary_continuity"] is not True):
        raise EvidenceBindingError("evidence_stability_or_continuity_unconfirmed")
    frames = descriptor["frames"]
    if not isinstance(frames, list) or not frames:
        raise EvidenceBindingError("evidence_frames_required")
    indices = []
    for frame in frames:
        if not isinstance(frame, dict) or set(frame) != {
                "source_frame", "content_sha256"}:
            raise EvidenceBindingError("evidence_frame_schema_invalid")
        number = frame["source_frame"]
        if not _is_int(number) or not start <= number <= end:
            raise EvidenceBindingError("evidence_frame_outside_window")
        if not _is_digest(frame["content_sha256"]):
            raise EvidenceBindingError("evidence_content_digest_required")
        indices.append(number)
    if (indices != sorted(set(indices))
            or indices[0] != start or indices[-1] != end):
        raise EvidenceBindingError("evidence_frames_must_order_and_bound_window")
    return deepcopy(descriptor)


def _unique_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise StoreIntegrityError("confirmation:duplicate_json_key:" + key)
        value[key] = item
    return value


class _StoreLock:
    """Best-effort exclusive lock; enough for single-user local tooling."""

    def __init__(self, lock_path):
        self._lock_path = Path(lock_path)
        self._acquired = False

    def __enter__(self):
        deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
        while True:
            try:
                handle = os.open(str(self._lock_path),
                                 os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(handle)
                self._acquired = True
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise StoreLockError(
                        "store_lock_unavailable:" + str(self._lock_path))
                time.sleep(0.02)

    def __exit__(self, exc_type, exc, traceback):
        if self._acquired:
            try:
                os.remove(str(self._lock_path))
            except FileNotFoundError:
                pass
            self._acquired = False
        return False


class ConfirmationStore:
    """Append-only JSONL store for one real-hand confirmation namespace."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._confirmations = self.path / CONFIRMATION_FILE
        self._lock_path = self.path / LOCK_FILE

    def lock(self):
        return _StoreLock(self._lock_path)

    def read_confirmations(self):
        if not self._confirmations.exists():
            return []
        rows = []
        seen = set()
        previous = None
        try:
            raw_text = self._confirmations.read_text(encoding="utf-8")
        except UnicodeError as exc:
            raise StoreIntegrityError("confirmation:invalid_utf8") from exc
        for lineno, raw in enumerate(raw_text.splitlines(), start=1):
            if raw.strip() == "":
                raise StoreIntegrityError(f"confirmation:blank_line:{lineno}")
            try:
                envelope = json.loads(raw, object_pairs_hook=_unique_json_object)
            except (json.JSONDecodeError, ValueError) as exc:
                raise StoreIntegrityError(
                    f"confirmation:malformed_json:{lineno}") from exc
            if not isinstance(envelope, dict) or set(envelope) != {
                    "record", "record_digest", "prev_digest"}:
                raise StoreIntegrityError(f"confirmation:not_object:{lineno}")
            row = envelope.get("record")
            if not isinstance(row, dict):
                raise StoreIntegrityError(f"confirmation:missing_record:{lineno}")
            if row.get("schema_version") != SCHEMA_VERSION:
                raise StoreIntegrityError(
                    f"confirmation:unknown_schema_version:{lineno}")
            identifier = row.get("confirmation_id")
            if not _is_non_empty_str(identifier):
                raise StoreIntegrityError(
                    f"confirmation:missing_confirmation_id:{lineno}")
            if identifier in seen:
                raise StoreIntegrityError(
                    f"confirmation:duplicate_confirmation_id:{lineno}")
            seen.add(identifier)
            try:
                _validate_record(row)
                seal = digest(row)
            except (RealHandConfirmationError, TypeError, ValueError) as exc:
                raise StoreIntegrityError(
                    f"confirmation:invalid_record:{lineno}:{exc}") from exc
            if seal != envelope.get("record_digest"):
                raise StoreIntegrityError(
                    f"confirmation:record_digest_mismatch:{lineno}")
            if envelope.get("prev_digest") != previous:
                raise StoreIntegrityError(
                    f"confirmation:chain_break:{lineno}")
            previous = envelope.get("record_digest")
            rows.append(row)
        _validate_histories(rows)
        return rows

    def _append(self, row):
        envelope = {"record": row, "record_digest": digest(row),
                    "prev_digest": self._tail_digest()}
        line = canonical(envelope) + "\n"
        self._confirmations.parent.mkdir(parents=True, exist_ok=True)
        with self._confirmations.open("a", encoding="utf-8",
                                      newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def _tail_digest(self):
        tail = None
        if not self._confirmations.exists():
            return None
        for raw in self._confirmations.read_text(encoding="utf-8").splitlines():
            if raw.strip() == "":
                continue
            tail = json.loads(raw).get("record_digest")
        return tail


def _validate_cards(value, count, label):
    if not isinstance(value, list) or len(value) != count:
        raise FactValueError(f"{label}_must_be_a_list_of_{count}_cards")
    for item in value:
        if not isinstance(item, str) or not CARD.match(item):
            raise FactValueError(f"{label}_contains_an_invalid_card:{item}")
    if len({str(item) for item in value}) != count:
        raise FactValueError(f"{label}_contains_a_duplicate_card")
    return list(value)


def _validate_seat_list(value, label, *, allow_empty=False):
    if not isinstance(value, list):
        raise FactValueError(f"{label}_must_be_a_list_of_seats")
    if not value and not allow_empty:
        raise FactValueError(f"{label}_must_not_be_empty")
    for item in value:
        if not _is_seat(item):
            raise FactValueError(f"{label}_contains_an_invalid_seat:{item}")
    if len({int(item) for item in value}) != len(value):
        raise FactValueError(f"{label}_contains_a_duplicate_seat")
    return [int(item) for item in value]


def _validate_stack_map(value, label):
    if not isinstance(value, dict) or not value:
        raise FactValueError(f"{label}_must_be_a_non_empty_seat_to_stack_map")
    cleaned = {}
    for seat, stack in value.items():
        if not (_is_seat(seat) or isinstance(seat, str) and seat in "01234567"
                and len(seat) == 1):
            raise FactValueError(f"{label}_has_a_noncanonical_seat:{seat}")
        key = str(seat)
        if key in cleaned:
            raise FactValueError(f"{label}_has_a_duplicate_seat:{key}")
        if not _is_int(stack) or stack < 0:
            raise FactValueError(
                f"{label}_seat_{key}_stack_must_be_a_non_negative_integer")
        cleaned[key] = stack
    return cleaned


def _validate_rules(value):
    if not isinstance(value, dict):
        raise FactValueError("table_rules_must_be_an_object")
    missing = [key for key in RULE_KEYS if key not in value]
    if missing:
        raise FactValueError("table_rules_missing:" + ",".join(sorted(missing)))
    if set(value) != set(RULE_KEYS):
        raise FactValueError("table_rules_unknown_fields")
    if not _is_int(value["table_size"]) or not 6 <= value["table_size"] <= 8:
        raise FactValueError("table_rules_table_size_invalid")
    for key in ("small_blind", "big_blind", "ante", "straddle_amount"):
        if not _is_int(value[key]) or value[key] < 0:
            raise FactValueError(f"table_rules_{key}_must_be_a_non_negative_int")
    if not 0 < value["small_blind"] <= value["big_blind"]:
        raise FactValueError("table_rules_positive_ordered_blinds_required")
    if value["ante_mode"] not in ANTE_MODES:
        raise FactValueError("table_rules_ante_mode_unknown")
    if value["straddle_mode"] not in STRADDLE_MODES:
        raise FactValueError("table_rules_straddle_mode_unknown")
    if value["ante_mode"] == "none" and value["ante"] != 0:
        raise FactValueError("table_rules_none_ante_must_be_zero")
    if ((value["straddle_mode"] == "none" and value["straddle_amount"] != 0)
            or (value["straddle_mode"] != "none"
                and value["straddle_amount"] <= value["big_blind"])):
        raise FactValueError("table_rules_straddle_amount_mode_mismatch")
    if not _is_non_empty_str(value["revision"]):
        raise FactValueError("table_rules_revision_required")
    return dict(value)


def _validate_fact_value(fact_key, value):
    """Structural validation. A free-text note can never satisfy any of these.

    The intent is not cosmetic: if a reviewer's prose could occupy ``value``
    then ``confirmed_view`` would later hand that prose to the kernel as a fact.
    """
    if fact_key == "ended_hand_confirmed":
        if value is not True:
            raise FactValueError("ended_hand_confirmed_must_be_true")
        return True
    if fact_key in ("street_wagers_zero", "no_side_pot", "no_pending_action",
                    "hero_is_first_river_actor"):
        if value is not True:
            raise FactValueError(f"{fact_key}_must_be_true")
        return True
    if fact_key in ("hero_seat", "dealer_button_seat"):
        if not _is_seat(value):
            raise FactValueError(f"{fact_key}_must_be_a_seat_integer")
        return int(value)
    if fact_key == "hero_cards":
        return _validate_cards(value, 2, "hero_cards")
    if fact_key == "board_cards":
        return _validate_cards(value, 5, "board_cards")
    if fact_key == "seat_set":
        return _validate_seat_list(value, "seat_set")
    if fact_key == "active_seats":
        return _validate_seat_list(value, "active_seats")
    if fact_key == "all_in_seats":
        return _validate_seat_list(value, "all_in_seats", allow_empty=True)
    if fact_key == "folded_seats":
        return _validate_seat_list(value, "folded_seats", allow_empty=True)
    if fact_key == "action_order":
        return _validate_seat_list(value, "action_order")
    if fact_key in ("river_start_stacks", "opening_stacks"):
        return _validate_stack_map(value, fact_key)
    if fact_key == "stack_delta_assertions":
        if not isinstance(value, dict):
            raise FactValueError("stack_delta_assertions_must_be_an_object")
        if set(value) != set(STACK_DELTA_ASSERTIONS + STACK_DELTA_BINDINGS):
            raise FactValueError("stack_delta_assertions_exact_schema_required")
        missing = [key for key in STACK_DELTA_ASSERTIONS
                   if value.get(key) is not True]
        if missing:
            raise FactValueError(
                "stack_delta_assertions_not_confirmed:" + ",".join(missing))
        if not all(_is_digest(value[key]) for key in STACK_DELTA_BINDINGS):
            raise FactValueError("stack_delta_interval_bindings_required")
        return dict(value)
    if fact_key == "pot_display":
        if not isinstance(value, dict) or set(value) != {"raw", "value"}:
            raise FactValueError("pot_display_must_be_an_object")
        if not _is_non_empty_str(value.get("raw")):
            raise FactValueError("pot_display_raw_string_required")
        if not _is_int(value.get("value")) or value["value"] < 0:
            raise FactValueError("pot_display_value_must_be_a_non_negative_int")
        return {"raw": value["raw"], "value": value["value"]}
    if fact_key == "table_rules":
        return _validate_rules(value)
    if fact_key == "other_fees":
        if not isinstance(value, dict) or set(value) != {"value", "confirmed"}:
            raise FactValueError("other_fees_must_be_an_object")
        if value.get("confirmed") is not True:
            raise FactValueError(
                "other_fees_not_confirmed:an_unknown_fee_must_not_default_to_0")
        if not _is_int(value.get("value")) or value["value"] < 0:
            raise FactValueError(
                "other_fees_value_must_be_a_non_negative_integer")
        return {"value": value["value"], "confirmed": True}
    if fact_key == "straddle_posted_this_hand":
        if not isinstance(value, bool):
            raise FactValueError("straddle_posted_this_hand_must_be_a_boolean")
        return value
    raise FactValueError("unknown_fact_key:" + str(fact_key))


def _validate_record(row):
    """The immutable semantic contract shared by append and recovery."""
    if not isinstance(row, dict) or set(row) != RECORD_KEYS:
        raise StoreIntegrityError("confirmation_record_exact_schema_required")
    if row["schema_version"] != SCHEMA_VERSION:
        raise StoreIntegrityError("unsupported_confirmation_schema")
    for key in ("confirmation_id", "hand_id", "observed_epoch",
                "capture_session_id", "reviewer"):
        if not _is_non_empty_str(row[key]):
            raise EvidenceBindingError("confirmation_identity_required:" + key)
    if row["provenance"] != PROV_HUMAN:
        raise FactValueError("confirmation_provenance_must_be_human_confirmed")
    if (row["audit_note"] is not None
            and not isinstance(row["audit_note"], str)):
        raise FactValueError("audit_note_must_be_a_string_or_absent")
    try:
        stamp = datetime.fromisoformat(row["recorded_at"])
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("timezone_required")
    except (TypeError, ValueError) as exc:
        raise EvidenceBindingError("recorded_at_timezone_timestamp_required") from exc
    if not _is_int(row["revision"]) or row["revision"] < 1:
        raise RevisionError("positive_integer_revision_required")
    if row["supersedes"] is not None and not _is_non_empty_str(row["supersedes"]):
        raise RevisionError("supersedes_must_be_a_confirmation_id")
    identity, source_digest = _source_identity(row["source_ref"])
    if source_digest != row["source_digest"]:
        raise IdentityMismatchError("source_digest_does_not_match_source_ref")
    descriptor = _validate_evidence(row["evidence_descriptor"])
    for key in ("hand_id", "observed_epoch", "capture_session_id", "marker"):
        if row[key] != descriptor[key]:
            raise IdentityMismatchError("evidence_identity_mismatch:" + key)
    if identity != descriptor["source_ref"]:
        raise IdentityMismatchError("evidence_identity_mismatch:source_ref")
    seal = digest(descriptor)
    if row["evidence_digest"] != seal or row["snapshot_digest"] != seal:
        raise EvidenceBindingError("evidence_descriptor_digest_mismatch")
    if (not _is_int(row["source_frame"]) or row["source_frame"] not in
            [frame["source_frame"] for frame in descriptor["frames"]]):
        raise EvidenceBindingError("confirmation_frame_not_bound_by_evidence")
    key = row["fact_key"]
    if not isinstance(key, str) or key not in FACT_KEYS:
        raise FactValueError("unknown_fact_key:" + str(key))
    if row["marker"] != FACT_MARKERS[key]:
        raise EvidenceBindingError("fact_marker_role_mismatch:" + key)
    cleaned = _validate_fact_value(key, row["value"])
    if canonical(cleaned) != canonical(row["value"]):
        raise FactValueError("stored_fact_value_is_not_canonical:" + key)
    return row


def _validate_histories(rows):
    identities = {}
    groups = {}
    for row in rows:
        hand = row["hand_id"]
        identity = tuple(row[key] for key in (
            "observed_epoch", "capture_session_id", "source_digest"))
        if hand in identities and identities[hand] != identity:
            raise StoreIntegrityError("confirmation:inconsistent_hand_identity")
        identities[hand] = identity
        groups.setdefault((hand, row["fact_key"]), []).append(row)
    for (hand, key), history in groups.items():
        resolved = resolve_fact(history)
        if resolved["status"] != STATUS_CONFIRMED:
            raise StoreIntegrityError(
                f"confirmation:invalid_history:{hand}:{key}:"
                + resolved["reason"])


def confirm_fact(store, *, hand_id, observed_epoch, capture_session_id,
                 source_ref, marker, source_frame, evidence_digest, fact_key,
                 value, reviewer, evidence_descriptor, snapshot_digest=None,
                 supersedes=None, audit_note=None, recorded_at=None):
    """Append one human confirmation. Refuse anything not bound to evidence.

    The stored row keeps the reviewer's own words in ``audit_note`` and nowhere
    else: the note is carried for audit and is never read as the fact.
    """
    if not isinstance(store, ConfirmationStore):
        raise RealHandConfirmationError("confirmation_store_required")
    if not _is_non_empty_str(hand_id):
        raise IdentityMismatchError("hand_id_required")
    if not _is_non_empty_str(observed_epoch):
        raise IdentityMismatchError("observed_epoch_required")
    if not _is_non_empty_str(capture_session_id):
        raise IdentityMismatchError("capture_session_id_required")
    if marker not in MARKERS:
        raise EvidenceBindingError("unknown_marker:" + str(marker))
    if not _is_int(source_frame) or source_frame < 0:
        raise EvidenceBindingError("source_frame_must_be_a_non_negative_int")
    if not isinstance(evidence_digest, str) or not HEX64.match(evidence_digest):
        raise EvidenceBindingError(
            "evidence_digest_required:a_confirmation_without_a_bound_evidence_"
            "digest_cannot_be_confirmed")
    if snapshot_digest is not None and (
            not isinstance(snapshot_digest, str)
            or not HEX64.match(snapshot_digest)):
        raise EvidenceBindingError("snapshot_digest_invalid")
    if not _is_non_empty_str(reviewer):
        raise EvidenceBindingError("reviewer_required")
    if fact_key not in FACT_KEYS:
        raise FactValueError("unknown_fact_key:" + str(fact_key))
    if audit_note is not None and not isinstance(audit_note, str):
        raise FactValueError("audit_note_must_be_a_string_or_absent")
    cleaned_value = _validate_fact_value(fact_key, value)
    identity, source_digest = _source_identity(source_ref)

    with store.lock():
        rows = store.read_confirmations()
        same_hand = [row for row in rows if row.get("hand_id") == hand_id]
        for row in same_hand:
            if row.get("observed_epoch") != observed_epoch:
                raise IdentityMismatchError(
                    "observed_epoch_mismatch:" + str(row.get("observed_epoch")))
            if row.get("capture_session_id") != capture_session_id:
                raise IdentityMismatchError(
                    "capture_session_id_mismatch:"
                    + str(row.get("capture_session_id")))
            if row.get("source_digest") != source_digest:
                raise IdentityMismatchError("source_recording_mismatch")
        existing = [row for row in same_hand if row.get("fact_key") == fact_key]
        heads = _heads(existing)
        if supersedes is None:
            if existing:
                raise RevisionError(
                    "revision_required:this_fact_already_has_a_confirmation_"
                    "and_must_supersede_it")
            revision = 1
        else:
            if not _is_non_empty_str(supersedes):
                raise RevisionError("supersedes_must_be_a_confirmation_id")
            if not heads or supersedes != heads[0].get("confirmation_id"):
                raise RevisionError(
                    "stale_supersedes:the_named_confirmation_is_not_the_"
                    "current_head")
            revision = heads[0].get("revision", 0) + 1
        row = {
            "schema_version": SCHEMA_VERSION,
            "confirmation_id": "rhc-" + uuid.uuid4().hex,
            "hand_id": hand_id,
            "observed_epoch": observed_epoch,
            "capture_session_id": capture_session_id,
            "source_ref": identity,
            "source_digest": source_digest,
            "marker": marker,
            "source_frame": source_frame,
            "evidence_descriptor": deepcopy(evidence_descriptor),
            "evidence_digest": evidence_digest,
            "snapshot_digest": (evidence_digest if snapshot_digest is None
                                else snapshot_digest),
            "fact_key": fact_key,
            "value": cleaned_value,
            "reviewer": reviewer,
            "recorded_at": now() if recorded_at is None else recorded_at,
            "revision": revision,
            "supersedes": supersedes,
            "provenance": PROV_HUMAN,
            "audit_note": audit_note,
        }
        _validate_record(row)
        _validate_histories(rows + [row])
        store._append(row)
    return deepcopy(row)


def _heads(rows):
    """Rows no other row supersedes."""
    if not rows:
        return []
    superseded = {row.get("supersedes") for row in rows
                  if row.get("supersedes")}
    return [row for row in rows
            if row.get("confirmation_id") not in superseded]


def resolve_fact(rows):
    """Collapse one fact's confirmation history to a status and a value.

    Two live heads, a duplicate revision, a cycle or a head that is not the
    highest revision are all CONFLICT: an ambiguous fact is not a fact.
    """
    if not rows:
        return {"status": STATUS_UNCONFIRMED, "value": None,
                "reason": "no_confirmation_recorded", "confirmation_id": None,
                "conflicts": []}
    identifiers = [row.get("confirmation_id") if isinstance(row, dict)
                   else None for row in rows]

    def conflict(reason):
        return {"status": STATUS_CONFLICT, "value": None, "reason": reason,
                "confirmation_id": None,
                "conflicts": sorted(str(item) for item in identifiers)}

    try:
        for row in rows:
            _validate_record(row)
    except (RealHandConfirmationError, TypeError, ValueError) as exc:
        return conflict("invalid_record:" + str(exc))
    if len(set(identifiers)) != len(rows):
        return conflict("duplicate_confirmation_id")
    if len({tuple(row[key] for key in (
            "hand_id", "observed_epoch", "capture_session_id", "source_digest",
            "fact_key")) for row in rows}) != 1:
        return conflict("mixed_fact_or_hand_identity")
    if len({row["revision"] for row in rows}) != len(rows):
        return conflict("duplicate_revision")
    indexed = {row["confirmation_id"]: row for row in rows}
    positions = {identifier: index for index, identifier in enumerate(identifiers)}
    roots = [row for row in rows if row["supersedes"] is None]
    if len(roots) != 1:
        return conflict("one_revision_one_root_required")
    if roots[0]["revision"] != 1:
        return conflict("root_revision_must_be_one")
    children = {}
    for row in rows:
        parent = row["supersedes"]
        if parent is None:
            continue
        if parent not in indexed:
            return conflict("supersedes_unknown_confirmation")
        if parent in children:
            return conflict("two_revisions_supersede_the_same_parent")
        if row["revision"] != indexed[parent]["revision"] + 1:
            return conflict("revision_must_increment_parent_by_one")
        if positions[parent] >= positions[row["confirmation_id"]]:
            return conflict("parent_must_precede_child")
        children[parent] = row["confirmation_id"]
    visited = set()
    head = roots[0]
    while True:
        identifier = head["confirmation_id"]
        if identifier in visited:
            return conflict("revision_cycle")
        visited.add(identifier)
        if identifier not in children:
            break
        head = indexed[children[identifier]]
    if len(visited) != len(rows):
        return conflict("disconnected_revision_history")
    return {"status": STATUS_CONFIRMED, "value": head.get("value"),
            "reason": None, "confirmation_id": head.get("confirmation_id"),
            "conflicts": []}


def derive_hand_committed_by_stack_delta(*, opening_stacks, river_start_stacks,
                                         assertions, pot_display, seats=None,
                                         opening_evidence_digest=None,
                                         river_evidence_digest=None):
    """Per-seat river-start commitment as ``opening - river_start``.

    Allowed only as a CONFIRMED DERIVATION, and only when every condition
    behind it is itself confirmed. It confirms the aggregate each seat has put
    in; it does NOT decompose into ante / SB / BB / straddle / per-street, so it
    is not a substitute for the forced-bet and action facts a full-hand PHH
    needs.

    Nothing here averages, back-fills or zeroes. A single failed condition
    leaves the whole derivation UNCONFIRMED.
    """
    reasons = []
    if not isinstance(assertions, dict):
        return _unconfirmed(["stack_delta_assertions_missing"], opening_stacks,
                            river_start_stacks)
    for key in STACK_DELTA_ASSERTIONS:
        if assertions.get(key) is not True:
            reasons.append("stack_delta_assertion_not_confirmed:" + key)
    try:
        _validate_fact_value("stack_delta_assertions", assertions)
    except FactValueError as exc:
        reasons.append("stack_delta_assertions_invalid:" + str(exc))
    for key, expected in zip(STACK_DELTA_BINDINGS, (
            opening_evidence_digest, river_evidence_digest)):
        if not _is_digest(expected) or assertions.get(key) != expected:
            reasons.append("stack_delta_interval_binding_mismatch:" + key)
    opening = _stack_map_or_none(opening_stacks, "opening_stacks", reasons)
    river = _stack_map_or_none(river_start_stacks, "river_start_stacks",
                               reasons)
    if opening is not None and river is not None:
        if set(opening) != set(river):
            reasons.append("stack_delta_seat_identity_mismatch")
        else:
            if seats is not None:
                try:
                    wanted = {str(seat) for seat in
                              _validate_seat_list(seats, "seats")}
                except FactValueError as exc:
                    wanted = None
                    reasons.append("stack_delta_seats_invalid:" + str(exc))
                if set(opening) != wanted:
                    reasons.append(
                        "stack_delta_seats_do_not_match_the_confirmed_seat_set")
            deltas = {}
            for seat in sorted(opening):
                delta = opening[seat] - river[seat]
                if delta < 0:
                    reasons.append(f"stack_delta_negative:{seat}")
                deltas[str(seat)] = delta
            pot = _pot_value(pot_display)
            if pot is None:
                reasons.append("stack_delta_pot_display_unconfirmed")
            elif sum(deltas.values()) != pot:
                reasons.append(
                    "stack_delta_pot_mismatch:"
                    f"sum={sum(deltas.values())} pot={pot}")
            if not reasons:
                return {
                    "status": STATUS_CONFIRMED,
                    "per_seat": deltas,
                    "total": sum(deltas.values()),
                    "provenance": PROV_DERIVED,
                    "derivation": DERIVATION_STACK_DELTA,
                    "reasons": [],
                    "digest": digest(deltas),
                }
    return _unconfirmed(reasons, opening_stacks, river_start_stacks)


def _unconfirmed(reasons, opening_stacks, river_start_stacks):
    return {"status": STATUS_UNCONFIRMED, "per_seat": None, "total": None,
            "provenance": PROV_DERIVED, "derivation": DERIVATION_STACK_DELTA,
            "reasons": sorted(set(reasons)),
            "digest": None}


def _stack_map_or_none(value, label, reasons):
    try:
        return _validate_stack_map(value, label)
    except FactValueError as error:
        reasons.append(f"{label}_invalid:{error}")
        return None


def _pot_value(pot_display):
    if isinstance(pot_display, dict):
        value = pot_display.get("value")
        return value if _is_int(value) and value >= 0 else None
    return None


def _structural_checks(facts):
    """Cross-fact checks a single confirmation cannot carry on its own."""
    checks = {}
    blocking = []
    hero = facts.get("hero_seat")
    active = facts.get("active_seats") or []
    all_in = facts.get("all_in_seats") or []
    folded = facts.get("folded_seats") or []
    order = facts.get("action_order") or []
    seat_set = facts.get("seat_set") or []
    board = facts.get("board_cards") or []
    hero_cards = facts.get("hero_cards") or []
    opening = facts.get("opening_stacks") or {}
    river = facts.get("river_start_stacks") or {}

    checks["hero_seat_is_active"] = _is_int(hero) and hero in active
    if not checks["hero_seat_is_active"]:
        blocking.append("hero_seat_is_not_in_the_active_set")
    checks["exactly_three_active"] = len(active) == 3
    if not checks["exactly_three_active"]:
        blocking.append("river_start_must_have_exactly_three_active_players")
    checks["all_in_not_counted_as_active"] = not (set(all_in) & set(active))
    if not checks["all_in_not_counted_as_active"]:
        blocking.append("an_all_in_seat_was_counted_as_active")
    checks["no_all_in_seats"] = not all_in
    if not checks["no_all_in_seats"]:
        blocking.append("any_all_in_seat_is_outside_target_s_scope")
    checks["participant_states_are_complete_and_disjoint"] = (
        set(active) | set(all_in) | set(folded)) == set(seat_set) and not (
            set(active) & set(folded) or set(all_in) & set(folded)
            or set(active) & set(all_in))
    if not checks["participant_states_are_complete_and_disjoint"]:
        blocking.append("participant_states_must_partition_the_seat_set")
    checks["hero_is_not_folded_or_all_in"] = (
        hero not in folded and hero not in all_in)
    if not checks["hero_is_not_folded_or_all_in"]:
        blocking.append("hero_has_a_confirmed_folded_or_all_in_status")
    checks["active_stacks_are_positive"] = bool(active) and all(
        _is_int(river.get(str(seat))) and river[str(seat)] > 0
        for seat in active)
    if not checks["active_stacks_are_positive"]:
        blocking.append("every_active_seat_requires_positive_remaining_stack")
    rules = facts.get("table_rules") or {}
    checks["complete_dealt_seat_set"] = (
        len(seat_set) in (6, 7, 8) and len(seat_set) == rules.get("table_size"))
    if not checks["complete_dealt_seat_set"]:
        blocking.append("seat_set_must_cover_all_six_to_eight_dealt_seats")
    checks["dealer_is_in_seat_set"] = facts.get("dealer_button_seat") in seat_set
    if not checks["dealer_is_in_seat_set"]:
        blocking.append("dealer_button_seat_must_belong_to_the_seat_set")
    checks["action_order_matches_active"] = (
        sorted(order) == sorted(active) and len(order) == len(active))
    if not checks["action_order_matches_active"]:
        blocking.append("action_order_is_not_a_permutation_of_the_active_set")
    dealer = facts.get("dealer_button_seat")
    clockwise = (sorted(active, key=lambda seat: (seat - dealer) % 8 or 8)
                 if _is_seat(dealer) else None)
    checks["action_order_matches_clockwise_dealer"] = (
        bool(active) and order == clockwise)
    if not checks["action_order_matches_clockwise_dealer"]:
        blocking.append("action_order_disagrees_with_dealer_and_clockwise_active_seats")
    checks["hero_acts_first"] = bool(order) and order[0] == hero
    if not checks["hero_acts_first"]:
        blocking.append("hero_is_not_first_in_the_action_order")
    checks["board_has_five_distinct_cards"] = (
        len(board) == 5 and len({str(item) for item in board}) == 5)
    if not checks["board_has_five_distinct_cards"]:
        blocking.append("board_is_not_five_distinct_cards")
    checks["hero_cards_disjoint_from_board"] = not (
        set(str(item) for item in hero_cards)
        & set(str(item) for item in board))
    if not checks["hero_cards_disjoint_from_board"]:
        blocking.append("hero_cards_overlap_the_board")
    checks["seat_set_covers_players"] = (
        set(active) | set(all_in) | set(folded)) <= set(seat_set)
    if not checks["seat_set_covers_players"]:
        blocking.append("seat_set_does_not_cover_every_reported_seat")
    checks["stack_maps_cover_the_seat_set"] = (
        set(int(seat) for seat in opening) == set(seat_set)
        and set(int(seat) for seat in river) == set(seat_set))
    if not checks["stack_maps_cover_the_seat_set"]:
        blocking.append("opening_and_river_start_stacks_must_cover_the_seat_set")
    return checks, blocking


def _straddle_check(facts):
    """A per-hand straddle is an observation, never a rule lookup."""
    rules = facts.get("table_rules") or {}
    mode = rules.get("straddle_mode")
    if mode == "optional_explicit_utg":
        if not isinstance(facts.get("straddle_posted_this_hand"), bool):
            return False, ("optional_straddle_was_not_observed_for_this_hand:"
                           "the_amount_is_declared_but_the_posting_is_not")
        return True, None
    posted = facts.get("straddle_posted_this_hand")
    if mode == "none" and posted is not False:
        return False, "straddle_posted_conflicts_with_none_rule"
    if mode == "mandatory_utg" and posted is not True:
        return False, "straddle_not_posted_conflicts_with_mandatory_rule"
    return True, None


def _marker_checks(evidence_refs):
    windows = {}
    blocking = []
    for marker in MARKERS:
        refs = [ref for ref in evidence_refs.values()
                if ref["marker"] == marker]
        if not refs:
            continue
        if len({ref["evidence_digest"] for ref in refs}) != 1:
            blocking.append("incompatible_evidence_windows:" + marker)
        else:
            windows[marker] = refs[0]["evidence_descriptor"]
    for left, right in ((MARKER_HAND_START, MARKER_RIVER_START),
                        (MARKER_RIVER_START, MARKER_PAYOUT)):
        if left in windows and right in windows and not (
                windows[left]["window_end_frame"]
                < windows[right]["window_start_frame"]):
            blocking.append("marker_chronology_invalid:" + left + ":" + right)
    return blocking


def confirmed_view(store, hand_id):
    """Re-derive the whole TARGET-S fact view from the append-only history.

    Nothing here trusts a stored verdict: every fact is resolved from its
    confirmation rows, every derived value recomputed, and every structural
    check re-run.
    """
    if not isinstance(store, ConfirmationStore):
        raise RealHandConfirmationError("confirmation_store_required")
    if not _is_non_empty_str(hand_id):
        raise IdentityMismatchError("hand_id_required")
    store_error = None
    try:
        rows = [row for row in store.read_confirmations()
                if row.get("hand_id") == hand_id]
    except StoreIntegrityError as exc:
        rows = []
        store_error = str(exc)
    identity = rows[0] if rows else None
    confirmed = {}
    unconfirmed = {}
    evidence_refs = {}
    conflicts = ([{"fact_key": None, "reason": store_error,
                   "confirmation_ids": []}] if store_error else [])
    for key in TARGET_S_REQUIRED_FACTS:
        resolved = resolve_fact([row for row in rows
                                 if row.get("fact_key") == key])
        if resolved["status"] == STATUS_CONFIRMED:
            confirmed[key] = resolved["value"]
            evidence_refs[key] = _evidence_ref(
                [row for row in rows if row.get("fact_key") == key],
                resolved["confirmation_id"])
        else:
            unconfirmed[key] = (resolved["reason"]
                                or "not_confirmed")
            if resolved["status"] == STATUS_CONFLICT:
                conflicts.append({"fact_key": key,
                                  "reason": resolved["reason"],
                                  "confirmation_ids": resolved["conflicts"]})

    checks, blocking = _structural_checks(confirmed)
    marker_blocking = _marker_checks(evidence_refs)
    blocking.extend(marker_blocking)
    if store_error:
        blocking.append("store_integrity_error:" + store_error)
    derivations = {}
    derived = {}
    if marker_blocking:
        unconfirmed["hand_committed"] = ",".join(marker_blocking)
    elif "opening_stacks" in confirmed and "river_start_stacks" in confirmed:
        commitment = derive_hand_committed_by_stack_delta(
            opening_stacks=confirmed["opening_stacks"],
            river_start_stacks=confirmed["river_start_stacks"],
            assertions=confirmed.get("stack_delta_assertions"),
            pot_display=confirmed.get("pot_display"),
            seats=confirmed.get("seat_set"),
            opening_evidence_digest=evidence_refs["opening_stacks"]["evidence_digest"],
            river_evidence_digest=(
                evidence_refs["river_start_stacks"]["evidence_digest"]))
        derivations["hand_committed"] = commitment
        if commitment["status"] == STATUS_CONFIRMED:
            derived["hand_committed"] = commitment["per_seat"]
        else:
            unconfirmed["hand_committed"] = ",".join(commitment["reasons"])
    else:
        unconfirmed.setdefault(
            "hand_committed", "stack_delta_source_snapshots_unconfirmed")

    # The empty river history is a legal TARGET-S P0 design, but only because
    # Hero is confirmed first-to-act -- not because a caller wrote [].
    if (confirmed.get("hero_is_first_river_actor") is True and not blocking
            and confirmed.get("no_pending_action") is True
            and confirmed.get("street_wagers_zero") is True):
        derived["history"] = []
        derived["history_empty_because"] = "hero_is_first_river_actor"
    else:
        unconfirmed["history"] = ("river_history_is_only_empty_when_hero_is_"
                                  "confirmed_first_river_actor")

    straddle_ok, straddle_reason = _straddle_check(confirmed)
    if not straddle_ok:
        blocking.append(straddle_reason)
    # A side pot is also checked against the derived numbers, so ticking
    # ``no_side_pot`` cannot carry a hand whose own commitments disagree.
    if "hand_committed" in derived:
        levels = {derived["hand_committed"][str(seat)]
                  for seat in confirmed.get("active_seats") or []
                  if str(seat) in derived["hand_committed"]}
        if len(levels) != 1:
            blocking.append(
                "active_commitments_are_not_equal:side_pot_present")
        else:
            level = next(iter(levels))
            if any(value > level
                   for value in derived["hand_committed"].values()):
                blocking.append(
                    "a_seat_committed_above_the_active_level:side_pot_present")
    if confirmed.get("no_side_pot") is not True:
        blocking.append("no_side_pot_not_confirmed")
    if confirmed.get("street_wagers_zero") is not True:
        blocking.append("street_wagers_are_not_all_zero_at_river_start")
    if confirmed.get("no_pending_action") is not True:
        blocking.append("a_pending_action_or_animation_is_still_unresolved")
    if confirmed.get("ended_hand_confirmed") is not True:
        blocking.append("the_hand_is_not_confirmed_ended")

    ready = not unconfirmed and not conflicts and not blocking
    bundle = {
        "schema": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "target": TARGET,
        "scope": SCOPE,
        "hand_id": hand_id,
        "observed_epoch": (identity or {}).get("observed_epoch"),
        "capture_session_id": (identity or {}).get("capture_session_id"),
        "source_digest": (identity or {}).get("source_digest"),
        "status": "READY" if ready else "NOT_READY",
        "confirmed": confirmed,
        "derived": derived,
        "unconfirmed": unconfirmed,
        "conflicts": conflicts,
        "blocking": sorted(set(blocking)),
        "checks": checks,
        "derivations": derivations,
        "evidence_refs": evidence_refs,
    }
    bundle["bundle_digest"] = digest(bundle)
    return bundle


def _evidence_ref(rows, confirmation_id):
    for row in rows:
        if row.get("confirmation_id") == confirmation_id:
            return {"confirmation_id": row.get("confirmation_id"),
                    "schema_version": row["schema_version"],
                    "hand_id": row["hand_id"],
                    "observed_epoch": row["observed_epoch"],
                    "capture_session_id": row["capture_session_id"],
                    "source_ref": deepcopy(row["source_ref"]),
                    "source_digest": row["source_digest"],
                    "marker": row.get("marker"),
                    "source_frame": row.get("source_frame"),
                    "evidence_descriptor": deepcopy(row["evidence_descriptor"]),
                    "evidence_digest": row.get("evidence_digest"),
                    "snapshot_digest": row.get("snapshot_digest"),
                    "reviewer": row.get("reviewer"),
                    "recorded_at": row.get("recorded_at"),
                    "revision": row.get("revision"),
                    "supersedes": row["supersedes"],
                    "record_digest": digest(row),
                    "ancestry_digests": [digest(item) for item in rows],
                    "provenance": row.get("provenance"),
                    "audit_note": row.get("audit_note")}
    return None


def target_s_acceptance(store, hand_id):
    """The one gate that can produce ``TARGET_S_REAL_HAND_CONFIRMED``.

    The receipt proves factual readiness of a real hand for the threeway river
    contract. It deliberately does not prove PHH readiness and does not assess
    strategy: opponent ranges and response models remain manual assumptions.
    """
    view = confirmed_view(store, hand_id)
    receipt = {
        "schema": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "target": TARGET,
        "scope": SCOPE,
        "hand_id": hand_id,
        "observed_epoch": view["observed_epoch"],
        "capture_session_id": view["capture_session_id"],
        "source_digest": view["source_digest"],
        "confirmation_bundle_digest": view["bundle_digest"],
        "river_start_snapshot_digest": _snapshot_digest(view),
        "derived_commitment_digest": (
            (view["derivations"].get("hand_committed") or {}).get("digest")),
        "rules_digest": (digest(view["confirmed"]["table_rules"])
                         if "table_rules" in view["confirmed"] else None),
        "facts_digest": digest(view["confirmed"]),
        "phh_ready": PHH_READY,
        "strategy_assessed": STRATEGY_ASSESSED,
        "target_p_phh_ready": False,
        "manual_hypothesis_scope": MANUAL_HYPOTHESIS_SCOPE,
    }
    if view["status"] == "READY":
        receipt["status"] = ACCEPTED
        receipt["missing"] = []
    else:
        receipt["status"] = NOT_READY
        receipt["missing"] = sorted(view["unconfirmed"])
        receipt["blocking"] = view["blocking"]
        receipt["conflicts"] = view["conflicts"]
    receipt["receipt_digest"] = digest(receipt)
    return receipt


def _snapshot_digest(view):
    seals = {ref["evidence_digest"] for ref in view["evidence_refs"].values()
             if ref["marker"] == MARKER_RIVER_START}
    return next(iter(seals)) if len(seals) == 1 else None


def validate_target_s_receipt(store, hand_id, receipt):
    """A receipt authorizes no fact without matching the current valid history.

    Never validate only the self-reported digest or stored status. V1 receipts,
    changed flags, evidence-only revisions and malformed stores all return False.
    This is a point-in-time check; consumers must revalidate at consumption.
    """
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA_VERSION:
        return False
    try:
        current = target_s_acceptance(store, hand_id)
        return (current["status"] == ACCEPTED
                and canonical(current) == canonical(receipt))
    except (RealHandConfirmationError, TypeError, ValueError):
        return False
