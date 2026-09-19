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

SCHEMA_VERSION = "aa-real-hand-confirmation-v1"
IMPLEMENTATION_VERSION = "aa-real-hand-confirmation-p0-target-s-v1"

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
    "no_chip_side_fee_or_rake",
    "same_integer_precision",
)

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
    "seat_set", "active_seats", "all_in_seats", "action_order",
    "dealer_button_seat", "river_start_stacks", "opening_stacks",
    "stack_delta_assertions", "street_wagers_zero", "pot_display",
    "hero_is_first_river_actor", "table_rules", "other_fees",
    "straddle_posted_this_hand", "no_side_pot", "no_pending_action",
)


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
                      ensure_ascii=False)


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
    media = ref.get("media_sha256")
    recording = ref.get("recording_id")
    if not (_is_non_empty_str(media) or _is_non_empty_str(recording)):
        raise EvidenceBindingError(
            "source_ref_needs_media_sha256_or_recording_id")
    if media is not None and not (isinstance(media, str) and HEX64.match(media)):
        raise EvidenceBindingError("source_ref_media_sha256_invalid")
    cleaned = {}
    for key in sorted(ref):
        item = ref[key]
        if item is None or isinstance(item, (str, int, float, bool)):
            cleaned[key] = item
    if not cleaned:
        raise EvidenceBindingError("source_ref_has_no_identity_field")
    return cleaned, digest(cleaned)


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
        raw_text = self._confirmations.read_text(encoding="utf-8")
        for lineno, raw in enumerate(raw_text.splitlines(), start=1):
            if raw.strip() == "":
                raise StoreIntegrityError(f"confirmation:blank_line:{lineno}")
            try:
                envelope = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise StoreIntegrityError(
                    f"confirmation:malformed_json:{lineno}") from exc
            if not isinstance(envelope, dict):
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
            if digest(row) != envelope.get("record_digest"):
                raise StoreIntegrityError(
                    f"confirmation:record_digest_mismatch:{lineno}")
            if envelope.get("prev_digest") != previous:
                raise StoreIntegrityError(
                    f"confirmation:chain_break:{lineno}")
            previous = envelope.get("record_digest")
            rows.append(row)
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
        try:
            key = int(seat)
        except (TypeError, ValueError) as exc:
            raise FactValueError(f"{label}_has_a_non_integer_seat:{seat}") from exc
        if not _is_seat(key):
            raise FactValueError(f"{label}_has_an_out_of_range_seat:{seat}")
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
    if not _is_int(value["table_size"]) or value["table_size"] < 2:
        raise FactValueError("table_rules_table_size_invalid")
    for key in ("small_blind", "big_blind", "ante", "straddle_amount"):
        if not _is_int(value[key]) or value[key] < 0:
            raise FactValueError(f"table_rules_{key}_must_be_a_non_negative_int")
    if value["small_blind"] > value["big_blind"]:
        raise FactValueError("table_rules_small_blind_exceeds_big_blind")
    if value["ante_mode"] not in ANTE_MODES:
        raise FactValueError("table_rules_ante_mode_unknown")
    if value["straddle_mode"] not in STRADDLE_MODES:
        raise FactValueError("table_rules_straddle_mode_unknown")
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
        missing = [key for key in STACK_DELTA_ASSERTIONS
                   if value.get(key) is not True]
        if missing:
            raise FactValueError(
                "stack_delta_assertions_not_confirmed:" + ",".join(missing))
        return {key: True for key in STACK_DELTA_ASSERTIONS}
    if fact_key == "pot_display":
        if not isinstance(value, dict):
            raise FactValueError("pot_display_must_be_an_object")
        if not _is_non_empty_str(value.get("raw")):
            raise FactValueError("pot_display_raw_string_required")
        if not _is_int(value.get("value")) or value["value"] < 0:
            raise FactValueError("pot_display_value_must_be_a_non_negative_int")
        return {"raw": value["raw"], "value": value["value"]}
    if fact_key == "table_rules":
        return _validate_rules(value)
    if fact_key == "other_fees":
        if not isinstance(value, dict):
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


def confirm_fact(store, *, hand_id, observed_epoch, capture_session_id,
                 source_ref, marker, source_frame, evidence_digest, fact_key,
                 value, reviewer, snapshot_digest=None, supersedes=None,
                 audit_note=None, recorded_at=None):
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
            "evidence_digest": evidence_digest,
            "snapshot_digest": snapshot_digest,
            "fact_key": fact_key,
            "value": cleaned_value,
            "reviewer": reviewer,
            "recorded_at": recorded_at or now(),
            "revision": revision,
            "supersedes": supersedes,
            "provenance": PROV_HUMAN,
            "audit_note": audit_note,
        }
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
    identifiers = {row.get("confirmation_id") for row in rows}
    revisions = [row.get("revision") for row in rows]
    if len(set(revisions)) != len(revisions):
        return {"status": STATUS_CONFLICT, "value": None,
                "reason": "duplicate_revision", "confirmation_id": None,
                "conflicts": sorted(str(item) for item in identifiers)}
    parents = {}
    for row in rows:
        parent = row.get("supersedes")
        if parent is None:
            continue
        if parent not in identifiers:
            return {"status": STATUS_CONFLICT, "value": None,
                    "reason": "supersedes_unknown_confirmation",
                    "confirmation_id": None,
                    "conflicts": sorted(str(item) for item in identifiers)}
        if parent in parents:
            return {"status": STATUS_CONFLICT, "value": None,
                    "reason": "two_revisions_supersede_the_same_parent",
                    "confirmation_id": None,
                    "conflicts": sorted(str(item) for item in identifiers)}
        parents[parent] = row.get("confirmation_id")
    heads = _heads(rows)
    if not heads:
        return {"status": STATUS_CONFLICT, "value": None,
                "reason": "no_live_head", "confirmation_id": None,
                "conflicts": sorted(str(item) for item in identifiers)}
    if len(heads) > 1:
        return {"status": STATUS_CONFLICT, "value": None,
                "reason": "multiple_live_heads", "confirmation_id": None,
                "conflicts": sorted(str(item) for item in
                                    (row.get("confirmation_id")
                                     for row in heads))}
    head = heads[0]
    if head.get("revision") != max(revisions):
        return {"status": STATUS_CONFLICT, "value": None,
                "reason": "head_is_not_the_highest_revision",
                "confirmation_id": head.get("confirmation_id"),
                "conflicts": sorted(str(item) for item in identifiers)}
    return {"status": STATUS_CONFIRMED, "value": head.get("value"),
            "reason": None, "confirmation_id": head.get("confirmation_id"),
            "conflicts": []}


def derive_hand_committed_by_stack_delta(*, opening_stacks, river_start_stacks,
                                         assertions, pot_display, seats=None):
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
    opening = _stack_map_or_none(opening_stacks, "opening_stacks", reasons)
    river = _stack_map_or_none(river_start_stacks, "river_start_stacks",
                               reasons)
    if opening is not None and river is not None:
        if set(opening) != set(river):
            reasons.append("stack_delta_seat_identity_mismatch")
        else:
            if seats is not None:
                wanted = {int(seat) for seat in seats}
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
            "digest": digest({"opening": opening_stacks,
                              "river": river_start_stacks,
                              "reasons": sorted(set(reasons))})}


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
    checks["action_order_matches_active"] = (
        sorted(order) == sorted(active) and len(order) == len(active))
    if not checks["action_order_matches_active"]:
        blocking.append("action_order_is_not_a_permutation_of_the_active_set")
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
    return True, None


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
    rows = [row for row in store.read_confirmations()
            if row.get("hand_id") == hand_id]
    identity = rows[0] if rows else None
    confirmed = {}
    unconfirmed = {}
    evidence_refs = {}
    conflicts = []
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

    derivations = {}
    derived = {}
    if "opening_stacks" in confirmed and "river_start_stacks" in confirmed:
        commitment = derive_hand_committed_by_stack_delta(
            opening_stacks=confirmed["opening_stacks"],
            river_start_stacks=confirmed["river_start_stacks"],
            assertions=confirmed.get("stack_delta_assertions"),
            pot_display=confirmed.get("pot_display"),
            seats=confirmed.get("seat_set"))
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
    if confirmed.get("hero_is_first_river_actor") is True:
        derived["history"] = []
        derived["history_empty_because"] = "hero_is_first_river_actor"
    else:
        unconfirmed["history"] = ("river_history_is_only_empty_when_hero_is_"
                                  "confirmed_first_river_actor")

    checks, blocking = _structural_checks(confirmed)
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
    bundle["bundle_digest"] = digest(
        {key: bundle[key] for key in (
            "hand_id", "observed_epoch", "capture_session_id", "source_digest",
            "confirmed", "derived", "unconfirmed", "conflicts", "blocking")})
    return bundle


def _evidence_ref(rows, confirmation_id):
    for row in rows:
        if row.get("confirmation_id") == confirmation_id:
            return {"confirmation_id": row.get("confirmation_id"),
                    "marker": row.get("marker"),
                    "source_frame": row.get("source_frame"),
                    "evidence_digest": row.get("evidence_digest"),
                    "snapshot_digest": row.get("snapshot_digest"),
                    "reviewer": row.get("reviewer"),
                    "recorded_at": row.get("recorded_at"),
                    "revision": row.get("revision"),
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
    receipt["receipt_digest"] = digest(
        {key: receipt[key] for key in (
            "hand_id", "observed_epoch", "capture_session_id", "source_digest",
            "confirmation_bundle_digest", "river_start_snapshot_digest",
            "derived_commitment_digest", "rules_digest", "facts_digest",
            "status")})
    return receipt


def _snapshot_digest(view):
    confirmed = view["confirmed"]
    return digest({
        "hero_seat": confirmed.get("hero_seat"),
        "hero_cards": confirmed.get("hero_cards"),
        "board_cards": confirmed.get("board_cards"),
        "active_seats": confirmed.get("active_seats"),
        "river_start_stacks": confirmed.get("river_start_stacks"),
        "pot_display": confirmed.get("pot_display"),
    })
