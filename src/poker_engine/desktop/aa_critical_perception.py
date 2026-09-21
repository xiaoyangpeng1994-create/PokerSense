"""Source-bound critical AA8 candidates. This module grants no truth authority.

KNOWN means a internally consistent machine candidate, never a confirmed fact
or measured visual accuracy. Historical observations are retained in the row.
"""

from copy import deepcopy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re


FIELDS = ("board", "hero_participation", "participation", "all_in_seats",
          "river_first_actor")
SEMANTICS = "SOURCE_BOUND_MACHINE_CANDIDATES_NOT_TRUTH"
_CARD = re.compile(r"[2-9TJQKA][cdhs]")
_HASH = re.compile(r"[0-9a-f]{64}")
_STATES = {"active": "ACTIVE", "folded": "FOLDED", "all_in": "ALL_IN",
           "waiting": "WAITING", "empty": "EMPTY"}


def _binding(row):
    state = row.get("observed_state_v2") or {}
    return {"source_id": row.get("source_id"), "epoch": state.get("observed_epoch"),
            "frame": row.get("frame"), "source_frame": row.get("source_frame"),
            "pts_seconds": row.get("pts_seconds"),
            "source_sha256": row.get("source_sha256")}


def _valid_binding(binding):
    return (isinstance(binding["source_id"], str) and bool(binding["source_id"])
            and isinstance(binding["epoch"], str) and bool(binding["epoch"])
            and all(type(binding[k]) is int and binding[k] >= 0
                    for k in ("frame", "source_frame"))
            and type(binding["pts_seconds"]) in (int, float)
            and math.isfinite(binding["pts_seconds"]) and binding["pts_seconds"] >= 0
            and isinstance(binding["source_sha256"], str)
            and _HASH.fullmatch(binding["source_sha256"]) is not None)


def _digest(row):
    keys = ("source_id", "source_frame", "frame", "pts_seconds", "source_sha256",
            "scene_supported", "reader_gap_reset", "special_modes", "cards",
            "board_count",
            "observed_state_v2", "participation", "current_actor", "actor_evidence",
            "stacks", "dealer_seat", "dealer_evidence_v2", "dealer_observation_v2",
            "glyphs",
            "observed_actions_v2", "glyph_transitions", "action_history_candidate",
            "causal_street_wagers_v2", "street_wagers", "wager_visibility_v2",
            "observed_center_v2", "pot")
    raw = {key: row.get(key) for key in keys}
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _field(value=None, reason="not_supported", *, conflict=False):
    status = "CONFLICT" if conflict else "UNKNOWN" if value is None else "KNOWN"
    return {"status": status,
            "value": deepcopy(value), "reasons": [] if value is not None else [reason]}


def _blocked(row):
    modes = row.get("special_modes") or {}
    return (row.get("scene_supported") is not True
            or modes.get("block_state_updates") is True
            or modes.get("insurance") == "VISIBLE")


def _board(row):
    state, cards = row.get("observed_state_v2") or {}, row.get("cards") or {}
    stage = state.get("street_candidate")
    size = {"flop": 3, "turn": 4, "river": 5}.get(stage)
    values = cards.get("board_slots")
    geometry = state.get("positive_board_geometry") or {}
    if (size is None or row.get("board_count") != size
            or not isinstance(values, list) or len(values) != 5
            or not all(isinstance(c, str) and _CARD.fullmatch(c) for c in values[:size])
            or any(c is not None for c in values[size:])
            or len(set(values[:size])) != size
            or state.get("board_candidate") != values[:size]
            or geometry.get("last_frame") != row.get("frame")
            or type(geometry.get("streak")) is not int or geometry["streak"] < 2):
        return None
    hero = cards.get("hero") or []
    if not isinstance(hero, list) or set(hero) & set(values[:size]):
        return None
    return values[:size]


def _participation(row):
    state = row.get("observed_state_v2") or {}
    epoch, frame = state.get("observed_epoch"), row.get("frame")
    participants, cues = state.get("participants") or {}, (
        row.get("participation") or {}).get("slots") or {}
    conflicts = set(state.get("critical_status_conflicts") or [])
    result, unknown = {}, []
    for seat in map(str, range(8)):
        item, cue = participants.get(seat) or {}, cues.get(seat) or {}
        value = item.get("state")
        if cue.get("conflict") or seat in conflicts:
            return None, "participation_conflict:" + seat, True
        evidence_frame = item.get("evidence_frame")
        if (item.get("epoch") != epoch or value not in _STATES
                or type(evidence_frame) is not int or evidence_frame > frame):
            unknown.append(seat)
            continue
        # Terminal events persist within a hand. Active/occupancy observations
        # require current positive support; historical ACTIVE is not currentness.
        if value not in ("folded", "all_in") and evidence_frame != frame:
            unknown.append(seat)
            continue
        if value in ("active", "all_in"):
            stack = (row.get("stacks") or {}).get(seat) or {}
            try:
                text = stack.get("value")
                number = (Decimal(text) if isinstance(text, str) and len(text) <= 64
                          else None)
            except (InvalidOperation, TypeError, ValueError):
                number = None
            if (number is None or not number.is_finite()
                    or (number <= 0 if value == "active" else number != 0)):
                unknown.append(seat)
                continue
        result[seat] = _STATES[value]
    return result, "current_participation_missing:" + ",".join(unknown), False


def _empty_wagers(row):
    wagers = row.get("causal_street_wagers_v2") or {}
    vector = wagers.get("wagers")
    return (wagers.get("status") == "OBSERVED_STREET_WAGERS_CANDIDATE"
            and wagers.get("title_center_ledger_reconciled") is True
            and isinstance(vector, dict) and set(vector) == set(map(str, range(8)))
            and all(v in ("0", "0.0", "0.00") if isinstance(v, str)
                    else v == {"status": "NOT_APPLICABLE"} for v in vector.values()))


def _provisional_empty_wagers(row):
    """Only retain the first initializer frame; never publish UNKNOWN as zero."""
    causal = row.get("causal_street_wagers_v2") or {}
    raw, visibility = row.get("street_wagers"), row.get("wager_visibility_v2")
    if (causal.get("status") != "WAGERS_UNKNOWN"
            or causal.get("reason") != "two_stable_frames_required"
            or not isinstance(raw, dict) or set(raw) != set(map(str, range(8)))
            or not isinstance(visibility, dict)
            or set(visibility) != set(map(str, range(8)))
            or any(value is not None for value in raw.values())
            or any(v.get("status") != "VISIBLE_EMPTY_CANDIDATE"
                   for v in visibility.values())):
        return False
    try:
        values = [(row.get(key) or {}).get("value") for key in (
            "pot", "observed_center_v2")]
        if any(not isinstance(v, str) or len(v) > 64 for v in values):
            return False
        title, center = map(Decimal, values)
        return title.is_finite() and center.is_finite() and title == center >= 0
    except (InvalidOperation, TypeError, ValueError):
        return False


def _actor_supported(row):
    evidence = row.get("actor_evidence") or {}
    actor = row.get("current_actor")
    if actor == 4 and evidence.get("reason") == "hero_buttons":
        return evidence.get("hero_turn") is True
    return (type(actor) is int and evidence.get("actor") == actor
            and evidence.get("timer_suffix_verified") is True
            and evidence.get("reason") in (
                "unique_bright_ring_candidate",
                "unique_ring_and_suffix_union_candidate"))


def _no_river_actions(row, start):
    state = row.get("observed_state_v2") or {}
    lists = ("observed_actions_v2", "glyph_transitions", "action_history_candidate")
    if (state.get("pending_actions") != 0
            or any(not isinstance(row.get(key), list) for key in lists)
            or row["observed_actions_v2"] or row["glyph_transitions"]):
        return False
    # Old, still-visible fold badges are not new river actions. Both the event
    # onset and semantic street are checked, including zero-cost CHECK/FOLD.
    return not any(a.get("epoch") == state.get("observed_epoch")
                   and (a.get("street") == "river" or
                        type(a.get("frame")) is int and a["frame"] >= start)
                   for a in row["action_history_candidate"])


class CriticalPerceptionBoundary:
    """Causal qualification, with no image models, labels, or solver access."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.last = None
        self.board = None
        self.board_conflict = False
        self.river = None

    def observe(self, row):
        binding = _binding(row)
        fields = {key: _field(reason="source_or_current_context_missing")
                  for key in FIELDS}
        result = {"schema_version": 1, "semantics": SEMANTICS,
                  "candidate_only": True, "strategy_eligible": False,
                  "advice_emitted": False, "binding": binding,
                  "raw_evidence_digest": _digest(row), "window": [], "fields": fields}
        previous = self.last
        valid = _valid_binding(binding) and not _blocked(row)
        contiguous = (valid and previous is not None
                      and previous["binding"]["source_id"] == binding["source_id"]
                      and previous["binding"]["epoch"] == binding["epoch"]
                      and previous["binding"]["frame"] + 1 == binding["frame"]
                      and previous["binding"]["source_frame"] + 1 == (
                          binding["source_frame"])
                      and 0 < binding["pts_seconds"] - previous["binding"][
                          "pts_seconds"] <= 1
                      and row.get("reader_gap_reset") is not True)
        if not contiguous:
            self.reset()
        if not valid:
            return result
        board = _board(row)
        if board is not None:
            if self.board is not None and (len(board) < len(self.board)
                                           or board[:len(self.board)] != self.board):
                self.board_conflict = True
            if not self.board_conflict:
                self.board = list(board)
        fields["board"] = (_field(reason="board_changed_within_hand", conflict=True)
                           if self.board_conflict else _field(
                               board if board is not None and len(board) == 5 else None,
                               "current_complete_river_board_missing"))
        states, reason, conflict = _participation(row)
        fields["participation"] = _field(
            states if states is not None and len(states) == 8 else None,
            reason, conflict=conflict)
        hero = (states or {}).get("4")
        fields["hero_participation"] = _field(
            hero if hero in ("ACTIVE", "FOLDED", "ALL_IN") else None,
            reason, conflict=conflict)
        fields["all_in_seats"] = _field(
            sorted(int(s) for s, v in states.items() if v == "ALL_IN")
            if states is not None and len(states) == 8 else None,
            reason, conflict=conflict)
        state = row.get("observed_state_v2") or {}
        active = sorted(int(s) for s, v in (states or {}).items() if v == "ACTIVE")
        dealer, actor = row.get("dealer_seat"), row.get("current_actor")
        # Seat numbering is the fixed AA8 clockwise layout. The dealer evidence
        # must be current, and two actor observations must corroborate that order.
        dealer_evidence = row.get("dealer_evidence_v2") or {}
        first = (min(active, key=lambda s: (s - dealer) % 8 or 8)
                 if type(dealer) is int and 0 <= dealer < 8 and active else None)
        start = self.river["start"] if self.river else binding["frame"]
        no_actions = _no_river_actions(row, start)
        geometry = state.get("positive_board_geometry") or {}
        river_geometry = (state.get("street_candidate") in (None, "river")
                          and row.get("board_count") == 5
                          and geometry.get("last_frame") == binding["frame"]
                          and type(geometry.get("streak")) is int
                          and geometry["streak"] >= 1)
        wagers_ready = _empty_wagers(row)
        provisional = _provisional_empty_wagers(row)
        supported = (not self.board_conflict and len(states or {}) == 8
                     and river_geometry and no_actions
                     and (wagers_ready or provisional)
                     and _actor_supported(row) and actor == first
                     and dealer_evidence.get("dealer_seat") == dealer
                     and dealer_evidence.get("epoch") == binding["epoch"]
                     and dealer_evidence.get("frame") == binding["frame"]
                     and (row.get("dealer_observation_v2") or {}).get(
                         "dealer_seat") == dealer)
        if (contiguous and previous["stage"] == "turn" and previous["board"] is not None
                and len(previous["board"]) == 4 and supported):
            self.river = {"actor": actor, "start": binding["frame"],
                          "window": [previous["binding"], binding], "states": states,
                          "dealer": dealer, "turn_board": previous["board"],
                          "requires_initial_baseline": not wagers_ready}
        elif self.river is not None:
            if (not contiguous or not supported
                    or states != self.river["states"] or dealer != self.river["dealer"]
                    or actor != self.river["actor"]
                    or len(self.river["window"]) >= 256):
                self.river = None
            else:
                self.river["window"].append(binding)
                baseline = (row.get("causal_street_wagers_v2") or {}).get(
                    "baseline_evidence_frames")
                initial_baseline = (not self.river["requires_initial_baseline"] or
                                    baseline == [self.river["start"],
                                                 self.river["start"] + 1])
                if (wagers_ready and initial_baseline
                        and fields["board"]["status"] == "KNOWN"
                        and board[:4] == self.river["turn_board"]):
                    fields["river_first_actor"] = _field(actor)
                    result["window"] = deepcopy(self.river["window"])
        self.last = {"binding": binding, "stage": state.get("street_candidate"),
                     "board": board}
        return result


def checked_view(row):
    """Validate a cached candidate projection against its current raw row.

    This is consistency checking, not authentication of machine observations.
    Evaluation must additionally pin producer and source artifacts externally.
    """
    view = row.get("critical_perception_v1")
    try:
        if (not isinstance(view, dict) or view.get("schema_version") != 1
                or view.get("semantics") != SEMANTICS
                or view.get("candidate_only") is not True
                or view.get("strategy_eligible") is not False
                or view.get("advice_emitted") is not False
                or view.get("binding") != _binding(row)
                or view.get("raw_evidence_digest") != _digest(row)
                or set(view.get("fields", {})) != set(FIELDS)):
            return None
        fields = view["fields"]
        for field in fields.values():
            if (not isinstance(field, dict)
                    or set(field) != {"status", "value", "reasons"}
                    or field["status"] not in ("KNOWN", "UNKNOWN", "CONFLICT")
                    or not isinstance(field["reasons"], list)
                    or (field["status"] == "KNOWN") != (field["value"] is not None)):
                return None
        if any(f["status"] == "KNOWN" for f in fields.values()):
            if not _valid_binding(view["binding"]) or _blocked(row):
                return None
        board = fields["board"]
        if board["status"] == "KNOWN" and (
                len(board["value"]) != 5 or board["value"] != _board(row)):
            return None
        states, _, conflict = _participation(row)
        expected = {"participation": states if len(states or {}) == 8 else None,
                    "hero_participation": (states or {}).get("4"),
                    "all_in_seats": (sorted(int(s) for s, v in (states or {}).items()
                                            if v == "ALL_IN")
                                     if len(states or {}) == 8 else None)}
        for key, value in expected.items():
            if fields[key]["status"] == "KNOWN" and (
                    conflict or fields[key]["value"] != value):
                return None
        actor = fields["river_first_actor"]
        if actor["status"] == "KNOWN":
            window = view.get("window")
            if (type(actor["value"]) is not int
                    or actor["value"] != row.get("current_actor")
                    or not isinstance(window, list) or len(window) < 3
                    or window[-1] != view["binding"] or board["status"] != "KNOWN"
                    or fields["participation"]["status"] != "KNOWN"):
                return None
            for before, after in zip(window, window[1:]):
                if (not _valid_binding(before) or not _valid_binding(after)
                        or before["source_id"] != after["source_id"]
                        or before["epoch"] != after["epoch"]
                        or before["frame"] + 1 != after["frame"]
                        or before["source_frame"] + 1 != after["source_frame"]
                        or not 0 < after["pts_seconds"] - before["pts_seconds"] <= 1):
                    return None
        return deepcopy(view)
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def target_s_candidate_screen(row):
    """Screen shape only. Never substitutes for human factual confirmation."""
    view = checked_view(row)
    reasons = []
    if view is None:
        reasons.append("critical_candidate_view_missing_or_invalid")
    else:
        fields = view["fields"]
        reasons.extend("critical_field_not_known:" + name for name in FIELDS
                       if fields[name]["status"] != "KNOWN")
        if not reasons:
            states = fields["participation"]["value"]
            if fields["hero_participation"]["value"] != "ACTIVE":
                reasons.append("hero_not_active")
            if sum(value == "ACTIVE" for value in states.values()) != 3:
                reasons.append("not_exactly_three_active")
            if fields["all_in_seats"]["value"]:
                reasons.append("all_in_at_river_start")
            if fields["river_first_actor"]["value"] != 4:
                reasons.append("hero_not_river_first_actor")
    return {"status": "SCREEN_BLOCKED" if reasons else "SCREEN_ELIGIBLE",
            "reasons": reasons, "candidate_only": True,
            "semantics": "TARGET_S_STRUCTURAL_CANDIDATE_NOT_ACCEPTANCE",
            "binding": deepcopy(view["binding"]) if view else None,
            "strategy_eligible": False, "advice_emitted": False,
            "real_hand_confirmed": False}
