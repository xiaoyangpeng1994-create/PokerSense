"""Development-only causal AA8 state evidence, separate from frozen V1.

Observed state and action evidence are not authoritative poker StateEvents.
"""

import argparse
from collections import deque
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

from poker_engine.state_engine.visual_timeline import VisualTimeline
from tools.aa8_action_transfer import inventory, sha


def amount(value):
    if isinstance(value, dict):
        value = value.get("value")
    if not isinstance(value, str):
        return None
    try:
        number = Decimal(value)
        return number if number.is_finite() and number >= 0 else None
    except InvalidOperation:
        return None


def current_cue(row, slot):
    value = row.get("participation", {}).get("slots", {}).get(str(slot), {})
    return "UNKNOWN" if value.get("conflict") else value.get("current", "UNKNOWN")


def positive_board_geometry(row):
    count = row.get("board_count")
    if count not in (3, 4, 5):
        return None
    evidence = row.get("cards", {}).get("evidence", {})
    accepted = ("nominal_face_boundary_supported", "measured_face_top_edge_candidate")
    for slot in range(5):
        item = evidence.get(f"board_{slot}", {})
        if slot < count:
            rect = item.get("rect")
            if (not isinstance(rect, (list, tuple)) or len(rect) != 4
                    or item.get("face_support") not in accepted):
                return None
        elif item.get("rect") is not None:
            return None
    return count


def posting_comparison(before, after):
    """Compare available stacks; only explicit absent/wait seats may be excluded."""
    debits, credits, excluded = {}, {}, []
    for s in map(str, range(8)):
        a, b = (amount(r.get("stacks", {}).get(s)) for r in (before, after))
        if a is None or b is None:
            missing = [r for r, v in ((before, a), (after, b)) if v is None]
            if any(current_cue(r, s) not in (
                    "EMPTY_CANDIDATE", "WAITING_CANDIDATE") for r in missing):
                return None
            excluded.append(s)
        elif a > b:
            debits[s] = str(a - b)
        elif b > a:
            credits[s] = str(b - a)
    if len(debits) < 3:
        return None
    return {"debits": debits, "credits": credits, "excluded_na_slots": excluded,
            "credit_semantics": "UNALLOCATED_POSSIBLE_REFILL_NOT_PROFIT"}


class AA8StateAdapterV2:
    @staticmethod
    def _stacks_equal(left, right):
        return left == right

    def __init__(self, pair_window=12):
        self.window = pair_window
        self.last = None
        self.history = deque(maxlen=4)
        self.epoch = None
        self.epoch_events = []
        self.deal_streak = 0
        self.pending_hand = None
        self.timeline = VisualTimeline(
            slot_count=8, hero_slot=4, pair_window=pair_window)
        self.participants = {}
        self.street = None
        self.board = None
        self.current_board_supported = False
        self.geometry_run = None
        self.pending = []
        self.actions = []
        self.cash = []
        self.pot_changes = []
        self.used_cash = set()
        self.credits = []
        self._credit_keys = set()
        self.credit_timeline = VisualTimeline(
            slot_count=8, hero_slot=4, pair_window=pair_window)

    def _credit(self, seat, first, confirmed, value, source, **evidence):
        key = (int(seat), first, value)
        if key not in self._credit_keys:
            self._credit_keys.add(key)
            self.credits.append({"seat": int(seat), "first_frame": first,
                                 "confirmed_frame": confirmed, "amount": value,
                                 "source": source, "epoch": self.epoch,
                                 "hand_attribution_verified": False,
                                 "semantics": "UNALLOCATED_NOT_PROFIT_OR_RAKE",
                                 **evidence})

    def _observe_positive_cash(self, row):
        # Monetary observation is independent of action-mode permission. Only
        # already accepted, visible stack readings enter; scene/gap resets still
        # apply inside VisualTimeline. No betting or fee attribution is produced.
        supported = row.get("scene_supported") is True
        converted = {
            "source_frame": row["frame"], "scene_reason": "supported_layout_candidate"
            if supported else "unsupported_scene", "actions": {}, "seat_presence": {},
            "hero": [], "stacks": {s: {
                "status": "valid" if amount(v) is not None else "unknown",
                "value": str(amount(v)) if amount(v) is not None else None}
                for s, v in row.get("stacks", {}).items()}}
        index = len(self.credit_timeline.events)
        self.credit_timeline.consume(converted)
        for event in self.credit_timeline.events[index:]:
            if event.kind == "balance_increase":
                self._credit(event.seat, event.first_frame, event.confirmed_frame,
                             event.amount, "stable_visual_balance_increase",
                             balance_before=event.balance_before,
                             balance_after=event.balance_after)

    def _new_epoch(self, frame, first, status, posting=None):
        self.epoch = "observed_deal_" + str(first)
        self.epoch_events.append({"frame": frame, "first_frame": first,
                                  "epoch": self.epoch, "status": status,
                                  "posting_comparison": posting,
                                  "authoritative_boundary": False})
        for seat, value in (posting or {}).get("credits", {}).items():
            self._credit(seat, first, frame, value, "new_post_comparison")
        self.timeline = VisualTimeline(
            slot_count=8, hero_slot=4, pair_window=self.window)
        self.participants = {}
        self.street, self.board = "preflop", []
        self.geometry_run = None
        self.pending, self.cash, self.pot_changes = [], [], []
        self.used_cash.clear()

    def observe(self, row):
        frame = row["frame"]
        modes = row.get("special_modes") or {}
        modal = bool(modes.get("block_state_updates")
                     or modes.get("insurance") == "VISIBLE")
        if (type(frame) is not int or frame < 0
                or self.last is not None and frame <= self.last):
            raise ValueError("strictly increasing source frames required")
        self._observe_positive_cash(row)
        if ((self.last is not None and frame != self.last + 1)
                or not row.get("scene_supported") or modal):
            self.history.clear()
            self.epoch = None
            self.participants, self.pending = {}, []
            self.street, self.board = None, None
            self.timeline = VisualTimeline(slot_count=8, hero_slot=4)
            self.pending_hand = None
            self.deal_streak = 0
            self.cash, self.pot_changes = [], []
            self.used_cash.clear()
            self.current_board_supported = False
            self.geometry_run = None
        self.last = frame
        if not row.get("scene_supported") or modal:
            return {**self.snapshot(frame), "observation_blocked": True,
                    "blocked_reason": "special_mode_or_overlay" if modal
                    else "unsupported_scene"}
        deal = row.get("hand_transition", {}).get("center_deal", {}).get(
            "visible") is True
        self.deal_streak = (self.deal_streak + 1
                            if deal and row.get("board_count") == 0 else 0)
        if self.pending_hand:
            pending = self.pending_hand
            if (deal and row.get("board_count") == 0 and amount(row.get("pot")) == 0
                    and self._stacks_equal(row.get("stacks"), pending["stacks"])):
                self._new_epoch(frame, pending["first"], "MULTI_POST_DEAL_CANDIDATE",
                                pending["comparison"])
            self.pending_hand = None
        if (len(self.history) >= 2 and deal and row.get("board_count") == 0
                and amount(row.get("pot")) == 0):
            a, b = list(self.history)[-2:]
            if (self._stacks_equal(a.get("stacks"), b.get("stacks"))
                    and a.get("board_count") == b.get("board_count") == 0):
                comparison = posting_comparison(b, row)
                if comparison:
                    self.pending_hand = {"first": frame, "stacks": row["stacks"],
                                         "comparison": comparison}
        if self.epoch is None and self.deal_streak == 2:
            self._new_epoch(frame, frame - 1, "UNANCHORED_DEAL_CONTEXT")
        cards = row.get("cards", {}).get("board_slots", [None] * 5)
        count = row.get("board_count")
        self.current_board_supported = (
            self.street == "preflop" and count == 0 and self.epoch is not None)
        geometry = positive_board_geometry(row)
        if geometry is None:
            self.geometry_run = None
        else:
            old = self.geometry_run
            self.geometry_run = {
                "count": geometry, "last_frame": frame,
                "first_frame": old["first_frame"] if old and old["count"] == geometry
                and old["last_frame"] == frame - 1 else frame,
                "streak": old["streak"] + 1 if old and old["count"] == geometry
                and old["last_frame"] == frame - 1 else 1}
            prior_count = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}.get(
                self.street, 0)
            if self.geometry_run["streak"] >= 2 and geometry >= prior_count:
                self.street = {3: "flop", 4: "turn", 5: "river"}[geometry]
                self.current_board_supported = True
                self.board = (cards[:geometry] if all(cards[:geometry])
                              and not any(cards[geometry:]) else None)
        for s in map(str, range(8)):
            cue = current_cue(row, s)
            state = {"DEALT_IN_CANDIDATE": "active",
                     "FOLDED_CANDIDATE": "folded",
                     "WAITING_CANDIDATE": "waiting",
                     "EMPTY_CANDIDATE": "empty"}.get(cue)
            previous = self.participants.get(s, {}).get("state")
            if state == "active" and previous in (
                    "folded", "waiting", "empty", "all_in"):
                continue
            if state in ("empty", "waiting") and previous in (
                    "active", "folded", "all_in"):
                continue
            if state is not None:
                self.participants[s] = {"state": state, "evidence_frame": frame,
                                        "evidence": cue, "epoch": self.epoch}
        present = {s: {"status": "valid", "value": p["state"] in (
            "active", "folded", "all_in")} for s, p in self.participants.items()}
        converted = {"source_frame": frame,
                     "scene_reason": "supported_layout_candidate",
                     "street_observation": {"status": "valid",
                                            "value": self.street},
                     "stacks": {s: {
                         "status": "valid" if amount(v) is not None else "unknown",
                         "value": str(amount(v)) if amount(v) is not None else None}
                                for s, v in row["stacks"].items()},
                     "seat_presence": present, "actions": {},
                     "hero": row.get("cards", {}).get("hero") or []}
        index = len(self.timeline.events)
        self.timeline.consume(converted)
        self.cash.extend(asdict(e) for e in self.timeline.events[index:]
                         if e.kind == "balance_decrease")
        if len(self.history) >= 3:
            a, b, c = list(self.history)[-3:]
            pots = [amount(r.get("pot")) for r in (a, b, c, row)]
            if (all(p is not None for p in pots) and pots[0] == pots[1]
                    and pots[2] == pots[3] and pots[2] > pots[1]):
                self.pot_changes.append({"first": c["frame"], "confirmed": frame,
                                         "amount": str(pots[2] - pots[1])})
        for glyph in row.get("glyph_transitions", []):
            self.pending.append({"frame": glyph["frame"], "slot": glyph["slot"],
                                 "glyph": glyph["glyph"], "epoch": self.epoch,
                                 "street": self.street,
                                 "source_sha256": row["source_sha256"]})
        new_actions = []
        for action in list(self.pending):
            zero = action["glyph"] in ("fold", "check")
            matches = [d for d in self.cash if d["seat"] == action["slot"]
                       and abs(d["first_frame"] - action["frame"]) <= self.window
                       and (d["seat"], d["first_frame"]) not in self.used_cash]
            debit = matches[0] if len(matches) == 1 else None
            if debit and any(d["seat"] != debit["seat"] and abs(
                    d["first_frame"] - debit["first_frame"]) <= 1 for d in self.cash):
                debit = None
            pot = [p for p in self.pot_changes
                   if debit and p["amount"] == debit["amount"]
                   and abs(p["first"] - debit["first_frame"]) <= self.window]
            if (zero or debit and len(pot) == 1
                    or frame - action["frame"] > self.window):
                paired = debit is not None and len(pot) == 1
                event = {**action, "confirmed_at": frame, "kind": action["glyph"],
                         "amount": "0" if zero else debit["amount"] if paired else None,
                         "status": "OBSERVED_GLYPH_CASH_CANDIDATE" if paired
                         else "OBSERVED_GLYPH",
                         "cash_evidence": debit if paired else None,
                         "pot_evidence": pot if paired else [],
                         "legal_action_verified": False}
                if event["cash_evidence"]:
                    self.used_cash.add((debit["seat"], debit["first_frame"]))
                self.pending.remove(action)
                self.actions.append(event)
                new_actions.append(event)
                s = str(action["slot"])
                state = None
                if action["glyph"] == "all_in" and amount(row["stacks"].get(s)) == 0:
                    state = "all_in"
                elif action["glyph"] == "fold":
                    state = "folded"
                elif paired and action["glyph"] in ("call", "aggressive"):
                    state = "active"
                if state is not None:
                    self.participants[s] = {
                        "state": state, "evidence_frame": frame,
                        "evidence": "confirmed_action_evidence", "epoch": self.epoch}
        short = (row.get("continuous_context") or {}).get(
            "unmarked_action_candidate", {})
        if short.get("action") == "call":
            event = {"frame": short["evidence_frames"][-2], "confirmed_at": frame,
                     "slot": short["actor"], "kind": "call", "glyph": None,
                     "amount": short["debit"], "epoch": self.epoch,
                     "street": self.street,
                     "status": "UNMARKED_SHORT_CALL_CANDIDATE",
                     "legal_action_verified": False,
                     "cash_evidence": short, "source_sha256": row["source_sha256"]}
            self.actions.append(event)
            new_actions.append(event)
            self.participants[str(short["actor"])] = {
                "state": "all_in", "evidence_frame": frame,
                "evidence": "short_call_candidate", "epoch": self.epoch}
        self.history.append(row)
        return {**self.snapshot(frame), "new_actions": new_actions}

    def snapshot(self, frame):
        return {"frame": frame, "observed_epoch": self.epoch,
                "street_candidate": (
                    self.street if self.current_board_supported else None),
                "board_candidate": self.board if self.current_board_supported else None,
                "positive_board_geometry": dict(self.geometry_run)
                if self.geometry_run else None,
                "participants": self.participants.copy(),
                "participant_semantics": "observed_history_not_current_visibility",
                "pending_actions": len(self.pending),
                "unallocated_positive_cash": list(self.credits),
                "authoritative_hand_boundary": False, "complete_legal_state": False,
                "strategy_eligible": False}


def compare_reviews(actions, reviews):
    from tools.aa8_action_reader import compare
    reports = []
    for index, path in enumerate(reviews):
        review = json.loads(path.read_text())
        if review.get("role") != "development":
            raise ValueError("only development references allowed")
        epochs = list(dict.fromkeys(a["epoch"] for a in actions))
        selected = [a for a in actions if a["epoch"] == epochs[index] and a["glyph"]]
        result = compare(selected, review)
        gold = [a for street in review["streets"] for a in street["actions"]]
        checks = []
        for match in result["matched"]:
            expected = next(a for a in gold if a["slot"] == match["slot"]
                            and a["window"] == match["window"])
            if "debit" in expected:
                actual = match["prediction"]["amount"]
                checks.append({"slot": match["slot"], "window": match["window"],
                               "expected": expected["debit"], "actual": actual,
                               "status": "UNKNOWN" if actual is None else
                               "MATCH" if actual == expected["debit"] else "MISMATCH"})
        reports.append({"reference_sha256": sha(path),
                        "matched": len(result["matched"]), "missed": result["missed"],
                        "extra": result["unmatched_proposals"],
                        "matching_latency_frames": 2, "cash_checks": checks})
        reports[-1]["street_mismatches"] = [
            m for m in result["matched"] if m["prediction"]["street"] != m["street"]]
        reports[-1]["unmarked_cash_checks"] = [{
            "slot": expected["slot"], "window": expected["window"],
            "expected_debit": expected["stack_before"],
            "matches": [a for a in actions if a["glyph"] is None
                        and a["slot"] == expected["slot"]
                        and expected["window"][0] < a["frame"] <= expected["window"][1]
                        and a["amount"] == expected["stack_before"]]}
            for expected in review.get("unmarked_action_evidence", [])]
    return reports


def run(observations, pools, output, reviews=()):
    if len(observations) != len(pools) or not observations:
        raise ValueError("one development pool per observation file required")
    adapter = AA8StateAdapterV2()
    output.mkdir(parents=True, exist_ok=False)
    manifest_hashes, snapshots = {}, []
    audit = None
    with (output / "states.jsonl").open("x") as stream:
        for path, pool in zip(observations, pools):
            metadata = json.loads((pool / "samples.json").read_text())
            if audit is not None and metadata["audit_sha256"] != audit:
                raise ValueError("different recording requires a new state adapter")
            audit = metadata["audit_sha256"]
            rows = inventory(pool, metadata["audit_sha256"])
            manifest_hashes[str(path)] = sha(path)
            manifest_hashes[str(pool / "samples.json")] = sha(pool / "samples.json")
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row["source_sha256"] != rows[row["frame"]]["sha256"]:
                    raise ValueError("development observations source mismatch")
                value = adapter.observe(row)
                stream.write(json.dumps(value) + "\n")
                if row["frame"] in (1320, 2700, 3319, 3400, 3536, 4825):
                    snapshots.append(value)
    result = {"actions": adapter.actions, "epochs": adapter.epoch_events,
              "unallocated_positive_cash": adapter.credits,
              "snapshots": snapshots, "inputs": manifest_hashes,
              "implementation_sha256": sha(Path(__file__)),
              "reference_comparison": compare_reviews(adapter.actions, reviews),
              "full_visual_acceptance": False, "independent_holdout": False}
    (output / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"actions": len(adapter.actions), "with_amount": sum(
        a["amount"] is not None for a in adapter.actions),
        "epochs": adapter.epoch_events}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, action="append", required=True)
    parser.add_argument("--pool", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path, action="append", default=[])
    args = parser.parse_args()
    run(args.observations, args.pool, args.output, args.review)
