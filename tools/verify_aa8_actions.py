"""Replay manually reviewed voluntary actions; never certify visual recognition."""

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from tools.verify_aa8_settlement import amount


def verify(review, cash):
    for key in ("hand_id", "audit_sha256", "registry_sha256", "opening_slots"):
        if review[key] != cash[key]:
            raise ValueError("action/cash provenance or roster mismatch")
    slots = review["opening_slots"]
    if slots != sorted(set(slots)) or any(s not in range(8) for s in slots):
        raise ValueError("invalid physical roster")
    balances = {int(s): amount(v) for s, v in cash["posted_balances"].items()}
    wagers = {int(s): amount(v) for s, v in review["seed_wagers"].items()}
    if set(wagers) != set(slots) or set(balances) != set(slots):
        raise ValueError("seed roster mismatch")
    pot = amount(review["seed_pot"])
    if sum(wagers.values()) + amount(review["seed_collected_pot"]) != pot:
        raise ValueError("seed pot mismatch")
    active, timeline = set(slots), []
    previous = review["seed_frame"]
    board = []
    if [s["street"] for s in review["streets"]] != [
            "preflop", "flop", "turn", "river"]:
        raise ValueError("complete ordered streets required")
    for index, street in enumerate(review["streets"]):
        expected_board = cash["board"][:(0, 3, 4, 5)[index]]
        if street["board"] != expected_board:
            raise ValueError("street board mismatch")
        if index:
            first_window = street["actions"][0]["window"][0]
            if not previous < street["board_frame"] <= first_window:
                raise ValueError("board timing mismatch")
            wagers = dict.fromkeys(slots, Decimal(0))
        board = street["board"]
        first = street["first_actor"]
        if first not in active:
            raise ValueError("invalid first actor")

        def following(actor):
            return sorted((s for s in active if s != actor and balances[s] > 0),
                          key=lambda s: (s - actor) % 8)

        pending = [first] + following(first)
        maximum = max(wagers.values())
        for action in street["actions"]:
            slot, kind = action["slot"], action["kind"]
            before, after = action["window"]
            if not previous <= before < after:
                raise ValueError("overlapping or unordered action evidence")
            previous = after
            if not pending or pending.pop(0) != slot:
                raise ValueError("out-of-order, folded, or waiting actor")
            debit = amount(action["debit"])
            if debit > balances[slot]:
                raise ValueError("debit exceeds stack")
            owed = maximum - wagers[slot]
            if kind == "fold":
                if debit:
                    raise ValueError("fold has debit")
                active.remove(slot)
            elif kind == "check":
                if debit or owed:
                    raise ValueError("illegal check")
            elif kind == "call":
                if owed <= 0 or debit != min(owed, balances[slot]):
                    raise ValueError("incorrect call amount")
            elif kind in ("bet", "raise"):
                if (kind == "bet") != (maximum == 0):
                    raise ValueError("bet/raise street semantics mismatch")
                if wagers[slot] + debit <= maximum:
                    raise ValueError("wager does not increase price")
                maximum = wagers[slot] + debit
                pending = following(slot)
            else:
                raise ValueError("unknown action")
            balances[slot] -= debit
            wagers[slot] += debit
            pot += debit
            if action.get("all_in", False) != (balances[slot] == 0):
                raise ValueError("all-in marker disagrees with stack")
            if pot != amount(action["pot_after"]):
                raise ValueError("action pot mismatch")
            timeline.append({"street": street["street"], "slot": slot,
                             "kind": kind, "window": action["window"],
                             "pot": str(pot), "balance": str(balances[slot]),
                             "board": list(board)})
        if pending or pot != amount(street["end_pot"]):
            raise ValueError("incomplete street")
    if balances != {int(s): amount(v) for s, v in cash["before_settlement"].items()}:
        raise ValueError("final balances differ from reviewed cash")
    if pot != amount(cash["displayed_pot"]):
        raise ValueError("final pot differs from reviewed cash")
    ordered = sorted(wagers.values(), reverse=True)
    unmatched = ordered[0] - ordered[1]
    if (unmatched != amount(review["expected_unmatched_return"]) or
            pot - unmatched != amount(review[
                "expected_contested_pot_before_unallocated_deduction"])):
        raise ValueError("unmatched amount mismatch")
    return {"reviewed_action_replay_consistent": True, "action_count": len(timeline),
            "timeline": timeline, "surviving_slots": sorted(active),
            "unmatched_return_under_reviewed_action_line": str(unmatched),
            "contested_pot_before_unallocated_deduction": str(pot - unmatched),
            "exact_event_frames_verified": False,
            "automated_visual_extraction": False, "full_visual_acceptance": False,
            "initial_post_components_verified": False, "rake_policy_verified": False}


def bind(review, pool, registry_path):
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    if (hashlib.sha256(registry_bytes).hexdigest() != review["registry_sha256"]
            or registry["hand_id"] != review["hand_id"]
            or registry["role"] != "development" or review["role"] != "development"):
        raise ValueError("wrong registry or non-development source")
    manifest = json.loads((pool / "samples.json").read_text(encoding="utf-8"))
    if manifest["audit_sha256"] != review["audit_sha256"]:
        raise ValueError("wrong recording")
    rows = {r["global_frame"]: r for r in manifest["samples"]}
    wanted = {review["seed_frame"]}
    for street in review["streets"]:
        if "board_frame" in street:
            wanted.add(street["board_frame"])
        for action in street["actions"]:
            wanted.update(action["window"])
    result = {}
    for frame in sorted(wanted):
        if not registry["start_global_frame"] <= frame <= registry["end_global_frame"]:
            raise ValueError("evidence outside hand")
        row = rows[frame]
        path = (pool / row["file"]).resolve()
        if (row["role"] != "development" or not path.is_relative_to(pool.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]):
            raise ValueError("evidence role/path/hash mismatch")
        result[str(frame)] = row
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("review", "cash", "pool", "registry", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    cash = json.loads(args.cash.read_text(encoding="utf-8"))
    result = verify(review, cash)
    result["evidence"] = bind(review, args.pool, args.registry)
    result["review_sha256"] = hashlib.sha256(args.review.read_bytes()).hexdigest()
    result["cash_sha256"] = hashlib.sha256(args.cash.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("timeline", "evidence")}))
