"""Source-bound seven-player cash reconciliation within an eight-slot AA layout."""

import argparse
from decimal import Decimal
import json
from pathlib import Path

from poker_engine.core.enums import Rank, Suit
from poker_engine.core.value_objects import Card
from poker_engine.equity.evaluator import evaluate
from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)


def amount(value):
    if not isinstance(value, str):
        raise ValueError("reviewed decimal strings required")
    result = Decimal(value)
    if not result.is_finite() or result < 0:
        raise ValueError("nonnegative finite amounts required")
    return result


def reconcile(review):
    slots = review["opening_slots"]
    if (any(type(s) is not int or not 0 <= s < 8 for s in slots)
            or len(slots) != len(set(slots))
            or review["excluded_waiting_slot"] in slots):
        raise ValueError("opening roster conflict")
    keys = {str(slot) for slot in slots}
    cash = {}
    cash_fields = ("starting_balances", "posted_balances",
                   "before_settlement", "after_settlement")
    for key in cash_fields:
        if set(review[key]) != keys:
            raise ValueError("balance coverage differs from opening roster")
        cash[key] = {s: amount(v) for s, v in review[key].items()}
    original, posted, before, after = (cash[k] for k in cash_fields)
    if any(posted[s] > original[s] for s in keys):
        raise ValueError("post interval contains an unmodeled inflow")
    pot = amount(review["displayed_pot"])
    if sum(map(amount, review["displayed_components"]), Decimal(0)) != pot:
        raise ValueError("displayed pot components do not sum to total")
    credits = {s: after[s] - before[s] for s in keys}
    if any(v < 0 for v in credits.values()):
        raise ValueError("settlement includes an unexplained debit")
    original_total = sum(original.values(), Decimal(0))
    contribution_gap = original_total - sum(before.values(), Decimal(0)) - pot
    payout_gap = pot - sum(credits.values(), Decimal(0))
    if contribution_gap < 0 or payout_gap < 0:
        raise ValueError("external inflow or inconsistent pot needs separate evidence")
    closing_gap = original_total - sum(after.values(), Decimal(0))
    hero = str(review["hero_slot"])
    observed_net = after[hero] - original[hero]
    external_post_hypothesis = (amount(review["displayed_mushroom_setting_bb"])
                                * amount(review["displayed_big_blind"]))
    checkpoints = []
    for checkpoint in review["monetary_checkpoints"]:
        if set(checkpoint["balances"]) != keys:
            raise ValueError("checkpoint seat coverage mismatch")
        observed = sum(map(amount, checkpoint["balances"].values()), Decimal(0))
        discrepancy = observed + amount(checkpoint["pot"]) + external_post_hypothesis
        discrepancy -= original_total
        checkpoints.append({
            "frame": checkpoint["frame"],
            "discrepancy_under_displayed_post_hypothesis": str(discrepancy)})
    all_codes = review["hero"] + review["opponent"] + review["board"]
    if (len(review["hero"]) != 2 or len(review["opponent"]) != 2
            or len(review["board"]) != 5 or len(set(all_codes)) != len(all_codes)
            or any(not isinstance(c, str) or len(c) != 2 for c in all_codes)):
        raise ValueError("complete distinct showdown cards required")

    def rank(codes):
        return evaluate(tuple(Card(Rank(c[0]), Suit(c[1])) for c in codes))

    return {"initial_post_debits": {
                s: str(original[s] - posted[s]) for s in sorted(keys)},
            "starting_cash_total": str(original_total),
            "monetary_checkpoints": checkpoints,
            "checkpoint_cash_and_pot_consistent_under_hypothesis": all(
                Decimal(r["discrepancy_under_displayed_post_hypothesis"]) == 0
                for r in checkpoints),
            "visible_settlement_credits": {s: str(credits[s]) for s in sorted(keys)},
            "outside_displayed_pot_before_settlement": str(contribution_gap),
            "payout_difference": str(payout_gap),
            "closing_cash_difference": str(closing_gap),
            "differences_reconcile": contribution_gap + payout_gap == closing_gap,
            "decomposition_is_algebraic_not_independent_validation": True,
            "outside_pot_matches_mushroom_setting": contribution_gap == (
                amount(review["displayed_mushroom_setting_bb"])
                * amount(review["displayed_big_blind"])),
            "hero_net_visible_change": str(observed_net),
            "hero_win_label_matches": (
                observed_net == amount(review["displayed_hero_win"])),
            "hero_beats_revealed_opponent": rank(review["hero"] + review["board"])
            > rank(review["opponent"] + review["board"]),
            "rake_policy_verified": False, "mushroom_routing_verified": False,
            "full_action_truth_verified": False, "automated_visual_extraction": False}


def run(audit, registry_dir, full_hand, opening_context, review_path, output):
    for folder in (audit, registry_dir, full_hand, opening_context):
        if verify_sha256sums(folder):
            raise ValueError("input integrity failure")
    review = json.loads(review_path.read_text())
    if (review["audit_sha256"] != sha256_file(audit / "report.json") or
            review["registry_sha256"] != sha256_file(registry_dir / "registry.json")):
        raise ValueError("review provenance mismatch")
    registry = json.loads((registry_dir / "registry.json").read_text())
    if registry["hand_id"] != review["hand_id"]:
        raise ValueError("wrong hand")
    boundary = review["evidence_frames"]
    if (boundary["starting"] != registry["start_global_frame"] - 1
            or boundary["posted"] != registry["start_global_frame"]
            or not boundary["posted"] < boundary["before_settlement"]
            < boundary["after_settlement"] <= registry["end_global_frame"]
            or review["opening_slots"] != registry["initial_posting_slots"]):
        raise ValueError("cash review conflicts with hand ownership")
    evidence = {}
    desired_frames = set(review["evidence_frames"].values()) | {
        r["frame"] for r in review["monetary_checkpoints"]}
    for pool in (opening_context, full_hand):
        manifest = json.loads((pool / "samples.json").read_text())
        if manifest["audit_sha256"] != review["audit_sha256"]:
            raise ValueError("wrong recording in image pool")
        for row in manifest["samples"]:
            if row["global_frame"] not in desired_frames:
                continue
            path = (pool / row["file"]).resolve()
            if (not path.is_relative_to(pool.resolve()) or
                    sha256_file(path) != row["sha256"]):
                raise ValueError("evidence hash/path mismatch")
            evidence[str(row["global_frame"])] = {**row, "path": str(path)}
    if len(evidence) != len(desired_frames):
        raise ValueError("missing reviewed evidence frame")
    result = reconcile(review)
    output.mkdir(parents=True, exist_ok=False)
    report = {**result, "evidence": evidence, "review_sha256": sha256_file(review_path)}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    write_sha256sums(output)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    keys = ("audit", "registry", "full-hand", "opening-context", "review", "output")
    for key in keys:
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.audit, args.registry, args.full_hand, args.opening_context,
        args.review, args.output)
