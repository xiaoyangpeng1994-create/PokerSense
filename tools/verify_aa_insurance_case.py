"""Reconcile one reviewed AA insurance window without inventing a rake policy."""

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


def cards(values):
    return tuple(Card(Rank(c[0]), Suit(c[1])) for c in values)


def river_outs(hero, opponent, board, other_known):
    known = hero + opponent + board + other_known
    if len(board) != 4 or len(set(known)) != len(known):
        raise ValueError("four-card turn and distinct known cards required")
    remaining = [r + s for r in "23456789TJQKA" for s in "cdhs" if r + s not in known]
    outs = [c for c in remaining if evaluate(cards(opponent + board + [c])) >
            evaluate(cards(hero + board + [c]))]
    return remaining, outs


def reconcile(review):
    if set(review["before_balances"]) != set(review["after_balances"]):
        raise ValueError("observed seat sets differ")
    if not (review["quote_frame"] < review["purchase_notice_frame"]
            < review["settlement_frame"]):
        raise ValueError("invalid evidence ordering")
    premium = Decimal(review["quote"]["premium"])
    if review["purchase_notice_text"] != "投保" + str(premium):
        raise ValueError("purchase notice does not agree with premium")
    deltas = {s: Decimal(review["after_balances"][s]) - Decimal(value)
              for s, value in review["before_balances"].items()}
    if any(value < 0 for value in deltas.values()):
        raise ValueError("additional debit needs separate transaction evidence")
    total = Decimal(review["displayed_total_pot"])
    if sum(map(Decimal, review["displayed_pot_components"])) != total:
        raise ValueError("displayed pot components do not reconcile")
    credits = sum(deltas.values())
    difference = total - credits
    unexplained = difference - premium
    remaining, outs = river_outs(review["hero"], review["insured_opponent"],
                                 review["turn_board"],
                                 review["other_revealed_cards_at_purchase_notice"])
    return {"credits_by_slot": {k: str(v) for k, v in deltas.items()},
            "total_visible_credits": str(credits), "pot_less_credits": str(difference),
            "premium_notice_matches_difference": difference == premium,
            "unexplained_after_premium": str(unexplained),
            "hero_insured_pot_less_premium_matches_credit": (
                Decimal(review["quote"]["insured_pot"]) - premium == deltas["5"]),
            "remaining_cards_at_purchase_notice": len(remaining),
            "calculated_outs": sorted(outs),
            "displayed_outs_match": set(outs) == set(review["displayed_outs"])
            and len(outs) == review["quote"]["outs"],
            "quote_rounding_residual": str(
                premium * Decimal(review["quote"]["odds_multiplier"])
                - Decimal(review["quote"]["displayed_compensation"])),
            "general_rake_policy_verified": False, "whole_hand_verified": False,
            "automated_visual_extraction": False}


def run(window, review_path, output):
    if verify_sha256sums(window):
        raise ValueError("source window integrity failure")
    source = json.loads((window / "samples.json").read_text())
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if source["source_sha256"] != review["source_sha256"]:
        raise ValueError("wrong source")
    samples = {r["source_frame"]: r for r in source["samples"]}
    evidence = {key: samples[review[key]] for key in (
        "quote_frame", "purchase_notice_frame", "settlement_frame")}
    for sample in evidence.values():
        path = (window / sample["file"]).resolve()
        if (not path.is_relative_to(window.resolve()) or
                sha256_file(path) != sample["sha256"]):
            raise ValueError("source evidence mismatch")
    result = reconcile(review)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps({
        **result, "evidence": evidence, "review_sha256": sha256_file(review_path),
        "source_sha256": source["source_sha256"]}, indent=2), encoding="utf-8")
    write_sha256sums(output)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("window", "review", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.window, args.review, args.output)
