"""Audit visible bomb contributions and unexplained extra debits, not a rake rule."""

import argparse
from decimal import Decimal
import json
from pathlib import Path

from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)


def audit_case(case):
    seats = case["occupied_slots"]
    if len(set(seats)) != len(seats) or len(seats) < 2:
        raise ValueError("distinct observed seats required")
    bb = Decimal(case["big_blind"])
    if not bb.is_finite() or bb <= 0:
        raise ValueError("positive BB required")
    per_player = bb * Decimal(case["bomb_bb_displayed"])
    expected_pool = per_player * len(seats)
    changes, extras, unknown = {}, {}, []
    for seat in seats:
        key = str(seat)
        before, after = case["before_balances"][key], case["after_balances"][key]
        if before is None or after is None:
            unknown.append(seat)
            continue
        debit = Decimal(before) - Decimal(after)
        changes[key] = str(debit)
        extras[key] = str(debit - per_player)
    dealer_extra = extras.get(str(case["new_dealer_slot"]))
    mushroom_change = (Decimal(case["mushroom_display_after"])
                       - Decimal(case["mushroom_display_before"]))
    return {"name": case["name"], "per_player_from_displayed_bb": str(per_player),
            "expected_collection": str(expected_pool),
            "collection_matches": expected_pool == Decimal(case["collected_display"]),
            "observed_debits": changes, "extra_above_bomb_by_slot": extras,
            "unknown_balance_slots": unknown,
            "new_dealer_extra_matches_mushroom_setting": (
                dealer_extra is not None and Decimal(dealer_extra)
                == bb * Decimal(case["mushroom_bb_displayed"])),
            "all_cash_observed": not unknown,
            "mushroom_display_change": str(mushroom_change),
            "mushroom_display_change_matches_dealer_extra": (
                dealer_extra is not None and mushroom_change == Decimal(dealer_extra)),
            "pot_title_lags_collection": case["pot_title_at_collection"] !=
            case["collected_display"],
            "mushroom_routing_verified": False, "rake_verified": False,
            "automatic_visual_extraction": False}


def run(window, review_path, output):
    if verify_sha256sums(window):
        raise ValueError("source integrity failure")
    source = json.loads((window / "samples.json").read_text())
    review = json.loads(review_path.read_text())
    if review["source_sha256"] != source["source_sha256"]:
        raise ValueError("source mismatch")
    samples = {r["source_frame"]: r for r in source["samples"]}
    results, evidence = [], {}
    for case in review["cases"]:
        for key in ("before_frame", "after_frame", "collection_frame"):
            row = samples[case[key]]
            path = (window / row["file"]).resolve()
            if (not path.is_relative_to(window.resolve()) or
                    sha256_file(path) != row["sha256"]):
                raise ValueError("reviewed frame mismatch")
            evidence[str(case[key])] = row
        results.append(audit_case(case))
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps({
        "results": results, "evidence": evidence,
        "review_sha256": sha256_file(review_path)}, indent=2))
    write_sha256sums(output)
    print(json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("window", "review", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.window, args.review, args.output)
