"""Score frozen checkpoints after continuous inference, without rerunning it."""

import argparse
from collections import Counter
import json
from pathlib import Path

from tools.capture_card_calibration.hashing import verify_sha256sums, write_sha256sums


def verdict(wanted, got):
    if wanted is None:
        return "negative_rejected" if got is None else "false_positive"
    return "correct" if wanted == got else "abstain" if got is None else "wrong"


def score(replay, truth_dir, output):
    if verify_sha256sums(replay) or verify_sha256sums(truth_dir):
        raise ValueError("input integrity failure")
    report = json.loads((replay / "report.json").read_text())
    truth = json.loads((truth_dir / "truth.json").read_text())
    if report["source_sha256"] != truth["source_sha256"]:
        raise ValueError("wrong source")
    predictions = {r["source_frame"]: r for r in map(json.loads,
                   (replay / "observations.jsonl").read_text().splitlines())}
    cards, money, pots, titles, rows = Counter(), Counter(), Counter(), Counter(), []
    for checkpoint in truth["checkpoints"]:
        frame = checkpoint["frame"]
        prediction = predictions[frame]
        for field, key, capacity in (("hero", "hero", 2), ("board", "board_slots", 5)):
            expected = checkpoint[field]
            if field == "board" and expected is None:
                continue
            expected = (expected or []) + [None] * (capacity - len(expected or []))
            got = prediction[key] or [None] * capacity
            if len(got) != capacity:
                raise ValueError("prediction slot count mismatch")
            for slot, (wanted, observed) in enumerate(zip(expected, got)):
                result = verdict(wanted, observed)
                cards[result] += 1
                rows.append({"frame": frame, "field": field, "slot": slot,
                             "expected": wanted, "got": observed, "verdict": result})
        money[verdict(checkpoint["cash"], prediction["stacks"]["5"])] += 1
        pots[verdict(checkpoint["pot"], prediction["pot"])] += 1
        titles[verdict(checkpoint["pot"], prediction.get("pot_title"))] += 1
    output.mkdir(parents=True, exist_ok=False)
    result = {"card_counts": dict(cards), "hero_cash_counts": dict(money),
              "pot_counts": dict(pots),
              "pot_title_counts": dict(titles),
              "pot_truth_semantics": "reviewed_display_title_not_canonical_ledger",
              "rows": rows, "whole_hand_verified": False, "independent_holdout": False}
    (output / "report.json").write_text(json.dumps(result, indent=2))
    write_sha256sums(output)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("replay", "truth", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    score(args.replay, args.truth, args.output)
