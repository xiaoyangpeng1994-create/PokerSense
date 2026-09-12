"""Read-only, repeatable WPK capture-card preflight. Never labels an incomplete
calibration or heuristic-only strategy as a production-ready product.

PYTHONPATH=src python -m tools.wpk_readiness --dataset-root G:/.../dataset
Exit 0 means these prerequisite checks pass, NOT proven profitable play.
Exit 2 reports remaining blockers. No labels, configs or hardware are changed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import sys

from poker_engine.desktop.live import load_measured_calibrations
from poker_engine.desktop.table_rules import load_user_table_rules, validate_table_rules
from poker_engine.strategy.registry import build_strategy_router


def inspect(repo: Path, dataset_root: Path | None = None) -> dict:
    blocked = []
    fields = load_measured_calibrations(
        repo / "configs/vision/wepoker_android_capture_card")
    expected = {"card", "board", "street", "amount", "stack", "dealer",
                "action", "occupancy", "actor"}
    blocked.extend(f"calibration_missing:{name}"
                   for name in sorted(expected - fields.keys()))
    calibration = json.loads((repo / "configs/vision/wepoker_android_capture_card"
                              / "calibration.json").read_text(encoding="utf-8"))
    if calibration.get("card_fused", {}).get("requires_revalidation"):
        blocked.append("card_model_revalidation_pending")
    rules = validate_table_rules(load_user_table_rules(
        repo / "configs/game/wpk-capture-card.json"))
    if rules.unavailable_reason:
        blocked.append(rules.unavailable_reason)
    providers = build_strategy_router().providers
    covered = {street.value for provider in providers
               for street in provider.capability.streets}
    blocked.extend(f"strategy_missing:{street}" for street in ("flop", "turn", "river")
                   if street not in covered)
    dataset = None
    if dataset_root is not None:
        labels = [json.loads(line) for line in
                  (dataset_root / "labels/frames.jsonl").read_text(encoding="utf-8")
                  .splitlines() if line.strip()]
        groups = defaultdict(set)
        for split in ("train", "eval"):
            members = set((dataset_root / f"splits/b4_{split}_frames.txt")
                          .read_text(encoding="utf-8").split())
            groups[split] = {(row["session_id"], row["hand_id"]) for row in labels
                             if row["frame"] in members}
        overlap = sorted(groups["train"] & groups["eval"])
        if overlap:
            blocked.append("train_eval_hand_overlap")
        dataset = {"labeled_frames": len(labels), "overlapping_hands": overlap}
    else:
        blocked.append("independent_dataset_not_checked")
    return {
        "scope": "WPK / capture-card / 6-8 seats",
        "status": "BLOCKED" if blocked else "PREREQUISITES_PASS",
        "blockers": blocked, "python": sys.version.split()[0],
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "opencv-python")},
        "calibrated_fields": sorted(fields), "dataset": dataset,
        "providers": [provider.provider_id for provider in providers],
        "not_verified": ["continuous raw video", "live capture-to-render latency",
                         "30-minute soak", "independent strategy quality",
                         "profitability"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    result = inspect(args.repo, args.dataset_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 2 if result["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
