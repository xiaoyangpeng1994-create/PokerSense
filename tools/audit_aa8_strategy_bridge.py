"""Audit saved AA8 observations against the shadow-strategy input contract."""

import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path

from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.aa8_shadow import assess_aa8_shadow_input
from poker_engine.strategy.contracts import GameConfig, GameType
from tools.aa8_action_transfer import sha


def audit(observations, player_count, output):
    if player_count not in (6, 7, 8):
        raise ValueError("player_count must be 6, 7 or 8")
    # Amounts are structural placeholders only. This audit never routes a
    # provider or produces advice; real table rules remain a separate gate.
    structural = GameConfig(
        "NLHE", GameType.CASH, 8, player_count,
        ChipAmount("1"), ChipAmount("2"),
        ante=ChipAmount("0"), rake_percent=Decimal("0"),
        rake_cap=ChipAmount("0"), minimum_chip=ChipAmount("1"),
    )
    reasons, classifications = Counter(), Counter()
    frames = 0
    with observations.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            blockers = assess_aa8_shadow_input(row, structural)
            frames += 1
            if any(reason.startswith("deferred_special_mode:") for reason in blockers):
                classifications["DEFERRED_SPECIAL_MODE"] += 1
            elif blockers:
                classifications["ABSTAIN"] += 1
            else:
                classifications["STRUCTURALLY_READY"] += 1
            reasons.update(blockers)
    result = {
        "schema_version": 1,
        "frames": frames,
        "player_count_parameter": player_count,
        "classifications": dict(classifications),
        "blocker_counts": dict(reasons),
        "observations_sha256": sha(observations),
        "structural_placeholder_rules_only": True,
        "table_rules_authorized_for_strategy": False,
        "provider_or_equity_executed": False,
        "advice_emitted": False,
        "independent_holdout": False,
        "strategy_eligible": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "frames": frames,
        "classifications": dict(classifications),
        "top_blockers": reasons.most_common(8),
    }))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--player-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.observations, args.player_count, args.output)
