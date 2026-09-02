#!/usr/bin/env python3
"""Verify PokerSense's HU blueprint Golden against an upstream checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


EXPECTED_REPOSITORY = "https://github.com/amaster97/poker_solver"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path, help="upstream poker_solver checkout")
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path(
            "tests/fixtures/strategy/provider/hu_preflop_blueprint_golden.json"
        ),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    golden = json.loads(args.golden.read_text())
    source = golden["source"]
    if source["repository"] != EXPECTED_REPOSITORY:
        raise SystemExit("Golden repository identity is unexpected")
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != source["commit"]:
        raise SystemExit(f"checkout commit {revision} != Golden {source['commit']}")

    asset_dir = repo / "assets/blueprints"
    manifest_bytes = (asset_dir / "manifest.json").read_bytes()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if digest != source["manifest_sha256"]:
        raise SystemExit(f"manifest SHA {digest} != Golden SHA")

    sys.path.insert(0, str(repo))
    from poker_solver.blueprint_loader import BlueprintLoader  # noqa: PLC0415

    loader = BlueprintLoader.from_dir(asset_dir, verify_sha256=True)
    entries = {
        (entry.stack_bb, float(entry.ante_bb)): entry.sha256
        for entry in loader.manifest.entries
    }
    spot = golden["spot"]
    if len(golden["hands"]) != 169:
        raise SystemExit("Golden root must contain all 169 hand classes")
    if entries[(spot["stack_bb"], spot["ante_bb"])] != source[
        "shard_sha256"
    ]:
        raise SystemExit("root shard SHA differs from manifest")
    labels = loader.actions(
        stack_bb=spot["stack_bb"],
        ante=spot["ante_bb"],
        action_history=spot["action_history"],
    )
    if labels != spot["action_labels"]:
        raise SystemExit("upstream action labels differ from Golden")
    for hand, expected in golden["hands"].items():
        actual = loader.lookup(
            stack_bb=spot["stack_bb"],
            ante=spot["ante_bb"],
            hand=hand,
            action_history=spot["action_history"],
        )
        rendered = [repr(float(value)) for value in actual]
        if rendered != expected:
            raise SystemExit(f"upstream probabilities differ for {hand}")
    for extra in golden.get("additional_spots", []):
        if entries[(extra["stack_bb"], extra["ante_bb"])] != extra[
            "shard_sha256"
        ]:
            raise SystemExit("additional spot shard SHA differs from manifest")
        labels = loader.actions(
            stack_bb=extra["stack_bb"],
            ante=extra["ante_bb"],
            action_history=extra["action_history"],
        )
        if labels != extra["action_labels"]:
            raise SystemExit(
                f"upstream action labels differ at {extra['action_history']}"
            )
        actual = loader.lookup(
            stack_bb=extra["stack_bb"],
            ante=extra["ante_bb"],
            hand=extra["hand"],
            action_history=extra["action_history"],
        )
        if [repr(float(value)) for value in actual] != extra["probabilities"]:
            raise SystemExit(
                f"upstream probabilities differ at {extra['action_history']}"
            )
    count = len(golden["hands"]) + len(golden.get("additional_spots", []))
    print(f"verified {count} Golden lookups at {revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
