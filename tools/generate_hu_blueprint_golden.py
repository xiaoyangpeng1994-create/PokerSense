#!/usr/bin/env python3
"""Generate pinned HU blueprint parity data from an upstream checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT / "tests/fixtures/strategy/provider/hu_preflop_blueprint_golden.json"
)
REPOSITORY = "https://github.com/amaster97/poker_solver"
PINNED_COMMIT = "f78f1b2bc338dd8cbb5226ecb8398bbdb3635676"
PACKAGE_VERSION = "1.11.0"
RANKS = "AKQJT98765432"
HANDS = tuple(
    [rank + rank for rank in RANKS]
    + [high + low + "s" for i, high in enumerate(RANKS) for low in RANKS[i + 1:]]
    + [high + low + "o" for i, high in enumerate(RANKS) for low in RANKS[i + 1:]]
)
EXTRA_QUERIES = (
    (100, 0.0, "c", "72o", "BB"),
    (100, 0.0, "b200", "AKs", "BB"),
    (100, 0.0, "b300", "AA", "BB"),
    (100, 0.0, "b500", "72o", "BB"),
    (100, 0.0, "b300r700", "AA", "BTN"),
    (100, 0.0, "b300r900", "AKs", "BTN"),
    (100, 0.5, "", "AA", "BTN"),
    (100, 0.5, "c", "AKs", "BB"),
    (100, 1.0, "", "AA", "BTN"),
    (20, 1.0, "", "72o", "BTN"),
    (20, 1.0, "b300", "AA", "BB"),
)


def _revision(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _probabilities(loader, query: dict[str, object]) -> list[str]:
    values = loader.lookup(**query)
    if values is None:
        raise RuntimeError(f"missing upstream lookup: {query}")
    return [repr(float(value)) for value in values]


def build(repo: Path) -> dict[str, object]:
    repo = repo.resolve()
    revision = _revision(repo)
    if revision != PINNED_COMMIT:
        raise RuntimeError(
            f"checkout commit {revision} != pinned {PINNED_COMMIT}"
        )
    asset_dir = repo / "assets/blueprints"
    manifest_bytes = (asset_dir / "manifest.json").read_bytes()
    sys.path.insert(0, str(repo))
    from poker_solver.blueprint_loader import BlueprintLoader  # noqa: PLC0415

    loader = BlueprintLoader.from_dir(asset_dir, verify_sha256=True)
    entries = {
        (entry.stack_bb, float(entry.ante_bb)): entry
        for entry in loader.manifest.entries
    }
    root_query = {"stack_bb": 100, "ante": 0.0, "action_history": ""}
    root_entry = entries[(100, 0.0)]
    root_labels = loader.actions(**root_query)
    hands = {
        hand: _probabilities(loader, {**root_query, "hand": hand})
        for hand in HANDS
    }
    additional = []
    for stack_bb, ante_bb, history, hand, position in EXTRA_QUERIES:
        query = {
            "stack_bb": stack_bb,
            "ante": ante_bb,
            "action_history": history,
        }
        labels = loader.actions(**query)
        if labels is None:
            raise RuntimeError(f"missing upstream actions: {query}")
        entry = entries[(stack_bb, ante_bb)]
        additional.append({
            "stack_bb": stack_bb,
            "ante_bb": ante_bb,
            "action_history": history,
            "position": position,
            "hand": hand,
            "action_labels": labels,
            "probabilities": _probabilities(loader, {**query, "hand": hand}),
            "shard_sha256": entry.sha256,
        })
    return {
        "fixture_version": "2.0.0",
        "source": {
            "repository": REPOSITORY,
            "commit": revision,
            "package_version": PACKAGE_VERSION,
            "license": "MIT",
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "manifest_schema": loader.manifest.schema_version,
            "asset_version": loader.manifest.premium_a_version,
            "shard_sha256": root_entry.sha256,
        },
        "spot": {
            "player_count": 2,
            "street": "preflop",
            "stack_bb": 100,
            "ante_bb": 0.0,
            "action_history": "",
            "position": "BTN",
            "action_labels": root_labels,
        },
        "hands": hands,
        "additional_spots": additional,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = (
        json.dumps(build(args.repo), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n"
    ).encode()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_bytes() != payload:
            raise SystemExit("HU blueprint Golden is out of date")
        print("HU blueprint Golden is current (180 lookups)")
        return 0
    OUTPUT.write_bytes(payload)
    print(f"generated HU blueprint Golden at {OUTPUT} (180 lookups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
