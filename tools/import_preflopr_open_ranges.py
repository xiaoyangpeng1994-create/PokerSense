#!/usr/bin/env python3
"""Create the reviewed 6/7/8/9-handed PreflopR RFI asset.

Upstream ``preflopR`` authors explicit range keys for 2, 6 and 9 players only.
For every other table size its ``get_range_key()`` walks a fallback chain
(``<n>_<pos>`` -> ``9_<pos>`` -> ``6_<pos>`` -> ``9_BTN``), and that last
resort silently renders a BB open-raise as the chart's widest BTN range.
PokerSense therefore refuses to replay the chain.

What this importer does instead:

* imports only keys explicitly authored upstream (``6_*`` and ``9_*``);
* rebuilds 7- and 8-handed coverage from upstream's own ``get_positions()``
  table, mapping each position to the **same-named** 9-handed key and nothing
  else -- the last-resort ``9_BTN`` step stays disabled;
* records every derived key in ``derived_ranges`` so the substitution is
  auditable at runtime instead of being hidden inside a lookup.

Positions come from parsing the pinned upstream source, never from a
hand-maintained list, so a change in upstream's position map fails ``--check``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path


RANKS = "AKQJT98765432"
SOURCE_URL = "https://github.com/bmorrow10/preflopR"


def _balanced_content(text: str, opening: int) -> tuple[str, int]:
    depth = 0
    quoted = False
    escaped = False
    for index in range(opening, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[opening + 1:index], index + 1
    raise ValueError("unterminated parenthesized expression")


def _split_calls(body: str) -> list[str]:
    values: list[str] = []
    start = 0
    depth = 0
    quoted = False
    escaped = False
    for index, char in enumerate(body):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            value = body[start:index].strip()
            if value:
                values.append(value)
            start = index + 1
    value = body[start:].strip()
    if value:
        values.append(value)
    return values


def _evaluate_call(value: str) -> list[str]:
    match = re.fullmatch(r'(pairs_down_to|suited|offsuit)\((.*)\)', value, re.S)
    if match is None:
        raise ValueError(f"unsupported upstream range expression: {value!r}")
    function, arguments = match.groups()
    ranks = re.findall(r'"([AKQJT2-9]+)"', arguments)
    if function == "pairs_down_to":
        if len(ranks) != 1:
            raise ValueError(f"invalid pairs_down_to expression: {value!r}")
        minimum = ranks[0][0]
        if minimum not in RANKS:
            raise ValueError(f"invalid pair rank: {minimum!r}")
        return [rank * 2 for rank in RANKS[:RANKS.index(minimum) + 1]]
    if len(ranks) != 2 or any(len(rank) != 1 for rank in ranks):
        raise ValueError(f"invalid suited/offsuit expression: {value!r}")
    suffix = "s" if function == "suited" else "o"
    return ["".join(ranks) + suffix]


def parse_upstream_positions(source: str) -> dict[int, tuple[str, ...]]:
    """Read upstream's ``get_positions()`` table straight from the source.

    Upstream defines the legal positions per table size here, so deriving 7/8
    handed coverage must start from this table rather than a local guess.
    """
    marker = "get_positions <- function"
    marker_index = source.find(marker)
    if marker_index < 0:
        raise ValueError("upstream get_positions() not found")
    body, _ = _balanced_content(source, source.index("{", marker_index))
    positions: dict[int, tuple[str, ...]] = {}
    pattern = re.compile(r'"(\d)"\s*=\s*c\(([^)]*)\)', re.S)
    for match in pattern.finditer(body):
        names = re.findall(r'"([^"]+)"', match.group(2))
        if not names:
            raise ValueError(f"empty upstream position list: {match.group(1)}")
        positions[int(match.group(1))] = tuple(names)
    if not positions:
        raise ValueError("upstream position table is empty")
    return positions


# Table sizes rebuilt from the same-named 9-handed key. 3-5 handed stays out:
# upstream maps those through the same fallback chain, and the owner's games
# are 6-8 handed, so there is no evidence-backed reason to widen the surface.
DERIVED_PLAYER_COUNTS = (7, 8)


def derive_same_name_ranges(
    ranges: Mapping[str, list[str]],
    positions: Mapping[int, tuple[str, ...]],
) -> dict[str, str]:
    """Map each 7/8-handed position onto the same-named 9-handed range key.

    Only an exact same-name key is accepted. A position whose 9-handed key is
    missing is a hard error instead of a silent fall back to ``6_*`` or to the
    widest chart (``9_BTN``) -- replaying either would reintroduce exactly the
    behaviour PokerSense's review rejected.
    """
    derived: dict[str, str] = {}
    for players in DERIVED_PLAYER_COUNTS:
        names = positions.get(players)
        if names is None:
            raise ValueError(f"upstream has no position table for {players} players")
        for name in names:
            if name == "BB":
                continue  # BB never faces an unopened RFI decision.
            source_key = f"9_{name}"
            if source_key not in ranges:
                raise ValueError(
                    f"no same-named 9-handed range for {players}_{name}; "
                    "refusing to fall back to another position"
                )
            derived[f"{players}_{name}"] = source_key
    return derived


def parse_explicit_ranges(source: str) -> dict[str, list[str]]:
    marker = "RANGES <- list("
    marker_index = source.find(marker)
    if marker_index < 0:
        raise ValueError("upstream RANGES list not found")
    opening = marker_index + len(marker) - 1
    body, _ = _balanced_content(source, opening)
    ranges: dict[str, list[str]] = {}
    pattern = re.compile(r'^\s*"([269]_[^"]+)"\s*=\s*c\(', re.M)
    for match in pattern.finditer(body):
        key = match.group(1)
        if not key.startswith(("6_", "9_")):
            continue
        call_body, _ = _balanced_content(body, match.end() - 1)
        hands: list[str] = []
        for call in _split_calls(call_body):
            hands.extend(_evaluate_call(call))
        if len(hands) != len(set(hands)):
            raise ValueError(f"duplicate hands in upstream range {key}")
        ranges[key] = hands
    expected = {
        "6_UTG", "6_HJ", "6_CO", "6_BTN", "6_SB",
        "9_UTG", "9_UTG+1", "9_UTG+2", "9_MP", "9_HJ",
        "9_CO", "9_BTN", "9_SB",
    }
    if set(ranges) != expected:
        raise ValueError(
            f"reviewed upstream key set changed: {sorted(set(ranges) ^ expected)}"
        )
    return ranges


def build_asset(source_path: Path, revision: str) -> dict[str, object]:
    source_bytes = source_path.read_bytes()
    source_text = source_bytes.decode("utf-8")
    ranges = parse_explicit_ranges(source_text)
    derived = derive_same_name_ranges(
        ranges, parse_upstream_positions(source_text)
    )
    return {
        "schema_version": 1,
        "asset_id": "preflopr-explicit-rfi-ranges",
        "asset_version": "2",
        "source_url": SOURCE_URL,
        "source_revision": revision,
        "source_file": source_path.name,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "license": "MIT",
        "scope": {
            "player_counts": [6, 7, 8, 9],
            "street": "preflop",
            "action_line": "unopened",
            "stack_bb": "100",
            "ante_bb": "0",
            "rake_percent": "0",
        },
        "limitations": [
            "heuristic_open_raise_chart_not_solver_derived",
            "binary_raise_or_fold_frequency_only",
            "no_raise_size_or_ev",
            "bb_has_no_open_raise_decision",
            "upstream_2_3_4_5_handed_ranges_excluded",
            "upstream_last_resort_9_btn_fallback_disabled",
            "7_and_8_handed_derived_from_same_named_9_handed_ranges",
        ],
        "ranges": ranges,
        "derived_ranges": derived,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = build_asset(args.source, args.revision)
    rendered = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("generated asset differs from committed asset")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
