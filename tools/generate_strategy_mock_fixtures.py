#!/usr/bin/env python3
"""Generate deterministic target-architecture strategy mock fixtures.

The generated corpus is synthetic contract data. It is intentionally not
eligible as real-provider Golden data, real-capture Replay evidence, or a
hardware performance acceptance result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0.0"
GENERATED_AT = "2026-08-22T00:00:00Z"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "tests" / "fixtures" / "strategy" / "v1"

REQUIREMENTS = {
    "REQ-IN-001", "REQ-IN-002", "REQ-ST-001", "REQ-ST-002",
    "REQ-CTX-001", "REQ-MET-001", "REQ-RNG-001", "REQ-EQ-001",
    "REQ-PRV-001", "REQ-RTR-001", "REQ-FUS-001", "REQ-OUT-001",
    "REQ-OUT-002", "REQ-UI-001", "REQ-PERF-001", "REQ-AUD-001",
    "REQ-TRN-001",
}

ADVICE_STATUSES = {"READY", "PARTIAL", "ABSTAIN", "STALE"}
FIXTURE_TYPES = {"synthetic", "benchmark"}
STREETS = {"preflop", "flop", "turn", "river"}
LOOKUP_STATES = {
    "NOT_CHECKED", "HIT_EXACT", "HIT_APPROXIMATE", "NOT_FOUND",
    "NOT_APPLICABLE", "REJECTED", "NO_STRATEGY",
}
POSITIONS = {
    2: ["BTN", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["CO", "BTN", "SB", "BB"],
    5: ["HJ", "CO", "BTN", "SB", "BB"],
    6: ["UTG", "HJ", "CO", "BTN", "SB", "BB"],
    7: ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    8: ["UTG", "UTG1", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    9: [
        "UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB",
    ],
}
ACTION_LINES = (
    "unopened", "limp", "multi_limp", "raise", "three_bet",
    "four_bet", "squeeze", "iso_raise", "all_in",
)
QUALITY_FIELDS = (
    "hero_cards", "board_cards", "dealer_button", "street", "pot",
    "player_stacks", "actor", "action_history", "blinds",
)


SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://pokersense.local/schema/strategy-fixture-v1.json",
    "title": "PokerSense strategy fixture v1",
    "type": "object",
    "required": [
        "fixture_schema_version", "fixture_id", "fixture_type", "title",
        "requirements", "function_ids", "test_ids", "tags", "input",
        "expected", "tolerances",
    ],
    "properties": {
        "fixture_schema_version": {"const": SCHEMA_VERSION},
        "fixture_id": {"type": "string", "pattern": "^MOCK-[A-Z0-9-]+$"},
        "fixture_type": {"enum": sorted(FIXTURE_TYPES)},
        "title": {"type": "string", "minLength": 1},
        "requirements": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": {"enum": sorted(REQUIREMENTS)},
        },
        "function_ids": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": {"type": "string"},
        },
        "test_ids": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": {"type": "string"},
        },
        "tags": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": {"type": "string"},
        },
        "input": {
            "type": "object",
            "required": [
                "game_config", "request_context", "observations", "state",
                "ranges", "providers", "fault_injection",
            ],
        },
        "expected": {
            "type": "object",
            "required": [
                "terminal_stage", "advice", "provider_lookups", "assertions",
            ],
        },
        "tolerances": {
            "type": "object",
            "required": ["probability_abs", "equity_abs", "money_abs"],
        },
    },
    "additionalProperties": False,
}


def _slug(value: str) -> str:
    return value.upper().replace("_", "-").replace("+", "PLUS")


def _money(value: str | int | Decimal) -> str:
    text = format(Decimal(value), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _seats(player_count: int) -> list[dict[str, Any]]:
    result = []
    for seat, position in enumerate(POSITIONS[player_count]):
        result.append({
            "seat_id": seat,
            "player_id": "hero" if seat == player_count - 1 else f"p{seat}",
            "position": position,
            "occupied": True,
            "status": "active",
            "is_hero": seat == player_count - 1,
            "is_dealer": position == "BTN",
            "starting_stack": "100",
            "stack": "100",
            "street_committed": "0",
            "hand_committed": "0",
        })
    return result


def _base_fixture(
    fixture_id: str,
    title: str,
    player_count: int = 2,
    street: str = "preflop",
    active_player_count: int | None = None,
) -> dict[str, Any]:
    active = active_player_count or player_count
    board_by_street = {
        "preflop": [],
        "flop": ["2c", "7d", "Jh"],
        "turn": ["2c", "7d", "Jh", "9s"],
        "river": ["2c", "7d", "Jh", "9s", "3h"],
    }
    seats = _seats(player_count)
    for seat in seats[:player_count - active]:
        seat["status"] = "folded"
    hero_seat = player_count - 1
    sb_seat = next(
        (seat["seat_id"] for seat in seats if seat["position"] == "SB"),
        next(seat["seat_id"] for seat in seats if seat["position"] == "BTN"),
    )
    bb_seat = next(
        seat["seat_id"] for seat in seats if seat["position"] == "BB"
    )
    if street == "preflop":
        for seat_id, amount in ((sb_seat, "0.5"), (bb_seat, "1")):
            seat = seats[seat_id]
            seat["stack"] = "99.5" if amount == "0.5" else "99"
            seat["street_committed"] = amount
            seat["hand_committed"] = amount
    else:
        for seat in seats:
            if seat["status"] != "folded":
                seat["stack"] = "97"
                seat["hand_committed"] = "3"
    provider_id = f"mock-preflop-{player_count}p-v1"
    if street != "preflop":
        provider_id = f"mock-postflop-{active}p-v1"
    return {
        "fixture_schema_version": SCHEMA_VERSION,
        "fixture_id": fixture_id,
        "fixture_type": "synthetic",
        "title": title,
        "requirements": ["REQ-CTX-001", "REQ-OUT-001"],
        "function_ids": ["CTX-001", "ADV-001"],
        "test_ids": ["T-CTX-001", "T-ADV-001"],
        "tags": [street, f"dealt-{player_count}", f"active-{active}"],
        "input": {
            "game_config": {
                "variant": "NLHE",
                "game_type": "cash",
                "max_seats": player_count,
                "dealt_player_count": player_count,
                "small_blind": "0.5",
                "big_blind": "1",
                "ante": "0",
                "rake_percent": "0",
                "rake_cap": "0",
                "minimum_chip": "0.5",
            },
            "request_context": {
                "hand_id": fixture_id.lower(),
                "state_version": 1,
                "request_id": f"req-{fixture_id.lower()}",
                "requested_at": "2026-08-22T00:00:00Z",
                "expires_at": "2026-08-22T00:00:02Z",
                "deadline_ms": 300,
            },
            "observations": {
                "window_identity": "mock-window-1",
                "frame_sequences": [100, 101],
                "field_quality": {
                    field: {
                        "status": "VALID",
                        "confidence": "0.99",
                        "source": "synthetic",
                        "evidence_ref": f"mock://{fixture_id}/{field}",
                    }
                    for field in QUALITY_FIELDS
                },
            },
            "state": {
                "street": street,
                "dealt_player_count": player_count,
                "active_player_count": active,
                "hero_seat": hero_seat,
                "actor_seat": hero_seat,
                "dealer_seat": next(
                    seat["seat_id"] for seat in seats if seat["is_dealer"]
                ),
                "hero_cards": ["As", "Kd"],
                "board_cards": board_by_street[street],
                "seats": seats,
                "pots": [{
                    "pot_id": "main",
                    "amount": "1.5" if street == "preflop"
                    else str(3 * active),
                    "eligible_seats": [
                        seat["seat_id"] for seat in seats
                        if seat["status"] != "folded"
                    ],
                }],
                "current_bet": "1" if street == "preflop" else "0",
                "to_call": "0",
                "legal_actions": [
                    {"action": "check", "min": "0", "max": "0"},
                    {"action": "bet", "min": "1", "max": "99"},
                ] if street != "preflop" else [
                    {"action": "check", "min": "0", "max": "0"},
                    {"action": "raise", "min": "2", "max": "100"},
                ],
                "action_history": [
                    {"sequence": 1, "street": "preflop", "seat_id": sb_seat,
                     "action": "post_sb", "amount": "0.5"},
                    {"sequence": 2, "street": "preflop", "seat_id": bb_seat,
                     "action": "post_bb", "amount": "1"},
                ],
            },
            "ranges": {
                "hero": {
                    "source": "known_cards", "version": "v1",
                    "combo_weights": {"AsKd": "1"}, "confidence": "1",
                },
                "villains": [
                    {
                        "seat_id": seat["seat_id"],
                        "source": "mock-population",
                        "version": "v1",
                        "combo_weights": {
                            "AA": "0.05", "AKs": "0.15", "AQo": "0.30",
                            "random": "0.50",
                        },
                        "entropy": "0.80",
                        "effective_sample_size": 0,
                        "confidence": "0.50",
                    }
                    for seat in seats
                    if not seat["is_hero"] and seat["status"] != "folded"
                ],
            },
            "providers": [{
                "provider_id": provider_id,
                "source_version": "mock-v1",
                "asset_hash": "sha256:mock-not-release-evidence",
                "fixture_eligibility": "synthetic-only",
                "capability": {
                    "player_counts": [active if street != "preflop"
                                      else player_count],
                    "streets": [street],
                    "stack_buckets_bb": [
                        10, 20, 30, 40, 60, 80, 100, 150, 200,
                    ],
                    "ante": ["0"],
                    "rake_profiles": ["zero"],
                    "action_lines": list(ACTION_LINES),
                    "match_kind": "exact",
                },
                "mock_result": {
                    "action_probabilities": {
                        "check": "0.40", "raise": "0.60",
                    } if street == "preflop" else {
                        "check": "0.40", "bet": "0.60",
                    },
                    "recommended_sizes": ["2.5"] if street == "preflop"
                    else ["2"],
                },
            }],
            "fault_injection": None,
        },
        "expected": {
            "terminal_stage": "advice",
            "advice": {
                "status": "READY",
                "match_kind": "exact",
                "strategy_source": provider_id,
                "strategy_version": "mock-v1",
                "action_probabilities": {
                    "check": "0.40", "raise": "0.60",
                } if street == "preflop" else {
                    "check": "0.40", "bet": "0.60",
                },
                "reason_codes": [],
                "must_expire": True,
            },
            "provider_lookups": [{
                "provider_id": provider_id, "state": "HIT_EXACT",
            }],
            "assertions": [
                "state_is_immutable", "evidence_chain_complete",
                "actions_are_legal", "probabilities_sum_to_one",
                "result_matches_hand_state_request",
            ],
        },
        "tolerances": {
            "probability_abs": "0.000000001",
            "equity_abs": "0.01",
            "money_abs": "0",
            "latency_ms": None,
        },
    }


def _set_requirements(
    fixture: dict[str, Any], requirements: list[str], functions: list[str],
    tests: list[str],
) -> None:
    fixture["requirements"] = requirements
    fixture["function_ids"] = functions
    fixture["test_ids"] = tests


def _set_abstain(
    fixture: dict[str, Any], reason: str, stage: str = "rejection_gate",
) -> None:
    fixture["expected"]["terminal_stage"] = stage
    fixture["expected"]["advice"].update({
        "status": "ABSTAIN",
        "match_kind": None,
        "strategy_source": None,
        "strategy_version": None,
        "action_probabilities": {},
        "reason_codes": [reason],
    })


def _preflop_action(fixture: dict[str, Any], action_line: str) -> None:
    state = fixture["input"]["state"]
    state["action_line"] = action_line
    history = state["action_history"]
    non_hero = [
        seat["seat_id"] for seat in state["seats"] if not seat["is_hero"]
    ]
    sequence = len(history) + 1
    templates = {
        "unopened": [],
        "limp": [(non_hero[0], "call", "1")],
        "multi_limp": [
            (seat, "call", "1") for seat in non_hero[:min(2, len(non_hero))]
        ],
        "raise": [(non_hero[0], "raise", "2.5")],
        "three_bet": [
            (non_hero[0], "raise", "2.5"),
            (non_hero[-1], "raise", "8"),
        ],
        "four_bet": [
            (non_hero[0], "raise", "2.5"),
            (non_hero[-1], "raise", "8"),
            (non_hero[0], "raise", "22"),
        ],
        "squeeze": [
            (non_hero[0], "raise", "2.5"),
            (non_hero[min(1, len(non_hero) - 1)], "call", "2.5"),
        ],
        "iso_raise": [
            (non_hero[0], "call", "1"),
            (non_hero[-1], "raise", "4"),
        ],
        "all_in": [(non_hero[0], "all_in", "100")],
    }
    for seat_id, action, amount in templates[action_line]:
        history.append({
            "sequence": sequence, "street": "preflop", "seat_id": seat_id,
            "action": action, "amount": amount,
        })
        seat = state["seats"][seat_id]
        seat["street_committed"] = amount
        seat["hand_committed"] = amount
        seat["stack"] = _money(Decimal("100") - Decimal(amount))
        if action == "all_in":
            seat["status"] = "all_in"
        sequence += 1
    commitments = [
        Decimal(seat["street_committed"]) for seat in state["seats"]
    ]
    current_bet = max(commitments)
    hero = state["seats"][state["hero_seat"]]
    to_call = max(Decimal("0"), current_bet - Decimal(hero["street_committed"]))
    state["current_bet"] = _money(current_bet)
    state["to_call"] = _money(to_call)
    state["pots"][0]["amount"] = _money(sum(commitments, Decimal("0")))
    if to_call == Decimal("0"):
        state["legal_actions"] = [
            {"action": "check", "min": "0", "max": "0"},
            {"action": "raise", "min": "2", "max": hero["stack"]},
        ]
    else:
        state["legal_actions"] = [
            {"action": "fold", "min": "0", "max": "0"},
            {"action": "call", "min": state["to_call"],
             "max": state["to_call"]},
        ]
        if current_bet < Decimal("100"):
            state["legal_actions"].append({
                "action": "raise",
                "min": _money(min(Decimal("100"), current_bet * 2)),
                "max": hero["stack"],
            })
    if to_call == Decimal("0"):
        probabilities = {"check": "0.40", "raise": "0.60"}
    elif current_bet >= Decimal("100"):
        probabilities = {"fold": "0.20", "call": "0.80"}
    else:
        probabilities = {"fold": "0.20", "call": "0.30", "raise": "0.50"}
    fixture["input"]["providers"][0]["mock_result"][
        "action_probabilities"
    ] = probabilities
    fixture["expected"]["advice"]["action_probabilities"] = probabilities


def generate_preflop() -> list[dict[str, Any]]:
    fixtures = []
    for player_count in range(2, 10):
        for action_line in ACTION_LINES:
            fixture_id = f"MOCK-PF-{player_count}P-{_slug(action_line)}"
            f = _base_fixture(
                fixture_id,
                f"{player_count}-player preflop {action_line}",
                player_count,
            )
            _preflop_action(f, action_line)
            f["tags"].extend(["positive", "router", action_line])
            _set_requirements(
                f,
                ["REQ-ST-002", "REQ-CTX-001", "REQ-PRV-001",
                 "REQ-RTR-001", "REQ-OUT-001"],
                ["ST-005", "CTX-001", "PRV-001", "RTR-002", "RTR-003",
                 "ADV-001"],
                ["T-ST-003", "T-RTR-005", "T-INT-011",
                 "E2E-001" if player_count == 2 else "E2E-003"],
            )
            fixtures.append(f)

    for player_count in range(3, 10):
        fixture_id = f"MOCK-PF-{player_count}P-HU-PROVIDER-ONLY"
        f = _base_fixture(fixture_id, "HU provider must not serve multiplayer",
                          player_count)
        f["input"]["providers"] = [
            copy.deepcopy(_base_fixture("MOCK-TEMP", "temp")["input"]
                          ["providers"][0])
        ]
        f["expected"]["provider_lookups"] = [{
            "provider_id": "mock-preflop-2p-v1",
            "state": "NOT_APPLICABLE",
        }]
        _set_abstain(f, "unsupported_player_count", "router")
        f["tags"].extend(["negative", "provider-mismatch"])
        _set_requirements(
            f, ["REQ-PRV-001", "REQ-RTR-001", "REQ-OUT-002"],
            ["PRV-001", "RTR-002", "FUS-004"],
            ["T-RTR-001", "T-INT-004", "E2E-002"],
        )
        fixtures.append(f)
    return fixtures


def generate_postflop() -> list[dict[str, Any]]:
    fixtures = []
    for street in ("flop", "turn", "river"):
        for active in range(2, 10):
            fixture_id = f"MOCK-{_slug(street)}-{active}WAY"
            f = _base_fixture(
                fixture_id, f"{active}-way {street}", active, street, active,
            )
            f["tags"].extend(["positive", "postflop", "multiway-equity"])
            _set_requirements(
                f,
                ["REQ-ST-002", "REQ-RNG-001", "REQ-EQ-001",
                 "REQ-PRV-001", "REQ-RTR-001"],
                ["ST-007", "RNG-005", "EQ-005", "PRV-004", "RTR-002"],
                ["T-ST-005", "T-RNG-004", "T-EQ-004", "T-INT-007"],
            )
            fixtures.append(f)

    for dealt in (6, 9):
        fixture_id = f"MOCK-FLOP-{dealt}P-TO-HU"
        f = _base_fixture(
            fixture_id, f"{dealt}-player preflop history reaches HU flop",
            dealt, "flop", 2,
        )
        f["tags"].extend(["lineage", "postflop-hu"])
        f["expected"]["assertions"].append(
            "ranges_preserve_original_multiplayer_history"
        )
        _set_requirements(
            f, ["REQ-ST-002", "REQ-RNG-001", "REQ-RTR-001"],
            ["ST-005", "RNG-003", "RTR-002"],
            ["E2E-004"],
        )
        fixtures.append(f)
    return fixtures


def generate_quality() -> list[dict[str, Any]]:
    fixtures = []
    for field in QUALITY_FIELDS:
        for status, confidence in (
            ("UNKNOWN", "0"), ("LOW_CONFIDENCE", "0.49"),
            ("CONFLICT", "0.50"),
        ):
            fixture_id = f"MOCK-QUALITY-{_slug(field)}-{status}"
            f = _base_fixture(fixture_id, f"{field} is {status}")
            quality = f["input"]["observations"]["field_quality"][field]
            quality["status"] = status
            quality["confidence"] = confidence
            f["tags"].extend(["negative", "input-quality", status.lower()])
            reason = f"{field}_{status.lower()}"
            _set_abstain(f, reason, "context_quality")
            f["expected"]["assertions"].extend([
                "unstable_input_does_not_mutate_state",
                "no_action_probabilities_are_exposed",
            ])
            _set_requirements(
                f, ["REQ-IN-001", "REQ-IN-002", "REQ-CTX-001",
                    "REQ-OUT-002"],
                ["ST-001", "CTX-002", "CTX-003", "FUS-004"],
                ["T-CTX-002", "T-FUS-002", "E2E-007"],
            )
            fixtures.append(f)
    return fixtures


def generate_provenance() -> list[dict[str, Any]]:
    """Cross-channel source, consensus, conflict, and absence fixtures."""
    cases = (
        (
            "VISION-ONLY", "pot", [
                {"source": "vision", "value": "12", "status": "VALID",
                 "confidence": "0.95", "evidence_ref": "frame://101/pot"},
            ], "VISION", "VALID", "READY", None,
        ),
        (
            "MANUAL-ONLY", "actor", [
                {"source": "manual", "value": 1, "status": "VALID",
                 "confidence": "1", "evidence_ref": "manual://session/actor"},
            ], "MANUAL", "VALID", "READY", None,
        ),
        (
            "CONFIG-ONLY", "blinds", [
                {"source": "config", "value": ["0.5", "1"],
                 "status": "VALID", "confidence": "1",
                 "evidence_ref": "config://table/blinds"},
            ], "CONFIG", "VALID", "READY", None,
        ),
        (
            "INFERRED-LOW", "actor", [
                {"source": "inferred", "value": 1,
                 "status": "LOW_CONFIDENCE", "confidence": "0.49",
                 "evidence_ref": "inference://action-sequence/actor"},
            ], "INFERRED", "LOW_CONFIDENCE", "ABSTAIN",
            "actor_low_confidence",
        ),
        (
            "SAME-VALUE-CONSENSUS", "pot", [
                {"source": "vision", "value": "12", "status": "VALID",
                 "confidence": "0.94", "evidence_ref": "frame://101/pot"},
                {"source": "manual", "value": "12", "status": "VALID",
                 "confidence": "1", "evidence_ref": "manual://session/pot"},
                {"source": "config", "value": "12", "status": "VALID",
                 "confidence": "1", "evidence_ref": "config://table/pot"},
            ], "MANUAL", "VALID", "READY", None,
        ),
        (
            "VISION-MANUAL-CONFLICT", "pot", [
                {"source": "vision", "value": "12", "status": "VALID",
                 "confidence": "0.94", "evidence_ref": "frame://101/pot"},
                {"source": "manual", "value": "13", "status": "VALID",
                 "confidence": "1", "evidence_ref": "manual://session/pot"},
            ], "MANUAL", "CONFLICT", "ABSTAIN", "pot_conflict",
        ),
        (
            "NULL-UNKNOWN", "actor", [
                {"source": "inferred", "value": None, "status": "VALID",
                 "confidence": "0", "evidence_ref": "inference://actor/null"},
            ], "INFERRED", "UNKNOWN", "ABSTAIN", "actor_unknown",
        ),
        (
            "EXPLICIT-CONFLICT", "street", [
                {"source": "vision", "value": "flop", "status": "CONFLICT",
                 "confidence": "0.7", "evidence_ref": "frame://101/street"},
                {"source": "manual", "value": "flop", "status": "VALID",
                 "confidence": "1", "evidence_ref": "manual://session/street"},
            ], "MANUAL", "CONFLICT", "ABSTAIN", "street_conflict",
        ),
    )
    fixtures = []
    for name, field, candidates, source, quality, advice, reason in cases:
        f = _base_fixture(
            f"MOCK-PROVENANCE-{name}",
            name.lower().replace("-", " "),
        )
        f["input"]["provenance_candidates"] = {
            field: candidates,
        }
        f["expected"]["resolved_provenance"] = [{
            "field_name": field,
            "source": source,
            "status": quality,
            "unique": True,
        }]
        f["expected"]["assertions"].extend([
            "one_provenance_record_per_field",
            "source_label_is_preserved",
            "candidate_order_does_not_change_resolution",
        ])
        f["tags"].extend(["input-provenance", source.lower(), quality.lower()])
        if advice == "ABSTAIN":
            _set_abstain(f, reason, "context_quality")
        _set_requirements(
            f,
            ["REQ-IN-001", "REQ-IN-002", "REQ-CTX-001", "REQ-OUT-002"],
            ["CTX-002", "CTX-003", "FUS-004"],
            ["T-CTX-002", "T-CTX-004", "E2E-007"],
        )
        fixtures.append(f)
    return fixtures


def generate_action_reconstruction() -> list[dict[str, Any]]:
    cases = (
        ("FOLD", "fold", "0", "0", "active", "folded", "0", "0",
         "EXACT", ["fold"], None),
        ("CHECK", None, "0", "0", "active", "active", "0", "0",
         "EXACT", ["check"], None),
        ("CALL", None, "10", "10", "active", "active", "10", "10",
         "EXACT", ["call"], None),
        ("BET", None, "20", "20", "active", "active", "0", "20",
         "EXACT", ["bet"], None),
        ("RAISE", None, "30", "30", "active", "active", "10", "30",
         "EXACT", ["raise"], None),
        ("SHORT-ALLIN-AMBIGUOUS", None, "40", "40", "active", "all_in",
         "100", "100", "AMBIGUOUS", ["all_in", "call"],
         "ambiguous_action_history"),
        ("SHORT-ALLIN-LABELED", "all_in", "40", "40", "active", "all_in",
         "100", "100", "EXACT", ["all_in"], None),
        ("CHIP-DELTA-MISMATCH", None, "10", "9", "active", "active",
         "10", "10", "INVALID", [], "chip_delta_mismatch"),
    )
    fixtures = []
    for (name, label, spent, pot_delta, before_status, after_status,
         before_bet, after_bet, status, candidates, reason) in cases:
        f = _base_fixture(
            f"MOCK-ACTION-RECONSTRUCTION-{name}",
            name.lower().replace("-", " "),
        )
        before_stack = "40" if "SHORT-ALLIN" in name else "100"
        before_commit = "0"
        after_stack = _money(Decimal(before_stack) - Decimal(spent))
        after_commit = spent
        f["input"]["action_transition"] = {
            "actor_seat": 0,
            "observed_action": label,
            "before": {
                "stack": before_stack,
                "street_committed": before_commit,
                "status": before_status,
                "pot": before_bet,
                "current_bet": before_bet,
                "villain_committed": before_bet,
            },
            "after": {
                "stack": after_stack,
                "street_committed": after_commit,
                "status": after_status,
                "pot": _money(Decimal(before_bet) + Decimal(pot_delta)),
                "current_bet": after_bet,
                "villain_committed": before_bet,
            },
        }
        reconstruction_reason = (
            "multiple_legal_action_interpretations"
            if reason == "ambiguous_action_history" else reason
        )
        f["expected"]["action_reconstruction"] = {
            "status": status,
            "candidates": candidates,
            "reason": reconstruction_reason,
            "event_required": status == "EXACT",
        }
        f["expected"]["assertions"].append(
            "non_exact_reconstruction_exposes_no_event"
        )
        f["tags"].extend(["action-reconstruction", status.lower()])
        if status != "EXACT":
            _set_abstain(f, reason, "state_event_reconstruction")
        _set_requirements(
            f,
            ["REQ-ST-002", "REQ-CTX-001", "REQ-OUT-002"],
            ["ST-005", "CTX-003", "FUS-004"],
            ["T-ST-003", "T-ST-006", "E2E-004"],
        )
        fixtures.append(f)
    return fixtures


def generate_temporal_consensus() -> list[dict[str, Any]]:
    cases = (
        (
            "POT-STABLE", "pot", 2,
            [(100, "10", "VALID"), (101, "10", "VALID")],
            [None, "10"], [101], [], None,
        ),
        (
            "POT-CHANGE-RESTART", "pot", 2,
            [(100, "10", "VALID"), (101, "11", "VALID"),
             (102, "11", "VALID")],
            [None, None, "11"], [102], [], None,
        ),
        (
            "FRAME-GAP", "pot", 2,
            [(100, "10", "VALID"), (102, "10", "VALID"),
             (103, "10", "VALID")],
            [None, None, "10"], [103], [], None,
        ),
        (
            "UNKNOWN-RESETS", "action", 2,
            [(100, "call", "VALID"), (101, None, "UNKNOWN"),
             (102, "call", "VALID"), (103, "call", "VALID")],
            [None, None, None, "call"], [103], [], None,
        ),
        (
            "CONFLICT-PRESERVED", "action", 2,
            [(100, None, "CONFLICT")],
            [None], [], [100], "action_conflict",
        ),
        (
            "SLOT-STACK-STABLE", "slot_stacks", 2,
            [(100, "100", "VALID", 4), (101, "100", "VALID", 4)],
            [None, "100"], [101], [], None,
        ),
        (
            "SLOT-DISAPPEARS", "slot_stacks", 2,
            [(100, "100", "VALID", 4), (101, None, "MISSING", 4),
             (102, "100", "VALID", 4), (103, "100", "VALID", 4)],
            [None, None, None, "100"], [103], [], None,
        ),
        (
            "THRESHOLD-ONE", "street", 1,
            [(100, "preflop", "VALID")],
            ["preflop"], [100], [], None,
        ),
    )
    fixtures = []
    for (name, field, threshold, frames, emitted, confirmed_frames,
         conflict_frames, reason) in cases:
        f = _base_fixture(
            f"MOCK-TEMPORAL-{name}", name.lower().replace("-", " "),
        )
        f["input"]["temporal_sequence"] = {
            "field": field,
            "confirmation_frames": threshold,
            "frames": [
                {
                    "frame_seq": item[0],
                    "value": item[1],
                    "status": item[2],
                    **({"slot_id": item[3]} if len(item) == 4 else {}),
                }
                for item in frames
            ],
        }
        f["expected"]["temporal_consensus"] = {
            "emitted_values": emitted,
            "confirmed_frames": confirmed_frames,
            "conflict_frames": conflict_frames,
            "pending_values_must_be_unknown": True,
        }
        f["expected"]["assertions"].extend([
            "confirmation_requires_consecutive_frame_seq",
            "pending_value_never_enters_canonical_state",
        ])
        f["tags"].extend(["temporal-consensus", field])
        if reason:
            _set_abstain(f, reason, "temporal_consensus")
        _set_requirements(
            f,
            ["REQ-IN-002", "REQ-ST-001", "REQ-OUT-002"],
            ["ST-001", "ST-002", "FUS-004"],
            ["T-ST-001", "T-FAIL-005"],
        )
        fixtures.append(f)
    return fixtures


def generate_hand_boundary_detection() -> list[dict[str, Any]]:
    cases = (
        (
            "HERO-CHANGED", "river", ["2c", "7d", "Jh", "9s", "3h"],
            "20", ["Qc", "Qd"], "preflop", [], "1.5", ["As", "Kd"],
            None, None, "CONFIRMED", ["hero_cards_changed"], [], None,
        ),
        (
            "POSTFLOP-RESET", "river", ["2c", "7d", "Jh", "9s", "3h"],
            "20", ["As", "Kd"], "preflop", [], "1.5", ["As", "Kd"],
            None, None, "CONFIRMED",
            ["street_reset_to_preflop", "board_cleared", "pot_reset"],
            [], None,
        ),
        (
            "WEAK-RESET", "river", ["2c", "7d", "Jh", "9s", "3h"],
            "20", ["As", "Kd"], "preflop", [], "20", ["As", "Kd"],
            None, None, "AMBIGUOUS",
            ["street_reset_to_preflop", "board_cleared"],
            ["insufficient_boundary_evidence"],
            "ambiguous_hand_boundary",
        ),
        (
            "CONFLICT", "river", ["2c", "7d", "Jh", "9s", "3h"],
            "20", ["As", "Kd"], "preflop", [], "1.5", ["As", "Kd"],
            "pot", None, "AMBIGUOUS", [], ["pot_conflict"],
            "ambiguous_hand_boundary",
        ),
        (
            "PREFLOP-MAPPED", "preflop", [], "20", ["As", "Kd"],
            "preflop", [], "1.5", ["As", "Kd"], None,
            {"dealer_slot_to_seat": {"7": 1},
             "stack_index_to_seat": [0, 1], "dealer_slot": 7,
             "stacks": ["150", "100"]},
            "CONFIRMED",
            ["pot_reset", "dealer_changed", "stack_reset_or_payout"],
            [], None,
        ),
        (
            "PREFLOP-UNMAPPED", "preflop", [], "20", ["As", "Kd"],
            "preflop", [], "1.5", ["As", "Kd"], None,
            {"dealer_slot": 7, "stacks": ["150", "100"]},
            "SAME_HAND", ["pot_reset"], [], None,
        ),
    )
    fixtures = []
    for (name, previous_street, previous_board, previous_pot, previous_hero,
         current_street, current_board, current_pot, current_hero,
         conflict_field, policy, status, evidence, reasons, refusal) in cases:
        f = _base_fixture(
            f"MOCK-HAND-BOUNDARY-{name}", name.lower().replace("-", " "),
        )
        f["input"]["hand_boundary"] = {
            "previous": {
                "street": previous_street,
                "board_cards": previous_board,
                "pot": previous_pot,
                "hero_cards": previous_hero,
                "dealer_seat": 0,
                "stacks": ["100", "100"],
            },
            "current": {
                "street": current_street,
                "board_cards": current_board,
                "pot": current_pot,
                "hero_cards": current_hero,
                "conflict_field": conflict_field,
            },
            "policy": policy or {},
        }
        f["expected"]["hand_boundary"] = {
            "status": status,
            "evidence": evidence,
            "reasons": reasons,
            "emit_hand_end": status == "CONFIRMED",
        }
        f["expected"]["assertions"].extend([
            "weak_boundary_never_closes_hand",
            "visual_indices_require_explicit_platform_mapping",
        ])
        f["tags"].extend(["hand-boundary", status.lower()])
        if refusal:
            _set_abstain(f, refusal, "hand_boundary")
        _set_requirements(
            f,
            ["REQ-IN-002", "REQ-ST-001", "REQ-OUT-002"],
            ["ST-004", "MEM-001", "FUS-004"],
            ["T-ST-004", "T-FAIL-005"],
        )
        fixtures.append(f)
    return fixtures


def generate_platform_mapping() -> list[dict[str, Any]]:
    """Synthetic slot-to-seat and candidate-state Replay catalog.

    These cases execute the generic mapping contract.  They are deliberately
    not evidence that any real WePoker layout has calibrated stack/action
    regions.
    """
    cases = (
        # name, action, actor slot, action slots, stacks, pot, dealer,
        # before profile, observation mutation, status, reason, event
        ("CHECK", "check", 10, [], [], None, 40, "default", {},
         "EXACT", None, "check"),
        ("FOLD", "fold", 10, [], [], None, 40, "default", {},
         "EXACT", None, "fold"),
        ("BET", "bet", 10, [], [[30, "80"]], "20", 40, "default", {},
         "EXACT", None, "bet"),
        ("CALL", "call", 10, [], [[30, "90"]], "20", 40, "facing10", {},
         "EXACT", None, "call"),
        ("RAISE", "raise", 10, [], [[30, "70"]], "40", 40, "facing10", {},
         "EXACT", None, "raise"),
        ("ALL-IN", "all_in", 10, [], [[30, "0"]], "140", 40,
         "short_all_in", {}, "EXACT", None, "all_in"),
        ("SLOT-ACTION", None, None, [[20, "check"]], [], None, 40,
         "default", {}, "EXACT", None, "check"),
        ("NO-ACTION", None, None, [], [], None, None, "default", {},
         "NO_ACTION", None, None),
        ("ACTOR-MISSING", "check", None, [], [], None, 40, "default", {},
         "AMBIGUOUS", "actor_missing", None),
        ("ACTOR-CONFLICT", None, 10, [[21, "check"]], [], None, 40,
         "default", {}, "AMBIGUOUS", "conflicting_actor_slots", None),
        ("ACTION-CONFLICT", "check", 10, [[20, "fold"]], [], None, 40,
         "default", {}, "AMBIGUOUS", "conflicting_action_labels", None),
        ("UNMAPPED-ACTOR", "check", 99, [], [], None, 40, "default", {},
         "INVALID", "unmapped_actor_slot", None),
        ("UNMAPPED-ACTION", None, 10, [[99, "check"]], [], None, 40,
         "default", {}, "INVALID", "unmapped_action_slot", None),
        ("UNMAPPED-STACK", "check", 10, [], [[99, "100"]], None, 40,
         "default", {}, "INVALID", "unmapped_stack_slot", None),
        ("UNMAPPED-DEALER", "check", 10, [], [], None, 99, "default", {},
         "INVALID", "unmapped_dealer_slot", None),
        ("DEALER-CONFLICT", "check", 10, [], [], None, 41, "default", {},
         "INVALID", "dealer_mapping_conflicts_with_state", None),
        ("MULTIPLE-STACK-CHANGE", "bet", 10, [], [[31, "90"]], "10", 40,
         "default", {}, "INVALID", "multiple_players_changed", None),
        ("MISSING-ACTOR-STACK", "bet", 10, [], [], "10", 40, "default", {},
         "INVALID", "actor_stack_missing_for_chip_action", None),
        ("MISSING-POT", "bet", 10, [], [[30, "90"]], None, 40, "default", {},
         "INVALID", "pot_missing_for_chip_action", None),
        ("CHIP-MISMATCH", "bet", 10, [], [[30, "90"]], "9", 40,
         "default", {}, "INVALID", "chip_delta_mismatch", None),
        ("MIXED-STREET", "check", 10, [], [], None, 40, "default",
         {"street": "flop"}, "INVALID",
         "cards_or_street_changed_during_action", None),
        ("MIXED-BOARD", "check", 10, [], [], None, 40, "default",
         {"board": ["2c", "7d", "Jh"]}, "INVALID",
         "cards_or_street_changed_during_action", None),
        ("FORCED-BLIND", "post_bb", 10, [], [], None, 40, "default", {},
         "INVALID", "forced_action_not_supported", None),
    )
    fixtures = []
    for (name, action, actor_slot, action_slots, stacks, pot, dealer_slot,
         before_profile, mutation, status, reason, event) in cases:
        fixture = _base_fixture(
            f"MOCK-PLATFORM-MAPPING-{name}",
            f"platform mapping {name.lower().replace('-', ' ')}",
            3,
        )
        fixture["input"]["platform_mapping_case"] = {
            "mapping": {
                "platform_id": "synthetic",
                "layout_id": "three-seat-v1",
                "version": "mock-1",
                "stack_slot_to_seat": {"30": 0, "31": 1, "32": 2},
                "action_slot_to_seat": {"20": 0, "21": 1, "22": 2},
                "actor_slot_to_seat": {"10": 0, "11": 1, "12": 2},
                "dealer_slot_to_seat": {"40": 0, "41": 1, "42": 2},
            },
            "before_profile": before_profile,
            "observation": {
                "actor_slot": actor_slot,
                "action": action,
                "action_slots": action_slots,
                "stack_slots": stacks,
                "pot": pot,
                "dealer_slot": dealer_slot,
                **mutation,
            },
        }
        fixture["expected"]["platform_mapping"] = {
            "status": status,
            "reason": reason,
            "event": event,
            "candidate_required": status == "EXACT",
        }
        fixture["expected"]["assertions"].extend([
            "visual_slot_is_not_a_player_identity",
            "non_exact_mapping_exposes_no_state_or_event",
        ])
        fixture["tags"].extend([
            "platform-mapping", "synthetic-replay", status.lower(),
        ])
        if status in {"AMBIGUOUS", "INVALID"}:
            _set_abstain(fixture, reason, "platform_mapping")
        elif status == "NO_ACTION":
            fixture["expected"]["advice"].update({
                "status": "PARTIAL",
                "action_probabilities": {},
                "reason_codes": ["no_action_transition"],
                "match_kind": "state_only",
            })
        _set_requirements(
            fixture,
            ["REQ-IN-002", "REQ-ST-002", "REQ-OUT-002"],
            ["ST-001", "ST-002", "ST-005", "FUS-004"],
            ["T-ST-002", "T-ST-003", "T-ST-006", "E2E-004"],
        )
        fixtures.append(fixture)
    return fixtures


def generate_atomic_memory() -> list[dict[str, Any]]:
    cases = (
        (
            "TRANSITION-COMMIT", "record_transition", None, True,
            "state and all events commit together",
        ),
        (
            "TRANSITION-WRONG-HAND", "record_transition", "wrong_event_hand",
            False, "wrong-hand event rolls back state and events",
        ),
        (
            "BOUNDARY-COMMIT", "replace_active_hand", None, True,
            "old hand completion and successor start commit together",
        ),
        (
            "BOUNDARY-EXISTING-SUCCESSOR", "replace_active_hand",
            "existing_successor", False,
            "existing successor rolls back the entire hand replacement",
        ),
    )
    fixtures = []
    for name, operation, fault, commits, title in cases:
        f = _base_fixture(f"MOCK-MEMORY-{name}", title)
        f["input"]["memory_operation"] = {
            "operation": operation,
            "fault": fault,
            "previous_hand_id": "h1",
            "previous_state_version": 0,
            "next_state_version": 1,
            "successor_hand_id": "h2",
            "event_types": (
                ["deal", "raise"]
                if operation == "record_transition" else ["hand_end"]
            ),
        }
        f["expected"]["memory"] = {
            "committed": commits,
            "active_hand_id": (
                "h2" if commits and operation == "replace_active_hand" else "h1"
            ),
            "previous_state_count": (
                2 if commits and operation == "record_transition" else 1
            ),
            "previous_event_count": (
                2 if commits and operation == "record_transition"
                else 1 if commits else 0
            ),
            "previous_completed": commits and operation == "replace_active_hand",
            "successor_exists": commits and operation == "replace_active_hand",
        }
        f["expected"]["assertions"].extend([
            "memory_commit_is_atomic",
            "failed_commit_leaves_active_hand_unchanged",
        ])
        f["tags"].extend(["memory", "atomic", "commit" if commits else "rollback"])
        _set_requirements(
            f,
            ["REQ-ST-001", "REQ-AUD-001"],
            ["MEM-001", "ST-003", "ST-004"],
            ["T-ST-001", "T-ST-004", "T-FAIL-004"],
        )
        fixtures.append(f)
    return fixtures


def generate_abstraction_matching() -> list[dict[str, Any]]:
    cases = (
        (
            "COMBINED-IN-RANGE", "BB", "95", "1.5", "2.5",
            [
                ["effective_stack_bb", "95", "100", "5", "10"],
                ["last_aggressive_total_bb", "2.5", "2", "0.5", "1"],
                ["pot_bb", "1.5", "1", "0.5", "1"],
            ], "HIT_APPROXIMATE", None,
        ),
        (
            "POT-OUTSIDE", "BB", "100", "10", "2", [],
            "NOT_APPLICABLE", "unsupported_pot",
        ),
        (
            "POSITION-UNSUPPORTED", "BB", "100", "1", "2", [],
            "NOT_APPLICABLE", "unsupported_hero_position",
        ),
        (
            "EXACT", "BTN", "100", "1", "2", [],
            "HIT_EXACT", None,
        ),
    )
    fixtures = []
    for name, position, stack, pot, size, dimensions, lookup, refusal in cases:
        f = _base_fixture(f"MOCK-ABSTRACTION-{name}", name.lower())
        f["input"]["abstraction_match"] = {
            "requested": {
                "hero_position": position,
                "effective_stack_bb": stack,
                "pot_bb": pot,
                "last_aggressive_total_bb": size,
            },
            "capability": {
                "hero_positions": ["BTN"] if "POSITION" in name else ["BTN", "BB"],
                "stack_buckets_bb": ["100"],
                "max_stack_distance_bb": "10",
                "pot_buckets_bb": ["1", "2"],
                "max_pot_distance_bb": "1",
                "aggressive_size_buckets_bb": ["2", "3"],
                "max_aggressive_size_distance_bb": "1",
            },
        }
        f["expected"]["match_dimensions"] = [
            {
                "name": item[0], "requested": item[1], "matched": item[2],
                "distance": item[3], "maximum_distance": item[4],
            }
            for item in dimensions
        ]
        f["expected"]["provider_lookups"] = [{
            "provider_id": "mock-preflop-2p-v1", "state": lookup,
        }]
        if dimensions:
            f["expected"]["advice"]["match_kind"] = "interpolated"
            f["expected"]["state_match_score"] = "0.5"
        if refusal:
            _set_abstain(f, refusal, "router")
        f["expected"]["assertions"].extend([
            "approximation_dimensions_are_structured",
            "distance_outside_threshold_never_routes",
        ])
        f["tags"].extend(["router", "abstraction", "approximation"])
        _set_requirements(
            f,
            ["REQ-PRV-001", "REQ-RTR-001", "REQ-OUT-002"],
            ["PRV-001", "RTR-004", "UI-002"],
            ["T-RTR-003", "T-PRV-001", "T-UI-002"],
        )
        fixtures.append(f)
    return fixtures


def generate_hard_gates() -> list[dict[str, Any]]:
    cases = (
        (
            "ALL-PASS",
            [("range_integrity", "PASS", []),
             ("numerical_integrity", "PASS", [])],
            "READY", [],
        ),
        (
            "RANGE-FAIL",
            [("range_integrity", "FAIL", ["range_card_collision"])],
            "ABSTAIN", ["range_card_collision"],
        ),
        (
            "NUMERICAL-FAIL",
            [("numerical_integrity", "FAIL", ["equity_ci_missing"])],
            "ABSTAIN", ["equity_ci_missing"],
        ),
        (
            "MULTIPLE-FAIL",
            [("range_integrity", "FAIL", ["range_unknown"]),
             ("numerical_integrity", "FAIL", ["equity_budget_exhausted"])],
            "ABSTAIN", ["range_unknown", "equity_budget_exhausted"],
        ),
    )
    fixtures = []
    for name, gates, status, reasons in cases:
        f = _base_fixture(f"MOCK-HARD-GATE-{name}", name.lower())
        f["input"]["hard_gates"] = [
            {"name": gate, "status": gate_status, "reasons": gate_reasons}
            for gate, gate_status, gate_reasons in gates
        ]
        f["expected"]["gate_results"] = copy.deepcopy(
            f["input"]["hard_gates"]
        )
        if status == "ABSTAIN":
            _set_abstain(f, reasons[0], "fusion")
            f["expected"]["advice"]["reason_codes"] = reasons
        f["expected"]["assertions"].extend([
            "failed_hard_gate_never_produces_ready",
            "gate_results_are_auditable",
        ])
        f["tags"].extend(["fusion", "hard-gate", status.lower()])
        _set_requirements(
            f,
            ["REQ-FUS-001", "REQ-OUT-002"],
            ["FUS-004", "ADV-001", "UI-001"],
            ["T-FUS-003", "T-UI-001"],
        )
        fixtures.append(f)
    return fixtures


def generate_fast_source_fallback() -> list[dict[str, Any]]:
    cases = (
        ("CACHE-HIT", [("cache", "HIT_EXACT")], "cache"),
        ("DB-HIT", [("cache", "NOT_FOUND"),
                    ("preflop_db", "HIT_EXACT")], "preflop_db"),
        ("PRESOLVED-HIT", [("cache", "NOT_FOUND"),
                           ("preflop_db", "NOT_FOUND"),
                           ("presolved", "HIT_EXACT")], "presolved"),
        ("MODEL-HIT", [("cache", "NOT_FOUND"),
                       ("preflop_db", "NOT_FOUND"),
                       ("presolved", "REJECTED"),
                       ("model", "HIT_APPROXIMATE")], "model"),
        ("ALL-MISS", [("cache", "NOT_FOUND"),
                      ("preflop_db", "NOT_FOUND"),
                      ("presolved", "NOT_FOUND"),
                      ("model", "NOT_FOUND")], None),
    )
    fixtures = []
    for name, layers, selected in cases:
        f = _base_fixture(f"MOCK-FAST-FALLBACK-{name}", name.lower())
        f["input"]["fast_source_layers"] = [
            {
                "layer": layer,
                "provider_id": f"mock-{layer}",
                "lookup_state": state,
            }
            for layer, state in layers
        ]
        f["expected"]["queried_layers"] = [layer for layer, _ in layers]
        f["expected"]["selected_layer"] = selected
        f["expected"]["provider_lookups"] = [
            {"provider_id": f"mock-{layer}", "state": state}
            for layer, state in layers
        ]
        if selected == "model":
            f["expected"]["advice"]["match_kind"] = "heuristic"
            f["expected"]["advice"]["strategy_source"] = "mock-model"
        elif selected is not None:
            f["expected"]["advice"]["strategy_source"] = f"mock-{selected}"
        else:
            _set_abstain(f, "no_fast_strategy", "router")
        f["expected"]["assertions"].extend([
            "fast_layers_are_queried_in_fixed_order",
            "lower_layers_are_not_queried_after_hit",
        ])
        f["tags"].extend(["router", "fast-fallback", name.lower()])
        _set_requirements(
            f,
            ["REQ-RTR-001", "REQ-OUT-001"],
            ["RTR-005", "RTR-008", "ADV-001"],
            ["T-RTR-001", "T-RTR-002", "T-INT-001"],
        )
        fixtures.append(f)
    return fixtures


def generate_strategy_asset_adapter() -> list[dict[str, Any]]:
    cases = (
        ("3P-PREFLOP-HIT", 3, 3, "preflop", "valid", "HIT_EXACT"),
        ("3WAY-FLOP-HIT", 6, 3, "flop", "valid", "HIT_EXACT"),
        ("4WAY-TURN-HIT", 9, 4, "turn", "valid", "HIT_EXACT"),
        ("NODE-MISS", 3, 3, "preflop", "missing", "NOT_FOUND"),
        ("MALFORMED-NODE", 3, 3, "preflop", "malformed", "REJECTED"),
        ("HASH-MISMATCH", 3, 3, "preflop", "bad_hash", "REGISTRATION_ERROR"),
    )
    fixtures = []
    for name, dealt, active, street, asset_state, outcome in cases:
        f = _base_fixture(
            f"MOCK-STRATEGY-ASSET-{name}", name.lower(), dealt, street, active
        )
        f["input"]["strategy_asset"] = {
            "schema_version": 1,
            "capability_id": f"synthetic-{active}way-{street}-v1",
            "provider_id": "synthetic-licensed-asset",
            "source_version": "synthetic-v1",
            "license_spdx": "LicenseRef-Synthetic-Test-Only",
            "asset_state": asset_state,
            "sha256_required": True,
        }
        f["expected"]["strategy_asset_adapter"] = {
            "outcome": outcome,
            "player_count": active,
            "street": street,
            "must_include_hash_license_version_evidence": outcome == "HIT_EXACT",
        }
        if outcome == "HIT_EXACT":
            f["expected"]["provider_lookups"] = [{
                "provider_id": "synthetic-licensed-asset",
                "state": "HIT_EXACT",
            }]
        elif outcome == "REGISTRATION_ERROR":
            _set_abstain(f, "strategy_asset_hash_mismatch", "provider")
            f["expected"]["provider_lookups"] = []
        else:
            f["expected"]["provider_lookups"] = [{
                "provider_id": "synthetic-licensed-asset",
                "state": outcome,
            }]
            _set_abstain(
                f,
                "strategy_asset_node_not_found"
                if outcome == "NOT_FOUND" else "invalid_strategy_asset_node",
                "provider",
            )
        f["expected"]["assertions"].extend([
            "strategy_asset_hash_is_verified_before_registration",
            "strategy_asset_never_invents_a_missing_node",
        ])
        f["tags"].extend([
            "provider", "strategy-asset", asset_state,
        ])
        _set_requirements(
            f,
            ["REQ-PRV-001", "REQ-RTR-001", "REQ-AUD-001"],
            [
                "PRV-003" if street == "preflop" else "PRV-004",
                "PRV-005",
                "ADV-003",
            ],
            ["T-PRV-001", "T-PRV-003", "T-INT-001"],
        )
        fixtures.append(f)
    return fixtures


def generate_gtopen_adapter() -> list[dict[str, Any]]:
    fixture = _base_fixture(
        "MOCK-GTOPEN-3P-ROOT-AKO",
        "local GTOpen three-player root AKo model strategy",
        3,
    )
    state = fixture["input"]["state"]
    state["hero_seat"] = 0
    state["actor_seat"] = 0
    for seat in state["seats"]:
        seat["is_hero"] = seat["seat_id"] == 0
        seat["player_id"] = (
            "hero" if seat["seat_id"] == 0 else f"p{seat['seat_id']}"
        )
        seat["starting_stack"] = "20"
        committed = Decimal(seat["hand_committed"])
        seat["stack"] = _money(Decimal("20") - committed)
    state["hero_cards"] = ["As", "Kd"]
    state["action_line"] = "unopened"
    state["to_call"] = "1"
    state["legal_actions"] = [
        {"action": "fold", "min": "0", "max": "0"},
        {"action": "call", "min": "1", "max": "1"},
        {"action": "raise", "min": "2", "max": "20"},
        {"action": "all_in", "min": "20", "max": "20"},
    ]
    fixture["input"]["gtopen"] = {
        "transport": "loopback-json-api",
        "configured_source_revision": (
            "4aee435bdeb155b25f0c8140e707a8342ce4356f"
        ),
        "server_revision_remotely_attested": False,
        "realization": "raw",
        "positions": ["BTN", "SB", "BB"],
        "posts_bb": ["0", "0.5", "1"],
        "equal_starting_stack_bb": "20",
        "path": [],
        "hero_class": "AKo",
        "hero_class_index": 155,
        "iterations": 100,
        "model_gap_bb": "0.008207490846030292",
        "tree": {
            "nodes": 4270,
            "action_nodes": 1710,
            "arena_mb": "5.771688",
        },
    }
    fixture["input"]["providers"] = [{
        "provider_id": "gtopen-local-preflop",
        "source_version": (
            "gtopen-adapter-v1:"
            "4aee435bdeb155b25f0c8140e707a8342ce4356f"
        ),
        "fixture_eligibility": "synthetic-local-model-only",
        "asset_hash": "not-applicable-local-api",
        "capability": {
            "player_counts": [3],
            "streets": ["preflop"],
            "stack_buckets_bb": [20],
            "ante": ["0"],
            "rake_profiles": ["zero"],
            "action_lines": ["unopened"],
            "match_kind": "heuristic",
        },
        "mock_result": {
            "action_probabilities": {
                "fold": "0.0000004925866227148439678986858293",
                "call": "0.001928847688945725019429796111",
                "raise": "0.3261888287416759352963990747",
                "all_in": "0.6718818309827556248402032305",
            },
            "recommended_sizes": {
                "raise": ["2", "2.5", "3"],
                "all_in": ["20"],
            },
        },
    }]
    for seat_id, distribution in zip(
        (1, 2), fixture["input"]["ranges"]["villains"]
    ):
        distribution["seat_id"] = seat_id
    fixture["expected"]["provider_lookups"] = [{
        "provider_id": "gtopen-local-preflop",
        "state": "HIT_APPROXIMATE",
    }]
    fixture["expected"]["advice"].update({
        "status": "READY",
        "match_kind": "heuristic",
        "strategy_source": "gtopen-local-preflop",
        "strategy_version": (
            "gtopen-adapter-v1:"
            "4aee435bdeb155b25f0c8140e707a8342ce4356f"
        ),
        "action_probabilities": fixture["input"]["providers"][0][
            "mock_result"
        ]["action_probabilities"],
        "reason_codes": [],
    })
    fixture["expected"]["gtopen"] = {
        "path_match": "exact-actor-kind-to",
        "hero_strategy_shape": "actions_x_169",
        "model_gap_threshold_met": True,
        "single_session_serialized": True,
        "timeout_stop_best_effort": True,
        "must_disclose": [
            "multiway_product_equity_approximation",
            "non_unique_multi_player_equilibrium",
            "server_revision_not_remotely_attested",
            "license_not_cleared_for_redistribution",
        ],
    }
    fixture["expected"]["assertions"].extend([
        "gtopen_path_matches_exactly",
        "hero_169_class_index_matches",
        "multi_size_options_are_preserved",
        "model_result_is_not_labeled_exact",
        "source_is_not_release_asset_evidence",
    ])
    fixture["tags"].extend([
        "gtopen", "local-api", "multiway-preflop", "slow-path",
        "heuristic-model", "synthetic-only",
    ])
    _set_requirements(
        fixture,
        ["REQ-PRV-001", "REQ-RTR-001", "REQ-AUD-001"],
        ["PRV-003", "PRV-008", "RTR-006", "RTR-007", "ADV-003"],
        ["T-PRV-006", "T-INT-005", "T-INT-010"],
    )
    return [fixture]


def generate_boundaries() -> list[dict[str, Any]]:
    fixtures = []
    for stack in ("0", "0.5", "9.5", "10", "20", "30", "40", "60",
                  "80", "100", "150", "200", "200.5", "10000"):
        fixture_id = f"MOCK-STACK-{stack.replace('.', 'P')}BB"
        f = _base_fixture(fixture_id, f"stack boundary {stack} BB", 6)
        for seat in f["input"]["state"]["seats"]:
            seat["starting_stack"] = stack
            seat["stack"] = stack
        supported = stack in {
            "10", "20", "30", "40", "60", "80", "100", "150", "200",
        }
        f["tags"].extend(["boundary", "stack"])
        if supported:
            f["expected"]["advice"]["match_kind"] = "exact"
        elif stack in {"9.5", "200.5"}:
            f["expected"]["advice"]["status"] = "PARTIAL"
            f["expected"]["advice"]["match_kind"] = "approximate"
            f["expected"]["provider_lookups"][0]["state"] = (
                "HIT_APPROXIMATE"
            )
        else:
            f["expected"]["provider_lookups"][0]["state"] = "NOT_FOUND"
            _set_abstain(f, "unsupported_stack", "router")
        _set_requirements(
            f, ["REQ-MET-001", "REQ-PRV-001", "REQ-RTR-001"],
            ["MET-001", "PRV-001", "RTR-004"],
            ["T-MET-001", "T-RTR-003"],
        )
        fixtures.append(f)

    for pot, call, expected in (
        ("0", "0", "0"), ("100", "0", "0"), ("100", "25", "0.2"),
        ("100", "50", "0.3333333333333333"), ("0.5", "0.5", "0.5"),
        ("999999999", "1", "0.000000001"),
    ):
        fixture_id = f"MOCK-POT-ODDS-{len(fixtures):03d}"
        f = _base_fixture(fixture_id, f"pot odds P={pot}, C={call}")
        f["input"]["state"]["pots"][0]["amount"] = pot
        f["input"]["state"]["to_call"] = call
        f["expected"]["math"] = {"required_equity": expected}
        f["tags"].extend(["boundary", "pot-odds"])
        _set_requirements(
            f, ["REQ-MET-001"], ["MET-003"], ["T-MET-002", "T-EV-001"],
        )
        fixtures.append(f)
    return fixtures


def generate_side_pots() -> list[dict[str, Any]]:
    cases = (
        ("TWO-ALLIN", ["20", "50", "100"], ["60", "60"], "50"),
        ("THREE-ALLIN", ["10", "30", "80", "100"],
         ["40", "60", "100"], "20"),
        ("FOLDED-CONTRIBUTOR", ["25", "25", "100"], ["75"], "75"),
        ("EQUAL-ALLIN", ["50", "50", "50", "50"], ["200"], "0"),
    )
    fixtures = []
    for name, commitments, expected_pots, uncalled_return in cases:
        player_count = len(commitments)
        fixture_id = f"MOCK-SIDE-POT-{name}"
        f = _base_fixture(fixture_id, name.lower().replace("-", " "),
                          player_count, "river", player_count)
        for seat, commitment in zip(f["input"]["state"]["seats"], commitments):
            seat["starting_stack"] = "100"
            seat["stack"] = str(100 - int(commitment))
            seat["street_committed"] = commitment
            seat["hand_committed"] = commitment
            seat["status"] = "all_in" if commitment != "100" else "active"
        if name == "FOLDED-CONTRIBUTOR":
            f["input"]["state"]["seats"][0]["status"] = "folded"
        f["input"]["state"]["pots"] = []
        f["expected"]["pot_amounts"] = expected_pots
        f["expected"]["uncalled_return"] = uncalled_return
        f["expected"]["assertions"].extend([
            "chip_conservation_exact", "side_pot_eligibility_exact",
        ])
        f["tags"].extend(["boundary", "all-in", "side-pot"])
        _set_requirements(
            f, ["REQ-ST-002", "REQ-EQ-001"],
            ["ST-007", "EQ-005"],
            ["T-ST-005", "T-EQ-004", "E2E-006"],
        )
        fixtures.append(f)
    return fixtures


def generate_faults() -> list[dict[str, Any]]:
    cases = (
        ("DUPLICATE-HERO-CARD", "state_validation", "duplicate_card"),
        ("HERO-BOARD-OVERLAP", "state_validation", "card_overlap"),
        ("BAD-BOARD-COUNT", "state_validation", "invalid_board_count"),
        ("NEGATIVE-STACK", "state_validation", "negative_stack"),
        ("CHIP-NON-CONSERVATION", "state_validation", "chip_non_conservation"),
        ("ACTOR-NOT-ACTIVE", "state_validation", "invalid_actor"),
        ("IMPOSSIBLE-CHECK", "state_validation", "illegal_action"),
        ("RAISE-BELOW-MIN", "state_validation", "illegal_raise_size"),
        ("AMBIGUOUS-ACTION", "context_quality", "ambiguous_action_history"),
        ("MISSING-POSITION", "context_quality", "missing_position"),
        ("STALE-HAND", "stale_filter", "stale_hand_id"),
        ("STALE-STATE", "stale_filter", "stale_state_version"),
        ("STALE-REQUEST", "stale_filter", "stale_request_id"),
        ("EXPIRED-DEADLINE", "stale_filter", "expired_advice"),
        ("BAD-ASSET-HASH", "provider_registry", "asset_integrity_error"),
        ("BAD-PROVIDER-SCHEMA", "provider_registry", "provider_schema_error"),
        ("RESOLVER-TIMEOUT", "resolver", "resolver_timeout"),
        ("RESOLVER-CRASH", "resolver", "resolver_crash"),
        ("INVALID-FREQUENCY", "fusion", "invalid_probability"),
        ("ILLEGAL-PROVIDER-ACTION", "fusion", "illegal_provider_action"),
        ("RANGE-CARD-COLLISION", "range_tracker", "range_card_collision"),
        ("EMPTY-RANGE", "equity", "empty_range"),
        ("WINDOW-IDENTITY-CHANGE", "capture_binding", "window_identity_mismatch"),
        ("SYSTEM-CLOCK-JUMP", "stale_filter", "monotonic_deadline_required"),
    )
    fixtures = []
    for name, stage, reason in cases:
        fixture_id = f"MOCK-FAULT-{name}"
        f = _base_fixture(fixture_id, name.lower().replace("-", " "), 6)
        f["input"]["fault_injection"] = {
            "fault": name.lower().replace("-", "_"), "at_stage": stage,
        }
        state = f["input"]["state"]
        request = f["input"]["request_context"]
        provider = f["input"]["providers"][0]
        if name == "DUPLICATE-HERO-CARD":
            state["hero_cards"] = ["As", "As"]
        elif name == "HERO-BOARD-OVERLAP":
            state["street"] = "flop"
            state["board_cards"] = ["As", "7d", "Jh"]
        elif name == "BAD-BOARD-COUNT":
            state["street"] = "flop"
            state["board_cards"] = ["2c", "7d"]
        elif name == "NEGATIVE-STACK":
            state["seats"][0]["stack"] = "-1"
        elif name == "CHIP-NON-CONSERVATION":
            state["pots"][0]["amount"] = "999"
        elif name == "ACTOR-NOT-ACTIVE":
            state["actor_seat"] = 0
            state["seats"][0]["status"] = "folded"
        elif name == "IMPOSSIBLE-CHECK":
            state["to_call"] = "2"
            state["action_history"].append({
                "sequence": 3, "street": "preflop", "seat_id": 5,
                "action": "check", "amount": "0",
            })
        elif name == "RAISE-BELOW-MIN":
            state["action_history"].append({
                "sequence": 3, "street": "preflop", "seat_id": 0,
                "action": "raise", "amount": "1.5", "minimum": "2",
            })
        elif name == "AMBIGUOUS-ACTION":
            f["input"]["ambiguous_event"] = {
                "sequence": 3,
                "candidates": [
                    {"action": "call", "amount": "1"},
                    {"action": "raise", "amount": "2"},
                ],
            }
        elif name == "MISSING-POSITION":
            state["seats"][state["hero_seat"]]["position"] = "UNKNOWN"
        elif name == "STALE-HAND":
            f["input"]["current_context"] = {
                "hand_id": "new-hand", "state_version": 1,
                "request_id": request["request_id"],
            }
        elif name == "STALE-STATE":
            f["input"]["current_context"] = {
                "hand_id": request["hand_id"], "state_version": 2,
                "request_id": request["request_id"],
            }
        elif name == "STALE-REQUEST":
            f["input"]["current_context"] = {
                "hand_id": request["hand_id"], "state_version": 1,
                "request_id": "req-new",
            }
        elif name == "EXPIRED-DEADLINE":
            f["input"]["runtime_clock"] = {
                "now": "2026-08-22T00:00:03Z",
                "monotonic_elapsed_ms": 3000,
            }
        elif name == "BAD-ASSET-HASH":
            provider["asset_hash"] = "sha256:corrupt"
            provider["expected_asset_hash"] = "sha256:expected"
        elif name == "BAD-PROVIDER-SCHEMA":
            del provider["capability"]["player_counts"]
        elif name == "INVALID-FREQUENCY":
            provider["mock_result"]["action_probabilities"] = {
                "check": "-0.1", "raise": "1.1",
            }
        elif name == "ILLEGAL-PROVIDER-ACTION":
            provider["mock_result"]["action_probabilities"] = {
                "bet": "1",
            }
        elif name == "RANGE-CARD-COLLISION":
            f["input"]["ranges"]["villains"][0]["combo_weights"] = {
                "AsQc": "1",
            }
        elif name == "EMPTY-RANGE":
            f["input"]["ranges"]["villains"][0]["combo_weights"] = {}
        elif name == "WINDOW-IDENTITY-CHANGE":
            f["input"]["observations"]["window_identities"] = [
                "mock-window-1", "mock-window-2",
            ]
        elif name == "SYSTEM-CLOCK-JUMP":
            f["input"]["runtime_clock"] = {
                "wall_start": "2026-08-22T00:00:01Z",
                "wall_end": "2026-08-21T23:59:59Z",
                "monotonic_elapsed_ms": 200,
            }
        f["tags"].extend(["negative", "fault-injection", stage])
        if name.startswith("STALE-") or name == "EXPIRED-DEADLINE":
            f["expected"]["terminal_stage"] = stage
            f["expected"]["advice"].update({
                "status": "STALE", "action_probabilities": {},
                "reason_codes": [reason],
            })
        elif name in {"RESOLVER-TIMEOUT", "RESOLVER-CRASH"}:
            f["expected"]["terminal_stage"] = "advice"
            f["expected"]["advice"]["status"] = "PARTIAL"
            f["expected"]["advice"]["match_kind"] = "equity_only"
            f["expected"]["advice"]["action_probabilities"] = {}
            f["expected"]["advice"]["reason_codes"] = [reason]
        else:
            _set_abstain(f, reason, stage)
        if name in {"BAD-ASSET-HASH", "BAD-PROVIDER-SCHEMA",
                    "RESOLVER-TIMEOUT", "RESOLVER-CRASH"}:
            f["expected"]["provider_lookups"] = [{
                "provider_id": f["input"]["providers"][0]["provider_id"],
                "state": "REJECTED",
            }]
        _set_requirements(
            f,
            ["REQ-ST-001", "REQ-CTX-001", "REQ-PRV-001",
             "REQ-FUS-001", "REQ-OUT-002"],
            ["ST-002", "CTX-003", "PRV-001", "RTR-007", "FUS-004"],
            ["T-FUS-002", f"T-FAIL-{len(fixtures) + 1:03d}"],
        )
        if name.startswith("STALE-") or name == "EXPIRED-DEADLINE":
            f["test_ids"].append("E2E-009")
            f["test_ids"].append("T-RTR-004")
        if name == "STALE-STATE":
            f["test_ids"].append("T-INT-006")
        if name == "CHIP-NON-CONSERVATION":
            f["test_ids"].append("T-ST-002")
        if name == "BAD-ASSET-HASH":
            f["test_ids"].append("E2E-010")
        if name == "RESOLVER-CRASH":
            f["test_ids"].append("T-INT-010")
        if name in {"RESOLVER-TIMEOUT", "RESOLVER-CRASH"}:
            f["test_ids"].append("T-PRV-005")
        if name == "RANGE-CARD-COLLISION":
            f["test_ids"].append("T-RNG-002")
        fixtures.append(f)
    return fixtures


def generate_equity() -> list[dict[str, Any]]:
    cases = (
        ("NUTS-RIVER", ["As", "Ks"], ["Qs", "Js", "Ts", "5h", "3d"],
         [["2d", "2c"]], "1", "0", "0", "1"),
        ("FORCED-TIE", ["As", "Ah"], ["2h", "7c", "9d", "Js", "Qh"],
         [["Ad", "Ac"]], "0", "1", "0", "0.5"),
        ("THREE-WAY-PARTIAL-TIE", ["2d", "3c"],
         ["As", "Ah", "Kd", "2c", "3d"],
         [["2h", "3h"], ["2s", "4c"]], "0", "1", "0", "0.5"),
        ("HERO-LOSES", ["2c", "3d"], ["As", "Ks", "Qs", "Js", "9h"],
         [["Ts", "Ad"]], "0", "0", "1", "0"),
    )
    fixtures = []
    for name, hero, board, opponents, win, tie, loss, equity in cases:
        count = len(opponents) + 1
        f = _base_fixture(f"MOCK-EQUITY-{name}", name.lower().replace("-", " "),
                          count, "river", count)
        f["input"]["state"]["hero_cards"] = hero
        f["input"]["state"]["board_cards"] = board
        f["input"]["known_opponent_cards"] = opponents
        f["expected"]["math"] = {
            "method": "enumeration", "win": win, "tie": tie,
            "loss": loss, "equity": equity, "samples": 1,
        }
        f["expected"]["advice"]["status"] = "PARTIAL"
        f["expected"]["advice"]["match_kind"] = "equity_only"
        f["expected"]["advice"]["action_probabilities"] = {}
        f["tags"].extend(["equity", "exact", "known-cards"])
        _set_requirements(
            f, ["REQ-EQ-001", "REQ-OUT-002"],
            ["EQ-001", "EQ-005", "ADV-001"],
            ["T-EQ-001", "T-EQ-004", "E2E-005"],
        )
        fixtures.append(f)
    for seed, trials in ((0, 1), (1, 100), (7, 20000), (2147483647, 50000)):
        f = _base_fixture(
            f"MOCK-EQUITY-MC-S{seed}-N{trials}",
            f"Monte Carlo seed {seed} trials {trials}", 3, "flop", 3,
        )
        f["input"]["equity_request"] = {
            "method": "montecarlo", "seed": seed, "trials": trials,
        }
        f["expected"]["math"] = {
            "method": "montecarlo", "must_be_replayable": True,
            "confidence_behavior": "decreases_with_low_trials",
        }
        f["tags"].extend(["equity", "montecarlo", "statistical"])
        _set_requirements(
            f, ["REQ-EQ-001"], ["EQ-003", "EQ-004"],
            ["T-EQ-002", "T-EQ-003"],
        )
        fixtures.append(f)
    return fixtures


def generate_output_states() -> list[dict[str, Any]]:
    fixtures = []
    for status in ("READY", "PARTIAL", "ABSTAIN", "STALE"):
        f = _base_fixture(f"MOCK-ADVICE-{status}", f"Advice {status}")
        f["tags"].extend(["advice-contract", status.lower()])
        f["expected"]["advice"]["status"] = status
        if status == "PARTIAL":
            f["expected"]["advice"]["match_kind"] = "equity_only"
            f["expected"]["advice"]["action_probabilities"] = {}
            f["expected"]["advice"]["reason_codes"] = ["strategy_unavailable"]
        elif status == "ABSTAIN":
            _set_abstain(f, "critical_input_missing")
        elif status == "STALE":
            f["expected"]["advice"]["action_probabilities"] = {}
            f["expected"]["advice"]["reason_codes"] = ["expired_advice"]
        _set_requirements(
            f, ["REQ-OUT-001", "REQ-OUT-002", "REQ-UI-001"],
            ["ADV-001", "ADV-002", "SER-001", "UI-001"],
            ["T-ADV-001", "T-ADV-002", "T-SER-001", "E2E-011"],
        )
        if status == "PARTIAL":
            f["test_ids"].append("E2E-008")
            f["test_ids"].append("T-INT-003")
        fixtures.append(f)
    return fixtures


def generate_audit_and_training() -> list[dict[str, Any]]:
    fixtures = []

    audit = _base_fixture(
        "MOCK-AUDIT-EVIDENCE-COMPLETE", "complete Advice evidence chain", 6,
    )
    audit["input"]["evidence_chain"] = [
        "mock://frame/101/hero_cards",
        "mock://state/mock-audit-evidence-complete/1",
        "mock://range/p0/v1",
        "mock://provider/mock-preflop-6p-v1/mock-v1",
    ]
    audit["expected"]["assertions"].append("all_evidence_refs_resolve")
    audit["expected"]["evidence_audit"] = {
        "complete": True,
        "missing": [],
        "confidence_limit": "1",
        "required_stages": ["input", "state", "range", "provider"],
        "chain_id_algorithm": "sha256",
    }
    audit["tags"].extend(["audit", "evidence-chain"])
    _set_requirements(
        audit, ["REQ-AUD-001", "REQ-OUT-001"],
        ["CTX-002", "ADV-003", "SER-001"],
        ["T-ADV-001", "T-ADV-003", "T-SER-001", "E2E-001"],
    )
    fixtures.append(audit)

    broken_audit = _base_fixture(
        "MOCK-AUDIT-EVIDENCE-MISSING-PROVIDER",
        "broken provider evidence caps otherwise ready Advice confidence",
        6,
    )
    broken_audit["input"]["evidence_chain"] = [
        "mock://frame/101/hero_cards",
        "mock://state/mock-audit-evidence-missing-provider/1",
        "mock://range/p0/v1",
    ]
    broken_audit["expected"]["evidence_audit"] = {
        "complete": False,
        "missing": ["provider:mock-preflop-6p-v1"],
        "confidence_limit": "0.49",
        "must_disclose_incomplete_assumption": True,
    }
    broken_audit["tags"].extend([
        "audit", "evidence-chain", "negative", "confidence-cap",
    ])
    _set_requirements(
        broken_audit,
        ["REQ-AUD-001", "REQ-OUT-001"],
        ["ADV-003", "FUS-005", "SER-001"],
        ["T-ADV-003"],
    )
    fixtures.append(broken_audit)

    for complete_ev in (True, False):
        suffix = "COMPLETE-EV" if complete_ev else "MISSING-EV"
        f = _base_fixture(
            f"MOCK-TRAIN-{suffix}",
            "debrief with complete EV" if complete_ev
            else "debrief without counterfactual EV",
            6,
        )
        f["input"]["actual_action"] = {
            "action": "call", "amount": "2.5",
            "observed_at": "2026-08-22T00:00:01Z",
            "advice_request_id": f["input"]["request_context"]["request_id"],
            "confidence": "0.99",
        }
        f["input"]["counterfactual_ev"] = (
            {"fold": "0", "call": "1.2", "raise": "2.0"}
            if complete_ev else None
        )
        f["input"]["hand_decisions"] = [
            {
                "state_version": 1,
                "request_id": f["input"]["request_context"]["request_id"],
                "actual_action": "call",
                "ev_loss": "0.8" if complete_ev else None,
            },
            {
                "state_version": 2,
                "request_id": f"{f['input']['request_context']['request_id']}-2",
                "actual_action": "raise",
                "ev_loss": "0",
            },
        ]
        f["expected"]["debrief"] = {
            "actual_action": "call",
            "show_ev_loss": complete_ev,
            "ev_loss": "0.8" if complete_ev else None,
            "show_strategy_deviation": True,
        }
        f["expected"]["hand_review"] = {
            "decision_count": 2,
            "known_ev_loss_total": "0.8" if complete_ev else "0",
            "ev_loss_complete": complete_ev,
            "ev_unavailable_count": 0 if complete_ev else 1,
            "max_loss_state_version": 1 if complete_ev else 2,
            "must_not_infer_missing_ev": True,
        }
        f["tags"].extend(["training", "debrief"])
        _set_requirements(
            f, ["REQ-TRN-001", "REQ-AUD-001"],
            ["TRN-001", "TRN-002", "ADV-003"],
            ["T-TRN-001", "T-INT-008", "E2E-012"],
        )
        fixtures.append(f)
    return fixtures


def generate_contract_flows() -> list[dict[str, Any]]:
    fixtures = []

    noop = _base_fixture("MOCK-STATE-NOOP", "duplicate observation is a no-op")
    noop["input"]["previous_state"] = copy.deepcopy(noop["input"]["state"])
    noop["expected"]["state_version_delta"] = 0
    noop["expected"]["persist_event_count"] = 0
    noop["tags"].extend(["state", "no-op", "idempotency"])
    _set_requirements(
        noop, ["REQ-ST-001"], ["ST-001", "ST-003"], ["T-ST-001"],
    )
    fixtures.append(noop)

    boundary = _base_fixture(
        "MOCK-STATE-HAND-BOUNDARY", "two stable frames start a new hand",
    )
    boundary["input"]["previous_state"] = copy.deepcopy(
        boundary["input"]["state"]
    )
    boundary["input"]["previous_state"]["hero_cards"] = ["Qc", "Qd"]
    boundary["input"]["observations"]["hero_card_sequence"] = [
        {"frame_seq": 100, "cards": ["As", "Kd"], "status": "VALID"},
        {"frame_seq": 101, "cards": ["As", "Kd"], "status": "VALID"},
    ]
    boundary["expected"]["new_hand"] = True
    boundary["expected"]["closed_previous_hand"] = True
    boundary["tags"].extend(["state", "hand-boundary", "temporal"])
    _set_requirements(
        boundary, ["REQ-IN-002", "REQ-ST-001"],
        ["ST-004", "MEM-001"], ["T-ST-004"],
    )
    fixtures.append(boundary)

    provenance = _base_fixture(
        "MOCK-CONTEXT-MIXED-PROVENANCE",
        "vision cards and manually entered stacks",
    )
    provenance["input"]["observations"]["field_quality"]["player_stacks"].update({
        "source": "manual", "confidence": "1",
        "evidence_ref": "manual://session/stack-form",
    })
    provenance["expected"]["provenance_badges"] = {
        "hero_cards": "vision", "player_stacks": "manual",
    }
    provenance["tags"].extend(["context", "manual-input", "provenance"])
    _set_requirements(
        provenance, ["REQ-IN-001", "REQ-CTX-001", "REQ-UI-001"],
        ["CTX-001", "CTX-002", "UI-002"],
        ["T-CTX-001", "T-CTX-003", "T-INT-002"],
    )
    fixtures.append(provenance)

    parity = _base_fixture(
        "MOCK-PROVIDER-ADAPTER-PARITY", "fake upstream and adapter parity",
    )
    parity["input"]["providers"][0]["direct_upstream_result"] = copy.deepcopy(
        parity["input"]["providers"][0]["mock_result"]
    )
    parity["expected"]["assertions"].append(
        "adapter_result_equals_direct_upstream_result"
    )
    parity["tags"].extend(["provider", "adapter-parity", "synthetic"])
    _set_requirements(
        parity, ["REQ-PRV-001"], ["PRV-001", "PRV-002"],
        ["T-PRV-001", "T-PRV-002", "T-INT-001"],
    )
    fixtures.append(parity)

    priority = _base_fixture(
        "MOCK-ROUTER-EXACT-BEATS-HEURISTIC",
        "exact Provider outranks heuristic Provider", 6,
    )
    heuristic = copy.deepcopy(priority["input"]["providers"][0])
    heuristic["provider_id"] = "mock-heuristic-6p-v1"
    heuristic["capability"]["match_kind"] = "heuristic"
    priority["input"]["providers"].insert(0, heuristic)
    priority["expected"]["provider_lookups"] = [
        {"provider_id": "mock-heuristic-6p-v1", "state": "HIT_APPROXIMATE"},
        {"provider_id": "mock-preflop-6p-v1", "state": "HIT_EXACT"},
    ]
    priority["tags"].extend(["router", "priority", "multiple-candidates"])
    _set_requirements(
        priority, ["REQ-RTR-001", "REQ-FUS-001"],
        ["RTR-003", "FUS-002"], ["T-RTR-002"],
    )
    fixtures.append(priority)

    exploit = _base_fixture(
        "MOCK-FUSION-KL-OPPONENT-ADJUSTMENT",
        "KL-bounded adjustment of an exact baseline from a trusted profile",
        6,
    )
    exploit["input"]["fusion_adjustment"] = {
        "baseline_action_probabilities": {
            "fold": "0.5", "call": "0.3", "raise": "0.2",
        },
        "action_q_values": {"fold": "-1", "call": "0", "raise": "2"},
        "profile": {"sample_size": 500, "quality": "0.8"},
        "policy": {
            "minimum_sample_size": 100,
            "minimum_profile_quality": "0.6",
            "maximum_profile_weight": "0.5",
            "kl_budget": "0.05",
            "maximum_logit_shift": "4",
        },
    }
    exploit["expected"]["exploit_adjustment"] = {
        "status": "APPLIED",
        "match_kind": "heuristic",
        "maximum_kl_divergence": "0.05",
        "probability_direction": {
            "raise": "increase", "fold": "decrease",
        },
        "preserve_zero_support": True,
        "metadata_required": [
            "profile_sample_size", "profile_quality", "kl_divergence",
        ],
    }
    exploit["expected"]["assertions"].extend([
        "adjusted_probabilities_sum_to_one",
        "kl_divergence_does_not_exceed_budget",
        "adjusted_candidate_is_not_labeled_exact",
    ])
    exploit["tags"].extend([
        "fusion", "opponent-profile", "kl-budget", "exploitative",
    ])
    _set_requirements(
        exploit,
        ["REQ-FUS-001", "REQ-AUD-001"],
        ["FUS-002", "FUS-003", "FUS-005", "ADV-003"],
        ["T-FUS-004", "T-FUS-005"],
    )
    fixtures.append(exploit)

    resolver = _base_fixture(
        "MOCK-LOCAL-RESOLVER-CONVERGED",
        "versioned local resolver process response upgrades Slow Advice",
    )
    resolver["input"]["local_resolver_request"] = {
        "protocol_schema_version": 1,
        "provider_id": "local-cfr",
        "source_version": "solver-v3",
        "command_transport": "argv-no-shell-json-stdin-stdout",
        "timeout_ms": 500,
        "identity": {
            "hand_id": resolver["input"]["request_context"]["hand_id"],
            "state_version": 1,
            "request_id": resolver["input"]["request_context"]["request_id"],
        },
    }
    resolver["input"]["local_resolver_response"] = {
        "status": "CONVERGED",
        "iterations": 1200,
        "exploitability_bb100": "0.02",
        "action_probabilities": {"check": "0.25", "raise": "0.75"},
        "recommended_sizes": {"raise": ["2.5"]},
        "action_ev": {"check": "0", "raise": "1.5"},
    }
    resolver["expected"]["provider_lookups"] = [{
        "provider_id": "local-cfr", "state": "HIT_APPROXIMATE",
    }]
    resolver["expected"]["slow_refinement"] = {
        "state": "APPLIED",
        "identity_must_match": True,
        "convergence_metadata_required": True,
        "expired_result_must_be_discarded": True,
    }
    resolver["tags"].extend([
        "resolver", "subprocess", "slow-path", "converged",
    ])
    _set_requirements(
        resolver,
        ["REQ-PRV-001", "REQ-RTR-001", "REQ-AUD-001"],
        ["PRV-007", "RTR-006", "RTR-007", "ADV-003"],
        ["T-PRV-005", "T-INT-005", "T-INT-010"],
    )
    fixtures.append(resolver)

    heuristic_rfi = _base_fixture(
        "MOCK-PREFLOPR-6P-CO-AKO",
        "reviewed 6-handed CO AKo heuristic RFI asset hit",
        6,
    )
    hero_seat = heuristic_rfi["input"]["state"]["hero_seat"]
    seats = heuristic_rfi["input"]["state"]["seats"]
    hero = seats[hero_seat]
    co = next(seat for seat in seats if seat["position"] == "CO")
    hero["position"], co["position"] = co["position"], hero["position"]
    heuristic_rfi["input"]["state"]["legal_actions"] = [
        {"action": "fold", "min": "0", "max": "0"},
        {"action": "raise", "min": "2", "max": "100"},
    ]
    heuristic_rfi["input"]["state"]["action_line"] = "unopened"
    asset_path = (
        ROOT / "src" / "poker_engine" / "strategy" / "assets"
        / "preflopr-explicit-rfi-ranges.json"
    )
    asset_sha = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    provider = heuristic_rfi["input"]["providers"][0]
    provider.update({
        "provider_id": "preflopr-explicit-rfi-heuristic",
        "source_version": (
            "preflopR/aed511d0451aea33a14f7e9204595fc2211f233f"
            f":asset/1@{asset_sha}"
        ),
        "asset_hash": f"sha256:{asset_sha}",
    })
    provider["capability"].update({
        "player_counts": [6, 9],
        "stack_buckets_bb": [100],
        "action_lines": ["unopened"],
        "match_kind": "heuristic",
    })
    provider["mock_result"] = {
        "action_probabilities": {"fold": "0", "raise": "1"},
        "recommended_sizes": [],
        "action_ev": {},
        "confidence": "0.4",
        "assumptions": [
            "heuristic_open_raise_chart_not_solver_derived",
            "no_raise_size_or_ev",
        ],
    }
    heuristic_rfi["expected"]["provider_lookups"] = [{
        "provider_id": provider["provider_id"],
        "state": "HIT_APPROXIMATE",
    }]
    heuristic_rfi["expected"]["advice"].update({
        "match_kind": "heuristic",
        "strategy_source": provider["provider_id"],
        "strategy_version": provider["source_version"],
        "action_probabilities": {"fold": "0", "raise": "1"},
    })
    heuristic_rfi["expected"]["assertions"].extend([
        "source_and_asset_hash_are_disclosed",
        "raise_size_and_ev_are_not_invented",
    ])
    heuristic_rfi["tags"].extend([
        "provider", "heuristic", "preflopr", "asset-backed",
    ])
    _set_requirements(
        heuristic_rfi,
        ["REQ-PRV-001", "REQ-RTR-001", "REQ-AUD-001"],
        ["PRV-006", "RTR-002", "RTR-003", "ADV-003"],
        ["T-PRV-003", "T-INT-012"],
    )
    fixtures.append(heuristic_rfi)

    for players, suffix, reason in (
        (8, "8P-UNSUPPORTED", "unsupported_player_count"),
        (9, "9P-BB", "unsupported_open_raise_position"),
    ):
        rejected = _base_fixture(
            f"MOCK-PREFLOPR-{suffix}",
            "PreflopR heuristic refuses synthetic player-count/BB fallback",
            players,
        )
        rejected["input"]["state"]["action_line"] = "unopened"
        rejected_provider = copy.deepcopy(provider)
        rejected["input"]["providers"] = [rejected_provider]
        rejected["expected"]["provider_lookups"] = [{
            "provider_id": rejected_provider["provider_id"],
            "state": "NOT_APPLICABLE",
        }]
        _set_abstain(rejected, reason, "router")
        rejected["tags"].extend([
            "provider", "heuristic", "negative", "no-fallback",
        ])
        _set_requirements(
            rejected,
            ["REQ-PRV-001", "REQ-RTR-001", "REQ-OUT-002"],
            ["PRV-003", "PRV-006", "RTR-002", "FUS-004"],
            ["T-PRV-004"],
        )
        fixtures.append(rejected)

    bayes = _base_fixture(
        "MOCK-RANGE-BAYES-UPDATE", "deterministic two-combo Bayesian update",
        6,
    )
    bayes["input"]["range_update"] = {
        "prior": {"AsAh": "0.5", "KsKh": "0.5"},
        "action_likelihood": {"AsAh": "0.8", "KsKh": "0.2"},
        "observed_action": "raise",
        "profile_sample_size": 100,
    }
    bayes["expected"]["posterior"] = {"AsAh": "0.8", "KsKh": "0.2"}
    bayes["tags"].extend(["range", "bayes", "shrinkage"])
    _set_requirements(
        bayes, ["REQ-RNG-001"], ["RNG-001", "RNG-003", "RNG-004"],
        ["T-RNG-001", "T-RNG-002", "T-RNG-003"],
    )
    fixtures.append(bayes)

    range_prior = _base_fixture(
        "MOCK-RANGE-PRIOR-6P-CO-OPEN",
        "versioned concrete-combo prior for a 6-handed CO first-in raise",
        6,
    )
    range_prior["input"]["range_prior_query"] = {
        "seat_id": 2,
        "player_count": 6,
        "position": "CO",
        "effective_stack_bb": "100",
        "action_line": "open_raise",
        "known_cards": ["As", "Kd"],
        "source_asset": "preflopr-explicit-rfi-ranges.json",
    }
    range_prior["expected"]["range_prior"] = {
        "state": "HIT",
        "combo_format": "concrete-four-character",
        "weights_sum": "1",
        "confidence": "0.4",
        "effective_sample_size": 0,
        "known_card_collisions": 0,
        "source_version_contains": "uniform-combo-prior/v1",
    }
    range_prior["expected"]["assertions"].extend([
        "range_contains_only_concrete_combos",
        "range_has_no_known_card_collision",
        "range_does_not_use_random_fallback",
    ])
    range_prior["tags"].extend([
        "range", "prior", "asset-backed", "blockers",
    ])
    _set_requirements(
        range_prior,
        ["REQ-RNG-001", "REQ-AUD-001"],
        ["RNG-001", "RNG-002", "ADV-003"],
        ["T-RNG-005"],
    )
    fixtures.append(range_prior)

    range_unknown = _base_fixture(
        "MOCK-RANGE-PRIOR-8P-NO-FALLBACK",
        "unsupported 8-player prior remains unknown instead of inheriting",
        8,
    )
    range_unknown["input"]["range_prior_query"] = {
        "seat_id": 2,
        "player_count": 8,
        "position": "CO",
        "effective_stack_bb": "100",
        "action_line": "open_raise",
        "known_cards": [],
    }
    range_unknown["expected"]["range_prior"] = {
        "state": "NOT_APPLICABLE",
        "distribution": None,
        "reasons": ["unsupported_player_count"],
    }
    range_unknown["tags"].extend([
        "range", "prior", "negative", "no-random-fallback",
    ])
    _set_requirements(
        range_unknown,
        ["REQ-RNG-001"],
        ["RNG-001"],
        ["T-RNG-005"],
    )
    fixtures.append(range_unknown)

    fusion = _base_fixture(
        "MOCK-FUSION-LEGALIZE", "remove illegal action and renormalize",
    )
    fusion["input"]["candidate_before_legalization"] = {
        "check": "0.2", "raise": "0.3", "bet": "0.5",
    }
    fusion["expected"]["candidate_after_legalization"] = {
        "check": "0.4", "raise": "0.6",
    }
    fusion["tags"].extend(["fusion", "legalization", "boundary"])
    _set_requirements(
        fusion, ["REQ-FUS-001"], ["FUS-001", "FUS-005"],
        ["T-FUS-001", "T-FUS-003"],
    )
    fixtures.append(fusion)

    ev = _base_fixture("MOCK-EV-BRANCH-TREE", "known bet EV branch tree")
    ev["input"]["ev_tree"] = {
        "bet": "10", "pot": "20", "fold_probability": "0.4",
        "call_probability": "0.5", "raise_probability": "0.1",
        "call_continuation_ev": "4", "raise_continuation_ev": "-10",
    }
    ev["expected"]["action_ev"] = {"bet": "9"}
    ev["tags"].extend(["ev", "branch-tree", "numeric"])
    _set_requirements(
        ev, ["REQ-FUS-001"], ["EV-002", "EV-003"],
        ["T-EV-002"],
    )
    fixtures.append(ev)

    refinement = _base_fixture(
        "MOCK-FAST-SLOW-REFINEMENT", "same-version slow result refines Fast Advice",
    )
    refinement["input"]["candidate_sequence"] = [
        {"path": "fast", "state_version": 1, "status": "PARTIAL",
         "match_kind": "equity_only", "latency_ms": 80},
        {"path": "slow", "state_version": 1, "status": "READY",
         "match_kind": "exact", "latency_ms": 900},
    ]
    refinement["expected"]["ui_sequence"] = ["PARTIAL", "READY"]
    refinement["tags"].extend(["fast-slow", "refinement", "ui-sequence"])
    _set_requirements(
        refinement, ["REQ-RTR-001", "REQ-UI-001"],
        ["RTR-006", "UI-003"], ["T-INT-005"],
    )
    fixtures.append(refinement)

    reconnect = _base_fixture(
        "MOCK-WEBSOCKET-RECONNECT", "reconnect restores only current Advice",
    )
    reconnect["input"]["stored_advice"] = [
        {"request_id": "req-old", "status": "STALE"},
        {"request_id": reconnect["input"]["request_context"]["request_id"],
         "status": "READY"},
    ]
    reconnect["expected"]["restored_request_ids"] = [
        reconnect["input"]["request_context"]["request_id"]
    ]
    reconnect["tags"].extend(["websocket", "reconnect", "stale"])
    _set_requirements(
        reconnect, ["REQ-UI-001", "REQ-OUT-001"],
        ["SER-001", "UI-003"], ["T-INT-009", "E2E-011"],
    )
    fixtures.append(reconnect)
    return fixtures


def generate_benchmarks() -> list[dict[str, Any]]:
    fixtures = []
    cases = (
        ("CTX-2P", 2, "preflop", 10000, 10, ["P-001", "PERF-CTX"]),
        ("LOOKUP-9P", 9, "preflop", 10000, 10,
         ["P-002", "PERF-LOOKUP"]),
        ("MULTIWAY-EQUITY-4P", 4, "flop", 200, None, ["P-003"]),
        ("FIRST-ADVICE-6P", 6, "preflop", 2000, 300,
         ["P-004", "PERF-001", "PERF-FIRST"]),
        ("OBS-STABILIZATION", 2, "preflop", 2000, None, ["P-005"]),
        ("SLOW-RESOLVER", 2, "river", 100, None,
         ["P-006", "PERF-SLOW"]),
        ("UI-PAINT", 6, "preflop", 5000, 25, ["P-007", "PERF-UI"]),
        ("LONG-SESSION", 9, "river", 100000, None,
         ["P-008", "PERF-STABILITY"]),
    )
    for name, count, street, iterations, latency, test_ids in cases:
        f = _base_fixture(f"MOCK-BENCH-{name}", name.lower().replace("-", " "),
                          count, street, count)
        f["fixture_type"] = "benchmark"
        f["input"]["benchmark"] = {
            "iterations": iterations, "warmup_iterations": 100,
            "measure": name.lower(),
        }
        f["tolerances"]["latency_ms"] = latency
        f["tags"].extend(["benchmark", "performance"])
        _set_requirements(
            f, ["REQ-PERF-001"], ["CTX-004", "RTR-005", "UI-003"],
            test_ids,
        )
        fixtures.append(f)
    return fixtures


def generate_all() -> list[dict[str, Any]]:
    fixtures = (
        generate_preflop() + generate_postflop() + generate_quality()
        + generate_provenance() + generate_action_reconstruction()
        + generate_temporal_consensus()
        + generate_hand_boundary_detection()
        + generate_platform_mapping()
        + generate_atomic_memory()
        + generate_abstraction_matching()
        + generate_hard_gates()
        + generate_fast_source_fallback()
        + generate_strategy_asset_adapter()
        + generate_gtopen_adapter()
        + generate_boundaries()
        + generate_side_pots() + generate_faults()
        + generate_equity() + generate_output_states()
        + generate_audit_and_training() + generate_contract_flows()
        + generate_benchmarks()
    )
    fixtures.sort(key=lambda item: item["fixture_id"])
    validate_dataset(fixtures)
    return fixtures


def validate_fixture(fixture: dict[str, Any]) -> None:
    required = set(SCHEMA["required"])
    if set(fixture) != required:
        raise ValueError(
            f"{fixture.get('fixture_id')}: top-level fields differ: "
            f"missing={required - set(fixture)}, extra={set(fixture) - required}"
        )
    if fixture["fixture_schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{fixture['fixture_id']}: schema version mismatch")
    if fixture["fixture_type"] not in FIXTURE_TYPES:
        raise ValueError(f"{fixture['fixture_id']}: bad fixture type")
    if not fixture["requirements"] or not set(fixture["requirements"]) <= REQUIREMENTS:
        raise ValueError(f"{fixture['fixture_id']}: unknown/empty requirements")
    for key in ("function_ids", "test_ids", "tags"):
        if not fixture[key] or len(fixture[key]) != len(set(fixture[key])):
            raise ValueError(f"{fixture['fixture_id']}: invalid {key}")
    for key in (
        "game_config", "request_context", "observations", "state", "ranges",
        "providers", "fault_injection",
    ):
        if key not in fixture["input"]:
            raise ValueError(f"{fixture['fixture_id']}: missing input.{key}")
    expected = fixture["expected"]
    if set(("terminal_stage", "advice", "provider_lookups", "assertions")) \
            - set(expected):
        raise ValueError(f"{fixture['fixture_id']}: incomplete expected")
    if expected["advice"]["status"] not in ADVICE_STATUSES:
        raise ValueError(f"{fixture['fixture_id']}: invalid Advice status")
    for lookup in expected["provider_lookups"]:
        if lookup["state"] not in LOOKUP_STATES:
            raise ValueError(f"{fixture['fixture_id']}: bad lookup state")
    state = fixture["input"]["state"]
    if state["street"] not in STREETS:
        raise ValueError(f"{fixture['fixture_id']}: bad street")
    dealt = state["dealt_player_count"]
    active = state["active_player_count"]
    if not (2 <= dealt <= 9 and 2 <= active <= dealt):
        raise ValueError(f"{fixture['fixture_id']}: invalid player counts")


def validate_dataset(fixtures: list[dict[str, Any]]) -> None:
    ids = [fixture["fixture_id"] for fixture in fixtures]
    if len(ids) != len(set(ids)):
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        raise ValueError(f"duplicate fixture IDs: {duplicates}")
    for fixture in fixtures:
        validate_fixture(fixture)
    covered_requirements = {
        req for fixture in fixtures for req in fixture["requirements"]
    }
    if covered_requirements != REQUIREMENTS:
        raise ValueError(
            f"requirement coverage mismatch: missing="
            f"{sorted(REQUIREMENTS - covered_requirements)}"
        )
    preflop_counts = {
        fixture["input"]["state"]["dealt_player_count"]
        for fixture in fixtures if fixture["input"]["state"]["street"] == "preflop"
    }
    if preflop_counts != set(range(2, 10)):
        raise ValueError("preflop player-count coverage must be 2..9")
    statuses = {fixture["expected"]["advice"]["status"] for fixture in fixtures}
    if statuses != ADVICE_STATUSES:
        raise ValueError("all Advice statuses must be covered")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n").encode("utf-8")


def _jsonl_bytes(fixtures: list[dict[str, Any]]) -> bytes:
    lines = [json.dumps(item, ensure_ascii=False, sort_keys=True)
             for item in fixtures]
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_artifacts() -> dict[Path, bytes]:
    fixtures = generate_all()
    fixture_bytes = _jsonl_bytes(fixtures)
    by_status = Counter(f["expected"]["advice"]["status"] for f in fixtures)
    by_type = Counter(f["fixture_type"] for f in fixtures)
    by_street = Counter(f["input"]["state"]["street"] for f in fixtures)
    by_count: dict[str, int] = defaultdict(int)
    for fixture in fixtures:
        by_count[str(fixture["input"]["state"]["dealt_player_count"])] += 1
    manifest = {
        "fixture_schema_version": SCHEMA_VERSION,
        "generated_at": GENERATED_AT,
        "generator": "tools/generate_strategy_mock_fixtures.py",
        "fixture_file": "fixtures.jsonl",
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "fixture_count": len(fixtures),
        "coverage": {
            "requirements": sorted(REQUIREMENTS),
            "advice_statuses": dict(sorted(by_status.items())),
            "fixture_types": dict(sorted(by_type.items())),
            "streets": dict(sorted(by_street.items())),
            "dealt_player_counts": dict(sorted(by_count.items())),
            "preflop_action_lines": list(ACTION_LINES),
            "quality_fields": list(QUALITY_FIELDS),
        },
        "not_release_acceptance_evidence": [
            "real_provider_golden_parity",
            "real_platform_capture_replay",
            "hardware_specific_performance_result",
            "clean_install_platform_acceptance",
        ],
    }
    return {
        OUTPUT_DIR / "schema.json": _json_bytes(SCHEMA),
        OUTPUT_DIR / "fixtures.jsonl": fixture_bytes,
        OUTPUT_DIR / "manifest.json": _json_bytes(manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true",
        help="fail when generated artifacts differ from committed files",
    )
    args = parser.parse_args()
    artifacts = build_artifacts()
    if args.check:
        mismatches = [
            str(path.relative_to(ROOT)) for path, content in artifacts.items()
            if not path.exists() or path.read_bytes() != content
        ]
        if mismatches:
            raise SystemExit("out-of-date strategy fixtures: " + ", ".join(mismatches))
        print(f"strategy mock fixtures are current ({len(generate_all())} fixtures)")
        return 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, content in artifacts.items():
        path.write_bytes(content)
    print(f"generated {len(generate_all())} fixtures in {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
