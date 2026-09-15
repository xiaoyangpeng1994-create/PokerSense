"""Analyze a manually specified terminal multiway river call/fold decision."""

import argparse
from dataclasses import fields, is_dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from poker_engine.core.enums import PlayerStatus, Position, Rank, Suit
from poker_engine.core.value_objects import Card, ChipAmount
from poker_engine.strategy.aa_rules_v2 import AARuleProfileV2
from poker_engine.strategy.contracts import DecisionSeat, RangeDistribution
from poker_engine.strategy.terminal_multiway_v1 import (
    TerminalScenario, analyze_terminal_multiway,
)


def exact_string(value):
    if not isinstance(value, str):
        raise ValueError("amounts and weights require exact decimal strings")
    amount = Decimal(value)
    if not amount.is_finite() or amount < 0:
        raise ValueError("finite nonnegative decimal required")
    return amount


def card(value):
    if not isinstance(value, str) or len(value) != 2:
        raise ValueError("two-character card code required")
    return Card(Rank(value[0]), Suit(value[1]))


def scenario_from_dict(data):
    required = {"schema_version", "mode", "range_assumptions", "rules", "seats",
                "hero_seat", "actor_seat", "hero_cards", "board_cards",
                "current_bet", "pot_before", "ranges", "other_fees", "split_policy"}
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError("scenario requires exact declared fields")
    if (type(data["schema_version"]) is not int or data["schema_version"] != 1
            or data["mode"] != "manual_hypothesis"
            or not isinstance(data["range_assumptions"], str)
            or not data["range_assumptions"].strip()):
        raise ValueError("explicit manual-hypothesis disclosure required")
    if type(data["hero_seat"]) is not int or type(data["actor_seat"]) is not int:
        raise ValueError("integer seat identities required")
    if not isinstance(data["seats"], list) or not isinstance(data["ranges"], list):
        raise ValueError("seat and range lists required")
    seats = []
    for item in data["seats"]:
        if not isinstance(item, dict) or set(item) != {
            "seat_id", "stack", "street_committed", "hand_committed", "status"
        }:
            raise ValueError("seat requires exact fields")
        if type(item["seat_id"]) is not int or not isinstance(item["status"], str):
            raise ValueError("seat identity/status invalid")
        try:
            status = PlayerStatus[item["status"]]
        except KeyError:
            raise ValueError("unknown player status") from None
        seats.append(DecisionSeat(
            item["seat_id"], f"manual-seat-{item['seat_id']}", Position.UNKNOWN,
            *(ChipAmount(exact_string(item[name])) for name in (
                "stack", "street_committed", "hand_committed")),
            status=status, is_hero=item["seat_id"] == data["hero_seat"],
        ))
    ranges = []
    for item in data["ranges"]:
        if (not isinstance(item, dict) or set(item) != {"seat_id", "combos"}
                or type(item["seat_id"]) is not int
                or not isinstance(item["combos"], dict)):
            raise ValueError("range requires seat_id and concrete combo weights")
        ranges.append(RangeDistribution(
            item["seat_id"], {key: exact_string(value)
                              for key, value in item["combos"].items()},
            "manual_terminal_assumption", f"manual:seat{item['seat_id']}",
            confidence=0, effective_sample_size=0,
        ))
    if not all(isinstance(data[key], list) for key in ("hero_cards", "board_cards")):
        raise ValueError("card lists required")
    return TerminalScenario(
        seats=tuple(seats), hero_seat=data["hero_seat"], actor_seat=data["actor_seat"],
        hero_cards=tuple(map(card, data["hero_cards"])),
        board_cards=tuple(map(card, data["board_cards"])),
        current_bet=ChipAmount(exact_string(data["current_bet"])),
        pot_before=ChipAmount(exact_string(data["pot_before"])), ranges=tuple(ranges),
        rules=AARuleProfileV2.from_dict(data["rules"]),
        other_fees=exact_string(data["other_fees"]), split_policy=data["split_policy"],
    )


def encode(value):
    if isinstance(value, Fraction):
        with localcontext() as context:
            context.prec = 28
            approximate = str(Decimal(value.numerator) / Decimal(value.denominator))
        return {"exact": str(value), "decimal": approximate}
    if isinstance(value, Decimal):
        return str(value)
    if is_dataclass(value):
        return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate input field")
        result[key] = value
    return result


def analyze_file(path, max_joint_assignments=4096):
    raw = path.read_bytes()
    data = json.loads(raw, object_pairs_hook=unique_object)
    scenario = scenario_from_dict(data)
    result = analyze_terminal_multiway(
        scenario, max_joint_assignments=max_joint_assignments)
    return {
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "MANUAL_CONDITIONAL_TERMINAL_RIVER_ANALYSIS_NOT_LIVE_ADVICE",
        "table_players": scenario.rules.table_size,
        "all_in_opponents": len(scenario.ranges),
        "range_assumptions": data["range_assumptions"],
        "result": encode(result),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-joint-assignments", type=int, default=4096)
    args = parser.parse_args()
    try:
        report = analyze_file(args.input, args.max_joint_assignments)
    except (OSError, TypeError, ValueError, ArithmeticError) as exc:
        parser.exit(2, f"terminal analysis rejected: {exc}\n")
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return 0 if report["result"]["status"] == "COMPLETE_CONDITIONAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
