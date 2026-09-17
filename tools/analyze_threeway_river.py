"""Manual three-player river action study and named-model sensitivity check."""

import argparse
from copy import deepcopy
from dataclasses import fields, is_dataclass
from decimal import Decimal
from fractions import Fraction
import hashlib
import inspect
from itertools import product
import json
from pathlib import Path

from poker_engine.core.enums import PlayerStatus, Position
from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.aa_rules_v2 import AARuleProfileV2
from poker_engine.strategy.contracts import DecisionSeat, RangeDistribution
from poker_engine.strategy.threeway_river_v1 import (
    ResponseModel, RiverAction, ThreewayRiverScenario, analyze_threeway_river,
)
from tools.analyze_terminal_multiway import (
    card, encode as scalar_encode, exact_string, unique_object,
)


# Untrained illustrative weights, not measured action frequencies. Each weight
# applies to each matching legal action; adding sizes changes normalization.
STYLE_WEIGHTS = {
    "tight": {
        "default": (6, 8, 3, 1, 1), "weak": (12, 10, 1, 1, 0),
        "strong": (1, 2, 7, 7, 4),
    },
    "loose_passive": {
        "default": (1, 10, 9, 1, 1), "weak": (2, 10, 8, 1, 0),
        "strong": (0, 3, 10, 5, 2),
    },
    "aggressive": {
        "default": (2, 2, 5, 7, 6), "weak": (4, 3, 2, 8, 6),
        "strong": (0, 1, 5, 10, 10),
    },
}
WEIGHT_KEYS = ("fold", "check", "call", "bet", "raise")


def style_model(seat_id, style):
    values = STYLE_WEIGHTS[style]

    def table(name):
        return dict(zip(WEIGHT_KEYS, map(str, values[name])))
    return {
        "seat_id": seat_id, "name": style + "-illustrative-v2",
        "source": "manual_example", "weights": table("default"),
        "category_weights": {str(c): table("weak" if c == 0 else "strong")
                             for c in (0, 3, 4, 5, 6, 7, 8)},
        "price_multipliers": [
            {"upper_ratio": "0.2", "weights": {}},
            {"upper_ratio": "1", "weights": {
                "tight": {"fold": "3", "call": "0.5", "raise": "1"},
                "loose_passive": {"fold": "1.5", "call": "0.8", "raise": "0.8"},
                "aggressive": {"fold": "1", "call": "1", "raise": "1.2"},
            }[style]},
        ],
    }


def encode(value):
    if is_dataclass(value):
        return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return scalar_encode(value)


def weight_table(data, *, allow_empty=False):
    if not isinstance(data, dict) or not data and not allow_empty:
        raise ValueError("explicit model weight mapping required")
    return tuple((key, Fraction(exact_string(value))) for key, value in data.items())


def scenario_from_dict(data):
    keys = {"schema_version", "mode", "range_start", "range_assumptions",
            "model_assumptions", "rules", "seats", "hero_seat", "hero_cards",
            "board_cards", "action_order", "ranges", "models",
            "aggression_targets", "max_aggressions", "history", "other_fees"}
    if not isinstance(data, dict) or set(data) != keys:
        raise ValueError("exact threeway scenario fields required")
    if (type(data["schema_version"]) is not int or data["schema_version"] != 1
            or data["mode"] != "manual_hypothesis"
            or data["range_start"] != "river_start"
            or type(data["hero_seat"]) is not int
            or type(data["max_aggressions"]) is not int
            or any(not isinstance(data[k], str) or not data[k].strip()
                   for k in ("range_assumptions", "model_assumptions"))):
        raise ValueError("manual river-start range and response hypotheses required")
    for name in ("seats", "hero_cards", "board_cards", "action_order", "ranges",
                 "models", "aggression_targets", "history"):
        if not isinstance(data[name], list):
            raise ValueError("explicit scenario lists required")
    if any(type(v) is not int for v in data["action_order"]):
        raise ValueError("integer action order required")
    seats = []
    for item in data["seats"]:
        if (not isinstance(item, dict) or set(item) != {
                "seat_id", "stack", "hand_committed", "status"}
                or type(item["seat_id"]) is not int):
            raise ValueError("river-start seat fields required; no current-street debt")
        try:
            status = PlayerStatus[item["status"]]
        except (KeyError, TypeError):
            raise ValueError("invalid seat status") from None
        seats.append(DecisionSeat(
            item["seat_id"], f"manual-seat-{item['seat_id']}", Position.UNKNOWN,
            ChipAmount(exact_string(item["stack"])), ChipAmount("0"),
            ChipAmount(exact_string(item["hand_committed"])), status,
            is_hero=item["seat_id"] == data["hero_seat"],
        ))
    ranges = []
    for item in data["ranges"]:
        if (not isinstance(item, dict) or set(item) != {"seat_id", "combos"}
                or type(item["seat_id"]) is not int
                or not isinstance(item["combos"], dict)):
            raise ValueError("manual opponent concrete ranges required")
        ranges.append(RangeDistribution(
            item["seat_id"], {k: exact_string(v) for k, v in item["combos"].items()},
            "manual_river_start_assumption", f"manual:seat{item['seat_id']}",
            confidence=0, effective_sample_size=0,
        ))
    models = []
    for item in data["models"]:
        if (not isinstance(item, dict) or set(item) - {"price_multipliers"} != {
                "seat_id", "name", "source", "weights", "category_weights"}
                or type(item["seat_id"]) is not int
                or item["source"] != "manual_example"
                or not isinstance(item["name"], str) or not item["name"].strip()
                or not isinstance(item["category_weights"], dict)):
            raise ValueError("explicit manual response model required")
        categories = []
        for category, values in item["category_weights"].items():
            if category not in tuple(map(str, range(9))):
                raise ValueError("category must be canonical 0..8")
            categories.append((int(category), weight_table(values)))
        bands = item.get("price_multipliers", [])
        if not isinstance(bands, list):
            raise ValueError("price multiplier bands must be a list")
        prices = []
        for band in bands:
            if not isinstance(band, dict) or set(band) != {"upper_ratio", "weights"}:
                raise ValueError("explicit price ratio/weights required")
            prices.append((Fraction(exact_string(band["upper_ratio"])),
                           weight_table(band["weights"], allow_empty=True)))
        models.append(ResponseModel(item["seat_id"], weight_table(item["weights"]),
                                    tuple(categories), tuple(prices)))
    history = []
    for item in data["history"]:
        if (not isinstance(item, dict) or set(item) != {"actor", "kind", "target"}
                or type(item["actor"]) is not int or not isinstance(item["kind"], str)):
            raise ValueError("public history actor/kind/target required")
        history.append(RiverAction(
            item["actor"], item["kind"], exact_string(item["target"])))
    return ThreewayRiverScenario(
        seats=tuple(seats), hero_seat=data["hero_seat"],
        hero_cards=tuple(map(card, data["hero_cards"])),
        board_cards=tuple(map(card, data["board_cards"])), ranges=tuple(ranges),
        rules=AARuleProfileV2.from_dict(data["rules"]),
        action_order=tuple(data["action_order"]), models=tuple(models),
        aggression_targets=tuple(map(exact_string, data["aggression_targets"])),
        history=tuple(history), max_aggressions=data["max_aggressions"],
        other_fees=exact_string(data["other_fees"]), range_start=data["range_start"],
    )


def analyze_file(path, *, sensitivity=False, max_joint_assignments=128,
                 max_nodes=20000):
    raw = path.read_bytes()
    data = json.loads(raw, object_pairs_hook=unique_object)
    scenario = scenario_from_dict(data)
    result = analyze_threeway_river(
        scenario, max_joint_assignments=max_joint_assignments, max_nodes=max_nodes)
    report = {
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "implementation_sha256": {
            "engine": hashlib.sha256(Path(inspect.getfile(
                analyze_threeway_river)).read_bytes()).hexdigest(),
            "cli": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "scope": "MANUAL_THREEWAY_RIVER_RESPONSE_MODEL_STUDY_NOT_GTO",
        "range_start": "river_start",
        "table_players": scenario.rules.table_size,
        "river_start_players": 3,
        "root_players": len(set(scenario.action_order) - {
            a.actor for a in scenario.history if a.kind == "fold"})
        if result.best_action is not None else None,
        "range_assumptions": data["range_assumptions"],
        "model_assumptions": data["model_assumptions"], "models": data["models"],
        "result": encode(result),
    }
    if sensitivity:
        if result.best_action is None:
            # Invalid baseline contracts must not be repaired by swapping models;
            # they may also lack the two seats needed by the sweep loops below.
            for name in ("sensitivity", "local_sensitivity", "range_sensitivity"):
                report[name] = {
                    "status": "INCOMPLETE", "cases": [],
                    "reason": "baseline_analysis_blocked",
                    "qualification": "no_sweep_on_invalid_or_unfinished_baseline",
                }
            return report
        cases, choices = [], set()
        for names in product(STYLE_WEIGHTS, repeat=2):
            variant = deepcopy(data)
            variant["models"] = [style_model(r.seat_id, name)
                                 for r, name in zip(scenario.ranges, names)]
            value = analyze_threeway_river(
                scenario_from_dict(variant),
                max_joint_assignments=max_joint_assignments,
                max_nodes=max_nodes)
            case = {"styles": names, "models": variant["models"],
                    "result": encode(value)}
            cases.append(case)
            if value.best_action is not None:
                choices.add((value.best_action.kind, str(value.best_action.target)))
        all_complete = all(c["result"]["best_action"] is not None for c in cases)
        report["sensitivity"] = {
            "status": "INCOMPLETE" if not all_complete else (
                "MODEL_SENSITIVE" if len(choices) > 1
                else "SAME_ACTION_IN_TESTED_MODELS"),
            "tested_model_pairs": len(cases), "cases": cases,
            "range_anchor": "river_start_history_reconditioned_for_each_model_pair",
            "qualification": "untrained_illustrative_styles_not_empirical_robustness",
            "no_model_robust_policy_claim": True,
        }
        local = []
        for index, kind, factor in product(range(2), ("fold", "call", "bet", "raise"),
                                           (Decimal("0.9"), Decimal("1.1"))):
            variant = deepcopy(data)
            model = variant["models"][index]
            for table in [model["weights"], *model["category_weights"].values()]:
                for key in table:
                    if key.split(":", 1)[0] == kind:
                        table[key] = str(exact_string(table[key]) * factor)
            value = analyze_threeway_river(
                scenario_from_dict(variant),
                max_joint_assignments=max_joint_assignments,
                max_nodes=max_nodes)
            local.append({"seat_id": model["seat_id"], "kind": kind,
                          "weight_factor": str(factor), "result": encode(value)})
        baseline_action = report["result"]["best_action"]
        report["local_sensitivity"] = {
            "status": "INCOMPLETE" if baseline_action is None or any(
                c["result"]["best_action"] is None for c in local) else (
                    "MODEL_SENSITIVE" if any(c["result"]["best_action"] != (
                        baseline_action) for c in local) else
                    "SAME_ACTION_IN_TESTED_LOCAL_PERTURBATIONS"),
            "relative_weight_changes": ["-10%", "+10%"], "cases": local,
            "qualification": "finite_model_sensitivity_not_empirical_stability",
        }
        range_cases = []
        range_count = 2 * sum(len(r["combos"]) for r in data["ranges"])
        if range_count <= 64:
            for index, hypothesis in enumerate(data["ranges"]):
                for combo, weight in hypothesis["combos"].items():
                    for factor in (Decimal("0.9"), Decimal("1.1")):
                        variant = deepcopy(data)
                        variant["ranges"][index]["combos"][combo] = str(
                            exact_string(weight) * factor)
                        value = analyze_threeway_river(
                            scenario_from_dict(variant),
                            max_joint_assignments=max_joint_assignments,
                            max_nodes=max_nodes)
                        range_cases.append({
                            "seat_id": hypothesis["seat_id"], "combo": combo,
                            "weight_factor": str(factor), "result": encode(value),
                        })
        report["range_sensitivity"] = {
            "status": "INCOMPLETE" if range_count > 64 or baseline_action is None
            or any(c["result"]["best_action"] is None for c in range_cases) else (
                "MODEL_SENSITIVE" if any(c["result"]["best_action"] != baseline_action
                                         for c in range_cases)
                else "SAME_ACTION_IN_TESTED_RANGE_PERTURBATIONS"),
            "planned_cases": range_count, "max_cases": 64, "cases": range_cases,
            "reason": "range_sweep_budget_exceeded" if range_count > 64 else None,
            "qualification": "declared_combo_weights_only_not_full_range_robustness",
        }
    return report


def summary(report):
    result = report["result"]
    output = {k: result[k] for k in (
        "status", "reasons", "root_actions", "best_action",
        "best_ev", "nodes", "joint_assignments")}
    output.update({
        "scope": report["scope"], "range_start": report["range_start"],
        "table_players": report["table_players"],
        "river_start_players": report["river_start_players"],
        "root_players": report["root_players"],
        "assumptions": result["assumptions"],
        "strategy_eligible": result["strategy_eligible"],
        "advice_emitted": result["advice_emitted"],
        "model_qualification": "manual_untrained_models_not_empirical_strategy",
        "no_model_robust_policy_claim": True,
    })
    ordered = sorted((Fraction(v["ev"]["exact"]) for v in result["root_actions"]),
                     reverse=True)
    output["selected_model_best_vs_second_margin"] = encode(
        ordered[0] - ordered[1]) if len(ordered) > 1 else None
    if "sensitivity" in report:
        output["sensitivity"] = {
            "status": report["sensitivity"]["status"],
            "qualification": report["sensitivity"]["qualification"],
            "cases": [{"model_pair": [
                {"seat_id": m["seat_id"], "style": name}
                for m, name in zip(c["models"], c["styles"])],
                       "best_action": c["result"]["best_action"],
                       "best_ev": c["result"]["best_ev"]}
                      for c in report["sensitivity"]["cases"]],
        }
        output["local_sensitivity"] = {
            "status": report["local_sensitivity"]["status"],
            "cases": len(report["local_sensitivity"]["cases"]),
            "qualification": report["local_sensitivity"]["qualification"],
        }
        output["range_sensitivity"] = {
            "status": report["range_sensitivity"]["status"],
            "cases": len(report["range_sensitivity"]["cases"]),
            "qualification": report["range_sensitivity"]["qualification"],
        }
    return output


def complete_report(report):
    return (report["result"]["best_action"] is not None
            and all(report.get(key, {}).get("status") != "INCOMPLETE"
                    for key in ("sensitivity", "local_sensitivity",
                                "range_sensitivity")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--max-joint-assignments", type=int, default=128)
    parser.add_argument("--max-nodes", type=int, default=20000)
    args = parser.parse_args()
    try:
        report = analyze_file(args.input, sensitivity=args.sensitivity,
                              max_joint_assignments=args.max_joint_assignments,
                              max_nodes=args.max_nodes)
    except (OSError, TypeError, ValueError, ArithmeticError) as exc:
        parser.exit(2, f"threeway input rejected: {exc}\n")
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps(summary(report), ensure_ascii=False, indent=2))
    return 0 if complete_report(report) else 2


if __name__ == "__main__":
    raise SystemExit(main())
