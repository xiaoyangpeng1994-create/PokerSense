"""Differential checks for the offline PokerKit terminal reference."""

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
import json
from pathlib import Path

import pytest

from poker_engine.core.enums import PlayerStatus, Position
from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.aa_rules_v2 import AARuleProfileV2
from poker_engine.strategy.contracts import DecisionSeat
from tools.analyze_terminal_multiway import scenario_from_dict
from tools import strategy_eval_independent_oracle_v1 as oracle


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "configs/strategy/examples/terminal-multiway-river-manual.json"


def scenario():
    return scenario_from_dict(json.loads(EXAMPLE.read_text(encoding="utf-8")))


def require_reference():
    pytest.importorskip("pokerkit")


def test_existing_six_player_sidepots_and_configured_rake_match():
    require_reference()
    result = oracle.compare_terminal(scenario())
    assert result["status"] == "MATCH"
    assert result["pokerkit_version"] == "0.7.5"
    assert result["reference"]["call_net_ev"] == 256
    assert [p[0] for p in result["reference"]["call_pots"]] == [110, 130, 80]
    assert result["reference"]["rake_by_pot"] == (
        Fraction(11, 8), Fraction(13, 8), Fraction(1))


def test_short_hero_call_refund_and_outer_pot_match():
    require_reference()
    base = scenario()
    hero = replace(base.seats[0], stack=ChipAmount("20"))
    result = oracle.compare_terminal(replace(base, seats=(hero, *base.seats[1:])))
    assert result["status"] == "MATCH"
    assert result["reference"]["call_refunds"] == ((1, Fraction(40)),)
    assert result["reference"]["call_cost"] == 20


@pytest.mark.parametrize("rounding,distribution,charge", (
    ("floor_to_chip", "main_pot_first", 4),
    ("ceil_to_chip", "proportional_all_pots", 5),
))
def test_rounding_and_rake_distribution_match(rounding, distribution, charge):
    require_reference()
    base = scenario()
    data = base.rules.to_dict()
    data.update(rake_percent="0.013", rake_cap_bb="100",
                rake_rounding=rounding, rake_distribution=distribution)
    result = oracle.compare_terminal(replace(
        base, rules=AARuleProfileV2.from_dict(data)))
    assert result["status"] == "MATCH"
    assert sum(result["reference"]["rake_by_pot"]) == charge


def test_pokerkit_showdown_detects_zero_hero_equity():
    require_reference()
    from tools.analyze_terminal_multiway import card
    base = scenario()
    result = oracle.compare_terminal(replace(
        base, hero_cards=(card("8d"), card("6d"))))
    assert result["status"] == "MATCH"
    assert result["reference"]["hero_share_by_pot"] == (0, 0, 0)
    assert result["reference"]["call_net_ev"] == -60


def test_weighted_card_hypotheses_and_showdown_tie_match():
    require_reference()
    base = scenario()
    first = replace(base.ranges[0], combo_weights={
        "KhKd": Decimal(1), "JhJd": Decimal(2),
        "AsKc": Decimal(50),  # blocked by Hero's As
    })
    weighted = oracle.compare_terminal(replace(
        base, ranges=(first, *base.ranges[1:])))
    assert weighted["status"] == "MATCH"
    assert weighted["reference"]["joint_assignments"] == 2
    assert 0 < weighted["reference"]["hero_share_by_pot"][0] < 1

    # A shared royal flush makes every surviving player tie on each eligible pot.
    from tools.analyze_terminal_multiway import card
    royal = replace(base, hero_cards=(card("4s"), card("5s")),
                    board_cards=tuple(map(card, (
                        "As", "Ks", "Qs", "Js", "Ts"))))
    tied = oracle.compare_terminal(royal)
    assert tied["status"] == "MATCH"
    assert tied["reference"]["hero_share_by_pot"] == (
        Fraction(1, 4), Fraction(1, 3), Fraction(1, 2))


@pytest.mark.parametrize("table_size", (7, 8))
def test_dealt_table_size_is_distinct_from_pot_eligibility(table_size):
    require_reference()
    base = scenario()
    seat_rows = list(base.seats)
    for number in range(6, table_size):
        seat_rows.append(DecisionSeat(
            number, f"folded-{number}", Position.UNKNOWN, ChipAmount("50"),
            ChipAmount("0"), ChipAmount("0"), PlayerStatus.FOLDED))
    rule_data = base.rules.to_dict()
    rule_data["table_size"] = table_size
    compared = oracle.compare_terminal(replace(
        base, seats=tuple(seat_rows),
        rules=AARuleProfileV2.from_dict(rule_data)))
    assert compared["status"] == "MATCH"
    assert compared["reference"]["hero_share_by_pot"] == (1, 1, 1)


def test_comparison_detects_injected_baseline_settlement_error(monkeypatch):
    require_reference()
    original = oracle.analyze_terminal_multiway

    def wrong(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, call_net_ev=result.call_net_ev + 1)

    monkeypatch.setattr(oracle, "analyze_terminal_multiway", wrong)
    result = oracle.compare_terminal(scenario())
    assert result["status"] == "MISMATCH"
    assert result["mismatches"] == ("call_net_ev",)


def test_missing_reference_is_explicit(monkeypatch):
    def unavailable():
        raise RuntimeError("pokerkit_0_7_5_not_installed")

    monkeypatch.setattr(oracle, "_pokerkit_hand", unavailable)
    assert oracle.compare_terminal(scenario())["status"] == "REFERENCE_UNAVAILABLE"


def test_unknown_rake_blocks_before_oracle_is_consulted(monkeypatch):
    def forbidden():
        raise AssertionError("reference should not receive blocked baseline")

    monkeypatch.setattr(oracle, "_pokerkit_hand", forbidden)
    base = scenario()
    data = base.rules.to_dict()
    data["rake_distribution"] = "unverified"
    result = oracle.compare_terminal(replace(
        base, rules=AARuleProfileV2.from_dict(data)))
    assert result["status"] == "BASELINE_BLOCKED"
    assert "rake_policy_unknown_or_invalid" in result["reasons"]
