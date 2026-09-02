"""Contract and coverage checks for generated strategy mock fixtures."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from poker_engine.core.enums import (
    ActionType,
    PlayerStatus,
    Position,
    Rank,
    Street,
    Suit,
)
from poker_engine.core.observation import (
    ObservationField,
    RawObservation,
    SlotObservation,
    ValidationStatus,
)
from poker_engine.core.opponents import PlayerState
from poker_engine.core.state import PokerState
from poker_engine.core.value_objects import Card, ChipAmount
from poker_engine.equity import EnumerationEquity
from poker_engine.strategy.contracts import QualityStatus
from poker_engine.strategy.input_provenance import (
    SuppliedInput,
    collect_input_provenance,
)
from poker_engine.state_engine.action_reconstruction import (
    reconstruct_action_event,
)
from poker_engine.state_engine.platform_mapping import (
    PlatformSeatMapping,
    map_action_candidate,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "tests" / "fixtures" / "strategy" / "v1"
GENERATOR = ROOT / "tools" / "generate_strategy_mock_fixtures.py"

spec = importlib.util.spec_from_file_location(
    "generate_strategy_mock_fixtures", GENERATOR,
)
generator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = generator
spec.loader.exec_module(generator)


def _load_fixtures():
    with (DATA_DIR / "fixtures.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture(scope="module")
def fixtures():
    return _load_fixtures()


def _card(value):
    return Card(Rank(value[0]), Suit(value[1]))


def test_generated_artifacts_are_current():
    for path, expected in generator.build_artifacts().items():
        assert path.exists(), path
        assert path.read_bytes() == expected, path


def test_manifest_count_hash_and_exclusions(fixtures):
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    fixture_bytes = (DATA_DIR / "fixtures.jsonl").read_bytes()
    assert manifest["fixture_count"] == len(fixtures)
    assert len(fixtures) >= 200
    assert manifest["fixture_sha256"] == hashlib.sha256(
        fixture_bytes
    ).hexdigest()
    assert set(manifest["not_release_acceptance_evidence"]) == {
        "real_provider_golden_parity",
        "real_platform_capture_replay",
        "hardware_specific_performance_result",
        "clean_install_platform_acceptance",
    }


def test_schema_and_semantic_validation(fixtures):
    schema = json.loads((DATA_DIR / "schema.json").read_text())
    assert schema["properties"]["fixture_schema_version"]["const"] == "1.0.0"
    generator.validate_dataset(fixtures)


def test_all_product_requirements_have_fixture_coverage(fixtures):
    covered = {req for fixture in fixtures for req in fixture["requirements"]}
    assert covered == generator.REQUIREMENTS


def test_all_regression_e2e_ids_have_fixture_coverage(fixtures):
    covered = {test for fixture in fixtures for test in fixture["test_ids"]}
    assert {f"E2E-{number:03d}" for number in range(1, 13)} <= covered


def test_every_documented_test_id_has_fixture_coverage(fixtures):
    pattern = re.compile(
        r"\b(?:T-[A-Z]+-\d{3}|E2E-\d{3}|P-\d{3}|PERF-[A-Z0-9-]+)\b"
    )
    documented = set()
    for name in (
        "strategy-requirements-matrix.md",
        "strategy-regression-test-matrix.md",
    ):
        documented.update(pattern.findall((ROOT / "docs" / name).read_text()))
    covered = {test for fixture in fixtures for test in fixture["test_ids"]}
    assert documented <= covered, sorted(documented - covered)


@pytest.mark.parametrize("player_count", range(2, 10))
@pytest.mark.parametrize("action_line", generator.ACTION_LINES)
def test_every_preflop_player_count_and_action_line_exists(
    fixtures, player_count, action_line,
):
    fixture_id = (
        f"MOCK-PF-{player_count}P-"
        f"{generator._slug(action_line)}"
    )
    fixture = next(item for item in fixtures if item["fixture_id"] == fixture_id)
    assert fixture["input"]["state"]["dealt_player_count"] == player_count
    assert fixture["input"]["state"]["action_line"] == action_line
    assert fixture["expected"]["advice"]["status"] == "READY"


def test_positive_preflop_fixtures_balance_committed_chips(fixtures):
    positives = [
        item for item in fixtures
        if "positive" in item["tags"] and item["input"]["state"]["street"]
        == "preflop"
    ]
    assert len(positives) == 72
    for fixture in positives:
        state = fixture["input"]["state"]
        committed = sum(
            (Decimal(seat["street_committed"]) for seat in state["seats"]),
            Decimal("0"),
        )
        assert committed == Decimal(state["pots"][0]["amount"])
        legal = {action["action"] for action in state["legal_actions"]}
        advised = set(fixture["expected"]["advice"]["action_probabilities"])
        assert advised <= legal


@pytest.mark.parametrize("player_count", range(3, 10))
def test_hu_provider_mismatch_exists_for_every_multiplayer_count(
    fixtures, player_count,
):
    fixture_id = f"MOCK-PF-{player_count}P-HU-PROVIDER-ONLY"
    fixture = next(item for item in fixtures if item["fixture_id"] == fixture_id)
    assert fixture["expected"]["advice"]["status"] == "ABSTAIN"
    assert fixture["expected"]["advice"]["reason_codes"] == [
        "unsupported_player_count"
    ]
    assert fixture["expected"]["provider_lookups"][0]["state"] == (
        "NOT_APPLICABLE"
    )


@pytest.mark.parametrize("street", ("flop", "turn", "river"))
@pytest.mark.parametrize("active_count", range(2, 10))
def test_every_postflop_active_count_exists(fixtures, street, active_count):
    fixture_id = f"MOCK-{street.upper()}-{active_count}WAY"
    fixture = next(item for item in fixtures if item["fixture_id"] == fixture_id)
    state = fixture["input"]["state"]
    assert state["street"] == street
    assert state["active_player_count"] == active_count


@pytest.mark.parametrize("field", generator.QUALITY_FIELDS)
@pytest.mark.parametrize(
    "status", ("UNKNOWN", "LOW_CONFIDENCE", "CONFLICT"),
)
def test_every_critical_field_has_quality_failure(fixtures, field, status):
    fixture_id = f"MOCK-QUALITY-{generator._slug(field)}-{status}"
    fixture = next(item for item in fixtures if item["fixture_id"] == fixture_id)
    quality = fixture["input"]["observations"]["field_quality"][field]
    assert quality["status"] == status
    assert fixture["expected"]["advice"]["status"] == "ABSTAIN"
    assert fixture["expected"]["advice"]["action_probabilities"] == {}


def test_all_provenance_mock_cases_execute_against_collector(fixtures):
    cases = [item for item in fixtures if "input-provenance" in item["tags"]]
    assert len(cases) == 8
    now = datetime(2026, 8, 22, tzinfo=timezone.utc)
    for fixture in cases:
        channels = {
            "observations": {},
            "manual_inputs": {},
            "config_inputs": {},
            "derived_inputs": {},
            "inferred_inputs": {},
        }
        for field_name, candidates in fixture["input"][
            "provenance_candidates"
        ].items():
            for candidate in candidates:
                source = candidate["source"]
                if source == "vision":
                    channels["observations"][field_name] = ObservationField(
                        candidate["value"],
                        float(candidate["confidence"]),
                        "mock-vision-adapter",
                        {"evidence_ref": candidate["evidence_ref"]},
                        now,
                        ValidationStatus(candidate["status"].lower()),
                    )
                    continue
                channel = {
                    "manual": "manual_inputs",
                    "config": "config_inputs",
                    "derived": "derived_inputs",
                    "inferred": "inferred_inputs",
                }[source]
                channels[channel][field_name] = SuppliedInput(
                    candidate["value"],
                    float(candidate["confidence"]),
                    QualityStatus(candidate["status"]),
                    candidate["evidence_ref"],
                    now,
                )
        actual = collect_input_provenance(**channels).provenance
        expected = fixture["expected"]["resolved_provenance"]
        assert len(actual) == len(expected)
        assert [item.field_name for item in actual] == [
            item["field_name"] for item in expected
        ]
        assert [item.source.name for item in actual] == [
            item["source"] for item in expected
        ]
        assert [item.status.value for item in actual] == [
            item["status"] for item in expected
        ]


def test_all_action_reconstruction_mock_cases_execute(fixtures):
    cases = [
        item for item in fixtures if "action-reconstruction" in item["tags"]
    ]
    assert len(cases) == 8
    now = datetime(2026, 8, 22, tzinfo=timezone.utc)

    def make_player(seat, data, *, hero=False):
        status = PlayerStatus(data["status"])
        committed = data["street_committed"]
        return PlayerState(
            "hero" if hero else f"p{seat}",
            seat,
            Position.BB if hero else Position.BTN,
            ChipAmount(data["stack"]),
            ChipAmount(committed),
            ChipAmount(committed),
            status,
            status is not PlayerStatus.FOLDED,
            hero,
            not hero,
        )

    for fixture in cases:
        transition = fixture["input"]["action_transition"]
        states = []
        for version, key in ((1, "before"), (2, "after")):
            data = transition[key]
            actor = make_player(0, data)
            villain = make_player(1, {
                "stack": "100",
                "street_committed": data["villain_committed"],
                "status": "active",
            }, hero=True)
            states.append(PokerState(
                version,
                fixture["fixture_id"],
                Street.PREFLOP,
                (_card("As"), _card("Kd")),
                (),
                (actor, villain),
                ChipAmount(data["pot"]),
                ChipAmount(data["current_bet"]),
                ChipAmount("0"),
                0,
            ))
        label = transition["observed_action"]
        actual = reconstruct_action_event(
            states[0],
            states[1],
            actor_seat=transition["actor_seat"],
            observed_action=ActionType(label) if label else None,
            timestamp=now,
        )
        expected = fixture["expected"]["action_reconstruction"]
        assert actual.status.value == expected["status"]
        assert [item.value for item in actual.candidates] == expected[
            "candidates"
        ]
        assert (actual.event is not None) is expected["event_required"]
        if expected["reason"]:
            assert expected["reason"] in actual.reasons


def test_all_platform_mapping_mock_replays_execute(fixtures):
    cases = [item for item in fixtures if "platform-mapping" in item["tags"]]
    assert len(cases) == 23
    now = datetime(2026, 8, 23, tzinfo=timezone.utc)

    def obs_field(value=None, status=ValidationStatus.UNKNOWN):
        return ObservationField(
            value, 1.0, "synthetic-replay", {}, now, status
        )

    def make_player(
        seat, *, stack="100", committed="0",
        status=PlayerStatus.ACTIVE, hero=False,
    ):
        return PlayerState(
            "hero" if hero else f"p{seat}",
            seat,
            (Position.BTN, Position.SB, Position.BB)[seat],
            ChipAmount(stack),
            ChipAmount(committed),
            ChipAmount(committed),
            status,
            status is not PlayerStatus.FOLDED,
            hero,
            seat == 0,
        )

    for fixture in cases:
        case = fixture["input"]["platform_mapping_case"]
        profile = case["before_profile"]
        actor = make_player(0, stack="40" if profile == "short_all_in" else "100")
        villain = make_player(
            1,
            stack="0" if profile == "short_all_in" else (
                "90" if profile == "facing10" else "100"
            ),
            committed="100" if profile == "short_all_in" else (
                "10" if profile == "facing10" else "0"
            ),
            status=(
                PlayerStatus.ALL_IN
                if profile == "short_all_in" else PlayerStatus.ACTIVE
            ),
        )
        pot = "100" if profile == "short_all_in" else (
            "10" if profile == "facing10" else "0"
        )
        current_bet = pot
        previous = PokerState(
            1,
            fixture["fixture_id"],
            Street.PREFLOP,
            (_card("As"), _card("Kd")),
            (),
            (actor, villain, make_player(2, hero=True)),
            ChipAmount(pot),
            ChipAmount(current_bet),
            ChipAmount("0"),
            2,
        )
        mapping_data = case["mapping"]
        mapping = PlatformSeatMapping(
            mapping_data["platform_id"],
            mapping_data["layout_id"],
            mapping_data["version"],
            {int(k): v for k, v in mapping_data["stack_slot_to_seat"].items()},
            {int(k): v for k, v in mapping_data["action_slot_to_seat"].items()},
            {int(k): v for k, v in mapping_data["actor_slot_to_seat"].items()},
            {int(k): v for k, v in mapping_data["dealer_slot_to_seat"].items()},
        )
        observed = case["observation"]
        action = observed["action"]
        board = observed.get("board")
        street = observed.get("street")
        observation = RawObservation(
            101,
            now,
            obs_field((_card("As"), _card("Kd")), ValidationStatus.VALID),
            obs_field(
                tuple(_card(value) for value in board),
                ValidationStatus.VALID,
            ) if board is not None else obs_field(),
            obs_field(
                ChipAmount(observed["pot"]), ValidationStatus.VALID
            ) if observed["pot"] is not None else obs_field(),
            obs_field(),
            obs_field(),
            obs_field(
                ActionType(action), ValidationStatus.VALID
            ) if action is not None else obs_field(),
            obs_field(
                Street(street), ValidationStatus.VALID
            ) if street is not None else obs_field(),
            obs_field(
                observed["dealer_slot"], ValidationStatus.VALID
            ) if observed["dealer_slot"] is not None else obs_field(),
            obs_field(
                observed["actor_slot"], ValidationStatus.VALID
            ) if observed["actor_slot"] is not None else obs_field(),
            1.0,
            tuple(
                SlotObservation(
                    slot, obs_field(ChipAmount(value), ValidationStatus.VALID)
                )
                for slot, value in observed["stack_slots"]
            ),
            tuple(
                SlotObservation(
                    slot, obs_field(ActionType(value), ValidationStatus.VALID)
                )
                for slot, value in observed["action_slots"]
            ),
        )
        actual = map_action_candidate(previous, observation, mapping)
        expected = fixture["expected"]["platform_mapping"]
        assert actual.status.value == expected["status"]
        assert (actual.state is not None) is expected["candidate_required"]
        assert (actual.event is not None) is expected["candidate_required"]
        if expected["reason"]:
            assert actual.reasons == (expected["reason"],)
        if expected["event"]:
            assert actual.event.event_type.value == expected["event"]


def test_all_advice_states_and_lookup_states_are_represented(fixtures):
    advice = {item["expected"]["advice"]["status"] for item in fixtures}
    lookups = {
        lookup["state"]
        for item in fixtures
        for lookup in item["expected"]["provider_lookups"]
    }
    assert advice == generator.ADVICE_STATUSES
    assert {"HIT_EXACT", "HIT_APPROXIMATE", "NOT_FOUND",
            "NOT_APPLICABLE", "REJECTED"} <= lookups


def test_negative_fault_catalog_has_required_reasons(fixtures):
    reasons = {
        reason
        for item in fixtures if "fault-injection" in item["tags"]
        for reason in item["expected"]["advice"]["reason_codes"]
    }
    assert {
        "duplicate_card", "card_overlap", "invalid_board_count",
        "negative_stack", "chip_non_conservation", "invalid_actor",
        "illegal_action", "illegal_raise_size", "ambiguous_action_history",
        "missing_position", "stale_hand_id", "stale_state_version",
        "stale_request_id", "expired_advice", "asset_integrity_error",
        "provider_schema_error", "resolver_timeout", "resolver_crash",
        "invalid_probability", "illegal_provider_action",
        "range_card_collision", "empty_range", "window_identity_mismatch",
        "monotonic_deadline_required",
    } <= reasons


def test_fault_fixtures_contain_real_malformed_or_mismatch_payloads(fixtures):
    by_id = {item["fixture_id"]: item for item in fixtures}
    assert by_id["MOCK-FAULT-DUPLICATE-HERO-CARD"]["input"]["state"][
        "hero_cards"
    ] == ["As", "As"]
    assert by_id["MOCK-FAULT-NEGATIVE-STACK"]["input"]["state"]["seats"][0][
        "stack"
    ] == "-1"
    assert "player_counts" not in by_id[
        "MOCK-FAULT-BAD-PROVIDER-SCHEMA"
    ]["input"]["providers"][0]["capability"]
    assert by_id["MOCK-FAULT-INVALID-FREQUENCY"]["input"]["providers"][0][
        "mock_result"
    ]["action_probabilities"]["check"] == "-0.1"
    assert by_id["MOCK-FAULT-RANGE-CARD-COLLISION"]["input"]["ranges"][
        "villains"
    ][0]["combo_weights"] == {"AsQc": "1"}


def test_side_pot_fixtures_account_for_uncalled_returns(fixtures):
    side_pots = [item for item in fixtures if "side-pot" in item["tags"]]
    assert len(side_pots) == 4
    for fixture in side_pots:
        commitments = sum(
            (Decimal(seat["hand_committed"])
             for seat in fixture["input"]["state"]["seats"]),
            Decimal("0"),
        )
        distributed = sum(
            (Decimal(value) for value in fixture["expected"]["pot_amounts"]),
            Decimal("0"),
        ) + Decimal(fixture["expected"]["uncalled_return"])
        assert distributed == commitments


def test_exact_equity_anchor_fixtures_match_current_engine(fixtures):
    anchors = [
        item for item in fixtures
        if "known-cards" in item["tags"] and "math" in item["expected"]
    ]
    assert len(anchors) == 4
    for fixture in anchors:
        state = fixture["input"]["state"]
        result = EnumerationEquity().estimate(
            tuple(_card(card) for card in state["hero_cards"]),
            tuple(
                tuple(_card(card) for card in holding)
                for holding in fixture["input"]["known_opponent_cards"]
            ),
            tuple(_card(card) for card in state["board_cards"]),
        )
        expected = fixture["expected"]["math"]
        assert result.win == pytest.approx(float(expected["win"]), abs=1e-12)
        assert result.tie == pytest.approx(float(expected["tie"]), abs=1e-12)
        assert result.loss == pytest.approx(float(expected["loss"]), abs=1e-12)
        assert result.equity == pytest.approx(
            float(expected["equity"]), abs=1e-12,
        )


def test_generated_money_values_are_decimal_strings(fixtures):
    for fixture in fixtures:
        config = fixture["input"]["game_config"]
        assert all(
            isinstance(config[key], str)
            for key in (
                "small_blind", "big_blind", "ante", "rake_percent",
                "rake_cap", "minimum_chip",
            )
        )
        for seat in fixture["input"]["state"]["seats"]:
            assert all(
                isinstance(seat[key], str)
                for key in (
                    "starting_stack", "stack", "street_committed",
                    "hand_committed",
                )
            )
