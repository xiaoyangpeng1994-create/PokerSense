from dataclasses import replace
from decimal import Decimal

from poker_engine.core.enums import Street
from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.aa_equity_shadow_v2 import (
    AAEquityShadowStatus,
    evaluate_aa_equity_shadow,
)
from poker_engine.strategy.aa_rules_v2 import (
    AARuleProfileV2,
    build_forced_bet_plan,
    reconcile_opening_debits,
)
from poker_engine.strategy.contracts import PotState, RangeDistribution
from poker_engine.strategy.aa_range_assets_v2 import (
    AARangeShadowSnapshotV2,
    assess_aa_range_readiness,
)

from .helpers import NOW, context


def rules(*, verification="live_verified", distribution="proportional_all_pots",
          application="all_pots", rounding="exact"):
    return AARuleProfileV2.from_dict({
        "schema_version": 2, "table_size": 6,
        "small_blind": "0.5", "big_blind": "1", "ante": "0",
        "ante_mode": "none", "straddle_mode": "mandatory_utg",
        "straddle_amount": "2", "rake_percent": "0.10",
        "rake_cap_bb": "100", "rake_application": application,
        "rake_rounding": rounding, "rake_distribution": distribution,
        "minimum_chip": "0.5", "verification_status": verification,
        "source": "test",
    })


def inputs(*, rule=None):
    profile = rule or rules()
    plan = build_forced_bet_plan(profile, tuple(range(6)), dealer_seat=3)
    observed = {str(seat): str(plan.contributions[seat]) for seat in range(8)}
    opening = reconcile_opening_debits(profile, plan, observed)
    base = context(6, street=Street.RIVER, active_count=3)
    ranges = (
        RangeDistribution(
            3, {"QcQd": Decimal("1")}, "test",
            f"aa-ranges-v2:{profile.fingerprint}:test", confidence=1.0,
        ),
        RangeDistribution(
            4, {"JcJd": Decimal("1")}, "test",
            f"aa-ranges-v2:{profile.fingerprint}:test", confidence=1.0,
        ),
    )
    pots = (
        PotState("main", ChipAmount("100"), (3, 4, 5)),
        PotState("side-1", ChipAmount("50"), (4, 5)),
    )
    ctx = replace(
        base, game_config=profile.game_config(), villain_ranges=ranges, pots=pots,
    )
    return profile, plan, opening, ctx


def test_live_verified_exact_river_reports_gross_and_proportional_net():
    profile, plan, opening, ctx = inputs()
    result = evaluate_aa_equity_shadow(
        ctx, profile, plan, opening, now=NOW, allow_untracked_ranges=True
    )
    assert result.status is AAEquityShadowStatus.COMPLETE
    assert result.equity_report.method.value == "exact"
    assert result.rake.amount == Decimal("15.00")
    assert result.rake_allocations == {
        "main": Decimal("10.00"), "side-1": Decimal("5.00"),
    }
    assert result.configured_net_expected_chips == (
        result.gross_expected_chips * Decimal("0.9")
    )
    assert not result.advice_emitted and not result.strategy_eligible


def test_main_pot_first_uses_each_pot_share_instead_of_global_scaling():
    profile, plan, opening, ctx = inputs(rule=rules(distribution="main_pot_first"))
    result = evaluate_aa_equity_shadow(
        ctx, profile, plan, opening, now=NOW, allow_untracked_ranges=True
    )
    assert result.rake_allocations == {
        "main": Decimal("15.00"), "side-1": Decimal("0"),
    }
    shares = {
        pot.pot_id: pot.expected_share for pot in result.equity_report.result.pots
    }
    expected = Decimal("85") * shares["main"] + Decimal("50") * shares["side-1"]
    assert result.configured_net_expected_chips == expected


def test_simulation_requires_explicit_opt_in_and_stays_simulation():
    profile, plan, opening, ctx = inputs(rule=rules(verification="simulation"))
    blocked = evaluate_aa_equity_shadow(
        ctx, profile, plan, opening, now=NOW, allow_untracked_ranges=True
    )
    assert blocked.status is AAEquityShadowStatus.BLOCKED
    assert "simulation_rules_not_explicitly_enabled" in blocked.reasons
    result = evaluate_aa_equity_shadow(
        ctx, profile, plan, opening, now=NOW, allow_simulation=True,
        allow_untracked_ranges=True,
    )
    assert result.status is AAEquityShadowStatus.SIMULATION
    assert result.reasons == (
        "simulation_rules_only", "untracked_ranges_explicit_test_only",
    )
    assert not result.rake.strategy_eligible


def test_missing_ranges_and_unreconciled_opening_block_before_math():
    profile, plan, opening, ctx = inputs()
    missing = replace(ctx, villain_ranges=())
    result = evaluate_aa_equity_shadow(
        missing, profile, plan, opening, now=NOW,
        allow_untracked_ranges=True,
    )
    assert result.status is AAEquityShadowStatus.BLOCKED
    assert result.equity_report is None
    assert "villain_ranges_missing" in result.reasons
    observed = {str(seat): str(plan.contributions[seat]) for seat in range(8)}
    observed["0"] = "99"
    bad = reconcile_opening_debits(profile, plan, observed)
    result = evaluate_aa_equity_shadow(
        ctx, profile, plan, bad, now=NOW, allow_untracked_ranges=True
    )
    assert "opening_forced_bets_not_exact" in result.reasons


def test_unknown_rake_distribution_blocks_net_equity():
    profile = rules(
        verification="simulation", distribution="unverified",
        application="unverified", rounding="unverified",
    )
    profile, plan, opening, ctx = inputs(rule=profile)
    result = evaluate_aa_equity_shadow(
        ctx, profile, plan, opening, now=NOW, allow_simulation=True,
        allow_untracked_ranges=True,
    )
    assert result.status is AAEquityShadowStatus.BLOCKED
    assert result.rake.amount is None
    assert result.equity_report is None


def test_expired_request_becomes_computation_blocker_not_exception():
    profile, plan, opening, ctx = inputs()
    result = evaluate_aa_equity_shadow(
        ctx, profile, plan, opening, now=ctx.request.expires_at,
        allow_untracked_ranges=True,
    )
    assert result.status is AAEquityShadowStatus.BLOCKED
    assert result.reasons == (
        "equity_input_or_computation_error:equity_deadline_expired",
    )


def test_range_tracker_snapshot_is_required_by_default_and_identity_bound():
    profile, plan, opening, ctx = inputs()
    blocked = evaluate_aa_equity_shadow(
        ctx, profile, plan, opening, now=NOW
    )
    assert blocked.status is AAEquityShadowStatus.BLOCKED
    assert "range_tracker_snapshot_missing" in blocked.reasons
    readiness = assess_aa_range_readiness(
        ctx.villain_ranges, hero_seat=ctx.hero_seat, pots=ctx.pots,
        rule_fingerprint=profile.fingerprint,
    )
    snapshot = AARangeShadowSnapshotV2(
        1, ctx.villain_ranges, (), (), readiness, True,
    )
    result = evaluate_aa_equity_shadow(
        ctx, profile, plan, opening, now=NOW, range_snapshot=snapshot
    )
    assert result.status is AAEquityShadowStatus.COMPLETE
    assert result.reasons == ()
    mismatch = replace(snapshot, distributions=tuple(reversed(
        snapshot.distributions
    )))
    blocked = evaluate_aa_equity_shadow(
        ctx, profile, plan, opening, now=NOW, range_snapshot=mismatch
    )
    assert "range_tracker_context_mismatch" in blocked.reasons
