from dataclasses import replace
from decimal import Decimal

import pytest

from poker_engine.core.enums import ActionType
from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.aa_rule_router_v2 import (
    AARuleBoundShadowRouter,
    AAShadowRouteStatus,
)
from poker_engine.strategy.aa_asset_binding_v2 import AAStrategyAssetBindingV2
from poker_engine.strategy.asset_provider import provider_capability_digest
from poker_engine.strategy.aa_rules_v2 import (
    AARuleProfileV2,
    build_forced_bet_plan,
    reconcile_opening_debits,
)
from poker_engine.strategy.contracts import LegalAction
from poker_engine.strategy.provider import FakeProvider
from poker_engine.strategy.router import StrategyRouter

from .helpers import NOW, candidate, capability, context, hit_result


def rules(*, verification="live_verified"):
    return AARuleProfileV2.from_dict({
        "schema_version": 2, "table_size": 8,
        "small_blind": "0.5", "big_blind": "1", "ante": "0",
        "ante_mode": "none", "straddle_mode": "mandatory_utg",
        "straddle_amount": "2", "rake_percent": "0",
        "rake_cap_bb": "0", "rake_application": "all_pots",
        "rake_rounding": "exact", "minimum_chip": "0.5",
        "rake_distribution": "proportional_all_pots",
        "verification_status": verification, "source": "test",
    })


def inputs(*, verification="live_verified"):
    profile = rules(verification=verification)
    plan = build_forced_bet_plan(profile, tuple(range(8)), dealer_seat=5)
    observed = {str(seat): str(plan.contributions[seat]) for seat in range(8)}
    opening = reconcile_opening_debits(profile, plan, observed)
    ctx = context(8, action_line="unopened")
    ctx = replace(
        ctx,
        game_config=profile.game_config(),
        legal_actions=(
            LegalAction(ActionType.FOLD, ChipAmount("0"), ChipAmount("0")),
            LegalAction(ActionType.CALL, ChipAmount("1"), ChipAmount("1")),
            LegalAction(ActionType.RAISE, ChipAmount("4"), ChipAmount("100")),
        ),
    )
    provider = FakeProvider(
        "aa-rule-bound-test", "v1",
        capability((8,), hero_positions=(ctx.seats[ctx.hero_seat].position,)),
        lambda value: hit_result(candidate(
            value, provider_id="aa-rule-bound-test", provider_version="v1",
            probabilities={
                ActionType.FOLD: Decimal("0.2"),
                ActionType.CALL: Decimal("0.3"),
                ActionType.RAISE: Decimal("0.5"),
            },
        )),
    )
    bound = AARuleBoundShadowRouter(
        StrategyRouter((provider,)), rule_fingerprint=profile.fingerprint,
        asset_binding_id="synthetic-test-only",
    )
    return profile, plan, opening, ctx, bound


class BoundProvider:
    def __init__(self, base):
        self.base = base
        self.provider_id = base.provider_id
        self.source_version = base.source_version
        self.capability = base.capability
        self.asset_sha256 = "a" * 64

    def query(self, context):
        return self.base.query(context)


def test_exact_rule_bound_provider_can_run_only_as_shadow_candidate():
    profile, plan, opening, ctx, bound = inputs()
    result = bound.evaluate(ctx, profile, plan, opening, now=NOW)
    assert result.status is AAShadowRouteStatus.SHADOW_CANDIDATE
    assert result.route.selected.provider_id == "aa-rule-bound-test"
    assert not result.advice_emitted
    assert not result.strategy_eligible


def test_simulation_rules_block_provider_even_when_forced_bets_match():
    profile, plan, opening, ctx, _ = inputs(verification="simulation")
    bound = AARuleBoundShadowRouter(
        StrategyRouter(), rule_fingerprint=profile.fingerprint,
        asset_binding_id="none",
    )
    result = bound.evaluate(ctx, profile, plan, opening, now=NOW)
    assert result.status is AAShadowRouteStatus.BLOCKED
    assert "aa_rules_not_live_verified" in result.reasons


def test_rule_fingerprint_mismatch_blocks_before_provider():
    profile, plan, opening, ctx, _ = inputs()
    bound = AARuleBoundShadowRouter(
        StrategyRouter(), rule_fingerprint="a" * 64,
        asset_binding_id="wrong",
    )
    result = bound.evaluate(ctx, profile, plan, opening, now=NOW)
    assert "strategy_asset_rule_fingerprint_mismatch" in result.reasons
    assert result.route is None


def test_unreconciled_opening_cannot_route():
    profile, plan, _, ctx, bound = inputs()
    observed = {str(seat): str(plan.contributions[seat]) for seat in range(8)}
    observed["7"] = "99"
    opening = reconcile_opening_debits(profile, plan, observed)
    result = bound.evaluate(ctx, profile, plan, opening, now=NOW)
    assert "opening_forced_bets_not_exact_or_verified" in result.reasons


def test_wrong_straddle_raise_floor_and_later_action_line_block():
    profile, plan, opening, ctx, bound = inputs()
    wrong = replace(ctx, legal_actions=tuple(
        replace(action, min_amount=ChipAmount("3"))
        if action.action is ActionType.RAISE else action
        for action in ctx.legal_actions
    ))
    assert "straddle_legal_sizing_not_bound" in bound.evaluate(
        wrong, profile, plan, opening, now=NOW
    ).reasons
    later = replace(ctx, action_line="raise")
    assert "post_opening_raise_increment_unverified" in bound.evaluate(
        later, profile, plan, opening, now=NOW
    ).reasons


def test_reviewed_binding_constructs_router_and_test_only_cannot():
    profile, plan, opening, ctx, original = inputs()
    provider = BoundProvider(original.router.providers[0])
    body = {
        "schema_version": 2,
        "asset_sha256": provider.asset_sha256,
        "capability_sha256": provider_capability_digest(provider.capability),
        "rule_fingerprint": profile.fingerprint,
        "provider_id": provider.provider_id,
        "source_version": provider.source_version,
        "player_counts": [8],
        "streets": ["preflop"],
        "asset_status": "shadow_reviewed",
        "source_url": "https://example.invalid",
        "source_revision": "test",
        "license_spdx": "LicenseRef-Test",
        "limitations": ["synthetic only"],
    }
    binding = AAStrategyAssetBindingV2.from_dict(body)
    bound = AARuleBoundShadowRouter.from_binding(provider, binding)
    assert bound.evaluate(
        ctx, profile, plan, opening, now=NOW
    ).status is AAShadowRouteStatus.SHADOW_CANDIDATE
    body["asset_status"] = "test_only"
    with pytest.raises(ValueError, match="not shadow reviewed"):
        AARuleBoundShadowRouter.from_binding(
            provider, AAStrategyAssetBindingV2.from_dict(body)
        )
