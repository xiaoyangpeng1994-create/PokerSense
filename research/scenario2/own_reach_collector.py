"""Exact own-reach increments for one frozen full-tree policy snapshot.

This is an independent statistic, not a trainer, policy, retrospective average,
or replacement for the native SIMPLE collector. Each information set contributes
once: weight * product(its owner's prior action probabilities) * current row.
Chance, the opponent's strategy, and hidden-member multiplicity are not factors.
No missing or zero-reach row is filled, normalized, or replaced by uniform play.
"""

from dataclasses import dataclass
from fractions import Fraction
import time
from types import MappingProxyType

from tools.aa_infoset_exact_br import Node, _Budget, _flatten, _profile


LABEL = "OWN_REACH_EXACT_SNAPSHOT_DELTA_NOT_HISTORY"
FORMULA = "delta[I,a] = weight * pi_owner[I] * frozen_profile[I,a]"


@dataclass(frozen=True)
class OwnReachPlan:
    """Validated immutable infoset signatures; no strategy or payoff inference."""

    information: object
    stats: object


def compile_plan(root, *, seconds=60.0, max_ops=100000,
                 clock=time.monotonic, deadline=None):
    """Validate every physical branch and perfect recall once, then retain I's.

    Shared Node objects still count once per physical occurrence. Validation
    includes branches that a later snapshot gives probability zero. Small trees
    are supported for independent analytical witnesses; acceptance counts for
    the frozen 1,488-infoset game belong to the calling diagnostic.
    """
    budget = _Budget(seconds, max_ops, clock, deadline)
    budget.require(type(root) is Node, "node_required")
    _, information = _flatten(root, budget)
    copied = {}
    hidden_members = 0
    duplicate_infosets = 0
    for key, info in information.items():
        budget.tick("compile_own_reach_infoset")
        owner, actions, sequence = info["signature"]
        for prior, action in sequence:
            budget.tick("compile_own_reach_ancestor")
            previous = information.get(prior)
            budget.require(previous is not None
                           and previous["signature"][0] == owner
                           and action in previous["signature"][1],
                           "invalid_own_recall_sequence")
        members = tuple(info["nodes"])
        hidden_members += len(members)
        duplicate_infosets += int(len(members) > 1)
        copied[key] = MappingProxyType({
            "signature": (owner, tuple(actions), tuple(sequence)),
            "nodes": members,
            "hidden_member_count": len(members),
            "duplicate_member_count": len(members) - 1,
        })
    budget.require(hidden_members == budget.stats["decision_nodes"],
                   "physical_decision_membership_mismatch")
    budget.tick("compile_own_reach_complete")
    stats = budget.snapshot()
    stats.update(hidden_member_count=hidden_members,
                 duplicate_member_count=hidden_members - len(copied),
                 infosets_with_hidden_duplicates=duplicate_infosets)
    stats["operations_by_phase"] = MappingProxyType(stats["operations_by_phase"])
    return OwnReachPlan(MappingProxyType(copied), MappingProxyType(stats))


def collect(plan, profile, *, weight=Fraction(1), seconds=60.0,
            max_ops=100000, clock=time.monotonic, deadline=None):
    """Return exact complete snapshot deltas, or raise without a partial result.

    The copied Fraction profile is frozen before collecting. For every I, the
    validated perfect-recall signature contains exactly its owner's prior
    (infoset, action) sequence. Multiplying only those factors gives own reach.
    A hidden-member registry is never traversed to add repeated increments.
    """
    budget = _Budget(seconds, max_ops, clock, deadline)
    budget.require(type(plan) is OwnReachPlan, "compiled_own_reach_plan_required")
    budget.require(type(weight) is Fraction and weight > 0,
                   "positive_exact_snapshot_weight_required")
    rows = _profile(profile, plan.information, budget)
    frozen_profile = {}
    for key, row in rows.items():
        budget.tick("freeze_own_reach_profile_row")
        frozen_profile[key] = MappingProxyType(row)
    frozen_profile = MappingProxyType(frozen_profile)

    delta, own_reach = {}, {}
    zero_mass_infosets = 0
    for key, info in plan.information.items():
        budget.tick("collect_own_reach_infoset")
        _, actions, sequence = info["signature"]
        reach = Fraction(1)
        for prior, action in sequence:
            budget.tick("multiply_prior_own_action")
            # Deliberately no physical-node, chance, or opponent factor.
            reach *= frozen_profile[prior][action]
        own_reach[key] = reach
        row = {}
        for action in actions:
            budget.tick("collect_exact_action_delta")
            row[action] = weight * reach * frozen_profile[key][action]
        budget.require(sum(row.values(), Fraction(0)) == weight * reach,
                       "own_reach_delta_mass_mismatch")
        if reach == 0:
            budget.require(all(value == 0 for value in row.values()),
                           "zero_own_reach_must_remain_zero")
            zero_mass_infosets += 1
        delta[key] = row
    budget.require(set(delta) == set(plan.information),
                   "own_reach_delta_infoset_coverage")
    budget.tick("own_reach_snapshot_complete")
    stats = budget.snapshot()
    stats.update(
        infosets=len(delta), collected_once_infosets=len(delta),
        zero_mass_infosets=zero_mass_infosets,
        positive_mass_infosets=len(delta) - zero_mass_infosets,
        hidden_member_count=plan.stats["hidden_member_count"],
        duplicate_member_count=plan.stats["duplicate_member_count"],
        infosets_with_hidden_duplicates=plan.stats["infosets_with_hidden_duplicates"],
        compiled_topology=plan.stats,
    )
    return dict(status="COMPLETE_SNAPSHOT_DELTA", label=LABEL, formula=FORMULA,
                weight=weight, delta=delta, own_reach=own_reach,
                frozen_profile=frozen_profile, stats=stats,
                history_reconstructed=False, historical_average_available=False,
                simulation_only=True, strategy_eligible=False)
