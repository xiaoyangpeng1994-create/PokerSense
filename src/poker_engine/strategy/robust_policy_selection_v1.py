"""Select one frozen policy under declared uncertainty, never empirical approval."""

from dataclasses import dataclass, replace
from fractions import Fraction

from .threeway_policy_evaluation_v1 import (
    PolicyBook, PolicyEvaluation, WorldResponseOverride, _json, _sha,
    evaluate_policy_book, policy_book_hash,
)
from .threeway_river_v1 import ThreewayRiverScenario


@dataclass(frozen=True)
class WorldHypothesis:
    world_id: str
    scenario: ThreewayRiverScenario
    overrides: tuple[WorldResponseOverride, ...] = ()


def world_hash(world):
    return _sha(_json((world.scenario, world.overrides)))


@dataclass(frozen=True)
class RobustScore:
    policy_id: str
    worst_baseline_delta_chips: Fraction
    worst_net_ev_chips: Fraction
    best_net_ev_chips: Fraction


@dataclass(frozen=True)
class RobustSelection:
    status: str
    books: tuple[PolicyBook, ...]
    calibration_worlds: tuple[WorldHypothesis, ...]
    evaluations: tuple
    scores: tuple[RobustScore, ...]
    selected_id: str | None
    max_nodes: int
    receipt_sha256: str = ""
    objective: str = "maximize_worst_delta_vs_stronger_check_fold_or_check_call"


@dataclass(frozen=True)
class RobustValidation:
    status: str
    selected_id: str | None
    selection_sha256: str
    evaluations: tuple
    worst_baseline_delta_chips: Fraction | None
    worst_net_ev_chips: Fraction | None
    reasons: tuple[str, ...]
    source_kind: str = "manual_hypothesis_study"
    strategy_eligible: bool = False
    advice_emitted: bool = False
    empirical_approval: bool = False

    def __post_init__(self):
        if (self.strategy_eligible is not False or self.advice_emitted is not False
                or self.empirical_approval is not False):
            raise ValueError("manual_uncertainty_cannot_approve_live_or_profitability")


def _worlds(worlds):
    if not isinstance(worlds, tuple) or not 1 <= len(worlds) <= 32:
        raise ValueError("one_to_32_immutable_worlds_required")
    ids, hashes = set(), set()
    for world in worlds:
        if (not isinstance(world, WorldHypothesis)
                or not isinstance(world.world_id, str) or not world.world_id.strip()
                or not isinstance(world.scenario, ThreewayRiverScenario)
                or not isinstance(world.overrides, tuple)):
            raise ValueError("explicit_named_world_required")
        identity = world_hash(world)
        if world.world_id in ids or identity in hashes:
            raise ValueError("duplicate_world_id_or_definition")
        ids.add(world.world_id)
        hashes.add(identity)


def _complete(result):
    return result.status == "COMPLETE_CONDITIONAL_FIXED_POLICY"


def _supported(result):
    return _complete(result) and result.metrics[0].probability_of_any_fallback == 0


def _delta(result):
    return min(result.delta_vs_check_fold_chips, result.delta_vs_check_call_chips)


def select_robust_policy(books, calibration_worlds, *, max_nodes=20000):
    """Maximin relative to both passive baselines; no validation input accepted."""
    _worlds(calibration_worlds)
    if (not isinstance(books, tuple) or not 2 <= len(books) <= 16
            or any(not isinstance(b, PolicyBook) for b in books)
            or len({b.policy_id for b in books}) != len(books)):
        raise ValueError("two_to_16_unique_frozen_policy_ids_required")
    if type(max_nodes) is not int or not 1 <= max_nodes <= 200000:
        raise ValueError("bounded_integer_node_budget_required")
    if any(policy_book_hash(b) != b.book_sha256 for b in books):
        raise ValueError("changed_policy_book")
    if len({b.public_conditions_json for b in books}) != 1:
        raise ValueError("policies_must_share_public_decision_conditions")
    evaluations, scores = [], []
    for book in books:
        results = []
        for world in calibration_worlds:
            result = evaluate_policy_book(book, world.scenario,
                                          world_overrides=world.overrides,
                                          max_nodes=max_nodes)
            evaluations.append((book.policy_id, world.world_id,
                                world_hash(world), result))
            results.append(result)
        if all(_supported(r) for r in results):
            scores.append(RobustScore(
                book.policy_id, min(_delta(r) for r in results),
                min(r.metrics[0].net_ev_chips for r in results),
                max(r.metrics[0].net_ev_chips for r in results)))
    # Do not silently discard an incomplete/unsupported candidate and pick another.
    complete = len(scores) == len(books)
    selected = min(scores, key=lambda s: (-s.worst_baseline_delta_chips,
                                          s.policy_id)).policy_id if complete else None
    result = RobustSelection(
        "SELECTED_CONDITIONAL" if complete else "INSUFFICIENT_EVIDENCE",
        books, calibration_worlds, tuple(evaluations), tuple(scores), selected,
        max_nodes)
    return replace(result, receipt_sha256=_sha(_json(result)))


def validate_robust_selection(selection, validation_worlds):
    """Recheck training receipt, then execute ONLY its fixed choice in new worlds."""
    if not isinstance(selection, RobustSelection):
        raise ValueError("selection_receipt_required")
    _worlds(validation_worlds)
    expected = select_robust_policy(selection.books, selection.calibration_worlds,
                                    max_nodes=selection.max_nodes)
    if selection != expected:
        raise ValueError("changed_selection_receipt")
    if ({w.world_id for w in selection.calibration_worlds}
            & {w.world_id for w in validation_worlds}
            or {world_hash(w) for w in selection.calibration_worlds}
            & {world_hash(w) for w in validation_worlds}):
        raise ValueError("calibration_validation_world_overlap")
    if selection.selected_id is None:
        skipped = tuple((w.world_id, world_hash(w), PolicyEvaluation(
            "BLOCKED", reasons=("no_calibration_selection_evaluation_not_run",)))
            for w in validation_worlds)
        return RobustValidation(
            "INSUFFICIENT_EVIDENCE", None, selection.receipt_sha256, skipped, None,
            None, ("calibration_worlds_incomplete_or_fallback",))
    book = next(b for b in selection.books if b.policy_id == selection.selected_id)
    rows = tuple((w.world_id, world_hash(w), evaluate_policy_book(
        book, w.scenario, world_overrides=w.overrides,
        max_nodes=selection.max_nodes)) for w in validation_worlds)
    if not all(_supported(r) for _, _, r in rows):
        return RobustValidation(
            "INSUFFICIENT_EVIDENCE", book.policy_id, selection.receipt_sha256,
            rows, None, None, ("validation_incomplete_or_fallback",))
    worst = min(_delta(r) for _, _, r in rows)
    raw = min(r.metrics[0].net_ev_chips for _, _, r in rows)
    status = "FAIL_DECLARED_WORLD_SCREEN" if worst < 0 else (
        "PASS_DECLARED_WORLD_SCREEN")
    return RobustValidation(status, book.policy_id, selection.receipt_sha256,
                            rows, worst, raw, (
                                "manual_worlds_not_representative_opponent_data",
                                "conditional_river_chips_not_full_hand_bb100",
                                "shared_kernel_not_independent_settlement",
                                "maximin_not_GTO_or_profitability_guarantee"))
