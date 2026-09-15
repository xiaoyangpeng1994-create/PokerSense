from dataclasses import replace
from fractions import Fraction

import pytest

from poker_engine.strategy import robust_policy_selection_v1 as module
from poker_engine.strategy.robust_policy_selection_v1 import (
    WorldHypothesis, select_robust_policy, validate_robust_selection,
)
from poker_engine.strategy.threeway_policy_evaluation_v1 import (
    PolicyEvaluation, PolicyMetric, compile_policy_book, policy_book_hash,
)
from .test_terminal_multiway_v1 import ranges
from .test_threeway_policy_evaluation_v1 import facing_bet


def setup():
    weak = facing_bet()
    strong = replace(weak, ranges=ranges({1: "JhJd", 2: "3c3d"}))
    middle = replace(weak, ranges=ranges({1: "KhKd", 2: "3c3d"}))
    books = (compile_policy_book(weak, policy_id="A"),
             compile_policy_book(strong, policy_id="B"))
    worlds = (WorldHypothesis("weak", weak), WorldHypothesis("strong", strong))
    return books, worlds, (WorldHypothesis("validation", middle),)


def fake(ev, fallback=0):
    value = Fraction(ev)
    return PolicyEvaluation(
        "COMPLETE_CONDITIONAL_FIXED_POLICY",
        metrics=(PolicyMetric("frozen_policy", value, value / 2,
                              Fraction(fallback), Fraction(fallback), (), 1),),
        delta_vs_check_fold_chips=value, delta_vs_check_call_chips=value)


def test_maximin_choice_frozen_when_validation_prefers_other_policy(monkeypatch):
    books, worlds, val = setup()
    calls = []

    def evaluate(book, scenario, **kwargs):
        index = next(i for i, w in enumerate((*worlds, *val))
                     if scenario == w.scenario)
        calls.append((book.policy_id, index))
        return fake({"A": [3, -2, 10], "B": [1, 1, -5]}[book.policy_id][index])

    monkeypatch.setattr(module, "evaluate_policy_book", evaluate)
    selected = select_robust_policy(books, worlds)
    assert selected.selected_id == "B"
    result = validate_robust_selection(selected, val)
    assert result.selected_id == "B"
    assert result.worst_baseline_delta_chips == -5
    assert result.status == "FAIL_DECLARED_WORLD_SCREEN"
    assert ("A", 2) not in calls
    assert not result.strategy_eligible and not result.empirical_approval


def test_real_fixed_policy_reports_absolute_loss_and_baseline_regret():
    books, worlds, val = setup()
    selected = select_robust_policy(books, worlds)
    assert selected.selected_id == "A"  # Worst regret -100 vs fold's -160.
    by_id = {s.policy_id: s for s in selected.scores}
    assert by_id["A"].worst_net_ev_chips == -100
    assert by_id["B"].worst_net_ev_chips == 0
    assert by_id["B"].worst_baseline_delta_chips == -160
    result = validate_robust_selection(selected, val)
    assert result.worst_net_ev_chips == -100
    assert result.status == "FAIL_DECLARED_WORLD_SCREEN"


@pytest.mark.parametrize("failure", ["blocked", "fallback"])
def test_incomplete_candidate_never_removed_to_select_another(monkeypatch, failure):
    books, worlds, val = setup()

    def evaluate(book, scenario, **kwargs):
        if book.policy_id == "A":
            return PolicyEvaluation("BLOCKED") if failure == "blocked" else fake(1, 1)
        return fake(10)

    monkeypatch.setattr(module, "evaluate_policy_book", evaluate)
    selected = select_robust_policy(books, worlds)
    assert selected.selected_id is None
    assert len(selected.evaluations) == 4
    assert validate_robust_selection(selected, val).status == "INSUFFICIENT_EVIDENCE"


def test_validation_budget_failure_retained(monkeypatch):
    books, worlds, val = setup()
    real = module.evaluate_policy_book

    def evaluate(book, scenario, **kwargs):
        if scenario == val[0].scenario:
            return PolicyEvaluation("BLOCKED", reasons=("budget",))
        return real(book, scenario, **kwargs)

    monkeypatch.setattr(module, "evaluate_policy_book", evaluate)
    report = validate_robust_selection(select_robust_policy(books, worlds), val)
    assert report.status == "INSUFFICIENT_EVIDENCE"
    assert len(report.evaluations) == 1
    assert report.worst_net_ev_chips is None


def test_receipt_tampering_rejected():
    books, worlds, val = setup()
    selected = select_robust_policy(books, worlds)
    with pytest.raises(ValueError, match="changed_selection"):
        validate_robust_selection(replace(selected, selected_id="B"), val)


@pytest.mark.parametrize("same_id", [True, False])
def test_world_overlap_by_name_or_definition_rejected(same_id):
    books, worlds, val = setup()
    overlapping = replace(val[0], world_id="weak") if same_id else (
        replace(worlds[0], world_id="renamed"))
    with pytest.raises(ValueError, match="world_overlap"):
        validate_robust_selection(select_robust_policy(books, worlds), (overlapping,))


def test_tie_break_is_lexical_not_input_order(monkeypatch):
    books, worlds, _ = setup()
    monkeypatch.setattr(module, "evaluate_policy_book", lambda *a, **k: fake(0))
    assert select_robust_policy(tuple(reversed(books)), worlds).selected_id == "A"


def test_stronger_check_call_baseline_controls_objective(monkeypatch):
    books, worlds, _ = setup()
    monkeypatch.setattr(module, "evaluate_policy_book", lambda *a, **k: replace(
        fake(2), delta_vs_check_call_chips=Fraction(-7)))
    selected = select_robust_policy(books, worlds)
    assert all(s.worst_baseline_delta_chips == -7 for s in selected.scores)


def test_different_public_conditions_rejected():
    books, worlds, _ = setup()
    wrong = replace(books[1], public_conditions_json="different")
    wrong = replace(wrong, book_sha256=policy_book_hash(wrong))
    with pytest.raises(ValueError, match="share_public"):
        select_robust_policy((books[0], wrong), worlds)
