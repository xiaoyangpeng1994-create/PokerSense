"""Small analytical witnesses; no poker training or strategy quality search."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F

import pytest

from tools.aa_infoset_exact_br import (
    ExactBRBudgetExceeded, ExactBRError, Node, evaluate,
)


def terminal(value):
    return Node('terminal', returns=(F(value), F(-value)))


def decision(actor, key, **children):
    return Node('decision', actor, key,
                tuple((action, F(1), child) for action, child in children.items()))


def chance(*children):
    return Node('chance', children=tuple(
        (str(i), probability, child)
        for i, (probability, child) in enumerate(children)))


def test_hidden_worlds_are_aggregated_before_the_response_maximum():
    root = chance(
        (F(1, 2), decision(1, 'hidden', left=terminal(1), right=terminal(-1))),
        (F(1, 2), decision(1, 'hidden', left=terminal(-1), right=terminal(1))))
    result = evaluate(root, {'hidden': {'left': F(1, 2), 'right': F(1, 2)}})
    assert result['expected'] == result['br'] == {1: F(0), 2: F(0)}
    assert result['nash_conv'] == 0
    assert result['best_response_policies'][1] == {'hidden': 'left'}
    # Maximizing separately in each hidden world would incorrectly return one.
    assert result['br'][1] != 1


def test_owner_zero_probability_branch_and_descendant_still_enter_br():
    root = decision(1, 'root', stop=terminal(0),
                    continue_=decision(1, 'later', bad=terminal(-1), good=terminal(3)))
    profile = {'root': {'stop': F(1), 'continue_': F(0)},
               'later': {'bad': F(1), 'good': F(0)}}
    result = evaluate(root, profile)
    assert result['expected'][1] == 0
    assert result['br'][1] == 3
    assert result['nash_conv_bb'] == F(3, 2)
    assert result['best_response_policies'][1] == {
        'later': 'good', 'root': 'continue_'}


def test_shared_opponent_infoset_uses_one_frozen_profile_row():
    root = chance(
        (F(1, 3), decision(2, 'opponent', a=terminal(3), b=terminal(0))),
        (F(2, 3), decision(2, 'opponent', a=terminal(-1), b=terminal(2))))
    result = evaluate(root, {'opponent': {'a': F(1, 4), 'b': F(3, 4)}})
    assert result['expected'] == {1: F(13, 12), 2: F(-13, 12)}
    assert result['br'] == {1: F(13, 12), 2: F(-1, 3)}
    assert result['nash_conv'] == F(3, 4)
    assert result['nash_conv_bb'] == F(3, 8)


def test_same_infoset_at_different_physical_depths_is_valid_perfect_recall():
    visible = decision(1, 'same', a=terminal(1), b=terminal(2))
    root = chance((F(1, 2), visible), (F(1, 2), chance((F(1), visible))))
    result = evaluate(root, {'same': {'a': F(1), 'b': F(0)}})
    assert result['expected'][1] == 1
    assert result['br'][1] == 2
    assert result['stats']['decision_nodes'] == 2
    assert result['stats']['infosets'] == 1


def test_forgetting_own_previous_action_is_refused_even_on_zero_branch():
    forgotten = decision(1, 'forgotten', a=terminal(0), b=terminal(1))
    root = decision(1, 'root', left=forgotten, right=forgotten)
    with pytest.raises(ExactBRError, match='imperfect_recall') as error:
        evaluate(root, {'root': {'left': F(1), 'right': F(0)},
                        'forgotten': {'a': F(1), 'b': F(0)}})
    assert error.value.status == 'NOT_SCORED'
    assert error.value.stats['operations'] > 0


def test_forgetting_previous_own_infoset_is_refused():
    end = decision(1, 'end', a=terminal(0), b=terminal(1))
    root = chance(
        (F(1, 2), decision(1, 'observed_left', go=end)),
        (F(1, 2), decision(1, 'observed_right', go=end)))
    with pytest.raises(ExactBRError, match='imperfect_recall'):
        evaluate(root, {})


def test_repeated_infoset_on_a_path_is_refused():
    root = decision(1, 'again', go=decision(1, 'again', end=terminal(0)))
    with pytest.raises(ExactBRError, match='infoset_cycle'):
        evaluate(root, {})


@pytest.mark.parametrize('other', [
    decision(1, 'same', changed=terminal(0)),
    decision(2, 'same', a=terminal(0)),
])
def test_shared_information_requires_same_owner_and_menu(other):
    root = chance((F(1, 2), decision(1, 'same', a=terminal(0))), (F(1, 2), other))
    with pytest.raises(ExactBRError, match='infoset_(menu|owner)_mismatch'):
        evaluate(root, {})


@pytest.mark.parametrize('profile', [
    {}, {'i': {'a': F(1)}}, {'i': {'a': F(1), 'b': F(1)}},
    {'i': {'a': F(-1), 'b': F(2)}}, {'i': {'a': 0.5, 'b': 0.5}},
    {'i': {'a': True, 'b': F(0)}},
    {'i': {'a': F(1), 'b': F(0)}, 'extra': {'a': F(1)}},
])
def test_profile_is_exact_complete_and_has_no_fallback(profile):
    with pytest.raises(ExactBRError):
        evaluate(decision(1, 'i', a=terminal(0), b=terminal(1)), profile)


@pytest.mark.parametrize('root', [
    chance((F(1, 3), terminal(0)), (F(1, 3), terminal(1))),
    chance((F(0), terminal(0)), (F(1), terminal(1))),
    Node('chance', children=(('a', 1.0, terminal(0)),)),
    Node('decision', 1, 'i', (('a', F(1, 2), terminal(0)),)),
    Node('decision', 1, 'i', (('a', F(1), terminal(0)),
                              ('a', F(1), terminal(1)))),
    Node('terminal', returns=(0, 0)),
    Node('terminal', actor=1, returns=(F(0), F(0))),
    Node('chance', children=[]),
    Node('decision', 1, [], (('a', F(1), terminal(0)),)),
])
def test_invalid_tree_structure_and_chance_fail_closed(root):
    with pytest.raises(ExactBRError):
        evaluate(root, {})


def test_fraction_chance_and_big_blind_units_remain_exact():
    root = chance((F(1, 3), decision(1, 'i', a=terminal(0), b=terminal(5))),
                  (F(2, 3), terminal(0)))
    result = evaluate(root, {'i': {'a': F(1), 'b': F(0)}}, big_blind=F(5, 2))
    assert result['nash_conv'] == F(5, 3)
    assert result['nash_conv_bb'] == F(2, 3)
    assert result['simulation_only'] is True
    assert result['strategy_eligible'] is False


@pytest.mark.parametrize('blind', [F(0), F(-1), 2, 2.0, True])
def test_big_blind_requires_positive_fraction(blind):
    with pytest.raises(ExactBRError, match='positive_exact_big_blind'):
        evaluate(terminal(0), {}, big_blind=blind)


def test_operation_budget_failure_retains_counters_and_no_partial_quality():
    with pytest.raises(ExactBRBudgetExceeded) as error:
        evaluate(decision(1, 'i', a=terminal(0)), {'i': {'a': F(1)}}, max_ops=2)
    assert error.value.status == 'NOT_SCORED'
    assert error.value.stats['operations'] == 2
    assert not hasattr(error.value, 'nash_conv')


def test_absolute_deadline_failure_is_not_a_partial_result():
    with pytest.raises(ExactBRBudgetExceeded, match='exact_deadline') as error:
        evaluate(terminal(0), {}, clock=lambda: 10.0, deadline=10.0)
    assert error.value.stats['operations'] == 0


def test_node_and_input_profile_are_not_modified():
    root = decision(1, 'i', a=terminal(0), b=terminal(1))
    profile = {'i': {'a': F(1, 2), 'b': F(1, 2)}}
    with pytest.raises(FrozenInstanceError):
        root.actor = 2
    evaluate(root, profile)
    assert profile == {'i': {'a': F(1, 2), 'b': F(1, 2)}}
    assert root.children[0][1] == 1
