"""Bounded exact best responses for immutable, two-player finite trees.

Chance and fixed-opponent reach are aggregated across every physical member of
an information set before choosing its one response action. Perfect recall is
checked on the complete tree, including zero-probability policy branches.
This is an offline mathematical diagnostic, never a policy or live admission.
"""

from dataclasses import dataclass
from fractions import Fraction
import math
import time
from collections.abc import Mapping


@dataclass(frozen=True)
class Node:
    kind: str
    actor: int | None = None
    infoset: object = None
    children: tuple = ()
    returns: tuple[Fraction, Fraction] | None = None


class ExactBRError(ValueError):
    """Failure receipt; no partial EV, best response, or quality is exposed."""

    def __init__(self, reason, stats):
        super().__init__(reason)
        self.reason = reason
        self.stats = stats
        self.status = 'NOT_SCORED'


class ExactBRBudgetExceeded(ExactBRError):
    pass


class _Budget:
    def __init__(self, seconds, max_ops, clock, deadline):
        self.clock = clock
        self.start = clock()
        self.stats = dict(operations=0, nodes=0, edges=0, decision_nodes=0,
                          chance_nodes=0, terminal_nodes=0, infosets=0,
                          operations_by_phase={})
        if (type(seconds) not in (int, float) or not math.isfinite(seconds)
                or not 0 < seconds <= 60 or type(max_ops) is not int
                or not 0 < max_ops <= 100000):
            self.fail('invalid_exact_budget')
        if (deadline is not None and (type(deadline) not in (int, float)
                                      or not math.isfinite(deadline))):
            self.fail('invalid_exact_deadline')
        self.deadline = min(self.start + seconds, deadline) if deadline is not None \
            else self.start + seconds
        self.max_ops = max_ops

    def snapshot(self):
        return {**self.stats,
                'operations_by_phase': dict(self.stats['operations_by_phase']),
                'elapsed_seconds': max(0.0, self.clock() - self.start)}

    def fail(self, reason, *, exhausted=False):
        error = ExactBRBudgetExceeded if exhausted else ExactBRError
        raise error(reason, self.snapshot())

    def require(self, condition, reason):
        if not condition:
            self.fail(reason)

    def tick(self, phase):
        if self.clock() >= self.deadline:
            self.fail('exact_deadline_exceeded', exhausted=True)
        if self.stats['operations'] >= self.max_ops:
            self.fail('exact_operation_budget_exceeded', exhausted=True)
        self.stats['operations'] += 1
        counts = self.stats['operations_by_phase']
        counts[phase] = counts.get(phase, 0) + 1


def _flatten(root, budget):
    """Expand shared Node objects by occurrence; reject physical path cycles."""
    records, information = [], {}
    stack = [(root, None, None, ((), ()), frozenset())]
    while stack:
        node, parent, incoming, memories, ancestors = stack.pop()
        budget.tick('validate_node')
        budget.require(type(node) is Node, 'node_required')
        budget.require(id(node) not in ancestors, 'physical_tree_cycle')
        budget.require(type(node.kind) is str and node.kind in {
            'chance', 'decision', 'terminal'}, 'invalid_node_kind')
        budget.require(type(node.children) is tuple, 'immutable_children_required')
        index = len(records)
        records.append(dict(node=node, edges=[]))
        if parent is not None:
            action, probability = incoming
            records[parent]['edges'].append((action, probability, index))
        budget.stats['nodes'] += 1
        budget.stats[node.kind + '_nodes'] += 1
        if node.kind == 'terminal':
            budget.require(node.actor is None and node.infoset is None
                           and not node.children and type(node.returns) is tuple
                           and len(node.returns) == 2
                           and all(type(v) is Fraction for v in node.returns),
                           'invalid_terminal')
            continue
        budget.require(node.returns is None and bool(node.children),
                       'invalid_nonterminal')
        if node.kind == 'chance':
            budget.require(node.actor is None and node.infoset is None,
                           'invalid_chance_identity')
        else:
            budget.require(type(node.actor) is int and node.actor in (1, 2)
                           and node.infoset is not None, 'invalid_decision_identity')
            try:
                hash(node.infoset)
            except TypeError:
                budget.fail('hashable_infoset_required')
        actions, total = [], Fraction(0)
        for edge in node.children:
            budget.tick('validate_edge')
            budget.require(type(edge) is tuple and len(edge) == 3,
                           'invalid_edge_schema')
            action, probability, child = edge
            budget.require(type(action) is str and bool(action)
                           and type(probability) is Fraction,
                           'exact_edge_probability_required')
            budget.require(probability > 0 if node.kind == 'chance'
                           else probability == 1, 'invalid_edge_probability')
            budget.require(action not in actions, 'duplicate_edge_action')
            actions.append(action)
            total += probability
        if node.kind == 'chance':
            budget.require(total == 1, 'chance_probability_mass')
        else:
            key = node.infoset
            sequence = memories[node.actor - 1]
            budget.require(all(prior != key for prior, _ in sequence),
                           'infoset_cycle')
            signature = (node.actor, tuple(sorted(actions)), sequence)
            existing = information.get(key)
            if existing is None:
                information[key] = dict(signature=signature, nodes=[index])
                budget.stats['infosets'] += 1
            else:
                budget.require(existing['signature'][0] == node.actor,
                               'infoset_owner_mismatch')
                budget.require(existing['signature'][1] == signature[1],
                               'infoset_menu_mismatch')
                budget.require(existing['signature'][2] == sequence,
                               'imperfect_recall')
                existing['nodes'].append(index)
        next_ancestors = ancestors | {id(node)}
        for action, probability, child in reversed(node.children):
            following = memories
            if node.kind == 'decision':
                updated = list(memories)
                updated[node.actor - 1] += ((node.infoset, action),)
                following = tuple(updated)
            stack.append((child, index, (action, probability), following,
                          next_ancestors))
            budget.stats['edges'] += 1
    return records, information


def _profile(profile, information, budget):
    budget.require(isinstance(profile, Mapping), 'profile_mapping_required')
    budget.require(set(profile) == set(information), 'profile_infoset_coverage')
    rows = {}
    for key, info in information.items():
        budget.tick('validate_profile_row')
        row = profile[key]
        actions = info['signature'][1]
        budget.require(isinstance(row, Mapping) and set(row) == set(actions),
                       'profile_action_coverage')
        copied, total = {}, Fraction(0)
        for action in actions:
            budget.tick('validate_profile_probability')
            value = row[action]
            budget.require(type(value) is Fraction and value >= 0,
                           'exact_profile_probability_required')
            copied[action] = value
            total += value
        budget.require(total == 1, 'profile_probability_mass')
        rows[key] = copied
    return rows


def _counterfactual_reach(records, rows, budget):
    reach = [(Fraction(0), Fraction(0)) for _ in records]
    reach[0] = (Fraction(1), Fraction(1))
    for index, record in enumerate(records):
        node = record['node']
        for action, probability, child in record['edges']:
            budget.tick('counterfactual_edge')
            if node.kind == 'chance':
                factors = (probability, probability)
            else:
                p = rows[node.infoset][action]
                factors = (Fraction(1) if node.actor == 1 else p,
                           Fraction(1) if node.actor == 2 else p)
            reach[child] = tuple(reach[index][i] * factors[i] for i in (0, 1))
    return reach


def _expected(records, rows, budget):
    values = [None] * len(records)
    for index in range(len(records) - 1, -1, -1):
        budget.tick('profile_value_node')
        node = records[index]['node']
        if node.kind == 'terminal':
            values[index] = node.returns
            continue
        total = [Fraction(0), Fraction(0)]
        for action, probability, child in records[index]['edges']:
            budget.tick('profile_value_edge')
            p = probability if node.kind == 'chance' else rows[node.infoset][action]
            for actor in (0, 1):
                total[actor] += p * values[child][actor]
        values[index] = tuple(total)
    return {1: values[0][0], 2: values[0][1]}


def _response(player, records, information, rows, reach, budget):
    choices, cache = {}, {}

    def value(start):
        # Explicit stack avoids coupling the diagnostic to Python recursion limits.
        stack = [(start, False)]
        while stack:
            index, ready = stack.pop()
            if index in cache:
                continue
            record = records[index]
            node = record['node']
            if not ready:
                budget.tick('br_value_node')
                if node.kind == 'terminal':
                    cache[index] = node.returns[player - 1]
                    continue
                edges = record['edges']
                if node.kind == 'decision' and node.actor == player:
                    budget.require(node.infoset in choices,
                                   'unresolved_descendant_infoset')
                    edges = [edge for edge in edges
                             if edge[0] == choices[node.infoset]]
                stack.append((index, True))
                stack.extend((child, False) for _, _, child in reversed(edges))
                continue
            if node.kind == 'decision' and node.actor == player:
                selected = choices[node.infoset]
                child = next(child for action, _, child in record['edges']
                             if action == selected)
                cache[index] = cache[child]
            else:
                total = Fraction(0)
                for action, probability, child in record['edges']:
                    budget.tick('br_value_edge')
                    p = (probability if node.kind == 'chance'
                         else rows[node.infoset][action])
                    total += p * cache[child]
                cache[index] = total
        return cache[start]

    groups = [(key, info) for key, info in information.items()
              if info['signature'][0] == player]
    # Perfect recall makes every descendant own infoset strictly deeper here,
    # even when members of the same infoset have different physical depths.
    groups.sort(key=lambda item: len(item[1]['signature'][2]), reverse=True)
    for key, info in groups:
        budget.tick('br_infoset')
        actions = info['signature'][1]
        totals = {action: Fraction(0) for action in actions}
        for index in info['nodes']:
            for action, _, child in records[index]['edges']:
                budget.tick('br_action_contribution')
                # Never prune the owner's original zero-probability action.
                totals[action] += reach[index][player - 1] * value(child)
        best = max(actions, key=lambda action: totals[action])
        choices[key] = best
        for index in info['nodes']:
            budget.tick('br_owner_assignment')
            child = next(child for action, _, child in records[index]['edges']
                         if action == best)
            cache[index] = cache[child]
    return value(0), choices


def evaluate(root, profile, *, big_blind=Fraction(2), seconds=60.0,
             max_ops=100000, clock=time.monotonic, deadline=None):
    """Return exact EV and infoset-consistent BRs, or a NOT_SCORED exception.

    ``profile`` has exactly one complete Fraction-valued row per information set.
    The budget includes topology/profile checks, both responses and final checks.
    Operation counts measure checked node, edge and information-set work, not
    individual big-integer instructions; the wall deadline also guards fractions.
    """
    budget = _Budget(seconds, max_ops, clock, deadline)
    try:
        budget.require(type(big_blind) is Fraction and big_blind > 0,
                       'positive_exact_big_blind_required')
        records, information = _flatten(root, budget)
        rows = _profile(profile, information, budget)
        reach = _counterfactual_reach(records, rows, budget)
        expected = _expected(records, rows, budget)
        responses, witnesses = {}, {}
        for player in (1, 2):
            responses[player], witnesses[player] = _response(
                player, records, information, rows, reach, budget)
        gains = {player: responses[player] - expected[player] for player in (1, 2)}
        budget.require(all(gain >= 0 for gain in gains.values()),
                       'negative_exact_response_gain')
        budget.tick('finalize')
        nash_conv = sum(gains.values(), Fraction(0))
        stats = budget.snapshot()
        stats['infosets_by_player'] = {
            player: sum(info['signature'][0] == player
                        for info in information.values()) for player in (1, 2)}
        return dict(status='COMPLETE_EXACT_BR', expected=expected, br=responses,
                    gains=gains, nash_conv=nash_conv,
                    nash_conv_bb=nash_conv / big_blind,
                    best_response_policies=witnesses, stats=stats,
                    simulation_only=True, strategy_eligible=False)
    except ExactBRError:
        raise
    except (TypeError, ValueError, KeyError, AttributeError, OverflowError) as exc:
        budget.fail('malformed_exact_game_or_profile:' + type(exc).__name__)
