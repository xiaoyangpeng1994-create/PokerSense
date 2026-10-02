"""One finite turn-to-river research game; never a production policy.

The second frozen river fixture is mechanically cut before its turn actions.
River chance has all 44 cards conditional on each active joint holding. Folded
holdings and burns are marginalized, not fixed blockers or a shortened deck.
The action tree is the explicitly frozen C/B2/B4 abstraction of legal NLHE.
"""
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import re
import time

from poker_engine.core.enums import Rank, Suit
from poker_engine.core.value_objects import Card
from poker_engine.equity.evaluator import evaluate
from poker_engine.strategy.aa_frozen_policy import canonical_hash
from poker_engine.strategy.aa_full_hand_arena import AAFullHandArena
from poker_engine.strategy.aa_rules_v2 import AARuleProfileV2
from tools.aa_infoset_exact_br import Node


SCOPE_ID = "aa-six-seat-two-active-full-chance-turn-river-v1"
ENCODER_ID = "research-exact-turn-river-visible-memory-v1"
BOARD = ("Ts", "Jh", "Qd", "Kc")
HOLDINGS = {1: {"STRAIGHT_A": ("As", "6h"), "SET_K": ("Kh", "Kd")},
            2: {"STRAIGHT_B": ("Ah", "7h"), "SET_Q": ("Qc", "Qh")}}
DEALS = (("STRAIGHT_A", "STRAIGHT_B"), ("STRAIGHT_A", "SET_Q"),
         ("SET_K", "STRAIGHT_B"), ("SET_K", "SET_Q"))
PREFIX = ((3, "preflop", "fold", "0"), (4, "preflop", "fold", "0"),
          (5, "preflop", "fold", "0"), (0, "preflop", "fold", "0"),
          (1, "preflop", "check_call", "2"),
          (2, "preflop", "check_call", "0"),
          (1, "flop", "check_call", "0"), (2, "flop", "check_call", "0"))
CONTRIBUTIONS = {0: 3, 1: 6, 2: 6, 3: 2, 4: 2, 5: 2}
STANDARD_DECK = tuple(rank + suit for rank in "23456789TJQKA" for suit in "cdhs")
C, B2, B4, F = "check_call", "raise_to:2", "raise_to:4", "fold"
ABSTRACT_ACTIONS = frozenset((C, B2, B4, F))
_VISIBLE_FIELDS = frozenset((
    "schema_version", "arena_version", "settlement_model", "strategy_eligible",
    "simulation_only", "rules_fingerprint", "rules", "table_size", "big_blind",
    "observing_seat", "actor", "occupied_seats", "dealer_seat", "straddler_seat",
    "street", "terminal", "own_hole", "board", "board_history", "stacks",
    "starting_stacks", "bets", "folded", "all_in", "contributions", "betting",
    "pot", "to_call", "public_history", "legal_actions", "own_memory"))


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("turn_river_deadline_no_partial_acceptance")


@lru_cache(maxsize=1)
def rules_profile():
    path = Path(__file__).resolve().parents[1] / "configs/game/aa-shadow-rules-v2.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["table_size"] = 6
    return AARuleProfileV2.from_dict(value)


def fixture_spec():
    return dict(scope_id=SCOPE_ID, board=list(BOARD),
                active_ranges={str(s): {name: list(cards)
                                        for name, cards in rows.items()}
                               for s, rows in HOLDINGS.items()},
                joint_deals=[list(deal) for deal in DEALS],
                joint_weights=[1, 1, 1, 1], normalizer=4,
                prefix=[list(row) for row in PREFIX], prefix_reach_probability="1",
                active_seats=[1, 2], dealer=5, table_size=6, starting_stacks="10",
                action_abstraction="legal-C-B2-B4-no-other-bet-sizes-v1",
                chance="44-rivers-per-active-joint-dead-and-burn-marginalized-v1",
                simulation_only=True, production_eligible=False)


def expected_binding():
    return dict(scope_id=SCOPE_ID, encoder_id=ENCODER_ID,
                fixture_sha256=canonical_hash(fixture_spec()),
                rules_fingerprint=rules_profile().fingerprint,
                simulation_only=True, strategy_eligible=False)


def _deal(deal):
    value = tuple(deal)
    if value not in DEALS:
        raise ValueError("unknown_active_joint")
    return value


def joint_rivers(deal):
    deal = _deal(deal)
    blocked = set(BOARD + HOLDINGS[1][deal[0]] + HOLDINGS[2][deal[1]])
    result = tuple(card for card in STANDARD_DECK if card not in blocked)
    if len(blocked) != 8 or len(result) != 44:
        raise ValueError("joint_card_collision")
    return result


def _deck_from_remainder(deal, river, remaining):
    holes = {1: HOLDINGS[1][deal[0]], 2: HOLDINGS[2][deal[1]]}
    for index, seat in enumerate((0, 3, 4, 5)):
        holes[seat] = remaining[2 * index:2 * index + 2]
    burns = remaining[8:11]
    deck = [card for seat in range(6) for card in holes[seat]]
    deck += [burns[0], *BOARD[:3], burns[1], BOARD[3], burns[2], river]
    deck += list(remaining[11:])
    if len(deck) != 52 or set(deck) != set(STANDARD_DECK):
        raise ValueError("complete_standard_deck_required")
    return deck


def compose_deck(deal, river, arrangement=0):
    """Two legal representatives of marginalized dead cards; not chance priors."""
    deal = _deal(deal)
    if river not in joint_rivers(deal) or type(arrangement) is not int:
        raise ValueError("invalid_river_or_dead_arrangement")
    if arrangement not in (0, 1):
        raise ValueError("invalid_dead_arrangement")
    remaining = [card for card in joint_rivers(deal) if card != river]
    if arrangement:
        remaining.reverse()
    return _deck_from_remainder(deal, river, remaining)


class TrackedTurnRiverArena:
    """Host-owned actual arena; only returned visible observations reach policy."""

    def __init__(self, arena, memory=None):
        self.arena = arena
        self._memory = (deepcopy(memory) if memory is not None
                        else {seat: [] for seat in arena.occupied_seats})

    @property
    def actor(self):
        return self.arena.actor

    @property
    def terminal(self):
        return self.arena.terminal

    @property
    def street(self):
        return self.arena.street

    def observe(self, seat):
        value = self.arena.observe(seat)
        value["own_memory"] = deepcopy(self._memory[seat])
        return value

    def legal_actions(self):
        return tuple(action for action in self.arena.legal_actions()
                     if action.id in ABSTRACT_ACTIONS)

    def step(self, action_id):
        if action_id not in tuple(action.id for action in self.legal_actions()):
            raise ValueError("outside_frozen_action_abstraction")
        actor = self.actor
        visible = self.arena.observe(actor)
        entry = dict(public_index=len(visible["public_history"]),
                     visible_observation_sha256=canonical_hash(visible),
                     action_id=action_id)
        self.arena.step(action_id)
        self._memory[actor].append(entry)
        return self

    def clone(self):
        return TrackedTurnRiverArena(self.arena.clone(), self._memory)

    def terminal_returns(self):
        return self.arena.terminal_returns()

    def terminal_result(self):
        return self.arena.terminal_result()


def _root_from_deck(deck, deadline=None):
    check_deadline(deadline)
    native = AAFullHandArena(rules_profile(), starting_stacks={s: 10 for s in range(6)},
                             occupied_seats=range(6), dealer_seat=5)
    native.reset(0, deck=deck)
    root = TrackedTurnRiverArena(native)
    for seat, street, action, paid in PREFIX:
        check_deadline(deadline)
        if (root.actor, root.street) != (seat, street):
            raise ValueError("native_prefix_actor_or_street")
        root.step(action)
        if root.observe(seat)["public_history"][-1]["paid"] != paid:
            raise ValueError("native_prefix_payment")
    if (root.actor, root.street, root.terminal) != (1, "turn", False):
        raise ValueError("native_turn_root")
    return root


def make_root(deal, river, arrangement=0, *, deadline=None):
    return _root_from_deck(compose_deck(deal, river, arrangement), deadline)


def sample_factory(seed):
    """Sample active joint, all 44 rivers, and a uniform remaining full deck."""
    if type(seed) is not int:
        raise ValueError("integer_factory_seed_required")
    rng = random.Random(seed)
    deal = rng.choice(DEALS)
    river = rng.choice(joint_rivers(deal))
    remaining = [card for card in joint_rivers(deal) if card != river]
    rng.shuffle(remaining)
    return _root_from_deck(_deck_from_remainder(deal, river, remaining))


@dataclass(frozen=True)
class PublicNode:
    history: tuple
    actor: int | None
    wagers: tuple
    folded: int | None = None
    children: tuple = ()

    @property
    def terminal(self):
        return self.actor is None

    @property
    def actions(self):
        return tuple(action for action, _ in self.children)


@lru_cache(maxsize=2)
def public_tree(cap=4):
    """Independent integer betting tree, without an arena/PokerKit import."""
    if cap not in (2, 4):
        raise ValueError("unknown_street_stack")

    def build(history, actor, wagers, checks=0):
        index = actor - 1
        current = max(wagers)
        owed = current - wagers[index]
        actions = ((F, C, B4) if current == 2 and cap == 4 else (F, C)) if owed else (
            (C, B2, B4) if cap == 4 else (C, B2))
        children = []
        for action in actions:
            following = history + (action,)
            if action == F:
                child = PublicNode(following, None, wagers, actor)
            elif action == C:
                if owed:
                    paid = list(wagers)
                    paid[index] = current
                    child = PublicNode(following, None, tuple(paid))
                elif checks:
                    child = PublicNode(following, None, wagers)
                else:
                    child = build(following, 3 - actor, wagers, 1)
            else:
                paid = list(wagers)
                paid[index] = int(action.split(":")[1])
                child = build(following, 3 - actor, tuple(paid))
            children.append((action, child))
        return PublicNode(history, actor, wagers, children=tuple(children))

    return build((), 1, (0, 0))


def _at_history(root, history):
    node = root
    for action in history:
        try:
            node = dict(node.children)[action]
        except KeyError as exc:
            raise ValueError("unknown_abstract_history") from exc
    return node


def symbolic_infoset(actor, own_type, turn_history, river_history=(), river=None):
    hole = tuple(sorted(HOLDINGS[actor][own_type]))
    street = "turn" if river is None else "river"
    board = BOARD if river is None else BOARD + (river,)
    return actor, hole, street, board, tuple(turn_history), tuple(river_history)


def infoset_id(observation):
    actor = observation.get("actor")
    if type(actor) is not int or actor not in HOLDINGS:
        raise ValueError("invalid_active_actor")
    own = tuple(sorted(observation.get("own_hole", ())))
    labels = [name for name, cards in HOLDINGS[actor].items()
              if tuple(sorted(cards)) == own]
    if len(labels) != 1:
        raise ValueError("unsupported_own_range")
    history = observation.get("public_history", ())
    if len(history) < len(PREFIX):
        raise ValueError("incomplete_prefix")
    turn = tuple(row["id"] for row in history[len(PREFIX):] if row["street"] == "turn")
    river = tuple(row["id"] for row in history[len(PREFIX):]
                  if row["street"] == "river")
    board = observation.get("board")
    if observation.get("street") == "turn" and board == list(BOARD) and not river:
        node = _at_history(public_tree(), turn)
        card = None
    elif (observation.get("street") == "river" and isinstance(board, list)
          and len(board) == 5 and tuple(board[:4]) == BOARD
          and board[4] in STANDARD_DECK and board[4] not in set(BOARD + own)):
        turn_end = _at_history(public_tree(), turn)
        if not turn_end.terminal or turn_end.folded or max(turn_end.wagers) == 4:
            raise ValueError("invalid_turn_to_river_history")
        node = _at_history(public_tree(4 - max(turn_end.wagers)), river)
        card = board[4]
    else:
        raise ValueError("unsupported_public_board_or_street")
    if node.actor != actor:
        raise ValueError("abstract_actor_mismatch")
    if tuple(row["id"] for row in observation["legal_actions"]) != node.actions:
        raise ValueError("abstract_menu_mismatch")
    return symbolic_infoset(actor, labels[0], turn, river, card)


def research_key(observation):
    """Hash trusted host-visible state and full own memory, not authenticate it.

    Only the native factory/context and its audited catalog supply this input;
    a caller-created JSON object is not proof of a legal source or past memory.
    """
    if not isinstance(observation, dict) or set(observation) != _VISIBLE_FIELDS:
        raise ValueError("exact_visible_observation_required")
    if (observation["simulation_only"] is not True
            or observation["strategy_eligible"] is not False
            or observation["terminal"] is not False
            or observation["observing_seat"] != observation["actor"]
            or observation["rules"] != rules_profile().to_dict()
            or observation["rules_fingerprint"] != rules_profile().fingerprint):
        raise ValueError("invalid_research_context")
    infoset_id(observation)
    actor = observation["actor"]
    required = [(index, row["id"]) for index, row in enumerate(
        observation["public_history"]) if row["actor"] == actor]
    memory = observation["own_memory"]
    if not isinstance(memory, list) or len(memory) != len(required):
        raise ValueError("complete_own_memory_required")
    for item, (index, action) in zip(memory, required):
        if (not isinstance(item, dict) or set(item) != {
                "public_index", "visible_observation_sha256", "action_id"}
                or item["public_index"] != index or item["action_id"] != action
                or not isinstance(item["visible_observation_sha256"], str)
                or re.fullmatch("[0-9a-f]{64}", item["visible_observation_sha256"])
                is None):
            raise ValueError("invalid_own_memory")
    return canonical_hash(dict(scope_id=SCOPE_ID, encoder_id=ENCODER_ID,
                               visible_observation=observation))


@lru_cache(maxsize=184)
def _strength(actor, own_type, river):
    cards = HOLDINGS[actor][own_type] + BOARD + (river,)
    return evaluate([Card(Rank(card[0]), Suit(card[1])) for card in cards])


def leaf_ledger(deal, turn_end, river_end=None, river=None):
    """Independent Fraction ledger, including all six seats' sunk contributions."""
    deal = _deal(deal)
    if not isinstance(turn_end, PublicNode) or not turn_end.terminal:
        raise ValueError("complete_turn_history_required")
    if _at_history(public_tree(), turn_end.history) != turn_end:
        raise ValueError("unknown_turn_terminal")
    turn_wagers = turn_end.wagers
    folded = turn_end.folded
    if folded:
        if river_end is not None or river is not None:
            raise ValueError("fold_has_no_future_chance")
        wagers = turn_wagers
    else:
        if river not in joint_rivers(deal):
            raise ValueError("legal_full_chance_river_required")
        if turn_wagers == (4, 4):
            if river_end is not None:
                raise ValueError("all_in_has_no_river_decision")
            wagers = turn_wagers
        else:
            if (not isinstance(river_end, PublicNode) or not river_end.terminal
                    or _at_history(public_tree(4 - turn_wagers[0]),
                                   river_end.history) != river_end):
                raise ValueError("complete_legal_river_history_required")
            wagers = tuple(turn_wagers[i] + river_end.wagers[i] for i in (0, 1))
            folded = river_end.folded
    matched = min(wagers)
    pot = Fraction(21 + 2 * matched)
    capped = min(pot * Fraction(3, 100), Fraction(4))
    rake = Fraction(capped.numerator // capped.denominator)
    if rake or pot > 29:
        raise ValueError("zero_rake_and_stack_bound")
    if folded:
        winners = (3 - folded,)
    else:
        values = {s: _strength(s, deal[s - 1], river) for s in (1, 2)}
        best = max(values.values())
        winners = tuple(s for s in (1, 2) if values[s] == best)
    refunds = {s: Fraction(0) for s in range(6)}
    returns = {s: Fraction(-v) for s, v in CONTRIBUTIONS.items()}
    for s in (1, 2):
        refunds[s] = Fraction(wagers[s - 1] - matched)
        returns[s] += refunds[s] - wagers[s - 1]
        if s in winners:
            returns[s] += (pot - rake) / len(winners)
    if sum(returns.values()) + rake != 0 or returns[1] + returns[2] != 9:
        raise ValueError("independent_ledger_conservation")
    return dict(returns=returns, refunds=refunds, rake=rake, settled_pot=pot,
                winners=winners, active_wagers=wagers,
                stacks={s: Fraction(10) + value for s, value in returns.items()})


def build_tree(*, deadline=None):
    """Compact exact chance tree: 7653 nodes, 4776 leaves, 1488 infosets."""
    def terminal(deal, turn, river_node=None, card=None):
        check_deadline(deadline)
        ledger = leaf_ledger(deal, turn, river_node, card)
        return Node("terminal", returns=(ledger["returns"][1], ledger["returns"][2]))

    def river_node(deal, turn, node, card):
        check_deadline(deadline)
        if node.terminal:
            return terminal(deal, turn, node, card)
        info = symbolic_infoset(node.actor, deal[node.actor - 1], turn.history,
                                node.history, card)
        return Node("decision", actor=node.actor, infoset=info, children=tuple(
            (action, Fraction(1), river_node(deal, turn, child, card))
            for action, child in node.children))

    def turn_node(deal, node):
        check_deadline(deadline)
        if node.terminal:
            if node.folded:
                return terminal(deal, node)
            children = []
            for card in joint_rivers(deal):
                child = (terminal(deal, node, card=card) if node.wagers == (4, 4)
                         else river_node(deal, node,
                                         public_tree(4 - node.wagers[0]), card))
                children.append((card, Fraction(1, 44), child))
            return Node("chance", children=tuple(children))
        info = symbolic_infoset(node.actor, deal[node.actor - 1], node.history)
        return Node("decision", actor=node.actor, infoset=info, children=tuple(
            (action, Fraction(1), turn_node(deal, child))
            for action, child in node.children))

    return Node("chance", children=tuple(
        ("/".join(deal), Fraction(1, 4), turn_node(deal, public_tree()))
        for deal in DEALS))


def _fraction(value):
    return Fraction(value["numerator"], value["denominator"])


def _audit_decision(arena, deal, turn, river_node, card):
    node = turn if river_node is None else river_node
    obs = arena.observe(arena.actor)
    prior = 0 if river_node is None else turn.wagers[0]
    expected_board = list(BOARD if river_node is None else BOARD + (card,))
    if (arena.actor != node.actor or obs["board"] != expected_board
            or tuple(a.id for a in arena.legal_actions()) != node.actions
            or obs["pot"] != str(21 + 2 * prior + sum(node.wagers))
            or obs["to_call"] != str(max(node.wagers) - node.wagers[node.actor - 1])
            or obs["folded"] != [0, 3, 4, 5]):
        raise ValueError("native_symbolic_decision_mismatch")
    for seat in (1, 2):
        wager = node.wagers[seat - 1]
        if (obs["stacks"][str(seat)] != str(4 - prior - wager)
                or obs["bets"][str(seat)] != str(wager)
                or obs["contributions"][str(seat)] != str(6 + prior + wager)):
            raise ValueError("native_symbolic_stack_or_contribution")
    turn_history = turn.history
    river_history = () if river_node is None else node.history
    expected = symbolic_infoset(node.actor, deal[node.actor - 1], turn_history,
                                river_history, None if river_node is None else card)
    if infoset_id(obs) != expected:
        raise ValueError("native_symbolic_infoset")
    return obs, expected


def audit_native(*, deadline, catalog=None, progress=None):
    """Enumerate both legal dead-card arrangements with honest partial counters.

    Predealt future cards duplicate turn-fold leaves. Report every physical
    terminal plus deduplicated compact leaves, and require the full denominator.
    Call once under an outer job budget; failure never becomes partial PASS.
    """
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("absolute_monotonic_deadline_required")
    started = time.monotonic()
    result = {} if progress is None else progress
    result.update(status="RUNNING", expected_terminal_checks=9552,
                  expected_physical_terminal_checks=11616, native_resets=0,
                  decision_checks=0, physical_terminal_checks=0,
                  unique_terminal_checks=0, infosets=0)
    catalog = {} if catalog is None else catalog
    if catalog:
        raise ValueError("empty_native_catalog_required")
    info_to_key, key_to_info, terminals = {}, {}, set()
    terminal_signatures = {}

    def finish_leaf(arena, deal, turn, river_node, card, arrangement):
        check_deadline(deadline)
        ledger = leaf_ledger(deal, turn, river_node, None if turn.folded else card)
        if not arena.terminal:
            raise ValueError("native_terminal_required")
        actual = arena.terminal_result()
        obs = arena.observe(1)
        if (arena.terminal_returns() != ledger["returns"]
                or _fraction(actual["rake"]) != ledger["rake"]
                or {int(s): _fraction(v) for s, v in actual["refunds"].items()}
                != ledger["refunds"]
                or sum((_fraction(row["amount"]) for row in actual["pots"]),
                       Fraction(0))
                != ledger["settled_pot"]
                or {int(s): Fraction(v) for s, v in obs["stacks"].items()}
                != ledger["stacks"]):
            raise ValueError("native_independent_terminal_ledger_mismatch")
        board = list(BOARD if turn.folded else BOARD + (card,))
        public_actions = tuple(row[2] for row in PREFIX) + turn.history
        if river_node is not None:
            public_actions += river_node.history
        if (obs["board"] != board or tuple(row["id"] for row in
                                           obs["public_history"]) != public_actions):
            raise ValueError("native_terminal_public_history")
        signature = canonical_hash([obs, arena.observe(2)])
        compact_id = (deal, turn.history, None if turn.folded else card,
                      None if river_node is None else river_node.history)
        previous = terminal_signatures.setdefault(compact_id, signature)
        if previous != signature:
            raise ValueError("dead_arrangement_changed_terminal_observation")
        result["physical_terminal_checks"] += 1
        terminal_id = (arrangement,) + compact_id
        terminals.add(terminal_id)
        result["unique_terminal_checks"] = len(terminals)

    def walk(arena, deal, turn, river_node, card, arrangement):
        check_deadline(deadline)
        if turn.terminal and river_node is None:
            if turn.folded or turn.wagers == (4, 4):
                finish_leaf(arena, deal, turn, None, card, arrangement)
                return
            river_node = public_tree(4 - turn.wagers[0])
        node = turn if river_node is None else river_node
        if node.terminal:
            finish_leaf(arena, deal, turn, river_node, card, arrangement)
            return
        obs, info = _audit_decision(arena, deal, turn, river_node, card)
        key = research_key(obs)
        if ((info in info_to_key and info_to_key[info] != key)
                or (key in key_to_info and key_to_info[key] != info)):
            raise ValueError("hidden_information_or_imperfect_recall")
        info_to_key[info], key_to_info[key] = key, info
        previous = catalog.get(key)
        if previous is not None and previous["visible_observation"] != obs:
            raise ValueError("same_key_changed_visible_state")
        catalog[key] = dict(visible_observation=obs, infoset=info)
        result["infosets"] = len(info_to_key)
        result["decision_checks"] += 1
        for action, child in node.children:
            check_deadline(deadline)
            following = arena.clone()
            following.step(action)
            walk(following, deal, child if river_node is None else turn,
                 None if river_node is None else child, card, arrangement)

    try:
        for arrangement in (0, 1):
            for deal in DEALS:
                for card in joint_rivers(deal):
                    check_deadline(deadline)
                    arena = make_root(deal, card, arrangement, deadline=deadline)
                    result["native_resets"] += 1
                    walk(arena, deal, public_tree(), None, card, arrangement)
        check_deadline(deadline)
        if (result["unique_terminal_checks"] != 9552
                or result["physical_terminal_checks"] != 11616
                or result["decision_checks"] != 8448
                or result["native_resets"] != 352 or result["infosets"] != 1488):
            raise ValueError("native_full_denominator_mismatch")
        result["status"] = "PASS"
        return result
    except BaseException as exc:
        result.update(status="STOP_ERROR_OR_BUDGET", error_type=type(exc).__name__,
                      error=str(exc))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["terminal_not_run"] = 9552 - result["unique_terminal_checks"]
