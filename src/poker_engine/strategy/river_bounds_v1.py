"""Distribution-free river payoff bounds under explicit pot-eligibility assumptions.

Not an opponent policy or a win-rate estimate. A counterexample hand proves a
zero lower bound; otherwise all legal single-opponent holdings are checked.
For multiple opponents, assuming all can tie gives a conservative share floor,
even when blockers make that many simultaneous ties impossible.
"""

from collections import Counter
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from functools import lru_cache
from itertools import combinations

from poker_engine.core.enums import Rank, Suit
from poker_engine.core.value_objects import Card
from poker_engine.equity.evaluator import evaluate


def _cards(values, count):
    if (not isinstance(values, (tuple, list)) or len(values) != count
            or any(not isinstance(v, str) or len(v) != 2 for v in values)):
        raise ValueError("完整且唯一的两张手牌和五张河牌必需")
    return tuple(Card(Rank(v[0]), Suit(v[1])) for v in values)


def _money(value):
    if not isinstance(value, str) or len(value) > 40:
        raise ValueError("金额必须为明确十进制字符串")
    try:
        result = Decimal(value)
    except InvalidOperation:
        raise ValueError("金额格式无效") from None
    if (not result.is_finite() or not 0 <= result <= Decimal("1e12")
            or result.as_tuple().exponent < -8):
        raise ValueError("金额超出范围")
    return Fraction(result)


def _encoded(number):
    with localcontext() as context:
        context.prec = 24
        return {"exact": str(number),
                "decimal": str(Decimal(number.numerator) / Decimal(number.denominator))}


def tying_opponents_upper_bound(holdings):
    """A valid vertex cover bounds the number of card-disjoint tying holdings.

The greedy cover need not be minimum: a larger cover only weakens our payoff
lower bound. The number of distinct available cards provides another upper bound.
"""
    edges = {frozenset(holding) for holding in holdings}
    if not edges:
        return 0
    vertices = set().union(*edges)
    cover = 0
    while edges:
        counts = Counter(card for edge in edges for card in edge)
        chosen = max(counts, key=lambda card: (counts[card], repr(card)))
        edges = {edge for edge in edges if chosen not in edge}
        cover += 1
    return min(cover, len(vertices) // 2)


@lru_cache(maxsize=128)
def _strength_bound(hero_codes, board_codes):
    hero, board = _cards(hero_codes, 2), _cards(board_codes, 5)
    if len(set(hero + board)) != 7:
        raise ValueError("牌面重复")
    hero_strength = evaluate(hero + board)
    deck = tuple(Card(Rank(rank), Suit(suit)) for rank in "23456789TJQKA"
                 for suit in "cdhs" if rank + suit not in hero_codes + board_codes)
    examined, tying = 0, []
    for holding in combinations(deck, 2):
        examined += 1
        strength = evaluate(holding + board)
        if strength > hero_strength:
            return {"unbeaten": False, "tie_possible": None,
                    "combos_examined": examined, "exhaustive": False}
        if strength == hero_strength:
            tying.append(holding)
    return {"unbeaten": True, "tie_possible": bool(tying),
            "tying_opponents_upper_bound": tying_opponents_upper_bound(tying),
            "combos_examined": examined, "exhaustive": True}


def river_payoff_bounds(hero_cards, board_cards, pot_before, call_cost,
                        opponents, *, max_hero_deduction=None):
    """Conditional on full displayed-pot eligibility and no further actions.

``max_hero_deduction`` is an explicitly supplied upper bound on ALL extra money
deducted from Hero's return, not an inferred rake percentage. Unknown stays null.
"""
    if type(opponents) is not int or not 1 <= opponents <= 7:
        raise ValueError("争池对手人数必须为1–7")
    pot, cost = _money(pot_before), _money(call_cost)
    if cost <= 0:
        raise ValueError("需要正的跟注金额")
    _cards(hero_cards, 2)
    _cards(board_cards, 5)
    strength = _strength_bound(tuple(sorted(hero_cards)), tuple(sorted(board_cards)))
    tie_limit = min(opponents, strength.get("tying_opponents_upper_bound", opponents))
    share = Fraction(1, tie_limit + 1) if strength["unbeaten"] else Fraction(0)
    lower = (pot + cost) * share - cost
    fee = _money(max_hero_deduction) if max_hero_deduction is not None else None
    return {"status": "COMPLETE_CONDITIONAL_BOUND", "opponents": opponents,
            "share_floor": _encoded(share), "call_gross_lower": _encoded(lower),
            "fold_reference": _encoded(Fraction(0)),
            "call_net_lower": _encoded(lower - fee) if fee is not None else None,
            "max_hero_deduction_for_nonnegative_bound": _encoded(lower)
            if lower >= 0 else None, "strength_evidence": strength,
            "assumptions": ["hero_eligible_for_all_displayed_pots",
                            "full_call_closes_all_actions",
                            "known_cards_and_price_are_correct",
                            "fractional_equal_split_without_odd_chip_effects"],
            "range_distribution_assumed": False, "win_probability": None,
            "strategy_eligible": False, "advice_emitted": False}
