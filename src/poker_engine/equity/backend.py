"""Optional open-source evaluation acceleration behind a comparable score."""

from poker_engine.equity.fast_evaluator import evaluate_fast


def select_evaluator(name: str = "auto"):
    if name not in ("auto", "python", "phevaluator"):
        raise ValueError("evaluator must be auto, python or phevaluator")
    if name != "python":
        try:
            from phevaluator import evaluate_cards
        except ImportError:
            if name == "phevaluator":
                raise RuntimeError("install the solver-tools extra for phevaluator")
        else:
            def evaluate(cards):
                # PH ranks 1 as strongest. Negate to retain higher-is-better
                # comparison semantics; never compare scores across backends.
                return (-evaluate_cards(*(str(card) for card in cards)),)
            return evaluate, "phevaluator"
    return evaluate_fast, "python-direct"
