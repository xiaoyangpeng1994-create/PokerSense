"""AA8 active Hero controls including the two-control short-all-in layout."""

from tools.aa_card_visibility import face_card_support
from tools.aa_hero_turn import hero_turn_candidate as three_control_candidate


def hero_turn_candidate(image):
    original = three_control_candidate(image)
    if original.get("hero_turn") is True:
        return {**original, "control_layout": "three_active_controls",
                "strategy_eligible": False}
    fractions = original.get("button_fractions")
    if fractions is None:
        return {**original, "control_layout": None, "strategy_eligible": False}
    holes = all(face_card_support(image, rect)[0]
                for rect in ((193, 938, 53, 78), (250, 938, 53, 78)))
    red, blue, green = fractions
    # Short callers have no raise control: a central avatar replaces blue.
    # Gray preselection controls cannot satisfy either red or green evidence.
    supported = red >= .55 and green >= .55 and blue < .20 and holes
    return {"hero_turn": True if supported else None,
            "button_fractions": fractions,
            "control_layout": "two_active_controls" if supported else None,
            "reason": "fold_and_call_controls_with_face_cards_candidate" if supported
            else "insufficient_turn_evidence", "strategy_eligible": False,
            "action_text_semantics_verified": False}
