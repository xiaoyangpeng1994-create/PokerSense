"""WePoker-style synthetic table renderer + training-dataset generator.

Generates *synthetic* WePoker-like table screenshots with FULL, exact ground
truth (hero cards, board cards, street, pot, bet, stack, action) and writes a
JSON manifest aligned with the real-adapter Golden format (small.json). Pure
rendering — no real platform is contacted or scraped.

Purpose: feed the Vision recognizers offline training/eval data in a WePoker
UI style, so the project can tune recognition without touching GGPoker/WePoker
(auto-scraping real platforms is out of scope by design).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# Rank -> single-character label for rendering (10 -> "T").
_RANK_LABELS = {
    "A": "A", "K": "K", "Q": "Q", "J": "J", "T": "T",
    "9": "9", "8": "8", "7": "7", "6": "6", "5": "5",
    "4": "4", "3": "3", "2": "2",
}
# Suit -> label character for rendering. Use the SAME letters as the recognizer
# templates (S/H/D/C) so the recognizer can read what we render (the whole
# point of the end-to-end self-check).
_SUIT_SYMBOLS = {"S": "S", "H": "H", "D": "D", "C": "C"}


def _render_card(rank: str, suit: str, w: int = 60, h: int = 84) -> np.ndarray:
    """Render a white card with black rank + suit letters (recognizer-matching).

    Both rank and suit are drawn in BLACK, identical to the recognizer's
    templates (which use `render_card` with black text). This lets the
    recognizer read exactly what we render — the end-to-end self-check.
    """
    import cv2

    img = np.full((h, w, 3), 255, dtype=np.uint8)
    label = _RANK_LABELS.get(rank, "?")
    suit_sym = _SUIT_SYMBOLS.get(suit, "?")
    cv2.putText(img, label, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, suit_sym, (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return img


def _render_empty_slot(w: int = 60, h: int = 84) -> np.ndarray:
    """Render an empty (no card) position as a dark patch."""
    return np.full((h, w, 3), 40, dtype=np.uint8)


def _draw_text(canvas, text, x, y, color=(255, 255, 255), scale=0.6):
    import cv2

    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def _draw_amount(canvas, txt, x0, y0, w=120, h=40):
    """Fill a white patch and draw black digits (aligns with amount recognizer)."""
    import cv2

    canvas[y0:y0 + h, x0:x0 + w] = 255
    for i, ch in enumerate(txt):
        cv2.putText(canvas, ch, (x0 + 4 + i * 20, y0 + h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def render_table(
    board_cards: tuple[str, ...],
    hero_cards: tuple[str, ...],
    pot: str,
    bet: str,
    stacks: tuple[str, ...],
    actions: tuple[str, ...],
    w: int = 600,
    h: int = 400,
) -> np.ndarray:
    """Render a WePoker-style table frame aligned to the stock 600x400 layout.

    Positions mirror ``tools/run_benchmark.build_frame`` so the existing
    ``build_engine``/``table_map`` can recognize this output unchanged (this is
    the whole point: the synthetic generator feeds the recognizer pipeline).
    Only the color scheme is WePoker-flavored (dark navy felt, gold pot text).
    Purely synthetic — no real platform asset is used.
    """
    # same felt background as build_frame (so occupancy evidence matches)
    canvas = np.full((h, w, 3), 60, dtype=np.uint8)

    def px(x, y):
        return int(x * w), int(y * h)

    # board: render only the present cards; empty slots stay at the felt
    # background (exactly like build_frame, so occupancy evidence matches).
    for i, c in enumerate(board_cards):
        x0, y0 = px(i * 0.2 + 0.01, 0.03)
        img = _render_card(c[0], c[1])
        canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img

    # hero: 2 cards bottom-left (same as build_frame)
    for i, c in enumerate(hero_cards):
        x0, y0 = px(0.08 + i * 0.2, 0.73)
        img = _render_card(c[0], c[1])
        canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img

    # pot + bet (same ROI), WePoker-gold text on white patch
    _draw_amount(canvas, pot, *px(0.40, 0.45), w=120, h=40)
    _draw_amount(canvas, bet, *px(0.40, 0.55), w=120, h=40)

    # stacks (same ROI)
    stack_xs = (0.55, 0.70, 0.85)
    for i, txt in enumerate(stacks):
        _draw_amount(canvas, txt, *px(stack_xs[i], 0.68), w=72, h=32)

    # actions (same ROI)
    act_xs = (0.55, 0.75)
    for i, txt in enumerate(actions):
        x0, y0 = px(act_xs[i], 0.80)
        canvas[y0:y0 + 24, x0:x0 + 90] = 255
        _draw_text(canvas, txt, x0 + 4, y0 + 18, (0, 0, 0), 0.5)

    return canvas


def _random_scenario(rng: np.random.Generator):
    """Generate one random scenario with known GT (deterministic from rng)."""
    ranks = list("AKQJT98765432")
    suits = "SHDC"

    def rand_card():
        r = rng.integers(0, len(ranks))
        s = rng.integers(0, 4)
        return ranks[r] + suits[s]

    street = ("PREFLOP", "FLOP", "TURN", "RIVER")[rng.integers(0, 4)]
    board_n = {"PREFLOP": 0, "FLOP": 3, "TURN": 4, "RIVER": 5}[street]
    board = tuple(rand_card() for _ in range(board_n))
    hero = tuple(rand_card() for _ in range(2))
    pot = str(int(rng.integers(1, 500)))
    bet = str(int(rng.integers(1, 200)))
    stacks = tuple(str(int(rng.integers(20, 1000))) for _ in range(3))
    action_pool = ("CHECK", "CALL", "BET", "RAISE", "FOLD")
    actions = tuple(action_pool[int(rng.integers(0, len(action_pool)))]
                    for _ in range(2))
    return board, hero, street, pot, bet, stacks, actions


def generate_dataset(out_dir: str, count: int, seed: int = 0) -> str:
    """Generate ``count`` WePoker-style frames + a small.json-aligned manifest.

    Returns the path to the written manifest JSON.
    """
    import cv2

    rng = np.random.default_rng(seed)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    samples = []
    for i in range(count):
        board, hero, street, pot, bet, stacks, actions = _random_scenario(rng)
        img = render_table(board, hero, pot, bet, stacks, actions)
        img_path = os.path.join(frames_dir, f"sample_{i}.png")
        cv2.imwrite(img_path, img)

        samples.append({
            "image_path": img_path,
            "platform_id": "wpk",
            "layout_id": "6max",
            "ground_truth": {
                "street": street,
                "pot": pot,
                "bet_size": bet,
                "hero_cards": list(hero),
                "board_cards": list(board),
                "stack": list(stacks),
                "action": list(actions),
            },
            "source": "synthetic-render",
            "sample_id": f"wepoker-{i}",
        })

    manifest = {
        "dataset": "wepoker-synthetic",
        "note": ("Synthetic WePoker-style table frames for recognizer training/"
                 "eval. NOT real platform data; does NOT represent real-world "
                 "accuracy. Auto-scraping real platforms is out of scope."),
        "samples": samples,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate WePoker-style synthetic training data"
    )
    parser.add_argument("--out", default="datasets/wepoker")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest_path = generate_dataset(args.out, args.count, args.seed)
    n = args.count
    print(f"generated {n} frames -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
