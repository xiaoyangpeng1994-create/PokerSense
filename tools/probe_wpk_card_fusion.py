"""Candidate-only regression probe for the source-bound AcQh video hand.

Explicitly constructs an offline diagnostic recognizer while production card
acceptance remains disabled pending recalibration. Emits no actions or advice.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2

from poker_engine.desktop.live import (
    CAPTURE_CARD_LAYOUT, CAPTURE_CARD_PLATFORM, _resource_root, load_calibration,
)
from poker_engine.perceptual.capture.base import Frame, WindowRect
from poker_engine.perceptual.capture.normalization import NormalizationConfig, normalize
from poker_engine.perceptual.vision.fused_card_adapter import FusedCardRecognizerAdapter
from poker_engine.perceptual.vision.fused_card_recognizer import (
    FusedCardRecognizer, load_card_heads,
)
from tools.verify_wpk_hand_trace import DEFAULT_TRACE, visible_board
from tools.wpk_video_dataset import pixels_digest


def probe(corpus: Path) -> dict:
    trace = json.loads(DEFAULT_TRACE.read_text(encoding="utf-8"))
    inventory = json.loads((corpus / "sources.json").read_text(encoding="utf-8"))
    source = next(s for s in inventory["sources"] if s["session"] == "session_002")
    path = Path(source["path"])
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (source["size_bytes"], source["mtime_ns"]):
        raise ValueError("frozen source changed")
    folder = _resource_root() / "configs/vision" / CAPTURE_CARD_PLATFORM
    calibration = json.loads((folder / "calibration.json").read_text(encoding="utf-8"))
    config = NormalizationConfig.from_json(
        (folder / "normalization.json").read_text(encoding="utf-8"))
    table, vision = load_calibration(CAPTURE_CARD_PLATFORM, CAPTURE_CARD_LAYOUT)
    fused = calibration["card_fused"]
    # Diagnostic instance only. Production's requires_revalidation flag is
    # untouched; callers must not use this helper in live decision routing.
    vision._card = FusedCardRecognizerAdapter(FusedCardRecognizer(
        load_card_heads(folder / "card_heads.npz"),
        rank_floor=fused["rank_floor"], suit_floor=fused["suit_floor"]))
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("cannot open source")
    checkpoints = {8400, 8440, 8500, 8520, 8580, 8600, 8700, 8760}
    counts = Counter()
    checked = []
    try:
        for index in range(8821):
            if index < 8360:
                if not capture.grab():
                    raise ValueError("source ended during sequential skip")
                continue
            ok, raw = capture.read()
            if not ok:
                raise ValueError("source ended in regression window")
            picture = normalize(raw, config)
            frame = Frame(index, datetime.now(timezone.utc), "offline-card-probe",
                          WindowRect(0, 0, 498, 1080), picture, 498, 1080)
            obs = vision.process(frame, table)
            hero = tuple(map(str, obs.hero_cards.value or ()))
            board = tuple(map(str, obs.board_cards.value or ()))
            if index >= 8400:
                status = obs.hero_cards.validation_status.value
                counts[status] += 1
                if status == "valid" and hero != ("Ac", "Qh"):
                    counts["wrong_accepted_hero"] += 1
            if index in checkpoints:
                expected = trace["evidence"][f"f{index}"]["pixel_sha256"]
                if pixels_digest(picture) != expected:
                    raise ValueError("source checkpoint pixel mismatch")
                checked.append({"source_frame": index, "hero": hero, "board": board,
                                "hero_matches": hero == ("Ac", "Qh"),
                                "board_matches": (
                                    board == visible_board(trace, index))})
    finally:
        capture.release()
    return {"candidate_only": True, "production_calibration_revalidated": False,
            "source_sha256": source["sha256"],
            "evaluated_identity_interval": [8400, 8820],
            "hero_status_counts": dict(counts), "checkpoints": checked,
            "passed": all(r["hero_matches"] and r["board_matches"] for r in checked)
            and not counts["wrong_accepted_hero"],
            "scope": "one manually reviewed hand; not independent full-corpus accuracy"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()
    report = probe(args.corpus)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
