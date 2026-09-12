"""Uncalibrated single-frame WPK-head transfer diagnostic on AA pixels.

No temporal or release claim: each slot gets a fresh one-image buffer. The
review is used only for scoring after predictions, never as model input.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from poker_engine.perceptual.vision.fused_card_recognizer import (
    FusedCardRecognizer, FusedSlotBuffer, load_card_heads,
)
from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)
from tools.wpk_video_dataset import read_image
from tools.aa_card_visibility import face_card_support, locate_face_card


def run(window, truth_dir, heads, output, *, face_gate=False, dynamic_board=False):
    face_gate = face_gate or dynamic_board
    if verify_sha256sums(window) or verify_sha256sums(truth_dir):
        raise ValueError("input integrity failure")
    samples = json.loads((window / "samples.json").read_text())
    truth = json.loads((truth_dir / "truth.json").read_text())
    if truth["samples_sha256"] != sha256_file(window / "samples.json"):
        raise ValueError("review is not bound to this window")
    head_hash = sha256_file(heads)
    model = FusedCardRecognizer(load_card_heads(heads), rank_floor=.5, suit_floor=.3)
    predictions = {}
    for sample in samples["samples"]:
        path = (window / sample["file"]).resolve()
        if not path.is_relative_to(window.resolve()):
            raise ValueError("sample escapes window")
        image = read_image(path)
        row = {}
        for group, xs, y in (("hero", (193, 250), 938),
                             ("board", (110, 166, 223, 279, 335), 473)):
            values, scores, reasons = [], [], []
            for x in xs:
                rect = (x, y, 53, 78)
                if dynamic_board and group == "board":
                    rect, reason = locate_face_card(image, rect)
                    supported = rect is not None
                else:
                    supported, reason = face_card_support(image, rect)
                reasons.append(reason if face_gate else "ungated_transfer_baseline")
                if face_gate and not supported:
                    values.append(None)
                    scores.append(0.)
                    continue
                rx, ry, rw, rh = rect
                patch = image[ry:ry + rh, rx:rx + rw]
                buffer = FusedSlotBuffer((0, 0, 53, 78), min_glyphs=1)
                buffer.ingest(patch)
                result = model.recognize(buffer)
                values.append(str(result.value[0]) if result.value else None)
                scores.append(result.raw_score)
            row[group], row[group + "_scores"] = values, scores
            row[group + "_visibility"] = reasons
        predictions[sample["source_frame"]] = row
    counts = Counter()
    comparisons = []
    for checkpoint in truth["checkpoints"]:
        frame = checkpoint["frame"]
        for group, capacity in (("hero", 2), ("board", 5)):
            expected = checkpoint[group]
            if group == "board" and expected is None:
                continue  # Moving/occluded board is not an empty-board label.
            expected = (expected or []) + [None] * (capacity - len(expected or []))
            pairs = zip(expected, predictions[frame][group])
            for slot, (want, got) in enumerate(pairs):
                if want is None:
                    verdict = "negative_rejected" if got is None else "false_positive"
                else:
                    verdict = ("correct" if got == want else
                               "abstain" if got is None else "wrong_identity")
                counts[verdict] += 1
                comparisons.append({"frame": frame, "group": group, "slot": slot,
                                    "expected": want, "predicted": got,
                                    "verdict": verdict})
    if head_hash != sha256_file(heads):
        raise ValueError("model changed during diagnostic")
    output.mkdir(parents=True, exist_ok=False)
    report = {"heads_sha256": head_hash, "source_sha256": samples["source_sha256"],
              "truth_sha256": sha256_file(truth_dir / "truth.json"),
              "method": "fresh single-image buffer; no temporal accumulation",
              "rank_floor": .5, "suit_floor": .3, "counts": dict(counts),
              "face_gate": face_gate,
              "dynamic_board": dynamic_board,
              "comparisons": comparisons, "predictions": predictions,
              "release_eligible": False, "independent_holdout": False}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    write_sha256sums(output)
    print(json.dumps(dict(counts)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("window", "truth", "heads", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--face-gate", action="store_true")
    parser.add_argument("--dynamic-board", action="store_true")
    args = parser.parse_args()
    run(args.window, args.truth, args.heads, args.output,
        face_gate=args.face_gate or args.dynamic_board,
        dynamic_board=args.dynamic_board)
