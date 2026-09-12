"""Controlled no-gap swaps of reviewed real card crops; not genuine recorded deals."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from poker_engine.desktop.live import (
    CAPTURE_CARD_LAYOUT, CAPTURE_CARD_PLATFORM, _resource_root, load_calibration,
)
from poker_engine.perceptual.capture.base import Frame, WindowRect
from poker_engine.perceptual.vision.engine import _crop_slot
from poker_engine.perceptual.vision.fused_card_adapter import FusedCardRecognizerAdapter
from poker_engine.perceptual.vision.fused_card_recognizer import (
    FusedCardRecognizer, FusedSlotBuffer, load_card_heads,
)
from poker_engine.perceptual.vision.roi import extract_roi
from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums
from tools.wpk_video_dataset import pixels_digest, read_image


def adapter(heads):
    return FusedCardRecognizerAdapter(FusedCardRecognizer(
        load_card_heads(heads), rank_floor=.5, suit_floor=.3))


def ingest(recognizer, crop, seq):
    begin = getattr(recognizer, "begin_frame", None)
    if callable(begin):
        stamp = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seq / 30)
        begin(seq, stamp,
              "controlled-source", {"hero": "fixed", "board": "fixed"})
    result = recognizer.recognize(crop, ("hero", 0))
    return (str(result.value[0]) if result.value and result.raw_score >= .3 else None)


def prepare(batch: Path, output: Path, heads: Path):
    if output.exists():
        raise ValueError("preserve prior handoff specification")
    spec = json.loads((batch / "corrected-score.json")
                      .read_text(encoding="utf-8"))["corrected_spec"]
    table, vision = load_calibration(CAPTURE_CARD_PLATFORM, CAPTURE_CARD_LAYOUT)
    samples, images, rejected = [], [], []
    seen = set()
    for point in spec["checkpoints"]:
        index = point["source_frame"]
        path = batch / "frames" / f"{index:06d}.png"
        picture = read_image(path)
        if pixels_digest(picture) != point["pixel_sha256"]:
            raise ValueError("reviewed source changed")
        frame = Frame(index, datetime(2000, 1, 1, tzinfo=timezone.utc), "review",
                      WindowRect(0, 0, 498, 1080), picture, 498, 1080)
        for group in ("hero", "board"):
            roi = next(r for r in table.rois if r.kind.value == f"{group}_cards")
            base = extract_roi(frame, roi)
            layout = vision._hero_layout if group == "hero" else vision._board_layout
            for slot, truth in enumerate(point[group]):
                crop = _crop_slot(base, layout.slots[slot])
                digest = pixels_digest(crop)
                if digest in seen:
                    continue
                seen.add(digest)
                reader = adapter(heads)
                predicted = [ingest(reader, crop, seq) for seq in range(8)][-1]
                row = {"source_frame": index, "group": group, "slot": slot,
                       "truth": truth, "source_image_sha256": sha256_file(path),
                       "pixel_sha256": digest, "shape": list(crop.shape)}
                if predicted != truth:
                    rejected.append({**row, "static_prediction": predicted})
                    continue
                samples.append(row)
                images.append(crop)
    pairs = []
    for i, a in enumerate(samples):
        for j, b in enumerate(samples):
            if a["truth"] == b["truth"] or a["shape"] != b["shape"]:
                continue
            if (a["group"], a["slot"]) != (b["group"], b["slot"]):
                continue
            h, w = images[i].shape[:2]
            first = FusedSlotBuffer._sig(images[i], (0, 0, w, h))
            second = FusedSlotBuffer._sig(images[j], (0, 0, w, h))
            diff = float(np.mean(cv2.absdiff(first, second)))
            if diff <= 10:
                pairs.append({"from": i, "to": j, "coarse_signature_difference": diff})
    pairs.sort(key=lambda row: (
        row["coarse_signature_difference"], row["from"], row["to"]))
    output.mkdir(parents=True)
    (output / "crops").mkdir()
    for index, (row, crop) in enumerate(zip(samples, images)):
        path = output / "crops" / f"{index:03d}.png"
        ok, encoded = cv2.imencode(".png", crop)
        if not ok:
            raise ValueError("cannot encode crop")
        path.write_bytes(encoded.tobytes())
        row.update(crop=path.relative_to(output).as_posix(), sha256=sha256_file(path))
    report = {"samples": samples, "pairs": pairs[:12], "eligible_pairs": len(pairs),
              "static_exclusions": rejected, "heads_sha256": sha256_file(heads),
              "selection": ("same slot/shape, different reviewed identities, static "
                            "reads correct; smallest signature differences, limit12"),
              "controlled_splices": True, "independent_validation": False}
    (output / "spec.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_sha256sums(output)
    return {"samples": len(samples), "eligible_pairs": len(pairs),
            "selected": len(pairs[:12]),
            "excluded": len(rejected)}


def prepare_remaining(spec_dir: Path, output: Path):
    """Freeze the other eligible swaps, never changing the initial12 cases."""
    if output.exists():
        raise ValueError("preserve remaining-pair specification")
    source = spec_dir / "spec.json"
    spec = json.loads(source.read_text(encoding="utf-8"))
    used = {(p["from"], p["to"]) for p in spec["pairs"]}
    samples = spec["samples"]
    images = []
    for row in samples:
        path = spec_dir / row["crop"]
        if sha256_file(path) != row["sha256"]:
            raise ValueError("crop changed")
        images.append(read_image(path))
    pairs = []
    for i, a in enumerate(samples):
        for j, b in enumerate(samples):
            if ((i, j) in used or a["truth"] == b["truth"] or a["shape"] != b["shape"]
                    or (a["group"], a["slot"]) != (b["group"], b["slot"])):
                continue
            h, w = images[i].shape[:2]
            first = FusedSlotBuffer._sig(images[i], (0, 0, w, h))
            second = FusedSlotBuffer._sig(images[j], (0, 0, w, h))
            diff = float(np.mean(cv2.absdiff(first, second)))
            if diff <= 10:
                pairs.append({"from": i, "to": j, "coarse_signature_difference": diff})
    output.mkdir(parents=True)
    shutil.copytree(spec_dir / "crops", output / "crops")
    spec.update(pairs=pairs, parent_spec_sha256=sha256_file(source),
                selection=("Remaining eligible pairs, excluded initial12; "
                           "same development images"))
    (output / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    write_sha256sums(output)
    return {"remaining_cases": len(pairs)}


def evaluate(spec_dir: Path, output: Path, heads: Path):
    if output.exists():
        raise ValueError("preserve first evaluation")
    spec_path = spec_dir / "spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if sha256_file(heads) != spec["heads_sha256"]:
        raise ValueError("candidate weights changed")
    crops = []
    for row in spec["samples"]:
        path = spec_dir / row["crop"]
        if sha256_file(path) != row["sha256"]:
            raise ValueError("source crop changed")
        crops.append(read_image(path))
    rows = []
    for pair in spec["pairs"]:
        a, b = pair["from"], pair["to"]
        reader = adapter(heads)
        for seq in range(64):
            before = ingest(reader, crops[a], seq)
        if before != spec["samples"][a]["truth"]:
            raise ValueError("warmup identity is not correct")
        predictions = [ingest(reader, crops[b], 64 + n) for n in range(12)]
        truth = spec["samples"][b]["truth"]
        wrong = sum(p is not None and p != truth for p in predictions)
        rows.append({**pair, "old": before, "new": truth, "predictions": predictions,
                     "wrong_accepted": wrong,
                     "correct_accepted": predictions.count(truth),
                     "abstentions": predictions.count(None)})
    output.mkdir(parents=True)
    repo = _resource_root()
    files = [repo / "src/poker_engine/perceptual/vision" / name for name in (
        "fused_card_adapter.py", "fused_card_recognizer.py")]
    files.append(Path(__file__))
    for path in files:
        shutil.copy2(path, output / path.name)
    report = {"cases": rows, "wrong_accepted": sum(r["wrong_accepted"] for r in rows),
              "spec_sha256": sha256_file(spec_path), "model_source_root": str(repo),
              "source_hashes": {str(p): sha256_file(p) for p in files},
              "controlled_splices": True, "production_calibration_revalidated": False}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_sha256sums(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, help="reviewed development batch")
    parser.add_argument("--remaining-from", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(args.prepare, args.output, args.heads), indent=2))
    elif args.remaining_from:
        print(json.dumps(prepare_remaining(args.remaining_from, args.output), indent=2))
    else:
        print(json.dumps(evaluate(args.spec, args.output, args.heads), indent=2))
