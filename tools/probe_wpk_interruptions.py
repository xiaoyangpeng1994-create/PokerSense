"""Inject lifecycle faults into a reviewed real frame; not actual device outages."""

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from poker_engine.desktop.live import (
    CAPTURE_CARD_LAYOUT, CAPTURE_CARD_PLATFORM, _resource_root, load_calibration,
)
from poker_engine.perceptual.capture.base import Frame, WindowRect
from poker_engine.perceptual.vision.fused_card_adapter import FusedCardRecognizerAdapter
from poker_engine.perceptual.vision.fused_card_recognizer import (
    FusedCardRecognizer, load_card_heads,
)
from poker_engine.perceptual.vision.table_map import ROIKind
from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums
from tools.validate_wpk_card_batch import RecordingRecognizer
from tools.wpk_video_dataset import read_image


def probe(window: Path, heads: Path, output: Path):
    if output.exists():
        raise ValueError("preserve previous injection result")
    review_path = window / "visual-review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    point = next(p for p in review["checkpoints"] if p["source_frame"] == 11340)
    image_path = window / "frames/011340.png"
    if sha256_file(image_path) != point["image_sha256"]:
        raise ValueError("reviewed image changed")
    image = read_image(image_path)
    cases = []
    for name in ("sequence_gap", "timestamp_gap", "source_change", "roi_missing",
                 "duplicate_frame", "explicit_reset"):
        table, vision = load_calibration(CAPTURE_CARD_PLATFORM, CAPTURE_CARD_LAYOUT)
        adapter = FusedCardRecognizerAdapter(FusedCardRecognizer(
            load_card_heads(heads), rank_floor=.5, suit_floor=.3))
        recorder = RecordingRecognizer(adapter, .3)
        vision._card = recorder
        records = []

        def observe(seq, seconds, source="phone", current_table=table):
            timestamp = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(
                seconds=seconds)
            recorder.reads.clear()
            vision.process(Frame(seq, timestamp, source, WindowRect(0, 0, 498, 1080),
                                 image, 498, 1080), current_table)
            cards = [recorder.reads.get(("hero", i), {}).get("card") for i in range(2)]
            records.append({"frame_seq": seq, "simulated_time_s": seconds,
                            "source": source, "hero": cards,
                            "glyph_counts": [adapter._buffers[("hero", i)].glyph_count
                                             if ("hero", i) in adapter._buffers else 0
                                             for i in range(2)]})
            return cards

        for seq in range(3):
            observe(seq, seq / 30)
        if records[-1]["hero"] != point["hero"]:
            raise ValueError("warmup did not accept reviewed hero")
        first, clock, source = 3, .1, "phone"
        if name == "sequence_gap":
            first = 10
        elif name == "timestamp_gap":
            clock = 5.0
        elif name == "source_change":
            source = "other-phone"
        elif name == "duplicate_frame":
            first, clock = 2, 2 / 30
        elif name == "explicit_reset":
            reset = getattr(vision, "reset_temporal", None)
            if callable(reset):
                reset()
        elif name == "roi_missing":
            missing = replace(table, rois=tuple(
                r for r in table.rois if r.kind is not ROIKind.HERO_CARDS))
            observe(first, clock, current_table=missing)
            first, clock = 4, 4 / 30
        recovery = [observe(first + i, clock + i / 30, source) for i in range(3)]
        passed = recovery[:2] == [[None, None], [None, None]]
        passed = passed and recovery[2] == point["hero"]
        cases.append({"scenario": name, "passed": passed, "records": records})
    repo = _resource_root()
    evidence_files = [
        repo / "src/poker_engine/perceptual/vision/fused_card_adapter.py",
        repo / "src/poker_engine/perceptual/vision/engine.py", heads,
        heads.with_suffix(".json"), review_path, image_path, Path(__file__),
    ]
    report = {"injected_faults": True, "real_outage_evidence": False,
              "same_reviewed_image_repeated": True, "expected_hero": point["hero"],
              "model_source_root": str(repo),
              "source_files": {str(p): sha256_file(p) for p in evidence_files},
              "cases": cases, "passed": all(case["passed"] for case in cases),
              "production_calibration_revalidated": False}
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_sha256sums(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = probe(args.window, args.heads, args.output)
    print(json.dumps({"passed": report["passed"], "cases": [
        {"scenario": c["scenario"], "passed": c["passed"]} for c in report["cases"]]},
        indent=2))
