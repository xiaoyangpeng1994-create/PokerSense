"""Private temporal-card diagnostics with unchanged ingestion results."""

import argparse
from collections import Counter
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
from poker_engine.perceptual.capture.normalization import NormalizationConfig, normalize
from poker_engine.perceptual.vision.fused_card_adapter import FusedCardRecognizerAdapter
from poker_engine.perceptual.vision.fused_card_recognizer import (
    FusedCardRecognizer, FusedSlotBuffer, load_card_heads,
)
from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)
from tools.validate_wpk_card_batch import RecordingRecognizer, score_slots
from tools.wpk_video_dataset import pixels_digest, read_image


def review(window: Path):
    output = window / "card-sheets"
    output.mkdir(exist_ok=False)
    samples = json.loads((window / "samples.json").read_text(encoding="utf-8"))
    for start in range(0, len(samples), 32):
        sheet = np.full((8 * 226, 4 * 300, 3), 235, np.uint8)
        for pos, row in enumerate(samples[start:start + 32]):
            pic = read_image(window / row["image"])
            x, y = (pos % 4) * 300, (pos // 4) * 226
            cv2.putText(sheet, str(row["source_frame"]), (x + 5, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
            sheet[y + 25:y + 115, x + 5:x + 290] = pic[470:560, 105:390]
            sheet[y + 120:y + 220, x + 5:x + 130] = pic[918:1018, 185:310]
        ok, encoded = cv2.imencode(".png", sheet)
        if not ok:
            raise ValueError("review sheet encoding failed")
        (output / f"page-{start // 32:02d}.png").write_bytes(encoded.tobytes())


def replay(corpus: Path, window: Path, output: Path, heads_path: Path):
    output.mkdir(parents=True, exist_ok=False)
    spec = json.loads((window / "visual-review.json").read_text(encoding="utf-8"))
    summary = json.loads((window / "summary.json").read_text(encoding="utf-8"))
    start, end = summary["requested_frame_window"]
    sources = json.loads((corpus / "sources.json").read_text(encoding="utf-8"))
    source = next(row for row in sources["sources"] if row["session"] == "session_002")
    if summary["source_sha256"] != source["sha256"]:
        raise ValueError("window/source mismatch")
    video = Path(source["path"])
    stat = video.stat()
    if (stat.st_size, stat.st_mtime_ns) != (source["size_bytes"], source["mtime_ns"]):
        raise ValueError("source changed")
    repo = _resource_root()
    folder = repo / "configs/vision" / CAPTURE_CARD_PLATFORM
    paths = list((repo / "src").rglob("*.py"))
    paths += [p for p in (repo / "configs").rglob("*") if p.is_file()]
    hashes = {}
    for path in paths:
        rel = path.relative_to(repo)
        target = output / "model-snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        hashes[rel.as_posix()] = sha256_file(path)
        shutil.copy2(path, target)
        if sha256_file(target) != hashes[rel.as_posix()]:
            raise ValueError("source changed while copying")
    frozen = {"source_sha256": source["sha256"], "model_files": hashes,
              "model_source_root": str(repo),
              "heads_sha256": sha256_file(heads_path), "rank_floor": .5,
              "heads_metadata_sha256": sha256_file(heads_path.with_suffix(".json")),
              "review_sha256": sha256_file(window / "visual-review.json"),
              "probe_sha256": sha256_file(Path(__file__))}
    (output / "frozen.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), output / "wpk_transition_probe.py")
    shutil.copy2(heads_path, output / "card_heads.npz")
    shutil.copy2(heads_path.with_suffix(".json"), output / "card_heads.json")
    table, vision = load_calibration(CAPTURE_CARD_PLATFORM, CAPTURE_CARD_LAYOUT)
    adapter = FusedCardRecognizerAdapter(FusedCardRecognizer(
        load_card_heads(heads_path), rank_floor=.5, suit_floor=.3))
    recorder = RecordingRecognizer(adapter, floor=.3)
    vision._card = recorder
    config = NormalizationConfig.from_json((folder / "normalization.json")
                                           .read_text(encoding="utf-8"))
    points = {r["source_frame"]: r for r in spec["checkpoints"]}
    original = FusedSlotBuffer.ingest

    def instrument(buffer, image):
        accepted = original(buffer, image)
        buffer._probe_fresh = accepted
        return accepted

    FusedSlotBuffer.ingest = instrument
    capture = cv2.VideoCapture(str(video))
    counts, checked = Counter(), []
    try:
        with (output / "frames.jsonl").open("w", encoding="utf-8") as trace:
            for index in range(end + 1):
                if index < start:
                    if not capture.grab():
                        raise ValueError("source ended during prefix")
                    continue
                ok, raw = capture.read()
                if not ok:
                    raise ValueError("source ended early")
                picture = normalize(raw, config)
                timestamp = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(
                    milliseconds=capture.get(cv2.CAP_PROP_POS_MSEC))
                recorder.reads.clear()
                obs = vision.process(Frame(index, timestamp, "offline",
                                           WindowRect(0, 0, 498, 1080), picture,
                                           498, 1080), table)
                row = {"source_frame": index, "hero": [], "board": []}
                for group, width in (("hero", 2), ("board", 5)):
                    for slot in range(width):
                        result = dict(recorder.reads.get((group, slot), {"card": None}))
                        buffer = adapter._buffers.get((group, slot))
                        fresh = bool(buffer is not None
                                     and getattr(buffer, "_probe_fresh", False))
                        result.update(fresh=fresh, glyph_count=buffer.glyph_count
                                      if buffer is not None else 0)
                        row[group].append(result)
                        if result["card"] is not None and not fresh:
                            counts["accepted_without_current_glyph"] += 1
                row["observation_status"] = {
                    "hero": obs.hero_cards.validation_status.value,
                    "board": obs.board_cards.validation_status.value}
                trace.write(json.dumps(row) + "\n")
                if index in points:
                    point = points[index]
                    expected_image = window / "frames" / f"{index:06d}.png"
                    if sha256_file(expected_image) != point["image_sha256"]:
                        raise ValueError("visually reviewed image changed")
                    if pixels_digest(picture) != pixels_digest(
                        read_image(expected_image),
                    ):
                        raise ValueError("source checkpoint pixels changed")
                    scored = {"source_frame": index}
                    for group, width in (("hero", 2), ("board", 5)):
                        values = [r["card"] for r in row[group]]
                        scored[group] = score_slots(point[group], values, width)
                        counts.update({f'{group}_{k}': v for k, v
                                       in scored[group]["counts"].items()})
                    checked.append(scored)
                if index % 200 == 0:
                    print(json.dumps({"frame": index}), flush=True)
    finally:
        capture.release()
        FusedSlotBuffer.ingest = original
    for rel, digest in hashes.items():
        if sha256_file(repo / rel) != digest:
            raise ValueError("model changed during probe")
    if sha256_file(heads_path) != frozen["heads_sha256"]:
        raise ValueError("heads changed during probe")
    if (sha256_file(heads_path.with_suffix(".json")) != frozen["heads_metadata_sha256"]
            or sha256_file(window / "visual-review.json") != frozen["review_sha256"]):
        raise ValueError("metadata/review changed during probe")
    after = video.stat()
    if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise ValueError("video changed during probe")
    if len(checked) != len(points):
        raise ValueError("not every visual checkpoint was scored")
    report = {"counts": dict(counts), "checkpoints": checked,
              "continuous_frames": end - start + 1, "independent_validation": False,
              "production_calibration_revalidated": False}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_sha256sums(output)
    return report


def rescore_trace(window: Path, run: Path, output: Path):
    """Score later pixel review against preserved predictions, without inference."""
    if output.exists():
        raise ValueError("preserve earlier rescore")
    if verify_sha256sums(run):
        raise ValueError("saved inference artifacts changed")
    review_path = window / "visual-review.json"
    review_data = json.loads(review_path.read_text(encoding="utf-8"))
    frozen = json.loads((run / "frozen.json").read_text(encoding="utf-8"))
    summary = json.loads((window / "summary.json").read_text(encoding="utf-8"))
    if summary["source_sha256"] != frozen["source_sha256"]:
        raise ValueError("review and inference refer to different videos")
    trace_path = run / "frames.jsonl"
    traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8")
              .splitlines()]
    by_frame = {row["source_frame"]: row for row in traces}
    if len(by_frame) != len(traces):
        raise ValueError("duplicate trace frames")
    rows, counts = [], Counter()
    for point in review_data["checkpoints"]:
        index = point["source_frame"]
        image = window / "frames" / f"{index:06d}.png"
        if sha256_file(image) != point["image_sha256"]:
            raise ValueError("reviewed pixels changed")
        trace = by_frame[index]
        scored = {"source_frame": index, "phase": point.get("phase")}
        for group, width in (("hero", 2), ("board", 5)):
            predictions = [r["card"] for r in trace[group]]
            scored[group] = score_slots(point[group], predictions, width)
            scored[group]["predictions"] = predictions
            counts.update({f'{group}_{k}': v for k, v
                           in scored[group]["counts"].items()})
        scored["hero_observation_status"] = trace["observation_status"]["hero"]
        rows.append(scored)
    report = {"new_inference": False, "independent_validation": False,
              "review_sha256": sha256_file(review_path),
              "trace_sha256": sha256_file(trace_path), "counts": dict(counts),
              "checkpoints": rows}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--review-sheets", action="store_true")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--heads", type=Path)
    parser.add_argument("--rescore-trace", type=Path)
    args = parser.parse_args()
    if args.review_sheets:
        review(args.window)
    elif args.rescore_trace:
        if not args.output:
            parser.error("rescore requires an output file")
        print(json.dumps(rescore_trace(args.window, args.rescore_trace, args.output)
                         ["counts"], indent=2))
    else:
        if not all((args.corpus, args.output, args.heads)):
            parser.error("replay requires --corpus, --output and --heads")
        print(json.dumps(replay(args.corpus, args.window, args.output, args.heads)
                         ["counts"], indent=2))
