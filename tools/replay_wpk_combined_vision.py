"""Sequential raw-video replay of combined visual candidates, never strategy."""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import time

import cv2

from poker_engine.perceptual.capture.base import Frame, WindowRect
from poker_engine.perceptual.capture.normalization import NormalizationConfig, normalize
from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums
from tools.probe_wpk_observation_fields import score_read
from tools.wpk_combined_vision import from_artifacts
from tools.wpk_video_dataset import pixels_digest
from tools.validate_wpk_card_batch import score_slots


def run(corpus, bank, output, start, end, *, review_path=None, card_review_path=None):
    if output.exists() or not 0 <= start <= end:
        raise ValueError("new output and valid frame interval required")
    repo = Path(__file__).resolve().parents[1]
    inventory = json.loads((corpus / "sources.json").read_text())
    source = next(s for s in inventory["sources"]
                  if s["session"] == "session_002")
    video_path = Path(source["path"])
    before_stat = video_path.stat()
    if (before_stat.st_size, before_stat.st_mtime_ns) != (
        source["size_bytes"], source["mtime_ns"]
    ):
        raise ValueError("source video changed")
    profile = repo / "configs/vision/wepoker_android_capture_card/normalization.json"
    config = NormalizationConfig.from_json(profile.read_text())
    truth_path = review_path or repo / (
        "tests/fixtures/wpk_reference_hands/aq_observation_v1.json")
    truth_path = truth_path.resolve()
    truth = json.loads(truth_path.read_text())
    card_path = card_review_path or repo / (
        "tests/fixtures/wpk_reference_hands/combined_card_check_v1.json")
    card_path = card_path.resolve()
    card_truth = json.loads(card_path.read_text())
    evidence_path = corpus / "hands/aq_allin_8240_8900/samples.json"
    evidence = json.loads(evidence_path.read_text())
    anchors = {s["source_frame"]: s["pixel_sha256"] for s in evidence}
    cash = json.loads((corpus / "settlement/credit_8780_8845/samples.json").read_text())
    anchors.update({s["source_frame"]: s["pixel_sha256"] for s in cash})
    anchors.update({p["source_frame"]: p["pixel_sha256"] for p in truth["checkpoints"]
                    if "pixel_sha256" in p})
    check_ids = {p["source_frame"] for p in truth["checkpoints"]
                 if start <= p["source_frame"] <= end}
    output.mkdir(parents=True)
    inputs = list((repo / "src").rglob("*.py")) + list((repo / "tools").rglob("*.py"))
    inputs += [p for p in (repo / "configs").rglob("*") if p.is_file()]
    inputs += [truth_path, card_path]
    hashed = {}
    for path in inputs:
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(repo)
        digest = sha256_file(path)
        destination = output / "source-snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if sha256_file(destination) != digest:
            raise ValueError("snapshot changed")
        hashed[str(path)] = digest
    artifact_paths = [bank / name for name in
                      ("inventory.json", "visual-review.json", "proposals.npz")]
    pot_bank = corpus / "gray_amount/session001_pot_proposals_v2"
    artifact_paths += [pot_bank / name for name in
                       ("inventory.json", "visual-review.json", "proposals.npz")]
    artifact_paths += [corpus / "training/rank_v6_candidate_01" / name
                       for name in ("card_heads.npz", "card_heads.json")]
    artifact_paths += [corpus / "hands/aq_allin_8240_8900/frames" / f"{i:06d}.png"
                       for i in (8500, 8700)]
    artifact_paths += [corpus / "transitions/scene_10400_11500/frames" / f"{i:06d}.png"
                       for i in (10700, 11000, 11250)]
    for i, path in enumerate(artifact_paths):
        hashed[str(path)] = sha256_file(path)
        shutil.copy2(path, output / f"asset-{i}-{path.name}")
    (output / "input-hashes.json").write_text(json.dumps(hashed, indent=2))
    pipeline = from_artifacts(corpus, bank)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("cannot open video")
    selected, counts, times, pts = {}, Counter(), [], []
    began = time.perf_counter()
    try:
        with (output / "frames.jsonl").open("w", encoding="utf-8") as stream:
            for index in range(end + 1):
                if index < start:
                    if not capture.grab():
                        raise ValueError("source ended during sequential skip")
                    continue
                ok, raw = capture.read()
                if not ok:
                    raise ValueError("source ended in required interval")
                picture = normalize(raw, config)
                timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
                if pts and timestamp <= pts[-1]:
                    raise ValueError("container PTS failed monotonicity")
                pts.append(timestamp)
                if index in anchors and pixels_digest(picture) != anchors[index]:
                    raise ValueError("raw frame disagrees with bound PNG")
                stamp = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(
                    milliseconds=timestamp)
                frame = Frame(index, stamp, "combined-raw-replay",
                              WindowRect(0, 0, 498, 1080), picture, 498, 1080)
                started = time.perf_counter()
                row = pipeline.process(frame)
                times.append(time.perf_counter() - started)
                stream.write(json.dumps(row) + "\n")
                if index in check_ids:
                    selected[index] = row
                counts["frames"] += 1
                counts["eight_numeric_slots_not_ground_truth"] += sum(
                    s["value"] is not None for s in row["stacks"].values()) == 8
                if (index - start) % 300 == 0:
                    elapsed = round(time.perf_counter() - began, 1)
                    progress = {"frame": index, "elapsed_s": elapsed}
                    print(json.dumps(progress), flush=True)
    finally:
        capture.release()
    scores = {key: Counter() for key in ("stack", "pot", "action")}
    for point in truth["checkpoints"]:
        if point["source_frame"] not in selected:
            continue
        row = selected[point["source_frame"]]
        scores["pot"][score_read(point["pot"], row["pot"], money=True)] += 1
        for slot in range(8):
            stack = row["stacks"].get(str(slot), {"value": None, "status": "unknown"})
            scores["stack"][score_read(point["stacks"][slot], stack, money=True)] += 1
            action = row["actions"][str(slot)]
            status = "valid" if action["accepted_candidate"] else "unknown"
            actual = {"value": action["value"], "status": status}
            scores["action"][score_read(point["actions"][slot], actual)] += 1
    card_scores = {"hero": Counter(), "board": Counter()}
    for point in card_truth["checkpoints"]:
        if point["source_frame"] not in selected:
            continue
        for field, width in (("hero", 2), ("board", 5)):
            observed_cards = selected[point["source_frame"]][field]
            result = score_slots(point[field], observed_cards, width)
            card_scores[field].update(result["counts"])
    report = {"counts": counts, "checkpoint_scores": scores,
              "card_checkpoint_scores": card_scores,
              "checkpoints": selected, "interval": [start, end],
              "source_sha256": source["sha256"], "sequential_source_decode": True,
              "processing_mean_ms": sum(times) / len(times) * 1000,
              "processing_max_ms": max(times) * 1000,
              "scope": "processing only, not real capture-to-render latency",
              "release_eligible": False, "state_event_chain_verified": False}
    if any(sha256_file(Path(p)) != h for p, h in hashed.items()):
        raise ValueError("input or implementation changed during replay")
    after = video_path.stat()
    if ((after.st_size, after.st_mtime_ns)
            != (before_stat.st_size, before_stat.st_mtime_ns)):
        raise ValueError("video changed during replay")
    (output / "report.json").write_text(json.dumps(report, indent=2))
    write_sha256sums(output)
    return {"counts": counts, "scores": scores, "card_scores": card_scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "bank", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--start", type=int, default=8360)
    parser.add_argument("--end", type=int, default=8845)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--card-review", type=Path)
    args = parser.parse_args()
    result = run(args.corpus, args.bank, args.output, args.start, args.end,
                 review_path=args.review, card_review_path=args.card_review)
    print(json.dumps(result, indent=2))
