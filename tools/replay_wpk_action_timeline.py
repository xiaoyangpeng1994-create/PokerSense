"""Infer visual action evidence from owned frames, then compare frozen screen truth."""

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import shutil

from poker_engine.perceptual.capture.base import Frame, WindowRect
from poker_engine.state_engine.visual_timeline import VisualTimeline
from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)
from tools.wpk_combined_vision import from_artifacts
from tools.wpk_hand_registry import assert_owned_frames
from tools.wpk_video_dataset import pixels_digest, read_image
from tools.wpk_field_candidate import crop


def compare_actions(truth, events):
    actions = [e for e in events if e["kind"] == "action_evidence"]
    matches, used = [], set()
    allowance = truth["comparison"]["max_additional_confirmation_frames"]
    for expected in truth["actions"]:
        candidates = [(i, row) for i, row in enumerate(actions) if i not in used
                      and (row["street"], row["seat"], row["action"]) ==
                      (expected["street"], expected["seat"], expected["action"])]
        if not candidates:
            matches.append({"order": expected["order"], "status": "missing"})
            continue
        index, got = candidates[0]
        used.add(index)
        earliest, latest = expected["visible_interval"]
        deadline = max(latest, expected["money_visible_by"]) + allowance
        timing = (earliest <= got["first_frame"] <= latest + allowance
                  and got["confirmed_frame"] <= deadline)
        money = (got["amount"] is not None
                 and Decimal(got["amount"]) == Decimal(expected["amount_additional"]))
        matches.append({"order": expected["order"],
                        "status": "match" if timing and money else "mismatch",
                        "timing_matches": timing, "amount_matches": money,
                        "expected": expected, "observed": got})
    unexpected = [row for i, row in enumerate(actions) if i not in used]
    observed_order = [row["order"] for row in sorted(
        (row for row in matches if "observed" in row),
        key=lambda row: row["observed"]["confirmed_frame"])]
    return {"matches": matches, "unexpected_actions": unexpected,
            "ordered_matches": observed_order == sorted(observed_order),
            "matched_count": sum(row["status"] == "match" for row in matches),
            "ordinary_actions_expected": len(truth["actions"])}


def run(corpus, registry_path, truth_path, output, *, improved_actions=False):
    if output.exists():
        raise ValueError("preserve prior replay results")
    registry_hash, truth_hash = sha256_file(registry_path), sha256_file(truth_path)
    registry = json.loads(registry_path.read_text())
    target = next(h for h in registry["entries"]
                  if h["role"] == "reconstruction_development")
    window = corpus / target["target_full_window"]
    if verify_sha256sums(window):
        raise ValueError("source window integrity failure")
    for name in ("samples.json", "summary.json"):
        key = str((window / name).resolve())
        if sha256_file(Path(key)) != registry["input_hashes"].get(key):
            raise ValueError("window metadata differs from frozen hand registry")
    samples = json.loads((window / "samples.json").read_text())
    samples = [s for s in samples
               if target["start"] <= s["source_frame"] <= target["end"]]
    assert_owned_frames(target, [s["source_frame"] for s in samples])
    if [s["source_frame"] for s in samples] != target["scoring_frame_ids"]:
        raise ValueError("owned frame inventory changed")
    repo = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True)
    shutil.copy2(truth_path, output / "truth.json")
    shutil.copy2(registry_path, output / "registry.json")
    hashes = {}
    for name in ("samples.json", "summary.json"):
        path = (window / name).resolve()
        hashes[str(path)] = sha256_file(path)
    for base in (repo / "src", repo / "tools", repo / "configs"):
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            digest = sha256_file(path)
            destination = output / "source-snapshot" / path.relative_to(repo)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            if sha256_file(destination) != digest:
                raise ValueError("source changed during freeze")
            hashes[str(path)] = digest
    bank = corpus / "gray_amount/session001_proposals_v1"
    artifact_files = [Path(p) for p in registry["frozen_model_hashes"]
                      if Path(p).is_relative_to(corpus.resolve())]
    artifact_files += [corpus / "hands/aq_allin_8240_8900/frames" / f"{i:06d}.png"
                       for i in (8500, 8700)]
    artifact_files += [corpus / "transitions/scene_10400_11500/frames" / f"{i:06d}.png"
                       for i in (10700, 11000, 11250)]
    check_patch = None
    if improved_actions:
        source_spec = repo / (
            "tests/fixtures/wpk_reference_hands/check_variant_source001_v1.json")
        source = json.loads(source_spec.read_text())
        source_folder = corpus.parent / (
            "capture_card_calibration_20260903/normalized/frames")
        path = source_folder / source["source_frame_file"]
        if sha256_file(path) != source["image_sha256"]:
            raise ValueError("reviewed check template changed")
        check_patch = crop(read_image(path), source["rect"])
        artifact_files += [path, source_spec]
    for index, path in enumerate(artifact_files):
        digest = sha256_file(path)
        hashes[str(path)] = digest
        shutil.copy2(path, output / f"asset-{index}-{path.name}")
    pipeline = from_artifacts(corpus, bank, check_variant=check_patch,
                              badge_scale_tolerance=improved_actions)
    timeline = VisualTimeline()
    # Truth has not been parsed and is never passed into the inference objects.
    with (output / "observations.jsonl").open("w", encoding="utf-8") as stream:
        for sample in samples:
            path = window / sample["image"]
            image = read_image(path)
            registry_digest = registry["input_hashes"].get(str(path.resolve()))
            if (sha256_file(path) != registry_digest
                    or registry_digest != sample["image_sha256"]
                    or pixels_digest(image) != sample["pixel_sha256"]):
                raise ValueError("source frame changed")
            stamp = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(
                milliseconds=sample["container_pts_ms"])
            frame = Frame(sample["source_frame"], stamp, "td3h-visual-timeline",
                          WindowRect(0, 0, 498, 1080), image, 498, 1080)
            observed = pipeline.process(frame)
            timeline.consume(observed)
            stream.write(json.dumps(observed) + "\n")
    events = [asdict(e) for e in timeline.events]
    truth = json.loads(truth_path.read_text())
    if truth["hand_id"] != target["hand_id"]:
        raise ValueError("truth belongs to another hand")
    comparison = compare_actions(truth, events)
    roster_matches = list(timeline.roster or ()) == truth["opening_seats"]
    report = {"hand_id": target["hand_id"], "owned_frames": len(samples),
              "registry_sha256": registry_hash, "truth_sha256": truth_hash,
              "summary": timeline.summary(), "events": events, "comparison": comparison,
              "opening_roster_matches": roster_matches,
              "independent_holdout": False, "source_hashes": hashes,
              "full_canonical_reconstruction_verified": False}
    report["improved_actions"] = improved_actions
    report["input_kind"] = "cached_source_bound_png_sequence"
    report["special_checks"] = {
        "departed_seat6_not_given_invented_fold": not any(
            e["kind"] == "action_evidence" and e["seat"] == 6 for e in events),
        "late_seat5_not_given_hand_action": not any(
            e["kind"] == "action_evidence" and e["seat"] == 5 for e in events),
        "hero_folded_before_end": 0 in timeline.folded,
    }
    if (sha256_file(truth_path) != truth_hash
            or sha256_file(registry_path) != registry_hash):
        raise ValueError("review changed during inference")
    if any(sha256_file(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("code/model changed during inference")
    (output / "report.json").write_text(json.dumps(report, indent=2))
    write_sha256sums(output)
    return {"summary": report["summary"], "roster_matches": roster_matches,
            "matched": comparison["matched_count"],
            "expected": comparison["ordinary_actions_expected"],
            "unexpected": len(comparison["unexpected_actions"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "registry", "truth", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--improved-actions", action="store_true")
    args = parser.parse_args()
    result = run(args.corpus, args.registry, args.truth, args.output,
                 improved_actions=args.improved_actions)
    print(json.dumps(result, indent=2))
