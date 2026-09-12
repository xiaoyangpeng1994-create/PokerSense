"""Freeze true hand ownership and audit prior exposure without relabeling it clean."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums
from tools.wpk_video_dataset import pixels_digest, read_image


def overlaps(left, right):
    return left[0] <= right[1] and right[0] <= left[1]


def validate_hands(hands):
    if not hands:
        raise ValueError("empty registry")
    ids = set()
    roles = {"reconstruction_development", "frozen_regression", "acceptance_candidate",
             "independent_acceptance"}
    for hand in hands:
        if hand["hand_id"] in ids:
            raise ValueError("duplicate hand id")
        ids.add(hand["hand_id"])
        start, end = hand["start"], hand["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start <= end:
            raise ValueError("invalid inclusive hand interval")
        if (hand["role"] not in roles or type(hand["opening_players"]) is not int
                or hand["opening_players"] not in (6, 7, 8)):
            raise ValueError("unsupported role or player count")
        if (hand["context_start"] != start - 1 or hand["context_end"] != end + 1
                or hand.get("boundary_reviewed") is not True):
            raise ValueError("adjacent reviewed boundary sentinels required")
    ordered = sorted(hands, key=lambda h: h["start"])
    for left, right in zip(ordered, ordered[1:]):
        if left["end"] >= right["start"]:
            raise ValueError("two hand owners overlap; do not split legacy aliases")


def audit_hand(hand, exposures, provenance_complete):
    interval = (hand["start"], hand["end"])
    hits = [e for e in exposures if overlaps(interval, (e["start"], e["end"]))]
    state = "KNOWN_EXPOSURE" if hits else "NO_KNOWN_EXPOSURE"
    qualified = (not hits and provenance_complete is True
                 and hand.get("boundary_reviewed") is True
                 and hand.get("full_action_truth_ready") is True)
    if not hits and provenance_complete is not True:
        state = "PROVENANCE_UNVERIFIED"
    if hand["role"] == "independent_acceptance" and not qualified:
        raise ValueError("cannot promote exposed/unverified/incomplete hand")
    independent = qualified and hand["role"] == "independent_acceptance"
    return {"exposure_status": state, "exposure_hits": hits,
            "independent_acceptance_eligible": independent,
            "action_scoring_ready": hand.get("full_action_truth_ready") is True}


def assert_owned_frames(hand, frame_ids):
    if not frame_ids or len(frame_ids) != len(set(frame_ids)) or any(
        type(frame) is not int or not hand["start"] <= frame <= hand["end"]
        for frame in frame_ids
    ):
        raise ValueError("context, adjacent hand or duplicate frame cannot be scored")


def collect_exposures(corpus, dataset):
    mapping_path = corpus / "session_002/reference-map.json"
    mapping = json.loads(mapping_path.read_text())
    exposures, hashes = [], {str(mapping_path): sha256_file(mapping_path)}
    for name in ("train", "calibration", "b4_train_frames"):
        path = dataset / "splits" / f"{name}.txt"
        hashes[str(path)] = sha256_file(path)
        frames = set(path.read_text().splitlines())
        for row in mapping:
            if row["frame"] in frames:
                for start, end in row["source_frame_ranges"]:
                    exposures.append({"start": start, "end": end, "kind": name,
                                      "frame": row["frame"],
                                      "mapping_status": row["mapping_status"]})
    paths = [corpus / "validation" / name / "frozen-spec.json"
             for name in ("v4_batch_02", "v6_batch_03")]
    for path in paths:
        spec = json.loads(path.read_text())
        hashes[str(path)] = sha256_file(path)
        exposures.append({"start": spec["warmup_start_frame"],
                          "end": max(p["source_frame"] for p in spec["checkpoints"]),
                          "kind": "consumed_card_development", "evidence": str(path)})
    # Documented recent raw replays / training references. Not an exhaustive
    # reconstruction of historical model ancestry: that remains explicitly false.
    for start, end, description in (
        (1180, 1240, "six-seat combined vision development"),
        (8240, 8900, "AcQh recognition and reviewed ledger development"),
        (10400, 11500, "transition/action/stack/layout development"),
    ):
        exposures.append({"start": start, "end": end,
                          "kind": "documented_development", "evidence": description})
    return exposures, hashes


def freeze(corpus, dataset, spec_path, output):
    if output.exists():
        raise ValueError("frozen registry output must be new")
    spec_hash = sha256_file(spec_path)
    spec = json.loads(spec_path.read_text())
    inventory_path = corpus / "sources.json"
    inventory = json.loads(inventory_path.read_text())
    session_source = next(s for s in inventory["sources"]
                          if s["session"] == "session_002")
    if spec["source_sha256"] != session_source["sha256"]:
        raise ValueError("this exposure audit is scoped to the indexed session002")
    validate_hands(spec["hands"])
    exposures, inputs = collect_exposures(corpus, dataset)
    inputs[str(spec_path.resolve())] = spec_hash
    inputs[str(inventory_path)] = sha256_file(inventory_path)
    entries = []
    for hand in spec["hands"]:
        bound = []
        expected = {"before_first_post": hand["context_start"],
                    "first_post": hand["start"], "last_owned_frame": hand["end"],
                    "next_hand_post": hand["context_end"]}
        if len(hand["boundary_evidence"]) != 4 or {
            row["purpose"] for row in hand["boundary_evidence"]
        } != set(expected):
            raise ValueError("four distinct boundary anchors required")
        for reference in hand["boundary_evidence"]:
            if reference["frame"] != expected[reference["purpose"]]:
                raise ValueError("boundary anchor does not match ownership")
            window = (corpus / reference["window"]).resolve()
            if not window.is_relative_to(corpus.resolve()):
                raise ValueError("boundary path escaped corpus")
            summary = json.loads((window / "summary.json").read_text())
            if summary["source_sha256"] != spec["source_sha256"]:
                raise ValueError("boundary belongs to another video")
            samples = json.loads((window / "samples.json").read_text())
            sample = next(s for s in samples if s["source_frame"] == reference["frame"])
            path = (window / sample["image"]).resolve()
            if not path.is_relative_to(window):
                raise ValueError("sample escaped boundary window")
            if sha256_file(path) != sample["image_sha256"] or (
                pixels_digest(read_image(path)) != sample["pixel_sha256"]
            ):
                raise ValueError("boundary bytes/pixels changed")
            inputs[str(path)] = sample["image_sha256"]
            for name in ("summary.json", "samples.json"):
                inputs[str(window / name)] = sha256_file(window / name)
            bound.append({**reference, **sample,
                          "boundary_review_state": "agent_visual_reviewed"})
        entry = {**hand, "boundaries": bound,
                 **audit_hand(hand, exposures, spec["component_provenance_complete"])}
        if hand.get("target_full_window"):
            window = (corpus / hand["target_full_window"]).resolve()
            if not window.is_relative_to(corpus.resolve()):
                raise ValueError("target escaped corpus")
            summary_path = window / "summary.json"
            target_summary = json.loads(summary_path.read_text())
            if target_summary["source_sha256"] != spec["source_sha256"]:
                raise ValueError("target full hand belongs to another source")
            inputs[str(summary_path)] = sha256_file(summary_path)
            inputs[str(window / "samples.json")] = sha256_file(window / "samples.json")
            samples = json.loads((window / "samples.json").read_text())
            owned = [s for s in samples
                     if hand["start"] <= s["source_frame"] <= hand["end"]]
            frame_ids = [s["source_frame"] for s in owned]
            assert_owned_frames(hand, frame_ids)
            if sorted(frame_ids) != list(range(hand["start"], hand["end"] + 1)):
                raise ValueError("target full hand has missing source frames")
            for sample in owned:
                path = (window / sample["image"]).resolve()
                if not path.is_relative_to(window):
                    raise ValueError("target image escaped window")
                if sha256_file(path) != sample["image_sha256"] or (
                    pixels_digest(read_image(path)) != sample["pixel_sha256"]
                ):
                    raise ValueError("target frame changed")
                inputs[str(path)] = sample["image_sha256"]
            entry.update(scoring_frame_ids=frame_ids, all_owned_frames_available=True,
                         all_owned_frames_field_labeled=False)
        entries.append(entry)
    if sum(h["role"] == "reconstruction_development" for h in entries) != 1:
        raise ValueError("choose exactly one current reconstruction target")
    model_files = {}
    repo = Path(__file__).resolve().parents[1]
    for base in (repo / "src", repo / "configs", repo / "tools"):
        for path in base.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                model_files[str(path)] = sha256_file(path)
    folders = ("gray_amount/session001_proposals_v1",
               "gray_amount/session001_pot_proposals_v2",
               "training/rank_v6_candidate_01")
    for folder in folders:
        for path in (corpus / folder).iterdir():
            if path.suffix in (".json", ".npz"):
                model_files[str(path)] = sha256_file(path)
    coverage = dict(Counter(str(h["opening_players"]) for h in entries))
    independent_count = sum(h["independent_acceptance_eligible"] for h in entries)
    report = {"schema_version": 1, "source_sha256": spec["source_sha256"],
              "entries": entries, "exposures": exposures,
              "input_hashes": inputs, "frozen_model_hashes": model_files,
              "registered_hands": len(entries),
              "opening_player_coverage": coverage,
              "independent_acceptance_hands": independent_count,
              "whole_visual_acceptance_ready": False,
              "classifier_executed_during_freeze": False}
    if any(sha256_file(Path(path)) != digest for path, digest in inputs.items()):
        raise ValueError("registry input changed during freeze")
    output.mkdir(parents=True)
    snapshots = {}
    for original, digest in model_files.items():
        path = Path(original)
        if path.is_relative_to(repo):
            relative = Path("model-snapshot/repo") / path.relative_to(repo)
        else:
            relative = (Path("model-snapshot/corpus")
                        / path.relative_to(corpus.resolve()))
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if sha256_file(destination) != digest or sha256_file(path) != digest:
            raise ValueError("model changed while snapshotting")
        snapshots[relative.as_posix()] = digest
    report["model_snapshots"] = snapshots
    shutil.copy2(spec_path, output / "spec.json")
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    (output / "registry.json").write_text(json.dumps(report, indent=2))
    write_sha256sums(output)
    keys = ("registered_hands", "opening_player_coverage",
            "independent_acceptance_hands")
    return {key: report[key] for key in keys}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "dataset", "spec", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.corpus, args.dataset, args.spec, args.output)
    print(json.dumps(result, indent=2))
