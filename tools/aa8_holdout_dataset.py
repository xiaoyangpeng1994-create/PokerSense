"""Whole-hand heldout extraction, only after explicit evaluation supplement freeze.

This adds no exception to the boundary sampler's three-second dense limit.
It has a separate complete-hand registry and lineage gate and no CLI.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

from tools.aa8_holdout_plan import sha, utc, validate_freeze
from tools.aa8_holdout_predict import bound_file, registry_range, validate_rows


def selected_index(index, registry):
    allowed = set()
    for hand in registry["hands"]:
        allowed.update(range(hand["preroll_first_frame"], hand["last_frame"] + 1))
    rows = {r["global_frame"]: {**r, "role": "holdout"}
            for r in index if r["global_frame"] in allowed}
    if len(rows) != sum(r["global_frame"] in allowed for r in index):
        raise ValueError("duplicate audit frame identity")
    validate_rows(rows, registry, registry["audit_sha256"], require_hash=False)
    return rows


def extract(*, audit, normalization_path, registry_path, freeze_path,
            supplement_path, training_manifest, output):
    """Authorized call reads video; importing/testing this module never does."""
    freeze = json.loads(freeze_path.read_text())
    supplement = json.loads(supplement_path.read_text())
    registry = json.loads(registry_path.read_text())
    started = datetime.now(timezone.utc).isoformat()
    failures = validate_freeze(freeze, freeze_path.parent, started)
    if failures:
        raise ValueError(failures)
    bound_file(training_manifest, freeze)
    training = json.loads(training_manifest.read_text())
    pinned = {p: sha(p) for p in (registry_path, supplement_path, normalization_path,
                                  Path(__file__), Path(__file__).with_name(
                                      "aa8_holdout_predict.py"))}
    chronology = utc(freeze["frozen_at_utc"]) < utc(
        supplement["frozen_at_utc"]) < utc(started)
    if (supplement.get("base_freeze_sha256") != sha(freeze_path)
            or registry.get("freeze_sha256") != sha(freeze_path)
            or supplement.get("dataset_harness_sha256") != sha(Path(__file__))
            or supplement.get("evaluation_harness_sha256") != sha(
                Path(__file__).with_name("aa8_holdout_predict.py"))
            or supplement.get("registry_sha256") != sha(registry_path)
            or supplement.get("normalization_sha256") != sha(normalization_path)
            or sha(normalization_path) != training["normalization_sha256"]
            or registry.get("audit_sha256") != training["audit_sha256"]
            or supplement.get("audit_index_sha256") != sha(audit / "frame_index.jsonl")
            or registry.get("audit_sha256") != sha(audit / "report.json")
            or not chronology):
        raise ValueError("whole-hand extraction lineage mismatch")
    split_path = next(Path(v["path"]) for v in freeze["files"] if Path(
        v["path"]).name == "aa8_recording_split_plan_20260909.json")
    split = json.loads(split_path.read_text())
    if split["source_audit_sha256"] != registry["audit_sha256"]:
        raise ValueError("holdout source differs from frozen split")
    lo, hi = registry_range(registry)
    if not any(r["role"] == "holdout_candidate" and
               float(r["start_inclusive"]) == lo and float(r["end_exclusive"]) == hi
               for r in split["ranges"]):
        raise ValueError("complete-hand registry not inside frozen holdout")
    source = json.loads((audit / "report.json").read_text())
    if source["state"] != "verified":
        raise ValueError("verified audit required")
    index = [json.loads(line) for line in (
        audit / "frame_index.jsonl").read_text().splitlines()]
    selected = selected_index(index, registry)
    import cv2
    from poker_engine.perceptual.capture.normalization import (
        NormalizationConfig, normalize,
    )
    config = NormalizationConfig.from_json(normalization_path.read_text())
    root = Path(source["recording"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    result = []
    for segment in source["segments"]:
        subset = {r["local_frame"]: r for r in selected.values()
                  if r["segment"] == segment["file"]}
        if not subset:
            continue
        path = (root / segment["file"]).resolve()
        if path.parent != root:
            raise ValueError("source path escape")
        before = path.stat()
        if (before.st_size, before.st_mtime_ns) != (
                segment["size_bytes"], segment["mtime_ns"]):
            raise ValueError("source attributes changed")
        capture = cv2.VideoCapture(str(path))
        try:
            for local in range(max(subset) + 1):
                ok, raw = capture.read()
                if not ok:
                    raise ValueError("unexpected source decode end")
                if local not in subset:
                    continue
                row = subset[local]
                canvas = normalize(raw, config)
                ok, data = cv2.imencode(".png", canvas)
                if not ok:
                    raise ValueError("PNG encode failed")
                target = output / "frames" / f"frame_{row['global_frame']:06d}.png"
                target.write_bytes(data.tobytes())
                result.append({**row, "file": target.relative_to(output).as_posix(),
                               "sha256": sha(target), "participation": "UNKNOWN"})
        finally:
            capture.release()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("source changed during extraction")
    rowmap = {r["global_frame"]: r for r in result}
    validate_rows(rowmap, registry, registry["audit_sha256"])
    for hand in registry["hands"]:
        for witness in hand.get("boundary_witnesses", {}).values():
            frame = witness["global_frame"]
            if frame in rowmap and rowmap[frame]["sha256"] != witness["sha256"]:
                raise ValueError("extracted frame differs from boundary witness")
    if (validate_freeze(freeze, freeze_path.parent, started)
            or any(sha(p) != value for p, value in pinned.items())):
        raise ValueError("base freeze changed during extraction")
    manifest = {"audit_sha256": registry["audit_sha256"],
                "normalization_sha256": sha(normalization_path),
                "role": "holdout", "samples": result,
                "extraction_start_utc": started, "registry_sha256": sha(registry_path),
                "dataset_harness_sha256": sha(Path(__file__)),
                "evaluation_supplement_sha256": sha(supplement_path),
                "base_freeze_sha256": sha(freeze_path), "prediction_performed": False}
    (output / "samples.json").write_text(json.dumps(manifest, indent=2))
    return manifest
