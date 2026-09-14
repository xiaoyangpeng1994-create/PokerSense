"""Metadata-only eligibility freeze and closed eight-seat confusion accounting.

Never imports a recognizer, opens media, or treats historical predictions as
holdout results. Positive eligibility relies on audited provenance declarations;
hashes bind those declarations, not their truth.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


CLASSES = ("fold", "muck", "all_in")
FLAGS = {"template", "tuning_or_debug", "manual_action_review",
         "template_episode_overlap"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def public_safe(value):
    if isinstance(value, dict):
        for key, item in value.items():
            public_safe(key)
            public_safe(item)
    elif isinstance(value, list):
        for item in value:
            public_safe(item)
    elif isinstance(value, str):
        if (any(c in value for c in (":", "/", "\\"))
                or ".." in value or value.lower().endswith(
                    (".png", ".jpg", ".mkv", ".mp4", ".env", ".pem"))
                or re.fullmatch(r"\+?\d{10,15}", value)
                or re.search(r"(?:ghp_|github_pat_|sk-)[A-Za-z0-9]", value)):
            raise ValueError("private path, media or sensitive identifier")


def interval(row):
    a, b = row["first"], row["last"]
    if type(a) is not int or type(b) is not int or not 0 <= a <= b:
        raise ValueError("invalid frame interval")
    return a, b


def overlap(a, b):
    return a["source"] == b["source"] and max(a["first"], b["first"]) <= min(
        a["last"], b["last"])


def eligibility(config):
    public_safe(config)
    if (type(config["version"]) is not int or config["version"] != 1
            or config["scope"] != (
            "template-disjoint intra-session holdout")):
        raise ValueError("wrong validation scope")
    source_ids = {s["id"] for s in config["sources"]}
    evidence_ids = {e["id"] for e in config["evidence"]}
    if (len(source_ids) != len(config["sources"])
            or len(evidence_ids) != len(config["evidence"])):
        raise ValueError("duplicate source or evidence")
    for episode in config["episodes"]:
        a, b = interval(episode)
        if (episode["source"] not in source_ids
                or not a <= episode["template_frame"] <= b):
            raise ValueError("invalid template episode enclosure")
    decisions, seen = [], []
    for row in config["candidates"]:
        interval(row)
        if (row["source"] not in source_ids or row["slots"] != list(range(8))
                or any(type(s) is not int for s in row["slots"])
                or set(row["exposure"]) != FLAGS
                or any(v not in ("YES", "NO", "UNKNOWN")
                       for v in row["exposure"].values())
                or not row["evidence"]
                or not set(row["evidence"]) <= evidence_ids
                or type(row["clean_proven"]) is not bool
                or type(row["episode_complete"]) is not bool):
            raise ValueError("incomplete provenance declaration")
        if any(p["id"] == row["id"] or overlap(p, row) for p in seen):
            raise ValueError("duplicate or overlapping candidates")
        seen.append(row)
        reasons = [key + "_" + value for key, value in row["exposure"].items()
                   if value != "NO"]
        if not row["clean_proven"]:
            reasons.append("NO_VERIFIABLE_CLEAN_PROVENANCE")
        if not row["episode_complete"]:
            reasons.append("EPISODE_BOUNDARIES_NOT_PROVEN")
        if any(overlap(row, e) for e in config["episodes"]):
            reasons.append("TEMPLATE_EPISODE_ENCLOSURE_OVERLAP")
        decisions.append({"id": row["id"], "source": row["source"],
                          "first": row["first"], "last": row["last"],
                          "eligible": not reasons, "reasons": reasons})
    return decisions


def verified_evidence(config, directory):
    root = Path(directory).resolve(strict=True)
    for row in config["evidence"]:
        if not re.fullmatch(r"[a-z0-9-]+\.json", row["file"]):
            raise ValueError("metadata JSON evidence only")
        path = root / row["file"]
        if path.is_symlink() or path.resolve().parent != root:
            raise ValueError("evidence escapes root")
        if digest(path) != row["sha256"]:
            raise ValueError("evidence hash mismatch")


def schedule(decisions):
    """Exhaustive temporal/seat order, no scores or prediction-driven selection."""
    return [(d["id"], frame, slot)
            for d in sorted(decisions, key=lambda r: (r["source"], r["first"]))
            if d["eligible"] for frame in range(d["first"], d["last"] + 1)
            for slot in range(8)]


def confusion(queue, labels, outputs):
    """Exhaustive frame-slot multiclass counts; not episode/event recall.

    Caller supplies human labels in the frozen non-model-driven queue order.
    missing/UNKNOWN labels or missing machine rows fail, never shrink denominator.
    """
    if (len(queue) != len(set(queue)) or len(labels) != len(queue)
            or len(outputs) != len(queue)):
        raise ValueError("exact census required")
    allowed = {*CLASSES, "none"}
    totals = {str(slot): {c: dict(
        TP=0, FN=0, FP=0, TN=0, opportunities=0, negatives=0)
                          for c in CLASSES} for slot in range(8)}
    for expected, human, machine in zip(queue, labels, outputs):
        if (tuple(human["key"]) != expected
                or tuple(machine["key"]) != expected
                or human["label"] not in allowed
                or machine["label"] not in allowed
                or type(expected[2]) is not int or not 0 <= expected[2] < 8):
            raise ValueError("census identity or label mismatch")
        for cls in CLASSES:
            pos = human["label"] == cls
            pred = machine["label"] == cls
            row = totals[str(expected[2])][cls]
            row["opportunities" if pos else "negatives"] += 1
            row["TP" if pos and pred else "FN" if pos else
                "FP" if pred else "TN"] += 1
    return totals


def freeze(config_path, evidence_dir, output):
    config = load(config_path)
    verified_evidence(config, evidence_dir)
    decisions = eligibility(config)
    value = {"schema_version": 1, "scope": config["scope"],
             "frozen_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
             "config_sha256": digest(config_path),
             "tool_sha256": digest(__file__), "decisions": decisions,
             "model_executed": False,
             "status": "ELIGIBLE" if any(d["eligible"] for d in decisions)
             else "NO_ELIGIBLE_HOLDOUT"}
    write_new(output, value)
    return value


def conclude(config_path, evidence_dir, manifest, expected_sha, out):
    if digest(manifest) != expected_sha:
        raise ValueError("frozen manifest hash mismatch")
    frozen = load(manifest)
    if (frozen["config_sha256"] != digest(config_path)
            or frozen["tool_sha256"] != digest(__file__)):
        raise ValueError("frozen config or implementation changed")
    config = load(config_path)
    verified_evidence(config, evidence_dir)
    decisions = eligibility(config)
    if decisions != frozen["decisions"]:
        raise ValueError("frozen decisions mismatch")
    if any(d["eligible"] for d in decisions):
        raise ValueError("eligible material needs separate label-first validation")
    target = Path(out)
    target.mkdir(exist_ok=False)
    for name in ("human-labels", "machine-output"):
        write_new(target / (name + ".json"), {
            "status": "NOT_RUN_NO_ELIGIBLE_HOLDOUT", "rows": [],
            "manifest_sha256": expected_sha})
    metrics = {str(s): {c: {"denominator": 0, "TP": None, "FN": None,
                            "FP": None, "TN": None} for c in CLASSES}
               for s in range(8)}
    result = {"status": "NO_ELIGIBLE_HOLDOUT", "scope": config["scope"],
              "finished_at": datetime.now(timezone.utc).isoformat(),
              "frozen_manifest_sha256": expected_sha,
              "eligible_frame_slots": 0, "metrics": metrics,
              "metrics_by_hand": {d["id"]: metrics for d in decisions},
              "labels_sha256": digest(target / "human-labels.json"),
              "output_sha256": digest(target / "machine-output.json"),
              "model_executed": False, "media_read": False,
              "decisions": decisions, "command": sys.argv,
              "python": sys.version, "code_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], text=True).strip(),
              "tool_sha256": digest(__file__)}
    # Version metadata only; no image/model module imports are needed.
    from importlib.metadata import version
    result["numpy"] = version("numpy")
    result["opencv"] = version("opencv-python")
    write_new(target / "report.json", result)
    write_new(target / "SHA256.json", {
        p.name: digest(p) for p in sorted(target.glob("*.json"))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "conclude"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.mode == "freeze":
        result = freeze(args.config, args.evidence_dir, args.output)
    else:
        if args.manifest is None or args.manifest_sha256 is None:
            parser.error("conclude requires externally pinned manifest")
        result = conclude(args.config, args.evidence_dir, args.manifest,
                          args.manifest_sha256, args.output)
    print(json.dumps({"status": result["status"], "model_executed": False}))


if __name__ == "__main__":
    main()
