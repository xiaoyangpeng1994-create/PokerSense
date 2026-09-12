"""Verify reviewed DEVELOPMENT dealer frames; never independent acceptance."""

import argparse
import json
from pathlib import Path

from tools.aa8_action_transfer import sha
from tools.aa8_dealer_v2 import AA8DealerReader
from tools.wpk_video_dataset import read_image


def run(gold_path, pools, output):
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    if (gold.get("role") != "development"
            or gold.get("independent_holdout") is not False):
        raise ValueError("development gold required")
    rows, audit, manifests = {}, None, {}
    for pool in pools:
        manifest_path = pool / "samples.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if audit is not None and manifest["audit_sha256"] != audit:
            raise ValueError("mixed audit identities")
        audit = manifest["audit_sha256"]
        manifests[str(manifest_path.resolve())] = sha(manifest_path)
        for row in manifest["samples"]:
            frame = row["global_frame"]
            if row.get("role") != "development" or frame in rows:
                raise ValueError("non-development or duplicate source frame")
            rows[frame] = (pool, row)
    reader = AA8DealerReader()
    comparisons = []
    for expected in gold["checkpoints"]:
        frame = expected["frame"]
        if frame not in rows:
            raise ValueError("gold frame outside supplied pools")
        pool, source = rows[frame]
        path = (pool / source["file"]).resolve()
        if (not path.is_relative_to(pool.resolve())
                or source["sha256"] != expected["source_sha256"]
                or sha(path) != source["sha256"]):
            raise ValueError("gold/source path or hash mismatch")
        actual = reader.read(read_image(path))
        comparisons.append({
            "frame": frame,
            "source_sha256": source["sha256"],
            "expected": expected["dealer_seat"],
            "actual": actual["dealer_seat"],
            "match": actual["dealer_seat"] == expected["dealer_seat"],
            "reason": actual["reason"],
        })
    result = {
        "schema_version": 1,
        "comparisons": comparisons,
        "matched": sum(item["match"] for item in comparisons),
        "total": len(comparisons),
        "gold_sha256": sha(gold_path),
        "source_manifest_sha256": manifests,
        "implementation_sha256": sha(Path(__file__).with_name("aa8_dealer_v2.py")),
        "independent_holdout": False,
        "canonical_verified": False,
        "full_visual_acceptance": False,
        "strategy_eligible": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "matched": result["matched"], "total": result["total"],
        "failures": [item for item in comparisons if not item["match"]],
    }))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pool", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.gold, args.pool, args.output)
