"""Freeze local candidate files and development exposure ledger without media reads."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from tools.aa8_holdout_plan import sha, validate_freeze


def freeze(repo, training_paths, models, parameters, output):
    repo = repo.resolve()
    implementations = list((repo / "src").rglob("*.py"))
    implementations += list((repo / "tools").rglob("*.py"))
    pairs = [(p, "implementation") for p in implementations]
    pairs += [(p, "training") for p in training_paths]
    pairs += [(p, "model") for p in models]
    pairs += [(p, "parameters") for p in parameters]
    entries = [{"path": str(p.resolve()), "kind": role, "sha256": sha(p)}
               for p, role in pairs]
    entries.sort(key=lambda r: Path(r["path"]).as_posix().casefold())
    result = {"id": output.name,
              "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "files": entries, "exposures": [{
                  "artifact_sha256": row["sha256"], "role": "development",
                  "used_for": "template_selection"} for row in entries
                  if row["kind"] == "training"],
              "prediction_performed": False, "full_visual_acceptance": False}
    problems = validate_freeze(result, output)
    if problems:
        raise ValueError(problems)
    output.mkdir(parents=True, exist_ok=False)
    (output / "freeze.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"id": result["id"], "files": len(entries),
                      "freeze_sha256": sha(output / "freeze.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--training", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--parameters", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.repo, args.training, args.model, args.parameters, args.output)
