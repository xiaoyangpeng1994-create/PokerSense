"""Audit a complete public decision-opportunity JSON ledger without fitting."""

import argparse
import hashlib
import json
from pathlib import Path

from poker_engine.strategy.decision_opportunities_v1 import (
    audit_decision_opportunities,
)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key:" + key)
        result[key] = value
    return result


def analyze_file(path):
    raw = path.read_bytes()
    data = json.loads(raw, object_pairs_hook=unique_object)
    return {
        "scope": "DECISION_OPPORTUNITY_AUDIT_NO_MODEL_OR_LIVE_USE",
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        **audit_decision_opportunities(data),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = analyze_file(args.dataset)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (OSError, ValueError, TypeError, ArithmeticError) as exc:
        parser.exit(2, f"decision opportunities rejected: {exc}\n")
    print(json.dumps({key: report[key] for key in (
        "engineering_contract_status", "data_readiness", "ledger_count",
        "detail_count", "eligible_target_count", "blockers")},
        ensure_ascii=False, indent=2))
    return 2 if report["data_readiness"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
