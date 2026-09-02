#!/usr/bin/env python3
"""Deterministic local-resolver protocol double used by process tests."""

from __future__ import annotations

import json
import sys
import time


def main() -> int:
    mode = sys.argv[1]
    request = json.loads(sys.stdin.buffer.read())
    context = request["context"]
    identity = {
        "hand_id": context["request"]["hand_id"],
        "state_version": context["request"]["state_version"],
        "request_id": context["request"]["request_id"],
    }
    response = {
        "schema_version": 1,
        "type": "ResolverResponse",
        "provider_id": request["provider_id"],
        "source_version": request["source_version"],
        "identity": identity,
        "status": "CONVERGED",
        "iterations": 1200,
        "exploitability_bb100": "0.02",
        "action_probabilities": {"check": "0.25", "raise": "0.75"},
        "recommended_sizes": {"raise": ["2.5"]},
        "action_ev": {"check": "0", "raise": "1.5"},
        "confidence": 0.85,
        "state_match_score": 0.9,
        "match_dimensions": [{
            "name": "resolver_tree_abstraction",
            "requested": "live_state",
            "matched": "solver_tree_v3",
            "distance": "0.1",
            "maximum_distance": "1",
        }],
    }
    if mode == "sleep":
        time.sleep(0.25)
    elif mode == "exit":
        return 7
    elif mode == "bad-json":
        sys.stdout.write("not-json")
        return 0
    elif mode == "large":
        sys.stdout.write("x" * 10_000)
        return 0
    elif mode == "not-converged":
        response["status"] = "NOT_CONVERGED"
    elif mode == "no-strategy":
        response["status"] = "NO_STRATEGY"
    elif mode == "wrong-identity":
        response["identity"]["request_id"] = "wrong"
    elif mode == "wrong-version":
        response["source_version"] = "wrong"
    elif mode == "bad-probabilities":
        response["action_probabilities"] = {"check": "0.9", "raise": "0.9"}
    elif mode == "missing-match-dimensions":
        response.pop("match_dimensions")
    elif mode == "overstated-match-score":
        response["state_match_score"] = 0.91
    elif mode == "high-exploitability":
        response["exploitability_bb100"] = "0.5"
    elif mode == "missing-exploitability":
        response["exploitability_bb100"] = None
    elif mode != "success":
        raise ValueError(f"unknown mode {mode}")
    sys.stdout.write(json.dumps(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
