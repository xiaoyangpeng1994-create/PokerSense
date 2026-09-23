"""Verify the frozen historical action-opportunity audit without opening media.

This verifies receipts and refusal claims; it does not turn action candidates,
actor cues, or frames into action opportunities.  The public-only mode is for
CI and explicitly cannot reverify the local historical evidence.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re


MANIFEST = Path(
    "configs/strategy/evaluation/action-likelihood-historical-audit-v1.json"
)
FROZEN_MANIFEST_SHA256 = (
    "d761e42050cb7d67ac9efd3eab1057bf9ed151ffe26ec033197bcd5c46eb3291"
)
DEFAULT_PRIVATE_ROOT = Path("G:/PokerSense_private")
RECEIPTS = (
    ("aa8_decision_readiness",
     "aa8_decision_opportunities_v1_20260914_v10/readiness.json"),
    ("visible_action_review",
     "aa8_action_truth_review_20260914_v1/result-v4.json"),
    ("actor_episode_census",
     "aa8_actor_episode_census_20260914_v6/episodes.json"),
    ("seven_pool_continuous_benchmark",
     "historical-benchmark-20260922/continuous-aggregate.json"),
)
EVIDENCE_UNITS = {
    "reviewed_visible_action_candidates": 36,
    "matched_visible_actions": 30,
    "matched_opponent_actions": 25,
    "matched_hero_actions": 5,
    "matched_river_actions_all_actors": 2,
    "actor_episode_candidates": 32,
    "episode_action_union_review_queue": 34,
    "same_rule_gameplay_sessions_with_candidate_actions": 1,
    "seven_pool_benchmark_frames": 120089,
    "seven_pool_river_retrieval_fragments": 57,
    "seven_pool_complete_legal_state_rows": 0,
}
MISSING = [
    "stable_opponent_and_cross_session_identity",
    "second_same_rule_gameplay_session",
    "decision_time_board_and_actor_and_action_order",
    "decision_time_pot_stacks_to_call_and_public_history",
    "complete_legal_action_menu_and_amount_semantics",
    "reviewed_room_rule_fingerprint_and_special_mode_exclusion",
    "action_onset_predecision_evidence",
    "all_opportunity_denominator_including_unmatched_and_unknown",
    "provenance_bound_offline_only_revealed_strength_labels",
]
OPPORTUNITY_CORPUS = {
    "complete_denominator_verified": False,
    "eligible_opportunities": [],
    "eligible_training_count": 0,
    "eligible_validation_count": 0,
    "all_actual_opportunities_reviewed": False,
    "later_revealed_strength_labels_usable": 0,
    "missing_or_unverified": MISSING,
}
MODEL_SUPPORT = {
    "player_specific": "INSUFFICIENT_DATA",
    "population_or_archetype": "INSUFFICIENT_DATA",
    "hand_strength_conditioned": "INSUFFICIENT_DATA",
}
FIXED_FIELDS = {
    "schema_version": 1,
    "audit_id": "ACTION_LIKELIHOOD_HISTORICAL_AUDIT_V1",
    "base_main_sha": "af67b4bb3ba56da2d4fdb70e0731583afc7aaf40",
    "scope": "existing structured historical A-poker evidence; no media decoding",
    "unit": (
        "one independently enumerated action opportunity, "
        "including UNKNOWN and refused rows"
    ),
    "distinct_evidence_units": EVIDENCE_UNITS,
    "opportunity_corpus": OPPORTUNITY_CORPUS,
    "model_support": MODEL_SUPPORT,
    "audit_status": "NO_EMPIRICAL_MODEL_FIT_PERMITTED",
    "strategy_eligible": False,
    "advice_emitted": False,
    "real_hand_acceptance_pending": True,
    "empirical_strategy": "NOT_ASSESSED",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest(raw, require_frozen_hash):
    if (require_frozen_hash
            and hashlib.sha256(raw).hexdigest() != FROZEN_MANIFEST_SHA256):
        raise ValueError("frozen_public_manifest_sha256_mismatch")
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("public_manifest_invalid_json") from exc
    if (not isinstance(document, dict)
            or set(document) != set(FIXED_FIELDS) | {"source_receipts"}):
        raise ValueError("public_manifest_schema_mismatch")
    for key, expected in FIXED_FIELDS.items():
        actual_json = json.dumps(document[key], sort_keys=True)
        expected_json = json.dumps(expected, sort_keys=True)
        if actual_json != expected_json:
            raise ValueError("public_manifest_field_mismatch:" + key)
    receipts = document["source_receipts"]
    if not isinstance(receipts, list) or len(receipts) != len(RECEIPTS):
        raise ValueError("public_manifest_receipts_mismatch")
    for index, ((role, relative_path), receipt) in enumerate(zip(RECEIPTS, receipts)):
        if (not isinstance(receipt, dict)
                or set(receipt) != {"role", "relative_private_path", "sha256"}
                or receipt.get("role") != role
                or receipt.get("relative_private_path") != relative_path
                or not isinstance(receipt.get("sha256"), str)
                or not SHA256_PATTERN.fullmatch(receipt["sha256"])):
            raise ValueError(f"public_manifest_receipt_schema_mismatch:{index}")
    return document


def audit(manifest_path=MANIFEST, private_root=DEFAULT_PRIVATE_ROOT,
          require_frozen_hash=True):
    """Return a deterministic refusal report; never parse private source data.

    ``private_root=None`` verifies public claims only and is suitable for CI.
    A private-file failure never changes the frozen zero-eligibility conclusion.
    """
    try:
        document = _validate_manifest(Path(manifest_path).read_bytes(),
                                      require_frozen_hash)
    except (OSError, ValueError) as exc:
        reason = (str(exc) if isinstance(exc, ValueError)
                  else "public_manifest_unreadable")
        return {
            "status": "REFUSED",
            "manifest_verified": False,
            "private_receipts_verified": False,
            "eligible_real_opportunities": 0,
            "model_support": MODEL_SUPPORT.copy(),
            "refusal_reasons": [reason],
        }

    report = {
        "status": ("PUBLIC_ONLY_NOT_LIVE_REVERIFIED" if private_root is None
                   else "VERIFIED_INSUFFICIENT_DATA"),
        "manifest_verified": True,
        "private_receipts_verified": False,
        "eligible_real_opportunities": 0,
        "model_support": document["model_support"],
        "refusal_reasons": list(
            document["opportunity_corpus"]["missing_or_unverified"]),
    }
    if private_root is None:
        report["refusal_reasons"].append("private_evidence_not_live_reverified")
        return report

    root = Path(private_root).resolve()
    receipt_failures = []
    for receipt in document["source_receipts"]:
        role = receipt["role"]
        path = (root / receipt["relative_private_path"]).resolve()
        if not path.is_relative_to(root):
            receipt_failures.append(f"private_receipt_outside_root:{role}")
        elif not path.is_file():
            receipt_failures.append(f"private_receipt_missing:{role}")
        else:
            try:
                if _sha256_file(path) != receipt["sha256"]:
                    receipt_failures.append(f"private_receipt_sha256_mismatch:{role}")
            except OSError:
                receipt_failures.append(f"private_receipt_unreadable:{role}")
    if receipt_failures:
        report["status"] = "REFUSED"
        report["refusal_reasons"] = receipt_failures + report["refusal_reasons"]
    else:
        report["private_receipts_verified"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument(
        "--public-only", action="store_true",
        help="verify public audit only; private evidence is not reverified")
    args = parser.parse_args()
    result = audit(args.manifest, None if args.public_only else args.private_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 1 if result["status"] == "REFUSED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
