"""Generate the predeclared, all-opportunity synthetic calibration control.

The challenge is a code-path control, never historical A-poker evidence.  Its
offline label sidecar is separate from every decision-time feature record.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random


PROTOCOL = Path(
    "configs/strategy/evaluation/action-likelihood-synthetic-challenge-v1.json")
OUTPUT = Path(
    "configs/strategy/evaluation/action-likelihood-synthetic-opportunities-v1.json")
LABELS = Path("configs/strategy/evaluation/action-likelihood-synthetic-labels-v1.json")
PROTOCOL_SHA256 = "61e21217fa7dfb41837538731fa9628f4f6abb45266679ebfc98a5c9177031dc"


def _encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2)
            + "\n").encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _draw(rng, legal, weights):
    mass = [weights[action.partition(":")[0]] for action in legal]
    draw = rng.randrange(sum(mass))
    for action, weight in zip(legal, mass):
        if draw < weight:
            return action
        draw -= weight
    raise AssertionError("synthetic_draw_outside_mass")


def generate(protocol_path=PROTOCOL):
    raw = Path(protocol_path).read_bytes()
    if _sha(raw) != PROTOCOL_SHA256:
        raise ValueError("frozen_synthetic_challenge_drift")
    protocol = json.loads(raw)
    if (protocol["source_kind"]
            != "synthetic_action_opportunities_not_historical_calibration"
            or protocol["opportunities_per_session"] != 48
            or len(protocol["partitions"]) != 3):
        raise ValueError("unexpected_synthetic_challenge_scope")
    contexts = {item["id"]: item for item in protocol["contexts"]}
    rows, labels = [], []
    for partition in protocol["partitions"]:
        rng = random.Random(partition["seed"])
        for session_id in partition["session_ids"]:
            for index in range(protocol["opportunities_per_session"]):
                opponent_id = protocol["opponents"][(index // 2) % 2]
                seat = 1 if opponent_id == "synthetic-A" else 2
                context = contexts["unopened" if index % 2 == 0 else "facing_bet"]
                prior_actor = 0 if seat == 1 else 1
                before = ([] if context["id"] == "unopened" else
                          [{"seat_id": prior_actor, "kind": "bet", "target": "20"}])
                order = ([seat, 0 if seat == 2 else 2, 0 if seat == 1 else 1]
                         if not before else [prior_actor, seat,
                                             2 if seat == 1 else 0])
                stacks = {"0": "200", "1": "200", "2": "200"}
                commitments = {"0": "0", "1": "0", "2": "0"}
                if before:
                    stacks[str(prior_actor)] = "180"
                    commitments[str(prior_actor)] = "20"
                opportunity_id = f"{session_id}-hand-{index:03d}-seat-{seat}"
                public = {
                    "platform_id": "AA_SYNTHETIC_CONTROL",
                    "rule_fingerprint": protocol["rule_fingerprint"],
                    "table_size": protocol["table_size"],
                    "active_seats": protocol["active_seats"],
                    "street": "river",
                    "board": protocol["board"],
                    "seat_id": seat,
                    "position": "BB" if seat == 1 else "BTN",
                    "action_order": order,
                    "pot": context["pot"],
                    "stacks": stacks,
                    "street_committed": commitments,
                    "to_call": context["to_call"],
                    "prior_public_actions": before,
                    "legal_actions": context["legal_actions"],
                }
                truth = protocol["generator_truth_weights_evaluator_only"][
                    partition["regime"]][opponent_id]
                action = _draw(rng, public["legal_actions"], truth)
                receipt = {"opportunity_id": opportunity_id,
                           "session_id": session_id,
                           "hand_id": f"{session_id}-hand-{index:03d}",
                           "opponent_id": opponent_id,
                           "public": public,
                           "actual_action": action}
                rows.append({**receipt, "source_kind": "synthetic",
                             "source_sha256": _sha(_encoded(receipt)),
                             "unknown_reasons": []})
                labels.append({"opportunity_id": opportunity_id,
                               "status": "LATER_STRENGTH_NOT_AVAILABLE",
                               "later_revealed_strength": None})
    output = {"schema_version": 1, "challenge_sha256": PROTOCOL_SHA256,
              "source_kind": "synthetic_complete_opportunity_control",
              "opportunities": rows}
    offline = {"schema_version": 1, "challenge_sha256": PROTOCOL_SHA256,
               "scope": "offline_only_not_a_decision_feature",
               "labels": labels}
    return _encoded(output), _encoded(offline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("write", "check"))
    args = parser.parse_args()
    output, labels = generate()
    for path, raw in ((OUTPUT, output), (LABELS, labels)):
        if args.mode == "write":
            with path.open("xb") as stream:
                stream.write(raw)
        elif path.read_bytes() != raw:
            raise ValueError("synthetic_challenge_fixture_drift:" + str(path))
    print(json.dumps({"opportunities_sha256": _sha(output),
                      "offline_labels_sha256": _sha(labels),
                      "opportunities": len(json.loads(output)["opportunities"])},
                     sort_keys=True))


if __name__ == "__main__":
    main()
