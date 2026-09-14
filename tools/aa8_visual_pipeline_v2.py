"""AA8 V2 offline observations with overlay-safe glyph de-duplication."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from poker_engine.perceptual.vision.gray_amount_recognizer import GrayAmountRecognizer
from tools.aa8_action_reader import AA8GlyphReader, TEMPLATES
from tools.aa8_action_transfer import inventory, load, sha
from tools.aa8_glyph_transitions_v2 import GlyphTransitionsV2
from tools.aa8_unmarked_money import pot_patch
from tools.aa_amount_candidate import stack_patch
from tools.aa_pot_candidate import colon_x, pot_mask, region
from tools.aa8_hero_turn import hero_turn_candidate


def unique_pot_patch(image, prefixes):
    """Different reviewed rasterizations must agree on one numeric crop."""
    candidates = {}
    for prefix in prefixes:
        patch = pot_patch(image, prefix)
        if patch is not None:
            key = (patch.shape, hashlib.sha256(patch.tobytes()).hexdigest())
            candidates[key] = patch
    return next(iter(candidates.values())) if len(candidates) == 1 else None


class ExactPatchCache:
    """Reuse only byte-identical current pixels, never last known field values."""

    def __init__(self, bank):
        self.bank = bank
        self.entries = {}

    def read(self, key, patch):
        if patch is None or patch.size == 0:
            self.entries.pop(key, None)
            return {"value": None, "reason": "empty"}
        fingerprint = (patch.shape, str(patch.dtype),
                       hashlib.sha256(patch.tobytes()).hexdigest())
        old = self.entries.get(key)
        if old is not None and old[0] == fingerprint:
            return old[1].copy()
        result = asdict(self.bank.diagnose(patch))
        self.entries[key] = (fingerprint, result)
        return result.copy()


def run(source, target, bank_path, profile_path, output, context_source=None,
        late_source=None, heads_path=None):
    from tools.aa8_continuous_state import CandidateReader, ContinuousEvidence
    from tools.aa8_special_modes import SOURCES
    from tools.aa8_special_spatial import AA8SpatialModeReader
    from tools.aa8_cards import AA8CardReader
    from tools.aa8_participation import ParticipationReader, RosterCandidates
    from tools.aa8_hand_transition import CenterDealCue, HandTransitionCandidates
    source_manifest = json.loads((source / "samples.json").read_text())
    audit = source_manifest["audit_sha256"]
    sources = inventory(source, audit)
    targets = inventory(target, audit)
    context_source = context_source or target
    context_rows = inventory(context_source, audit)
    profile = json.loads(profile_path.read_text())
    inputs = {str(p.resolve()): sha(p) for p in (
        source / "samples.json", target / "samples.json", bank_path, profile_path,
        Path(__file__), Path(__file__).with_name("aa8_action_reader.py"),
        Path(__file__).with_name("aa8_glyph_transitions_v2.py"),
        Path(__file__).with_name("aa8_unmarked_money.py"),
        Path(__file__).with_name("aa8_continuous_state.py"),
        Path(__file__).with_name("aa8_hero_turn.py"),
        Path(__file__).with_name("aa_hero_turn.py"),
        Path(__file__).with_name("aa8_special_modes.py"),
        Path(__file__).with_name("aa8_special_spatial.py"),
        Path(__file__).with_name("aa8_cards.py"),
        Path(__file__).with_name("aa8_card_preflight.py"),
        Path(__file__).with_name("aa8_hand_transition.py"),
        Path(__file__).with_name("aa8_participation.py"),
        context_source / "samples.json")}
    if heads_path is None:
        raise ValueError("explicit frozen card heads required")
    inputs[str(heads_path.resolve())] = sha(heads_path)
    images = {key: load(source, sources[frame])
              for key, (frame, _) in TEMPLATES.items()}
    reader = AA8GlyphReader(profile, images, load(source, sources[1320]))
    with np.load(bank_path, allow_pickle=False) as bank:
        amounts = GrayAmountRecognizer(bank["features"], bank["labels"], augment=True)
    cache = ExactPatchCache(amounts)
    references = [load(source, sources[f]) for f in (1500, 1650, 1950, 2550)]
    prefixes = []
    for reference in references:
        binary = pot_mask(region(reference))
        colon = colon_x(binary)
        if colon is None or colon < 34:
            raise ValueError("reviewed pot prefix missing")
        prefixes.append(binary[:, colon - 34:colon + 2])
    # These source images are now development training, never independent tests.
    context_reader = CandidateReader(profile, amounts, references[0],
                                     timer_reference=load(context_source,
                                                          context_rows[4755]))
    temporal = ContinuousEvidence()
    extra = {}
    late_rows = {}
    if late_source is not None:
        metadata = json.loads((late_source / "samples.json").read_text())
        if (metadata["audit_sha256"] != audit or
                any(r["role"] != "development" for r in metadata["samples"])):
            raise ValueError("late templates must be from the development recording")
        late_rows = {r["global_frame"]: r for r in metadata["samples"]}
        inputs[str((late_source / "samples.json").resolve())] = sha(
            late_source / "samples.json")
        extra = {"countdown_images": {
            s: load(late_source, late_rows[f]) for s, f in ((5, 25204), (7, 28654))},
            "buyin_overlay_image": load(late_source, late_rows[29104])}
    modes = AA8SpatialModeReader(
        {k: load(context_source, context_rows[f]) for k, f in SOURCES.items()},
        amount_bank=amounts, mushroom_references=[references[0]], **extra)
    cards = AA8CardReader(heads_path, preprocessing="gaussian_050")
    seats = ParticipationReader(profile, load(source, sources[1320]),
                                load(source, sources[3319]),
                                waiting_next=load(source, sources[2400]))
    roster = RosterCandidates()
    deal = CenterDealCue(load(source, sources[1263]))
    hands = HandTransitionCandidates()
    tracker, events, counters = GlyphTransitionsV2(), [], Counter()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    # Output is an experiment artifact, never a released acceptance result.
    with (output / "observations.jsonl").open("w", encoding="utf-8") as stream:
        for frame, sample in targets.items():
            image = load(target, sample)
            special = modes.recognize(image)
            glyphs = reader.recognize(image)
            supported = (reader.scene.recognize(image)["scene"] == "AA_TABLE_CANDIDATE"
                         and not special["block_state_updates"])
            if not supported:
                glyphs = dict.fromkeys(map(str, range(8)))
            stacks = {str(s): cache.read(("stack", s), stack_patch(
                image, profile["slots"][s]["stack"]) if supported else None)
                for s in range(8)}
            pot = cache.read("pot", unique_pot_patch(image, prefixes)
                             if supported else None)
            context = context_reader.read(image) if supported else None
            if context is not None:
                hero = hero_turn_candidate(image)
                if hero["hero_turn"] is True:
                    # An opponent ring conflicting with Hero buttons is ambiguous.
                    context["actor"] = 4 if context["actor"] is None else None
                    context["actor_evidence"] = {
                        **hero, "timer_suffix_verified": False,
                        "reason": "hero_buttons" if context["actor"] == 4
                        else "conflicting_actor_cues"}
                context["pot"] = pot["value"]
                context["stacks"] = {s: r["value"] for s, r in stacks.items()}
                continuous = temporal.observe(frame, context)
            else:
                temporal = ContinuousEvidence()
                continuous = None
            card_read = cards.read(image if supported else None, frame,
                                   float(sample["pts_seconds"]), audit)
            counters["supported_scene_frames"] += int(supported)
            counters["known_pot_frames"] += int(pot["value"] is not None)
            counters["known_stack_fields"] += sum(
                r["value"] is not None for r in stacks.values())
            counters["actor_candidate_frames"] += int(
                context is not None and context["actor"] is not None)
            counters["known_wager_fields"] += sum(v is not None for v in (
                context["wagers"].values() if context else ()))
            transitions = tracker.observe(frame, glyphs, suspended=not supported)
            events.extend(transitions)
            row = {"frame": frame, "pts_seconds": sample["pts_seconds"],
                   "source_sha256": sample["sha256"], "scene_supported": supported,
                   "glyphs": glyphs, "glyph_transitions": transitions,
                   "stacks": stacks, "pot": pot,
                   "current_actor": context["actor"] if context else None,
                   "actor_evidence": context["actor_evidence"] if context else None,
                   "street_wagers": context["wagers"] if context else None,
                   "board_count": context["board_count"] if context else None,
                   "continuous_context": continuous, "special_modes": special,
                   "cards": card_read,
                   "strategy_eligible": False}
            transition = hands.observe(row, deal.recognize(image) if supported else {})
            if transition["candidate"]:
                roster.begin_hand("candidate_" + str(
                    transition["candidate"]["first_candidate_frame"]))
            cues = seats.recognize(image if supported else None)
            row["participation"] = roster.observe(frame, cues, glyphs)
            row["hand_transition"] = transition
            stream.write(json.dumps(row) + "\n")
            if (frame - min(targets)) % 600 == 0:
                print(f"Observed through frame {frame}", flush=True)
    if any(sha(Path(path)) != expected for path, expected in inputs.items()):
        raise ValueError("input implementation changed during run; retain failure")
    report = {"input_hashes": inputs, "frames": len(targets),
              "coverage": dict(counters),
              "events": events, "elapsed_seconds": time.perf_counter() - started,
              "additional_development_templates": {
                  "timer": context_rows[4755], "special_modes": {
                      k: context_rows[f] for k, f in SOURCES.items()},
                  "late_spatial": {str(f): late_rows[f] for f in (
                      (25204, 28654, 29104) if late_rows else ())},
                  "pot_prefixes": [sources[f] for f in (1500, 1650, 1950, 2550)]},
              "coverage_is_not_accuracy": True, "independent_holdout": False,
              "card_preprocessing": "gaussian_050",
              "full_visual_acceptance": False, "strategy_eligible": False,
              "glyph_transition_policy": {
                  "stable_frames": tracker.required,
                  "clear_frames": tracker.clear_required,
                  "rapid_change_guard_frames": tracker.rapid_change_guard_frames,
                  "unsupported_scene_frames_suspended": True,
                  "unconfirmed_streaks_cross_suspension": False,
                  "stable_clear_required_after_suppression": True,
                  "hand_epoch_reset": (
                      "not_applied_without_authoritative_boundary")},
              "observations_sha256": sha(output / "observations.jsonl")}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("events", "input_hashes")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "target", "bank", "profile", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--context-source", type=Path)
    parser.add_argument("--late-source", type=Path)
    parser.add_argument("--heads", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.target, args.bank, args.profile, args.output,
        args.context_source, args.late_source, args.heads)
