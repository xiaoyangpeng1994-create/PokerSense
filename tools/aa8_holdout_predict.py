"""Evaluation-only harness created after the candidate freeze, before prediction.

No training, inventory relabeling or holdout tuning. Frozen components and their
parameters are composed with explicitly copied, reviewable per-frame logic.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

from tools.aa8_holdout_plan import sha, utc, validate_freeze


FIELDS = ("actor", "street_wagers", "actions", "pot", "stacks", "street",
          "hand", "participation", "hero_cards", "board_cards",
          "insurance", "mushroom", "bomb")


def registry_range(registry):
    if "candidate_range_seconds" in registry:
        return tuple(map(float, registry["candidate_range_seconds"]))
    return registry["allowed_start_seconds"], registry["allowed_end_seconds"]


def bound_file(path, freeze):
    path = Path(path).resolve()
    entries = [v for v in freeze["files"] if Path(v["path"]).resolve() == path]
    if len(entries) != 1 or sha(path) != entries[0]["sha256"]:
        raise ValueError("input is not bound to original candidate freeze")
    return path


def validate_rows(rows, registry, audit, *, require_hash=True):
    """Metadata only. Preroll belongs to holdout too, never crosses its range."""
    if registry.get("role") != "holdout" or registry.get("audit_sha256") != audit:
        raise ValueError("holdout registry/audit mismatch")
    hands = registry.get("hands", [])
    if not hands or len({h["id"] for h in hands}) != len(hands):
        raise ValueError("unique registered complete hands required")
    allowed = set()
    scored = set()
    for hand in hands:
        first, last, preroll = (hand[k] for k in (
            "first_frame", "last_frame", "preroll_first_frame"))
        if (hand.get("boundaries_verified") is not True or
                any(type(v) is not int or v < 0 for v in (first, last, preroll))
                or not preroll <= first <= last or first - preroll > 30):
            raise ValueError("invalid complete-hand/preroll bounds")
        owned = set(range(first, last + 1))
        if scored & owned:
            raise ValueError("hand ownership overlaps")
        scored |= owned
        allowed |= set(range(preroll, last + 1))
    if list(rows) != sorted(allowed):
        raise ValueError("exact ordered whole-hand plus bounded preroll rows required")
    previous_pts = None
    lo, hi = registry_range(registry)
    for frame, row in rows.items():
        pts = float(row["pts_seconds"])
        if (row.get("role") != "holdout" or row.get("global_frame") != frame
                or not lo <= pts < hi or
                previous_pts is not None and pts <= previous_pts):
            raise ValueError("invalid holdout identity/role/time")
        if require_hash and (not isinstance(row.get("sha256"), str)
                             or len(row["sha256"]) != 64):
            raise ValueError("source image hash required")
        previous_pts = pts
    return scored


def load_holdout(pool, row):
    """Only callable during authorized prediction; never imports dev inventory."""
    from tools.wpk_video_dataset import read_image
    if row.get("role") != "holdout":
        raise ValueError("not a holdout frame")
    path = (pool / row["file"]).resolve()
    if not path.is_relative_to(pool.resolve()) or sha(path) != row["sha256"]:
        raise ValueError("holdout path/hash mismatch")
    return read_image(path)


class FrozenPredictionState:
    """Explicit composition equivalent to frozen aa8_visual_pipeline.run."""
    def __init__(self, source, context_source, late_source, bank_path,
                 profile_path, heads_path, audit):
        import numpy as np
        from tools import aa8_visual_pipeline as p
        from tools.aa8_continuous_state import CandidateReader, ContinuousEvidence
        from tools.aa8_special_modes import SOURCES
        from tools.aa8_special_spatial import AA8SpatialModeReader
        from tools.aa8_cards import AA8CardReader
        from tools.aa8_participation import ParticipationReader, RosterCandidates
        from tools.aa8_hand_transition import CenterDealCue, HandTransitionCandidates
        self.p, self.audit = p, audit
        self.profile = json.loads(profile_path.read_text())
        sources = p.inventory(source, audit)
        contexts = p.inventory(context_source, audit)
        late = json.loads((late_source / "samples.json").read_text())
        if late["audit_sha256"] != audit or any(
                r["role"] != "development" for r in late["samples"]):
            raise ValueError("late training roles/audit mismatch")
        late_rows = {r["global_frame"]: r for r in late["samples"]}
        images = {key: p.load(source, sources[frame])
                  for key, (frame, _) in p.TEMPLATES.items()}
        self.reader = p.AA8GlyphReader(self.profile, images,
                                       p.load(source, sources[1320]))
        with np.load(bank_path, allow_pickle=False) as bank:
            amounts = p.GrayAmountRecognizer(
                bank["features"], bank["labels"], augment=True)
        self.cache = p.ExactPatchCache(amounts)
        references = [p.load(source, sources[f]) for f in (1500, 1650, 1950, 2550)]
        self.prefixes = []
        for reference in references:
            binary = p.pot_mask(p.region(reference))
            colon = p.colon_x(binary)
            if colon is None or colon < 34:
                raise ValueError("reviewed pot prefix missing")
            self.prefixes.append(binary[:, colon - 34:colon + 2])
        self.context_reader = CandidateReader(self.profile, amounts, references[0],
                                              timer_reference=p.load(
                                                  context_source, contexts[4755]))
        self.temporal = ContinuousEvidence()
        self.modes = AA8SpatialModeReader(
            {k: p.load(context_source, contexts[f]) for k, f in SOURCES.items()},
            amount_bank=amounts, mushroom_references=[references[0]],
            countdown_images={s: p.load(late_source, late_rows[f])
                              for s, f in ((5, 25204), (7, 28654))},
            buyin_overlay_image=p.load(late_source, late_rows[29104]))
        self.cards = AA8CardReader(heads_path, preprocessing="gaussian_050")
        self.seats = ParticipationReader(self.profile, p.load(source, sources[1320]),
                                         p.load(source, sources[3319]),
                                         waiting_next=p.load(source, sources[2400]))
        self.roster = RosterCandidates()
        self.deal = CenterDealCue(p.load(source, sources[1263]))
        self.hands = HandTransitionCandidates()
        self.tracker = p.GlyphTransitions()

    def read(self, image, frame, sample):
        from tools.aa8_continuous_state import ContinuousEvidence
        p = self.p
        special = self.modes.recognize(image)
        glyphs = self.reader.recognize(image)
        supported = (self.reader.scene.recognize(image)["scene"] == "AA_TABLE_CANDIDATE"
                     and not special["block_state_updates"])
        if not supported:
            glyphs = dict.fromkeys(map(str, range(8)))
        stacks = {str(s): self.cache.read(("stack", s), p.stack_patch(
            image, self.profile["slots"][s]["stack"]) if supported else None)
            for s in range(8)}
        pot = self.cache.read("pot", p.unique_pot_patch(image, self.prefixes)
                              if supported else None)
        context = self.context_reader.read(image) if supported else None
        if context is not None:
            hero = p.hero_turn_candidate(image)
            if hero["hero_turn"] is True:
                context["actor"] = 4 if context["actor"] is None else None
                context["actor_evidence"] = {
                    **hero, "timer_suffix_verified": False,
                    "reason": "hero_buttons" if context["actor"] == 4
                    else "conflicting_actor_cues"}
            context["pot"] = pot["value"]
            context["stacks"] = {s: r["value"] for s, r in stacks.items()}
            continuous = self.temporal.observe(frame, context)
        else:
            self.temporal = ContinuousEvidence()
            continuous = None
        card_read = self.cards.read(image if supported else None, frame,
                                    float(sample["pts_seconds"]), self.audit)
        transitions = self.tracker.observe(frame, glyphs)
        row = {"frame": frame, "pts_seconds": sample["pts_seconds"],
               "source_sha256": sample["sha256"], "scene_supported": supported,
               "glyphs": glyphs, "glyph_transitions": transitions,
               "stacks": stacks, "pot": pot,
               "current_actor": context["actor"] if context else None,
               "actor_evidence": context["actor_evidence"] if context else None,
               "street_wagers": context["wagers"] if context else None,
               "board_count": context["board_count"] if context else None,
               "continuous_context": continuous, "special_modes": special,
               "cards": card_read, "strategy_eligible": False}
        transition = self.hands.observe(
            row, self.deal.recognize(image) if supported else {})
        if transition["candidate"]:
            self.roster.begin_hand("candidate_" + str(
                transition["candidate"]["first_candidate_frame"]))
        cues = self.seats.recognize(image if supported else None)
        row["participation"] = self.roster.observe(frame, cues, glyphs)
        row["hand_transition"] = transition
        return row


def normalize_fields(row):
    """Accepted visual candidates only; no synthetic legal/fee/trigger truth."""
    fields = {k: {"status": "UNKNOWN", "value": None} for k in FIELDS}

    def known(key, value):
        fields[key] = {"status": "KNOWN", "value": value,
                       "semantics": "accepted_visual_candidate_not_legal_state"}

    if not row.get("scene_supported"):
        return fields
    if type(row.get("current_actor")) is int:
        known("actor", row["current_actor"])
    if row.get("pot", {}).get("value") is not None:
        known("pot", row["pot"]["value"])
    stacks = {s: v.get("value") for s, v in row.get("stacks", {}).items()}
    if set(stacks) == set(map(str, range(8))) and all(
            v is not None for v in stacks.values()):
        known("stacks", stacks)
    wagers = row.get("street_wagers")
    if isinstance(wagers, dict) and set(wagers) == set(map(str, range(8))) and all(
            v is not None for v in wagers.values()):
        known("street_wagers", wagers)
    cards = row.get("cards", {})
    if cards.get("hero") is not None:
        known("hero_cards", cards["hero"])
    count = row.get("board_count")
    board = cards.get("board_slots", [None] * 5)
    if count in (3, 4, 5) and all(board[:count]) and not any(board[count:]):
        known("board_cards", board[:count])
    if row.get("special_modes", {}).get("insurance") == "VISIBLE":
        known("insurance", {"active": True, "semantics": "visible_insurance_ui"})
    return fields


def predict_rows(*, freeze_path, supplement_path, registry_path, source,
                 context_source, late_source, target, bank_path, profile_path,
                 heads_path, rowmap, output):
    """Explicit authorized call only; this module intentionally has no CLI."""
    freeze = json.loads(freeze_path.read_text())
    supplement = json.loads(supplement_path.read_text())
    registry = json.loads(registry_path.read_text())
    started = datetime.now(timezone.utc).isoformat()
    errors = validate_freeze(freeze, freeze_path.parent, started)
    if errors:
        raise ValueError(errors)
    chronology = utc(freeze["frozen_at_utc"]) < utc(
        supplement["frozen_at_utc"]) < utc(started)
    if (supplement.get("base_freeze_sha256") != sha(freeze_path)
            or registry.get("freeze_sha256") != sha(freeze_path)
            or supplement.get("evaluation_harness_sha256") != sha(Path(__file__))
            or supplement.get("registry_sha256") != sha(registry_path)
            or supplement.get("target_manifest_sha256") != sha(target / "samples.json")
            or not chronology):
        raise ValueError("evaluation supplement lineage/hash/time mismatch")
    for path in (source / "samples.json", context_source / "samples.json",
                 late_source / "samples.json", bank_path, profile_path, heads_path):
        bound_file(path, freeze)
    manifest = json.loads((target / "samples.json").read_text())
    if manifest.get("registry_sha256") != sha(registry_path):
        raise ValueError("target manifest registry binding mismatch")
    if (len(rowmap) != len(manifest["samples"]) or
            rowmap != {r["global_frame"]: r for r in manifest["samples"]}):
        raise ValueError("rowmap differs from target manifest")
    scored = validate_rows(rowmap, registry, manifest["audit_sha256"])
    # The range itself must originate in the frozen split, not supplied ad hoc.
    split_path = next(Path(v["path"]) for v in freeze["files"] if Path(
        v["path"]).name == "aa8_recording_split_plan_20260909.json")
    split = json.loads(split_path.read_text())
    ranges = [r for r in split["ranges"] if r["role"] == "holdout_candidate"]
    lo, hi = registry_range(registry)
    if not any(float(r["start_inclusive"]) == lo and
               float(r["end_exclusive"]) == hi for r in ranges):
        raise ValueError("registry range does not match frozen holdout range")
    from tools import aa8_visual_pipeline as pipeline
    pipeline_path = bound_file(Path(pipeline.__file__), freeze)
    state = FrozenPredictionState(source, context_source, late_source, bank_path,
                                  profile_path, heads_path, manifest["audit_sha256"])
    output.mkdir(parents=True, exist_ok=False)
    with (output / "observations.jsonl").open("x", encoding="utf-8") as stream:
        for frame, sample in rowmap.items():
            row = state.read(load_holdout(target, sample), frame, sample)
            row["evaluation_fields"] = normalize_fields(row)
            row["scored_whole_hand_frame"] = frame in scored
            stream.write(json.dumps(row) + "\n")
    errors = validate_freeze(freeze, freeze_path.parent, started)
    if errors or sha(Path(__file__)) != supplement["evaluation_harness_sha256"]:
        raise ValueError("freeze changed during prediction; retain failure artifacts")
    report = {"base_freeze_sha256": sha(freeze_path),
              "evaluation_supplement_sha256": sha(supplement_path),
              "evaluation_harness_sha256": sha(Path(__file__)),
              "harness_created_after_base_freeze": True,
              "prediction_start_utc": started, "registry_sha256": sha(registry_path),
              "target_manifest_sha256": sha(target / "samples.json"),
              "copied_orchestration_source_sha256": sha(pipeline_path),
              "frames": len(rowmap),
              "scored_frames": len(scored), "labels_used_for_prediction": False,
              "model_or_parameter_changes": False, "target_role": "holdout",
              "observations_sha256": sha(output / "observations.jsonl"),
              "full_visual_acceptance": False, "strategy_eligible": False}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report
