"""Critical participation safeguards on the existing live candidate adapter."""

from copy import deepcopy

from .aa_live_context import LiveStateAdapter


class LiveStateAdapterV3(LiveStateAdapter):
    """Keep terminal state and contradictions within the same observed hand."""

    def __init__(self):
        super().__init__()
        self.status_conflicts = set()

    def observe(self, row):
        previous, epoch = deepcopy(self.participants), self.epoch
        value = super().observe(row)
        if self.epoch != epoch or value.get("observation_blocked"):
            self.status_conflicts.clear()
        else:
            for seat, old in previous.items():
                new = self.participants.get(seat) or {}
                if (old.get("state") in ("folded", "all_in")
                        and new.get("state") != old["state"]):
                    self.participants[seat] = old
                    self.status_conflicts.add(seat)
        for seat, cue in (row.get("participation") or {}).get("slots", {}).items():
            if cue.get("conflict"):
                self.status_conflicts.add(seat)
        # Full-size Hero cards can remain visible after folding. Positive current
        # action controls are a separate cue; card visibility alone never enters.
        if (self.epoch is not None and not value.get("observation_blocked")
                and row.get("current_actor") == 4
                and (row.get("actor_evidence") or {}).get("reason") == "hero_buttons"
                and (row.get("actor_evidence") or {}).get("hero_turn") is True):
            if self.participants.get("4", {}).get("state") in ("folded", "all_in"):
                self.status_conflicts.add("4")
            elif "4" not in self.status_conflicts:
                self.participants["4"] = {
                    "state": "active", "evidence_frame": row["frame"],
                    "evidence": "CURRENT_HERO_ACTION_CONTROLS", "epoch": self.epoch}
        value["participants"] = deepcopy(self.participants)
        value["critical_status_conflicts"] = sorted(self.status_conflicts)
        return value

    def snapshot(self, frame):
        value = super().snapshot(frame)
        value["critical_status_conflicts"] = sorted(
            getattr(self, "status_conflicts", set()))
        return value
