"""Explicit manual boundary context for private offline development replays."""

from tools.aa8_state_adapter_v2 import AA8StateAdapterV2


class ObserverDevelopmentContext(AA8StateAdapterV2):
    """A hand interval is context, never proof of opening money or legal state.

    Caller must verify the registry and source receipt hashes before constructing
    this object. It must create a fresh instance per registered hand.
    """

    def __init__(self, *, first_frame, last_frame, receipt_sha256,
                 registry_sha256):
        super().__init__()
        if (type(first_frame) is not int or type(last_frame) is not int
                or first_frame < 0 or last_frame < first_frame):
            raise ValueError("invalid development interval")
        for digest in (receipt_sha256, registry_sha256):
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("source and registry hashes required")
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.receipt_sha256 = receipt_sha256
        self.registry_sha256 = registry_sha256
        self.started = False

    def observe(self, row):
        frame = row.get("frame")
        if (type(frame) is not int
                or not self.first_frame <= frame <= self.last_frame):
            raise ValueError("frame outside registered development hand")
        if row.get("source_receipt_sha256") != self.receipt_sha256:
            raise ValueError("source receipt mismatch")
        if not self.started:
            if frame != self.first_frame:
                raise ValueError("must start at registered first frame")
            self.started = True
            modes = row.get("special_modes") or {}
            if (row.get("scene_supported") is True
                    and row.get("board_count") == 0
                    and not modes.get("block_state_updates")
                    and modes.get("insurance") != "VISIBLE"):
                self._new_epoch(frame, frame, "MANUAL_DEVELOPMENT_BOUNDARY")
        result = super().observe(row)
        result["manual_development_registry_sha256"] = self.registry_sha256
        result["complete_legal_state"] = False
        result["strategy_eligible"] = False
        return result
