"""Versioned AA8 completed-glyph transitions for offline development only."""


class GlyphTransitionsV2:
    """Emit stable visible glyph changes while failing closed around overlays."""

    def __init__(self, stable_frames=2, clear_frames=5,
                 rapid_change_guard_frames=5):
        if type(stable_frames) is not int or stable_frames < 1:
            raise ValueError("positive stability required")
        if type(clear_frames) is not int or clear_frames < stable_frames:
            raise ValueError("clear stability must be at least confirmation stability")
        if (type(rapid_change_guard_frames) is not int
                or rapid_change_guard_frames < stable_frames):
            raise ValueError("rapid change guard must cover confirmation stability")
        self.required = stable_frames
        self.clear_required = clear_frames
        self.rapid_change_guard_frames = rapid_change_guard_frames
        self.last_frame = None
        self.state = {}

    def begin_epoch(self):
        """Forget prior-hand glyph identity after a separate boundary detector."""
        self.state.clear()

    def observe(self, frame, values, *, suspended=False):
        if self.last_frame is not None and frame <= self.last_frame:
            raise ValueError("duplicate/backwards frame")
        if type(suspended) is not bool:
            raise ValueError("explicit suspended boolean required")
        if self.last_frame is None or frame != self.last_frame + 1:
            self.state.clear()
        self.last_frame = frame
        if suspended:
            # Preserve only confirmed/suppressed identities. A one-frame candidate
            # before an overlay cannot combine with one frame after the overlay.
            for slot, (_, _, emitted, emitted_frame, suppressed) in list(
                    self.state.items()):
                self.state[slot] = (
                    None, 0, emitted, emitted_frame, suppressed)
            return []
        events = []
        for slot, value in values.items():
            previous, count, emitted, emitted_frame, suppressed = self.state.get(
                slot, (None, 0, None, None, None))
            count = count + 1 if value == previous else 1
            threshold = self.clear_required if value is None else self.required
            if count >= threshold and value != emitted:
                if value is None:
                    emitted, emitted_frame, suppressed = None, None, None
                elif suppressed == value:
                    # A short clear or different transient does not re-arm an
                    # already rejected post-action glyph. Stable clear or an
                    # explicit hand epoch is required.
                    pass
                elif (emitted is not None and emitted_frame is not None
                      and frame - emitted_frame <= self.rapid_change_guard_frames):
                    suppressed = value
                else:
                    emitted, emitted_frame, suppressed = value, frame, None
                    events.append({"frame": frame, "slot": int(slot),
                                   "glyph": value})
            self.state[slot] = (
                value, count, emitted, emitted_frame, suppressed)
        return events
