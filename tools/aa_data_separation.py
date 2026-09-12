"""Protect reserved AA intervals without calling unreviewed intervals true hands."""


def validate_reservations(reservations):
    intervals = reservations["reserved_inclusive_intervals"]
    previous_end = -1
    exposed = reservations["known_exploration_frames"]
    development = reservations["known_development_inclusive_intervals"]
    for start, end in intervals:
        if (type(start) is not int or type(end) is not int or start < 0
                or end < start or start <= previous_end):
            raise ValueError("invalid or overlapping reservations")
        if any(start <= frame <= end for frame in exposed):
            raise ValueError("reservation contains an exposed frame")
        if any(start <= b and a <= end for a, b in development):
            raise ValueError("reservation overlaps development")
        previous_end = end


def assert_training_frames(frames, reservations):
    validate_reservations(reservations)
    for frame in frames:
        if type(frame) is not int or frame < 0:
            raise ValueError("invalid source frame")
        intervals = reservations["reserved_inclusive_intervals"]
        if any(a <= frame <= b for a, b in intervals):
            raise ValueError("reserved frame cannot enter training or tuning")


def eligible_hand(start, end, reservations, *, boundaries_reviewed=False):
    validate_reservations(reservations)
    if (type(start) is not int or type(end) is not int or start < 0 or end < start
            or boundaries_reviewed is not True):
        return False
    return any(a <= start <= end <= b
               for a, b in reservations["reserved_inclusive_intervals"])
