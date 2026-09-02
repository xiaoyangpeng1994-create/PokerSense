"""Desktop-layer recoverable errors without optional capture dependencies."""


class LiveCaptureError(RuntimeError):
    """A live capture problem the user can act on."""


__all__ = ["LiveCaptureError"]
