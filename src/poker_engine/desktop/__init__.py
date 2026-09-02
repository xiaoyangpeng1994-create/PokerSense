"""Desktop shell: local server + native companion for the ADB realtime engine.

Not part of ``poker_engine.core``'s frozen contract layer -- this is glue
between the engine and a UI process, allowed to depend on optional
third-party packages (fastapi/uvicorn/pywebview) that the core/domain/vision
layers deliberately avoid.
"""
