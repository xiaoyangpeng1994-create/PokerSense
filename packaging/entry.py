"""PyInstaller entry point: a single script, no ``python -m`` package resolution.

Kept separate from ``poker_engine.desktop.app`` (the real module) so the
package itself never has to know it's being frozen.
"""

from poker_engine.desktop.app import main

if __name__ == "__main__":
    main()
