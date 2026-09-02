PYTHON ?= python

.PHONY: install test lint clean run-desktop run-desktop-server package

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m flake8 src tests 2>/dev/null || echo "flake8 not installed (dev dep); run 'make install'"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache build dist *.egg-info

# Desktop shell (needs the `desktop` extra: pip install -e ".[dev,desktop]").
# Reads a real calibrated poker window; use tools/list_windows.py to choose a
# same-titled window explicitly when needed.
run-desktop:
	$(PYTHON) -m poker_engine.desktop.app $(ARGS)

# Server only (no native window) -- useful for previewing ui/ in a browser.
run-desktop-server:
	$(PYTHON) -m poker_engine.desktop.server $(ARGS)

# Build a local .app (macOS) / folder (Windows) via PyInstaller. Needs the
# `packaging` extra: pip install -e ".[dev,desktop,packaging]".
# CI builds + signs/notarizes both platforms on every push -- see
# .github/workflows/build-desktop.yml.
package:
	pyinstaller packaging/pokersense.spec --distpath dist --workpath build --noconfirm
