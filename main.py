"""Entry point for launching the Pygame dashboard."""

from __future__ import annotations

from dashboard import run


if __name__ == "__main__":
    # Delegate to the dashboard package so the entry point stays minimal.
    run()
