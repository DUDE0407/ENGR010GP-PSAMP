"""Legacy placeholder for the retired Streamlit dashboard."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    raise ImportError(
        "The Streamlit dashboard has been replaced by the Pygame implementation. "
        "Importing 'dashboard.sections' is no longer supported."
    )



