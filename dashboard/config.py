"""Centralized configuration for the dashboard application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DashboardConfig:
    """Encapsulate frequently used filesystem locations."""

    project_root: Path = Path(__file__).resolve().parents[1]

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "analysis_outputs" / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.project_root / "analysis_outputs" / "figures"

    @property
    def data_file(self) -> Path:
        return self.project_root / "power_system_data.csv"


CONFIG = DashboardConfig()
