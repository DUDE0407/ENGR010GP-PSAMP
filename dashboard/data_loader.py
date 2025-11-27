"""Data loading utilities for the dashboard."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict

import pandas as pd

from analysis_core import (
    GridStandards,
    calculate_basic_statistics,
    calculate_power_quality_indices,
    compare_to_standards,
    identify_load_patterns,
    load_power_data,
    perform_fault_analysis,
)

from .config import CONFIG


STANDARDS = GridStandards()


@lru_cache(maxsize=1)
def _load_dataset() -> pd.DataFrame:
    return load_power_data(CONFIG.data_file).copy()


@lru_cache(maxsize=1)
def _load_patterns() -> Dict[str, pd.DataFrame]:
    patterns = identify_load_patterns(_load_dataset())
    return {name: frame.copy() for name, frame in patterns.items()}


@lru_cache(maxsize=1)
def load_basic_statistics() -> pd.DataFrame:
    return calculate_basic_statistics(_load_dataset()).copy()


@lru_cache(maxsize=1)
def load_standard_comparison() -> pd.DataFrame:
    return compare_to_standards(_load_dataset(), STANDARDS).copy()


@lru_cache(maxsize=1)
def load_power_quality_indices() -> pd.DataFrame:
    return calculate_power_quality_indices(_load_dataset(), STANDARDS).copy()


@lru_cache(maxsize=1)
def load_fault_summary() -> pd.DataFrame:
    return perform_fault_analysis(_load_dataset(), STANDARDS).copy()


@lru_cache(maxsize=1)
def load_daily_pattern() -> pd.DataFrame:
    return _load_patterns()["daily"].copy()


@lru_cache(maxsize=1)
def load_weekly_pattern() -> pd.DataFrame:
    return _load_patterns()["weekly"].copy()


@lru_cache(maxsize=1)
def load_hourly_profile() -> pd.DataFrame:
    return _load_patterns()["hourly_profile"].copy()


def clear_cache() -> None:
    """Reset cached data so the UI can pick up refreshed files."""

    _load_dataset.cache_clear()
    _load_patterns.cache_clear()
    load_basic_statistics.cache_clear()
    load_standard_comparison.cache_clear()
    load_power_quality_indices.cache_clear()
    load_fault_summary.cache_clear()
    load_daily_pattern.cache_clear()
    load_weekly_pattern.cache_clear()
    load_hourly_profile.cache_clear()
