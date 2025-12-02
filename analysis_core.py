"""Reusable analysis utilities for power system datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GridStandards:
    voltage_min: float = 0.95
    voltage_max: float = 1.05
    voltage_fault_min: float = 0.90
    voltage_fault_max: float = 1.10
    current_max: float = 120.0
    power_factor_min: float = 0.90
    power_factor_fault: float = 0.80
    power_factor_target: float = 0.95


def load_power_data(csv_path: Path) -> pd.DataFrame:
    """Load and tidy the power system data file."""

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    return df


def calculate_basic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-station mean, median, and standard deviation for key fields."""

    metrics = [
        "voltage_pu",
        "current_pu",
        "real_power_mw",
        "reactive_power_mvar",
        "power_factor",
    ]
    grouped = df.groupby("station_id")[metrics]
    stats = grouped.agg(["mean", "median", "std"])
    # Flatten the multi-index columns so callers get simple metric labels.
    stats.columns = ["_".join(col) for col in stats.columns]
    return stats.reset_index()


def identify_load_patterns(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Capture daily, weekly, and diurnal load signatures for each station."""

    indexed = df.set_index("timestamp")
    # Averaging the dataset at daily and weekly resolution highlights long-term trends.
    grouped = indexed.groupby("station_id")

    daily_real = grouped["real_power_mw"].resample("D").mean().reset_index()
    daily_reactive = grouped["reactive_power_mvar"].resample("D").mean().reset_index()
    daily = daily_real.merge(
        daily_reactive,
        on=["station_id", "timestamp"],
        how="left",
    )

    weekly_real = grouped["real_power_mw"].resample("W").mean().reset_index()
    weekly_reactive = grouped["reactive_power_mvar"].resample("W").mean().reset_index()
    weekly = weekly_real.merge(
        weekly_reactive,
        on=["station_id", "timestamp"],
        how="left",
    )

    hourly_real = (
        indexed.groupby(["station_id", indexed.index.hour])["real_power_mw"]
        .mean()
        .rename("mean_real_power_mw")
        .reset_index()
    )
    hourly_reactive = (
        indexed.groupby(["station_id", indexed.index.hour])["reactive_power_mvar"]
        .mean()
        .rename("mean_reactive_power_mvar")
        .reset_index()
    )
    hourly_real.rename(columns={"level_1": "hour", "timestamp": "hour"}, inplace=True)
    hourly_reactive.rename(columns={"level_1": "hour", "timestamp": "hour"}, inplace=True)
    hourly_profile = hourly_real.merge(
        hourly_reactive,
        on=["station_id", "hour"],
        how="left",
    )

    # Hour index from the groupby comes back as "level_1"; normalize it for readability.
    hourly_profile["hour"] = hourly_profile["hour"].astype(int)
    return {
        "daily": daily,
        "weekly": weekly,
        "hourly_profile": hourly_profile,
    }


def compare_to_standards(df: pd.DataFrame, standards: GridStandards) -> pd.DataFrame:
    """Quantify compliance against simple grid standards per station."""

    results: list[dict[str, float | str]] = []
    for station, group in df.groupby("station_id"):
        voltage_in_band = group["voltage_pu"].between(
            standards.voltage_min, standards.voltage_max
        )
        pf_in_band = group["power_factor"] >= standards.power_factor_min
        current_in_band = group["current_pu"] <= standards.current_max
        results.append(
            {
                "station_id": station,
                "voltage_within_pct": voltage_in_band.mean() * 100,
                "power_factor_within_pct": pf_in_band.mean() * 100,
                "current_within_pct": current_in_band.mean() * 100,
                "voltage_min_observed": group["voltage_pu"].min(),
                "voltage_max_observed": group["voltage_pu"].max(),
                "current_max_observed": group["current_pu"].max(),
                "power_factor_min_observed": group["power_factor"].min(),
            }
        )
    return pd.DataFrame(results)


def calculate_power_quality_indices(df: pd.DataFrame, standards: GridStandards) -> pd.DataFrame:
    """Compute basic power quality metrics for each station."""

    work = df.copy()
    real_power = work["real_power_mw"].astype(float)
    reactive_power = work["reactive_power_mvar"].astype(float)
    complex_power = real_power + 1j * reactive_power
    work["apparent_power_mva"] = np.abs(complex_power)
    work["phase_angle_deg"] = np.degrees(np.angle(complex_power))
    work["voltage_deviation"] = (work["voltage_pu"] - 1.0).abs()

    def low_pf_pct(series: pd.Series) -> float:
        return (series < standards.power_factor_min).mean() * 100

    def poor_pf_pct(series: pd.Series) -> float:
        return (series < standards.power_factor_fault).mean() * 100

    grouped = work.groupby("station_id")
    # Aggregate several custom metrics in one pass so downstream consumers receive
    # a single table of per-station indices.
    summary = grouped.agg(
        avg_power_factor=("power_factor", "mean"),
        min_power_factor=("power_factor", "min"),
        low_power_factor_pct=("power_factor", low_pf_pct),
        very_low_power_factor_pct=("power_factor", poor_pf_pct),
        avg_voltage_deviation=("voltage_deviation", "mean"),
        voltage_std=("voltage_pu", "std"),
        avg_apparent_power_mva=("apparent_power_mva", "mean"),
        avg_phase_angle_deg=("phase_angle_deg", "mean"),
        phase_angle_std_deg=("phase_angle_deg", "std"),
        peak_demand_mw=("real_power_mw", "max"),
        avg_demand_mw=("real_power_mw", "mean"),
        avg_reactive_power_mvar=("reactive_power_mvar", "mean"),
    )
    summary["load_factor"] = summary["avg_demand_mw"] / summary["peak_demand_mw"].replace(0, np.nan)
    summary["kvar_to_kw_ratio"] = summary["avg_reactive_power_mvar"] / summary[
        "avg_demand_mw"
    ].replace(0, np.nan)
    return summary.reset_index()


def calculate_circuit_metrics(df: pd.DataFrame, standards: GridStandards) -> pd.DataFrame:
    """Derive RMS quantities and power-factor correction needs per station."""

    target_pf = float(np.clip(standards.power_factor_target, 0.0, 1.0))
    target_angle_rad = float(np.arccos(np.clip(target_pf, -1.0, 1.0))) if 0.0 < target_pf <= 1.0 else 0.0
    target_angle_deg = float(np.degrees(target_angle_rad))

    rows: list[dict[str, float | str]] = []
    for station, group in df.groupby("station_id"):
        if group.empty:
            continue

        voltage_samples = group["voltage_pu"].astype(float)
        current_samples = group["current_pu"].astype(float)
        real_samples = group["real_power_mw"].astype(float)
        reactive_samples = group["reactive_power_mvar"].astype(float)

        voltage_rms = float(np.sqrt(np.mean(np.square(voltage_samples))))
        current_rms = float(np.sqrt(np.mean(np.square(current_samples))))

        complex_samples = real_samples + 1j * reactive_samples
        average_complex_power = complex_samples.mean()
        avg_real = float(np.real(average_complex_power))
        avg_reactive = float(np.imag(average_complex_power))
        apparent_mag = float(np.abs(average_complex_power))

        existing_pf = float(avg_real / apparent_mag) if apparent_mag > 0 else np.nan
        existing_angle_deg = float(np.degrees(np.angle(average_complex_power)))

        target_reactive = avg_real * np.tan(target_angle_rad) if avg_real != 0 else 0.0
        reactive_correction = max(0.0, avg_reactive - target_reactive)
        expected_reactive = avg_reactive - reactive_correction
        expected_apparent = float(np.hypot(avg_real, expected_reactive))
        expected_pf = float(avg_real / expected_apparent) if expected_apparent > 0 else np.nan

        rows.append(
            {
                "station_id": station,
                "voltage_rms_pu": voltage_rms,
                "current_rms_pu": current_rms,
                "avg_real_power_mw": avg_real,
                "avg_reactive_power_mvar": avg_reactive,
                "existing_power_factor": existing_pf,
                "existing_phase_angle_deg": existing_angle_deg,
                "target_power_factor": target_pf,
                "target_phase_angle_deg": target_angle_deg,
                "required_reactive_correction_mvar": reactive_correction,
                "required_capacitor_bank_kvar": reactive_correction * 1000.0,
                "expected_power_factor": expected_pf,
            }
        )

    return pd.DataFrame(rows)


def perform_fault_analysis(df: pd.DataFrame, standards: GridStandards) -> pd.DataFrame:
    """Surface simple fault indicators based on threshold breaches."""

    records: list[dict[str, object]] = []
    for station, group in df.groupby("station_id"):
        voltage_sag = group[group["voltage_pu"] < standards.voltage_fault_min]
        voltage_swell = group[group["voltage_pu"] > standards.voltage_fault_max]
        over_current = group[group["current_pu"] > standards.current_max]
        very_low_pf = group[group["power_factor"] < standards.power_factor_fault]

        # Combine timestamps from all detected fault windows to report first/last events.
        fault_frames = [
            frame[["timestamp"]]
            for frame in [voltage_sag, voltage_swell, over_current, very_low_pf]
            if not frame.empty
        ]
        fault_timestamps = pd.concat(fault_frames) if fault_frames else None
        records.append(
            {
                "station_id": station,
                "voltage_sag_events": len(voltage_sag),
                "voltage_swell_events": len(voltage_swell),
                "over_current_events": len(over_current),
                "very_low_pf_events": len(very_low_pf),
                "first_fault": fault_timestamps["timestamp"].min()
                if fault_timestamps is not None
                else None,
                "last_fault": fault_timestamps["timestamp"].max()
                if fault_timestamps is not None
                else None,
            }
        )
    return pd.DataFrame(records)
