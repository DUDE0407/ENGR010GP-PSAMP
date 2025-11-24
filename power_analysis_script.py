"""Power grid analysis workflow for power_system_data.csv.

This script loads half-year station data, computes statistics, compares the
measurements to nominal grid standards, derives power quality indices, performs
simple fault detection, and produces a set of summary plots.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


sns.set_theme(style="whitegrid")


@dataclass(frozen=True)
class GridStandards:
	voltage_min: float = 0.95
	voltage_max: float = 1.05
	voltage_fault_min: float = 0.90
	voltage_fault_max: float = 1.10
	current_max: float = 120.0
	power_factor_min: float = 0.90
	power_factor_fault: float = 0.80


def load_power_data(csv_path: Path) -> pd.DataFrame:
	"""Load and tidy the power system data file."""

	df = pd.read_csv(csv_path, parse_dates=["timestamp"])  # type: ignore[arg-type]
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
	stats.columns = ["_".join(col) for col in stats.columns]
	return stats.reset_index()


def identify_load_patterns(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
	"""Capture daily, weekly, and diurnal load signatures for each station."""

	indexed = df.set_index("timestamp")
	daily = (
		indexed.groupby("station_id")["real_power_mw"].resample("D").mean().reset_index()
	)
	weekly = (
		indexed.groupby("station_id")["real_power_mw"].resample("W").mean().reset_index()
	)
	hourly_profile = (
		indexed.groupby(["station_id", indexed.index.hour])["real_power_mw"]
		.mean()
		.reset_index()
		.rename(
			columns={
				"level_1": "hour",
				"timestamp": "hour",
				"real_power_mw": "mean_real_power_mw",
			}
		)
	)
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
	work["apparent_power_mva"] = np.hypot(
		work["real_power_mw"], work["reactive_power_mvar"]
	)
	work["voltage_deviation"] = (work["voltage_pu"] - 1.0).abs()

	def low_pf_pct(series: pd.Series) -> float:
		return (series < standards.power_factor_min).mean() * 100

	def poor_pf_pct(series: pd.Series) -> float:
		return (series < standards.power_factor_fault).mean() * 100

	grouped = work.groupby("station_id")
	summary = grouped.agg(
		avg_power_factor=("power_factor", "mean"),
		min_power_factor=("power_factor", "min"),
		low_power_factor_pct=("power_factor", low_pf_pct),
		very_low_power_factor_pct=("power_factor", poor_pf_pct),
		avg_voltage_deviation=("voltage_deviation", "mean"),
		voltage_std=("voltage_pu", "std"),
		avg_apparent_power_mva=("apparent_power_mva", "mean"),
		peak_demand_mw=("real_power_mw", "max"),
		avg_demand_mw=("real_power_mw", "mean"),
		avg_reactive_power_mvar=("reactive_power_mvar", "mean"),
	)
	summary["load_factor"] = summary["avg_demand_mw"] / summary["peak_demand_mw"].replace(0, np.nan)
	summary["kvar_to_kw_ratio"] = summary["avg_reactive_power_mvar"] / summary[
		"avg_demand_mw"
	].replace(0, np.nan)
	return summary.reset_index()


def perform_fault_analysis(df: pd.DataFrame, standards: GridStandards) -> pd.DataFrame:
	"""Surface simple fault indicators based on threshold breaches."""

	records: list[dict[str, object]] = []
	for station, group in df.groupby("station_id"):
		voltage_sag = group[group["voltage_pu"] < standards.voltage_fault_min]
		voltage_swell = group[group["voltage_pu"] > standards.voltage_fault_max]
		over_current = group[group["current_pu"] > standards.current_max]
		very_low_pf = group[group["power_factor"] < standards.power_factor_fault]

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


def create_visualizations(
	df: pd.DataFrame,
	load_patterns: dict[str, pd.DataFrame],
	output_dir: Path,
) -> None:
	"""Generate time-series and pattern plots for each station."""

	output_dir.mkdir(parents=True, exist_ok=True)

	for station, group in df.groupby("station_id"):
		fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
		group.plot(x="timestamp", y="voltage_pu", ax=axes[0], legend=False)
		axes[0].set_ylabel("Voltage (p.u.)")
		group.plot(x="timestamp", y="current_pu", ax=axes[1], legend=False)
		axes[1].set_ylabel("Current (p.u.)")
		group.plot(x="timestamp", y="real_power_mw", ax=axes[2], legend=False)
		axes[2].set_ylabel("Real Power (MW)")
		group.plot(x="timestamp", y="power_factor", ax=axes[3], legend=False)
		axes[3].set_ylabel("Power Factor")
		axes[3].set_xlabel("Timestamp")
		fig.suptitle(f"Station {station} Time-Series", fontsize=14)
		plt.tight_layout()
		fig.savefig(output_dir / f"{station}_timeseries.png", dpi=150)
		plt.close(fig)

	daily = load_patterns["daily"].pivot(
		index="timestamp", columns="station_id", values="real_power_mw"
	)
	fig, ax = plt.subplots(figsize=(12, 6))
	daily.plot(ax=ax)
	ax.set_ylabel("Daily Mean Real Power (MW)")
	ax.set_title("Daily Load Profile")
	plt.tight_layout()
	fig.savefig(output_dir / "daily_load_profile.png", dpi=150)
	plt.close(fig)

	hourly = load_patterns["hourly_profile"].pivot(
		index="hour", columns="station_id", values="mean_real_power_mw"
	)
	fig, ax = plt.subplots(figsize=(12, 6))
	hourly.plot(ax=ax)
	ax.set_xlabel("Hour of Day")
	ax.set_ylabel("Average Real Power (MW)")
	ax.set_title("Typical Diurnal Load Shape")
	plt.tight_layout()
	fig.savefig(output_dir / "hourly_load_profile.png", dpi=150)
	plt.close(fig)

	weekly = load_patterns["weekly"]
	fig, ax = plt.subplots(figsize=(12, 6))
	sns.lineplot(
		data=weekly, x="timestamp", y="real_power_mw", hue="station_id", ax=ax
	)
	ax.set_ylabel("Weekly Mean Real Power (MW)")
	ax.set_title("Weekly Load Trend")
	plt.tight_layout()
	fig.savefig(output_dir / "weekly_load_trend.png", dpi=150)
	plt.close(fig)


def export_results(
	output_dir: Path,
	stats: pd.DataFrame,
	standard_comparison: pd.DataFrame,
	quality_indices: pd.DataFrame,
	fault_summary: pd.DataFrame,
	load_patterns: dict[str, pd.DataFrame],
) -> None:
	"""Persist tabular outputs to disk for downstream use."""

	output_dir.mkdir(parents=True, exist_ok=True)
	stats.to_csv(output_dir / "basic_statistics.csv", index=False)
	standard_comparison.to_csv(output_dir / "standard_comparison.csv", index=False)
	quality_indices.to_csv(output_dir / "power_quality_indices.csv", index=False)
	fault_summary.to_csv(output_dir / "fault_summary.csv", index=False)
	for name, frame in load_patterns.items():
		frame.to_csv(output_dir / f"load_pattern_{name}.csv", index=False)


def run_analysis(csv_path: Path, reports_dir: Path, figures_dir: Path) -> None:
	"""Execute the full analysis workflow."""

	standards = GridStandards()
	df = load_power_data(csv_path)
	stats = calculate_basic_statistics(df)
	load_patterns = identify_load_patterns(df)
	standard_comparison = compare_to_standards(df, standards)
	quality_indices = calculate_power_quality_indices(df, standards)
	fault_summary = perform_fault_analysis(df, standards)

	export_results(
		reports_dir,
		stats,
		standard_comparison,
		quality_indices,
		fault_summary,
		load_patterns,
	)
	create_visualizations(df, load_patterns, figures_dir)

	print("Basic Statistics:\n", stats)
	print("\nStandards Comparison:\n", standard_comparison)
	print("\nPower Quality Indices:\n", quality_indices)
	print("\nFault Summary:\n", fault_summary)
	print(f"\nReports exported to: {reports_dir}")
	print(f"Figures exported to: {figures_dir}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Power grid data analysis")
	parser.add_argument(
		"--data",
		type=Path,
		default=Path("power_system_data.csv"),
		help="Path to the CSV file containing the timeseries data.",
	)
	parser.add_argument(
		"--reports",
		type=Path,
		default=Path("analysis_outputs/reports"),
		help="Directory to store CSV reports.",
	)
	parser.add_argument(
		"--figures",
		type=Path,
		default=Path("analysis_outputs/figures"),
		help="Directory to store generated plots.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	run_analysis(args.data, args.reports, args.figures)


if __name__ == "__main__":
	main()
