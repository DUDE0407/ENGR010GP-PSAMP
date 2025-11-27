"""Power grid analysis workflow for power_system_data.csv.

This script loads half-year station data, computes statistics, compares the
measurements to nominal grid standards, derives power quality indices, performs
simple fault detection, and produces a set of summary plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis_core import (
	GridStandards,
	calculate_basic_statistics,
	calculate_power_quality_indices,
	compare_to_standards,
	identify_load_patterns,
	load_power_data,
	perform_fault_analysis,
)


# Apply a consistent Matplotlib aesthetic for all exported plots.
sns.set_theme(style="whitegrid")




def create_visualizations(
	df: pd.DataFrame,
	load_patterns: Dict[str, pd.DataFrame],
	output_dir: Path,
) -> None:
	"""Generate time-series and pattern plots for each station."""

	output_dir.mkdir(parents=True, exist_ok=True)

	for station, group in df.groupby("station_id"):
		# Plot the main electrical measurements together for an at-a-glance review.
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
	# Highlight typical diurnal behaviour by comparing average load per hour.
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
	# Weekly aggregation smooths volatility and reveals trend direction.
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
	load_patterns: Dict[str, pd.DataFrame],
) -> None:
	"""Persist tabular outputs to disk for downstream use."""

	output_dir.mkdir(parents=True, exist_ok=True)
	stats.to_csv(output_dir / "basic_statistics.csv", index=False)
	standard_comparison.to_csv(output_dir / "standard_comparison.csv", index=False)
	quality_indices.to_csv(output_dir / "power_quality_indices.csv", index=False)
	fault_summary.to_csv(output_dir / "fault_summary.csv", index=False)
	# Persist each load-pattern slice so downstream notebooks can decide which view to load.
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

	# Write results to disk before generating plots so artefacts exist even if plotting fails.
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
