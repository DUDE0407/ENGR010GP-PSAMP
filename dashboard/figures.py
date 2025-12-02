"""Utilities to render Matplotlib figures as Pygame surfaces."""

from __future__ import annotations

from io import BytesIO
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pygame


def _figure_to_surface(fig: plt.Figure) -> pygame.Surface:
    # Render the Matplotlib figure into an in-memory PNG so Pygame can blit it.
    buffer = BytesIO()
    fig.savefig(buffer, format="PNG", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buffer.seek(0)
    surface = pygame.image.load(buffer, "chart.png").convert_alpha()
    return surface


def _line_chart(
    data: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
) -> pygame.Surface:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    data.plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    legend_cols = min(3, max(1, data.shape[1]))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=legend_cols)
    fig.tight_layout()
    return _figure_to_surface(fig)


def _bar_chart(
    data: pd.DataFrame,
    title: str,
    ylabel: str,
) -> pygame.Surface:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    colors = plt.get_cmap("viridis")([0.2, 0.5, 0.8])
    for idx, column in enumerate(["voltage_within_pct", "power_factor_within_pct", "current_within_pct"]):
        # Encode category information inside the x-label to keep the chart compact.
        ax.bar(
            data["station_id"] + f"\n{column.split('_')[0].title()}",
            data[column],
            label=column.replace("_", " ").title(),
            color=colors[idx % len(colors)],
            alpha=0.8,
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _figure_to_surface(fig)


def _clustered_bar_chart(
    data: pd.DataFrame,
    title: str,
    ylabel: str,
) -> pygame.Surface:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    data.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(3, data.shape[1]))
    fig.tight_layout()
    return _figure_to_surface(fig)


def build_chart_surfaces(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    hourly: pd.DataFrame,
    compliance: pd.DataFrame,
    circuit_metrics: pd.DataFrame,
) -> List[Tuple[str, pygame.Surface]]:
    charts: List[Tuple[str, pygame.Surface]] = []

    if not daily.empty:
        if "real_power_mw" in daily:
            daily_real = (
                daily.pivot(index="timestamp", columns="station_id", values="real_power_mw")
                .sort_index()
            )
            charts.append(
                (
                    "Daily Mean Real Power",
                    _line_chart(daily_real, "Daily Mean Real Power", "Date", "MW"),
                )
            )
        if "reactive_power_mvar" in daily:
            daily_reactive = (
                daily.pivot(index="timestamp", columns="station_id", values="reactive_power_mvar")
                .sort_index()
            )
            charts.append(
                (
                    "Daily Mean Reactive Power",
                    _line_chart(
                        daily_reactive,
                        "Daily Mean Reactive Power",
                        "Date",
                        "MVAr",
                    ),
                )
            )

    if not weekly.empty:
        if "real_power_mw" in weekly:
            weekly_real = (
                weekly.pivot(index="timestamp", columns="station_id", values="real_power_mw")
                .sort_index()
            )
            charts.append(
                (
                    "Weekly Mean Real Power",
                    _line_chart(weekly_real, "Weekly Mean Real Power", "Week", "MW"),
                )
            )
        if "reactive_power_mvar" in weekly:
            weekly_reactive = (
                weekly.pivot(index="timestamp", columns="station_id", values="reactive_power_mvar")
                .sort_index()
            )
            charts.append(
                (
                    "Weekly Mean Reactive Power",
                    _line_chart(
                        weekly_reactive,
                        "Weekly Mean Reactive Power",
                        "Week",
                        "MVAr",
                    ),
                )
            )

    if not hourly.empty:
        if "mean_real_power_mw" in hourly:
            hourly_real = (
                hourly.pivot(index="hour", columns="station_id", values="mean_real_power_mw")
                .sort_index()
            )
            charts.append(
                (
                    "Diurnal Load Profile",
                    _line_chart(hourly_real, "Average Hourly Load", "Hour", "MW"),
                )
            )
        if "mean_reactive_power_mvar" in hourly:
            hourly_reactive = (
                hourly.pivot(index="hour", columns="station_id", values="mean_reactive_power_mvar")
                .sort_index()
            )
            charts.append(
                (
                    "Diurnal Reactive Load",
                    _line_chart(
                        hourly_reactive,
                        "Average Hourly Reactive Load",
                        "Hour",
                        "MVAr",
                    ),
                )
            )

    if not compliance.empty:
        charts.append(
            (
                "Standards Compliance",
                _bar_chart(
                    compliance[[
                        "station_id",
                        "voltage_within_pct",
                        "power_factor_within_pct",
                        "current_within_pct",
                    ]],
                    "Percentage Within Limits",
                    "Percent of Measurements",
                ),
            )
        )

    if not circuit_metrics.empty:
        pf_table = (
            circuit_metrics.set_index("station_id")[
                ["existing_power_factor", "expected_power_factor"]
            ]
        )
        charts.append(
            (
                "Power Factor Correction",
                _clustered_bar_chart(
                    pf_table,
                    "Existing vs Corrected Power Factor",
                    "Power Factor",
                ),
            )
        )

        correction_table = (
            circuit_metrics.set_index("station_id")[
                ["required_reactive_correction_mvar"]
            ]
            .rename(columns={"required_reactive_correction_mvar": "Required MVAr"})
        )
        charts.append(
            (
                "Capacitor Sizing",
                _clustered_bar_chart(
                    correction_table,
                    "Recommended Reactive Compensation",
                    "MVAr",
                ),
            )
        )

    return charts
