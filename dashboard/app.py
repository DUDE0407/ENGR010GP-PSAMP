"""Pygame dashboard for power system analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pandas as pd
import pygame

from analysis_core import (
    GridStandards,
    calculate_basic_statistics,
    calculate_power_quality_indices,
    compare_to_standards,
    identify_load_patterns,
    perform_fault_analysis,
)
from .data_loader import clear_cache, load_dataset
from .figures import build_chart_surfaces

BACKGROUND = (18, 20, 28)
SIDEBAR_BG = (26, 29, 39)
PANEL_BG = (32, 35, 46)
BORDER_COLOR = (12, 13, 18)
TEXT_COLOR = (232, 235, 240)
ACCENT_COLOR = (100, 149, 237)
BUTTON_BG = (57, 97, 200)
BUTTON_HOVER = (82, 122, 230)
BUTTON_BORDER = (21, 25, 36)
BUTTON_TEXT = (245, 247, 252)
BUTTON_DISABLED_BG = (44, 51, 76)
BUTTON_DISABLED_TEXT = (160, 168, 186)

SCREEN_SIZE = (1280, 720)
SIDEBAR_WIDTH = 480
PADDING = 24
BLOCK_PADDING = 14
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 44
BUTTON_SPACING = 16

VIEW_MODES: Tuple[str, ...] = ("overall", "monthly", "daily")
STANDARDS = GridStandards()


@dataclass
class Button:
    label: str
    rect: pygame.Rect
    action: str
    enabled: bool = True


@dataclass(frozen=True)
class PeriodOption:
    key: object
    label: str


def _wrap_text(text: str, font: pygame.font.Font, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.size(candidate)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _format_basic_statistics(stats: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    if stats.empty:
        return lines
    for _, row in stats.iterrows():
        lines.append(f"{row['station_id']}")
        lines.append(
            f"  Voltage: {_format_value(row['voltage_pu_mean'], '{:.3f}')} pu"
        )
        lines.append(
            f"  Current: {_format_value(row['current_pu_mean'], '{:.1f}')} pu"
        )
        lines.append(
            f"  Real Power: {_format_value(row['real_power_mw_mean'], '{:.1f}')} MW"
        )
        lines.append(
            f"  Reactive Power: {_format_value(row['reactive_power_mvar_mean'], '{:.1f}')} MVAr"
        )
        lines.append(
            f"  Power Factor: {_format_value(row['power_factor_mean'], '{:.3f}')}"
        )
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _format_compliance(comparison: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    if comparison.empty:
        return lines
    for _, row in comparison.iterrows():
        lines.append(f"{row['station_id']}")
        lines.append(
            f"  Voltage within band: {_format_value(row['voltage_within_pct'], '{:.1f}')}%"
        )
        lines.append(
            f"  PF within band: {_format_value(row['power_factor_within_pct'], '{:.1f}')}%"
        )
        lines.append(
            f"  Current within band: {_format_value(row['current_within_pct'], '{:.1f}')}%"
        )
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _format_quality(quality: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    if quality.empty:
        return lines
    for _, row in quality.iterrows():
        lines.append(f"{row['station_id']}")
        lines.append(
            "  Power Factor avg/min: "
            f"{_format_value(row['avg_power_factor'], '{:.3f}')} / "
            f"{_format_value(row['min_power_factor'], '{:.3f}')}"
        )
        lines.append(
            f"  Voltage deviation (std): {_format_value(row['voltage_std'], '{:.4f}')} pu"
        )
        lines.append(
            f"  Low PF hours: {_format_value(row['low_power_factor_pct'], '{:.1f}')}%"
        )
        lines.append(
            f"  Very low PF hours: {_format_value(row['very_low_power_factor_pct'], '{:.1f}')}%"
        )
        lines.append(
            f"  Avg reactive power: {_format_value(row['avg_reactive_power_mvar'], '{:.2f}')} MVAr"
        )
        lines.append(f"  Load factor: {_format_value(row['load_factor'], '{:.3f}')}")
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _safe_int(value: object) -> int:
    if value is None or pd.isna(value):
        return 0
    return int(value)


def _format_value(value: object, pattern: str) -> str:
    if value is None or pd.isna(value):
        return "-"
    return pattern.format(value)


def _format_faults(faults: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    if faults.empty:
        return lines
    for _, row in faults.iterrows():
        first = _format_timestamp(row.get("first_fault"))
        last = _format_timestamp(row.get("last_fault"))
        lines.append(f"{row['station_id']}")
        lines.append(f"  Voltage sag events: {_safe_int(row['voltage_sag_events'])}")
        lines.append(f"  Voltage swell events: {_safe_int(row['voltage_swell_events'])}")
        lines.append(f"  Overcurrent events: {_safe_int(row['over_current_events'])}")
        lines.append(f"  PF < 0.8 events: {_safe_int(row['very_low_pf_events'])}")
        lines.append(f"  First / Last: {first} – {last}")
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _format_timestamp(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    timestamp = pd.to_datetime(value)
    return timestamp.strftime("%Y-%m-%d")


def _render_text_block(
    surface: pygame.Surface,
    title: str,
    lines: Sequence[str],
    start_pos: Tuple[int, int],
    heading_font: pygame.font.Font,
    body_font: pygame.font.Font,
    max_width: int,
) -> int:
    x, y = start_pos
    title_surface = heading_font.render(title, True, ACCENT_COLOR)
    surface.blit(title_surface, (x, y))
    y += title_surface.get_height() + 6

    if not lines:
        lines = ("No data available.",)

    for line in lines:
        for wrapped in _wrap_text(line, body_font, max_width):
            text_surface = body_font.render(wrapped, True, TEXT_COLOR)
            surface.blit(text_surface, (x, y))
            y += text_surface.get_height() + 4
    return y + 10


def _measure_blocks_height(
    blocks: Sequence[Tuple[str, Sequence[str]]],
    heading_font: pygame.font.Font,
    body_font: pygame.font.Font,
    max_width: int,
) -> int:
    total = 0
    for title, lines in blocks:
        total += heading_font.get_height() + 6
        items = lines or ("No data available.",)
        for line in items:
            wrapped = _wrap_text(line, body_font, max_width)
            total += len(wrapped) * (body_font.get_height() + 4)
        total += 10
    return max(total, 1)


def _build_sidebar_surface(
    blocks: Sequence[Tuple[str, Sequence[str]]],
    heading_font: pygame.font.Font,
    body_font: pygame.font.Font,
    max_width: int,
) -> Tuple[pygame.Surface, int]:
    content_height = _measure_blocks_height(blocks, heading_font, body_font, max_width)
    surface = pygame.Surface((max_width, content_height), pygame.SRCALPHA)
    y = 0
    for title, lines in blocks:
        y = _render_text_block(
            surface,
            title,
            lines,
            (0, y),
            heading_font,
            body_font,
            max_width,
        )
    return surface, content_height


def _build_period_options(df: pd.DataFrame) -> dict[str, List[PeriodOption]]:
    periods: dict[str, List[PeriodOption]] = {"overall": [PeriodOption(key=None, label="All Data")]}

    if df.empty:
        periods["monthly"] = []
        periods["daily"] = []
        return periods

    monthly_periods = sorted(df["timestamp"].dt.to_period("M").unique())
    periods["monthly"] = [
        PeriodOption(key=period, label=period.strftime("%b %Y"))
        for period in monthly_periods
    ]

    daily_periods = sorted(df["timestamp"].dt.normalize().unique())
    periods["daily"] = [
        PeriodOption(key=pd.Timestamp(day), label=pd.Timestamp(day).strftime("%d %b %Y"))
        for day in daily_periods
    ]

    return periods


def _filter_dataframe(
    df: pd.DataFrame,
    mode: str,
    option: PeriodOption,
) -> pd.DataFrame:
    if mode == "overall" or option.key is None:
        return df.copy()

    if mode == "monthly":
        mask = df["timestamp"].dt.to_period("M") == option.key
    else:  # daily
        mask = df["timestamp"].dt.normalize() == option.key

    return df.loc[mask].copy()


def _format_subtitle(
    mode: str,
    option: PeriodOption,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    stations_label: str,
) -> str:
    if mode == "overall" and start is not None and end is not None:
        return (
            f"Monitoring period: {start:%b %Y} - {end:%b %Y} | Stations: {stations_label}"
        )

    mode_label = "Monthly" if mode == "monthly" else "Daily"
    detail = option.label if option.label else "(no selection)"
    return f"{mode_label} view: {detail} | Stations: {stations_label}"


def _create_buttons(y: int) -> List[Button]:
    buttons: List[Button] = []
    specs = [
        ("Prev Chart", "prev_chart"),
        ("Next Chart", "next_chart"),
        ("Prev Period", "prev_period"),
        ("Next Period", "next_period"),
        ("Cycle Mode", "cycle_mode"),
        ("Reload Data", "reload"),
        ("Exit", "exit"),
    ]
    total_width = len(specs) * BUTTON_WIDTH + (len(specs) - 1) * BUTTON_SPACING
    start_x = (SCREEN_SIZE[0] - total_width) // 2
    for idx, (label, action) in enumerate(specs):
        rect = pygame.Rect(
            start_x + idx * (BUTTON_WIDTH + BUTTON_SPACING),
            y,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
        )
        buttons.append(Button(label, rect, action))
    return buttons


def _draw_buttons(
    surface: pygame.Surface,
    buttons: Sequence[Button],
    font: pygame.font.Font,
    hovered: Button | None,
) -> None:
    for button in buttons:
        if not button.enabled:
            color = BUTTON_DISABLED_BG
            text_color = BUTTON_DISABLED_TEXT
        else:
            color = BUTTON_HOVER if button is hovered else BUTTON_BG
            text_color = BUTTON_TEXT
        pygame.draw.rect(surface, color, button.rect, border_radius=10)
        pygame.draw.rect(surface, BUTTON_BORDER, button.rect, width=1, border_radius=10)
        label_surface = font.render(button.label, True, text_color)
        label_rect = label_surface.get_rect(center=button.rect.center)
        surface.blit(label_surface, label_rect)


def run() -> None:
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("Power Analysis Dashboard")
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont("Segoe UI", 32, bold=True)
    heading_font = pygame.font.SysFont("Segoe UI", 22, bold=True)
    body_font = pygame.font.SysFont("Segoe UI", 18)
    info_font = pygame.font.SysFont("Segoe UI", 16)
    button_font = pygame.font.SysFont("Segoe UI", 18, bold=True)

    title_surface = title_font.render("Power Grid Analysis Dashboard", True, TEXT_COLOR)
    subtitle_surface = body_font.render(
        "Monitoring period: Jan-Jun 2024 | Stations: SUB_001, SUB_002, SUB_003",
        True,
        TEXT_COLOR,
    )

    button_y = SCREEN_SIZE[1] - PADDING - BUTTON_HEIGHT
    chart_top = PADDING + title_surface.get_height() + subtitle_surface.get_height() + 26
    chart_height = max(240, button_y - chart_top - 40)
    chart_width = SCREEN_SIZE[0] - SIDEBAR_WIDTH - (PADDING * 4)

    sidebar_rect = pygame.Rect(PADDING, chart_top, SIDEBAR_WIDTH - (PADDING // 2), chart_height)
    chart_rect = pygame.Rect(
        sidebar_rect.right + PADDING,
        chart_top,
        chart_width,
        chart_height,
    )

    caption_offset = heading_font.get_height() + 30

    sidebar_width = max(1, sidebar_rect.width - (BLOCK_PADDING * 2))
    sidebar_view_height = max(1, sidebar_rect.height - (BLOCK_PADDING * 2))
    sidebar_surface: pygame.Surface | None = None
    sidebar_content_height = 0
    sidebar_scroll_max = 0
    sidebar_offset = 0

    buttons = _create_buttons(button_y)

    dataset = pd.DataFrame()
    period_options: dict[str, List[PeriodOption]] = {mode: [] for mode in VIEW_MODES}
    period_options["overall"] = [PeriodOption(key=None, label="All Data")]
    dataset_start: pd.Timestamp | None = None
    dataset_end: pd.Timestamp | None = None
    stations_label = "N/A"

    view_mode = "overall"
    period_index = 0
    active_period = period_options["overall"][0]
    current_record_count = 0
    current_start: pd.Timestamp | None = None
    current_end: pd.Timestamp | None = None

    stats = pd.DataFrame()
    comparison = pd.DataFrame()
    quality = pd.DataFrame()
    faults = pd.DataFrame()
    charts: List[Tuple[str, pygame.Surface]] = []
    text_blocks: List[Tuple[str, Sequence[str]]] = []
    chart_index = 0

    def _mode_label(value: str) -> str:
        return {
            "overall": "Whole Dataset",
            "monthly": "Monthly",
            "daily": "Daily",
        }[value]

    def _current_period_list() -> List[PeriodOption]:
        return period_options.get(view_mode, [PeriodOption(key=None, label="All Data")])

    def _rebuild_sidebar_surface() -> None:
        nonlocal sidebar_surface, sidebar_content_height, sidebar_scroll_max, sidebar_offset
        # Pre-render the text blocks onto an off-screen surface; the main loop blits a view slice.
        sidebar_surface, sidebar_content_height = _build_sidebar_surface(
            text_blocks,
            heading_font,
            body_font,
            sidebar_width,
        )
        sidebar_scroll_max = max(0, sidebar_content_height - sidebar_view_height)
        sidebar_offset = max(0, min(sidebar_offset, sidebar_scroll_max))

    def _update_subtitle() -> None:
        nonlocal subtitle_surface
        subtitle_text = _format_subtitle(
            view_mode,
            active_period,
            dataset_start,
            dataset_end,
            stations_label,
        )
        subtitle_surface = body_font.render(subtitle_text, True, TEXT_COLOR)

    def _sync_button_states() -> None:
        period_list = _current_period_list()
        has_period_nav = view_mode != "overall" and len(period_list) > 1
        multi_charts = len(charts) > 1
        for button in buttons:
            if button.action in ("prev_chart", "next_chart"):
                button.enabled = multi_charts
            elif button.action in ("prev_period", "next_period"):
                button.enabled = has_period_nav
            else:
                button.enabled = True

    def _update_blocks() -> None:
        nonlocal text_blocks, sidebar_offset
        context_lines = [f"Mode: {_mode_label(view_mode)}"]
        if view_mode != "overall":
            context_lines.append(f"Period: {active_period.label}")
        if current_start is not None and current_end is not None:
            if current_start.date() == current_end.date():
                context_lines.append(f"Span: {current_start:%d %b %Y}")
            else:
                context_lines.append(
                    f"Span: {current_start:%d %b %Y} - {current_end:%d %b %Y}"
                )
        context_lines.append(f"Records: {current_record_count:,}")
        text_blocks = [
            ("View Context", context_lines),
            ("Basic Statistics", _format_basic_statistics(stats)),
            ("Standards Compliance", _format_compliance(comparison)),
            ("Power Quality Indices", _format_quality(quality)),
            ("Fault Overview", _format_faults(faults)),
        ]
        sidebar_offset = 0
        _rebuild_sidebar_surface()

    def _refresh_analysis() -> None:
        nonlocal stats, comparison, quality, faults, charts, chart_index, active_period
        nonlocal current_record_count, current_start, current_end
        period_list = _current_period_list()
        if not period_list:
            active_period = PeriodOption(key=None, label="All Data")
        else:
            nonlocal period_index
            period_index = max(0, min(period_index, len(period_list) - 1))
            active_period = period_list[period_index]

        filtered = _filter_dataframe(dataset, view_mode, active_period)
        current_record_count = len(filtered)
        current_start = filtered["timestamp"].min() if not filtered.empty else None
        current_end = filtered["timestamp"].max() if not filtered.empty else None

        if filtered.empty:
            stats = pd.DataFrame()
            comparison = pd.DataFrame()
            quality = pd.DataFrame()
            faults = pd.DataFrame()
            charts = []
        else:
            stats = calculate_basic_statistics(filtered)
            comparison = compare_to_standards(filtered, STANDARDS)
            quality = calculate_power_quality_indices(filtered, STANDARDS)
            faults = perform_fault_analysis(filtered, STANDARDS)
            patterns = identify_load_patterns(filtered)
            charts = build_chart_surfaces(
                patterns["daily"],
                patterns["weekly"],
                patterns["hourly_profile"],
                comparison,
            )
            if view_mode == "daily":
                excluded = {
                    "Daily Mean Real Power",
                    "Daily Mean Reactive Power",
                    "Weekly Mean Real Power",
                    "Weekly Mean Reactive Power",
                }
                charts = [
                    (name, surface)
                    for name, surface in charts
                    if name not in excluded
                ]

        chart_index = min(chart_index, len(charts) - 1) if charts else 0
        _update_blocks()
        _update_subtitle()
        _sync_button_states()

    def _reload_dataset() -> None:
        nonlocal dataset, period_options, dataset_start, dataset_end, stations_label
        nonlocal view_mode, period_index

        dataset = load_dataset()
        period_options = _build_period_options(dataset)
        period_options.setdefault("overall", [PeriodOption(key=None, label="All Data")])
        if not period_options["overall"]:
            period_options["overall"] = [PeriodOption(key=None, label="All Data")]

        if dataset.empty:
            dataset_start = None
            dataset_end = None
            stations_label = "N/A"
        else:
            dataset_start = dataset["timestamp"].min()
            dataset_end = dataset["timestamp"].max()
            stations = sorted(dataset["station_id"].unique())
            stations_label = ", ".join(stations)

        if view_mode != "overall" and not period_options.get(view_mode):
            view_mode = "overall"
            period_index = 0

        period_list = _current_period_list()
        if period_list:
            period_index = max(0, min(period_index, len(period_list) - 1))
        else:
            period_index = 0

        _refresh_analysis()

    _reload_dataset()

    status_message = ""
    status_timeout = 0
    running = True

    def _advance_mode() -> str:
        nonlocal view_mode, period_index
        start_idx = VIEW_MODES.index(view_mode)
        idx = start_idx
        for _ in range(len(VIEW_MODES)):
            idx = (idx + 1) % len(VIEW_MODES)
            candidate = VIEW_MODES[idx]
            if candidate == "overall" or period_options.get(candidate):
                view_mode = candidate
                period_index = 0
                _refresh_analysis()
                return _mode_label(view_mode)

        view_mode = "overall"
        period_index = 0
        _refresh_analysis()
        return _mode_label(view_mode)

    def _apply_action(action: str) -> None:
        nonlocal chart_index, period_index, running
        period_list = _current_period_list()

        if action == "next_chart" and charts:
            if len(charts) > 1:
                chart_index = (chart_index + 1) % len(charts)
                _set_status(f"Showing chart {chart_index + 1}/{len(charts)}")
            else:
                _set_status("Only one chart available", 1500)
        elif action == "prev_chart" and charts:
            if len(charts) > 1:
                chart_index = (chart_index - 1) % len(charts)
                _set_status(f"Showing chart {chart_index + 1}/{len(charts)}")
            else:
                _set_status("Only one chart available", 1500)
        elif action == "next_period" and view_mode != "overall" and period_list:
            if len(period_list) > 1:
                period_index = (period_index + 1) % len(period_list)
                _refresh_analysis()
                _set_status(f"Period: {active_period.label}")
        elif action == "prev_period" and view_mode != "overall" and period_list:
            if len(period_list) > 1:
                period_index = (period_index - 1) % len(period_list)
                _refresh_analysis()
                _set_status(f"Period: {active_period.label}")
        elif action == "cycle_mode":
            label = _advance_mode()
            _set_status(f"{label} view")
        elif action == "reload":
            clear_cache()
            _reload_dataset()
            _set_status("Data reloaded")
        elif action == "exit":
            running = False

    def _set_status(message: str, duration_ms: int = 2200) -> None:
        nonlocal status_message, status_timeout
        status_message = message
        status_timeout = pygame.time.get_ticks() + duration_ms

    instructions_text = (
        "Charts: buttons or arrow keys | Period: buttons or [ ] keys | Mode: button or M | "
        "Sidebar: wheel/PgUp/PgDn | Reload: R | Exit: Esc"
    )
    instructions_surface = info_font.render(instructions_text, True, TEXT_COLOR)

    while running:
        mouse_pos = pygame.mouse.get_pos()
        hovered_button = next(
            (b for b in buttons if b.enabled and b.rect.collidepoint(mouse_pos)),
            None,
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Keyboard shortcuts mirror the on-screen controls for power users.
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    _apply_action("exit")
                elif event.key == pygame.K_RIGHT:
                    _apply_action("next_chart")
                elif event.key == pygame.K_LEFT:
                    _apply_action("prev_chart")
                elif event.key == pygame.K_r:
                    _apply_action("reload")
                elif event.key == pygame.K_m:
                    _apply_action("cycle_mode")
                elif event.key == pygame.K_RIGHTBRACKET:
                    _apply_action("next_period")
                elif event.key == pygame.K_LEFTBRACKET:
                    _apply_action("prev_period")
                elif event.key == pygame.K_UP:
                    sidebar_offset = max(0, sidebar_offset - 28)
                elif event.key == pygame.K_DOWN:
                    sidebar_offset = min(sidebar_scroll_max, sidebar_offset + 28)
                elif event.key == pygame.K_PAGEUP:
                    sidebar_offset = max(0, sidebar_offset - sidebar_view_height // 2)
                elif event.key == pygame.K_PAGEDOWN:
                    sidebar_offset = min(
                        sidebar_scroll_max,
                        sidebar_offset + sidebar_view_height // 2,
                    )
                elif event.key == pygame.K_HOME:
                    sidebar_offset = 0
                elif event.key == pygame.K_END:
                    sidebar_offset = sidebar_scroll_max
            elif event.type == pygame.MOUSEWHEEL:
                if sidebar_scroll_max > 0:
                    sidebar_offset = max(
                        0,
                        min(
                            sidebar_scroll_max,
                            sidebar_offset - event.y * 32,
                        ),
                    )
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Buttons are the primary interaction target for casual users.
                for button in buttons:
                    if button.enabled and button.rect.collidepoint(event.pos):
                        _apply_action(button.action)
                        break

        screen.fill(BACKGROUND)

        pygame.draw.rect(screen, SIDEBAR_BG, sidebar_rect, border_radius=12)
        pygame.draw.rect(screen, PANEL_BG, chart_rect, border_radius=12)
        pygame.draw.rect(screen, BORDER_COLOR, chart_rect, width=1, border_radius=12)

        screen.blit(title_surface, (PADDING, PADDING))
        screen.blit(subtitle_surface, (PADDING, PADDING + title_surface.get_height() + 6))

        sidebar_x = sidebar_rect.left + BLOCK_PADDING
        sidebar_y = sidebar_rect.top + BLOCK_PADDING
        view_rect = pygame.Rect(0, sidebar_offset, sidebar_width, sidebar_view_height)
        if sidebar_surface is not None:
            # Only the visible slice of the sidebar surface is blitted each frame.
            screen.blit(sidebar_surface, (sidebar_x, sidebar_y), area=view_rect)
        else:
            placeholder = body_font.render("No data available.", True, TEXT_COLOR)
            screen.blit(placeholder, (sidebar_x, sidebar_y))

        if sidebar_scroll_max > 0:
            track_rect = pygame.Rect(
                sidebar_rect.right - 10,
                sidebar_rect.top + BLOCK_PADDING,
                4,
                sidebar_view_height,
            )
            pygame.draw.rect(screen, (54, 58, 72), track_rect, border_radius=2)
            knob_height = max(18, int(sidebar_view_height * sidebar_view_height / sidebar_content_height))
            knob_top = track_rect.top
            if sidebar_scroll_max > 0:
                knob_top += int((sidebar_offset / sidebar_scroll_max) * (sidebar_view_height - knob_height))
            knob_rect = pygame.Rect(track_rect.left, knob_top, track_rect.width, knob_height)
            # Draw a minimal scrollbar to communicate overflow without overwhelming the UI.
            pygame.draw.rect(screen, ACCENT_COLOR, knob_rect, border_radius=2)

        chart_canvas_rect = pygame.Rect(
            chart_rect.left + 30,
            chart_rect.top + caption_offset,
            max(240, chart_rect.width - 60),
            max(200, chart_rect.height - caption_offset - 60),
        )
        # The chart surface is scaled into this inner rectangle to preserve consistent margins.

        if charts:
            name, surface = charts[chart_index]
            caption_surface = heading_font.render(
                f"{chart_index + 1}/{len(charts)} | {name}", True, TEXT_COLOR
            )
            screen.blit(caption_surface, (chart_rect.left + 20, chart_rect.top + 18))
            inner_rect = chart_canvas_rect
            scaled = pygame.transform.smoothscale(surface, (inner_rect.width, inner_rect.height))
            screen.blit(scaled, inner_rect.topleft)
        else:
            empty_surface = heading_font.render("No charts available", True, TEXT_COLOR)
            empty_rect = empty_surface.get_rect(center=chart_rect.center)
            screen.blit(empty_surface, empty_rect)

        instructions_pos = (
            (SCREEN_SIZE[0] - instructions_surface.get_width()) // 2,
            button_y - instructions_surface.get_height() - 12,
        )
        screen.blit(instructions_surface, instructions_pos)

        _draw_buttons(screen, buttons, button_font, hovered_button)

        if status_message and pygame.time.get_ticks() < status_timeout:
            status_surface = info_font.render(status_message, True, ACCENT_COLOR)
            status_rect = status_surface.get_rect()
            status_rect.bottomright = (
                SCREEN_SIZE[0] - PADDING,
                button_y - 16,
            )
            screen.blit(status_surface, status_rect)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
