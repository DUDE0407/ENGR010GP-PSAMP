"""Pygame dashboard for power system analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pandas as pd
import pygame

from .data_loader import (
    clear_cache,
    load_basic_statistics,
    load_daily_pattern,
    load_fault_summary,
    load_hourly_profile,
    load_power_quality_indices,
    load_standard_comparison,
    load_weekly_pattern,
)
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

SCREEN_SIZE = (1280, 720)
SIDEBAR_WIDTH = 480
PADDING = 24
BLOCK_PADDING = 14
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 44
BUTTON_SPACING = 16


@dataclass
class Button:
    label: str
    rect: pygame.Rect
    action: str


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
        lines.append(
            (
                f"{row['station_id']}: V={row['voltage_pu_mean']:.3f} pu | "
                f"I={row['current_pu_mean']:.1f} pu | P={row['real_power_mw_mean']:.1f} MW | "
                f"PF={row['power_factor_mean']:.3f}"
            )
        )
    return lines


def _format_compliance(comparison: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    if comparison.empty:
        return lines
    for _, row in comparison.iterrows():
        lines.append(
            (
                f"{row['station_id']}: Voltage {row['voltage_within_pct']:.1f}% | "
                f"PF {row['power_factor_within_pct']:.1f}% | "
                f"Current {row['current_within_pct']:.1f}%"
            )
        )
    return lines


def _format_quality(quality: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    if quality.empty:
        return lines
    for _, row in quality.iterrows():
        lines.append(
            (
                f"{row['station_id']}: Avg PF {row['avg_power_factor']:.3f} (min {row['min_power_factor']:.3f}) | "
                f"Voltage std {row['voltage_std']:.4f} pu | Low PF hrs {row['low_power_factor_pct']:.1f}%"
            )
        )
    return lines


def _safe_int(value: object) -> int:
    if value is None or pd.isna(value):
        return 0
    return int(value)


def _format_faults(faults: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    if faults.empty:
        return lines
    for _, row in faults.iterrows():
        first = _format_timestamp(row.get("first_fault"))
        last = _format_timestamp(row.get("last_fault"))
        lines.append(
            (
                f"{row['station_id']}: Sag {_safe_int(row['voltage_sag_events'])} | "
                f"Swell {_safe_int(row['voltage_swell_events'])} | Overcurrent {_safe_int(row['over_current_events'])} | "
                f"PF<0.8 {_safe_int(row['very_low_pf_events'])} | First {first} | Last {last}"
            )
        )
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


def _create_buttons(y: int) -> List[Button]:
    buttons: List[Button] = []
    specs = [
        ("Prev Chart", "prev"),
        ("Next Chart", "next"),
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
        color = BUTTON_HOVER if button is hovered else BUTTON_BG
        pygame.draw.rect(surface, color, button.rect, border_radius=10)
        pygame.draw.rect(surface, BUTTON_BORDER, button.rect, width=1, border_radius=10)
        label_surface = font.render(button.label, True, BUTTON_TEXT)
        label_rect = label_surface.get_rect(center=button.rect.center)
        surface.blit(label_surface, label_rect)


def _load_assets():
    stats = load_basic_statistics()
    comparison = load_standard_comparison()
    quality = load_power_quality_indices()
    faults = load_fault_summary()
    daily = load_daily_pattern()
    weekly = load_weekly_pattern()
    hourly = load_hourly_profile()
    charts = build_chart_surfaces(daily, weekly, hourly, comparison)
    return stats, comparison, quality, faults, charts


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

    (
        stats,
        comparison,
        quality,
        faults,
        charts,
    ) = _load_assets()

    text_blocks: List[Tuple[str, Sequence[str]]] = [
        ("Basic Statistics", _format_basic_statistics(stats)),
        ("Standards Compliance", _format_compliance(comparison)),
        ("Power Quality Indices", _format_quality(quality)),
        ("Fault Overview", _format_faults(faults)),
    ]

    def _rebuild_sidebar_surface() -> None:
        nonlocal sidebar_surface, sidebar_content_height, sidebar_scroll_max, sidebar_offset
        sidebar_surface, sidebar_content_height = _build_sidebar_surface(
            text_blocks,
            heading_font,
            body_font,
            sidebar_width,
        )
        sidebar_scroll_max = max(0, sidebar_content_height - sidebar_view_height)
        sidebar_offset = max(0, min(sidebar_offset, sidebar_scroll_max))

    _rebuild_sidebar_surface()

    chart_index = 0
    status_message = ""
    status_timeout = 0
    running = True

    def _update_blocks() -> None:
        nonlocal text_blocks, sidebar_offset
        text_blocks = [
            ("Basic Statistics", _format_basic_statistics(stats)),
            ("Standards Compliance", _format_compliance(comparison)),
            ("Power Quality Indices", _format_quality(quality)),
            ("Fault Overview", _format_faults(faults)),
        ]
        sidebar_offset = 0
        _rebuild_sidebar_surface()

    def _set_status(message: str, duration_ms: int = 2200) -> None:
        nonlocal status_message, status_timeout
        status_message = message
        status_timeout = pygame.time.get_ticks() + duration_ms

    def _apply_action(action: str) -> None:
        nonlocal chart_index, stats, comparison, quality, faults, charts, running
        if action == "next" and charts:
            chart_index = (chart_index + 1) % len(charts)
            _set_status(f"Showing chart {chart_index + 1}/{len(charts)}")
        elif action == "prev" and charts:
            chart_index = (chart_index - 1) % len(charts)
            _set_status(f"Showing chart {chart_index + 1}/{len(charts)}")
        elif action == "reload":
            clear_cache()
            (
                stats,
                comparison,
                quality,
                faults,
                charts,
            ) = _load_assets()
            chart_index = 0
            _update_blocks()
            _set_status("Data reloaded")
        elif action == "exit":
            running = False

    instructions_text = (
        "Charts: buttons or arrow keys | Sidebar: wheel/PgUp/PgDn | R reloads data | Esc exits"
    )
    instructions_surface = info_font.render(instructions_text, True, TEXT_COLOR)

    while running:
        mouse_pos = pygame.mouse.get_pos()
        hovered_button = next((b for b in buttons if b.rect.collidepoint(mouse_pos)), None)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    _apply_action("exit")
                elif event.key == pygame.K_RIGHT:
                    _apply_action("next")
                elif event.key == pygame.K_LEFT:
                    _apply_action("prev")
                elif event.key == pygame.K_r:
                    _apply_action("reload")
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
                for button in buttons:
                    if button.rect.collidepoint(event.pos):
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
            pygame.draw.rect(screen, ACCENT_COLOR, knob_rect, border_radius=2)

        chart_canvas_rect = pygame.Rect(
            chart_rect.left + 30,
            chart_rect.top + caption_offset,
            max(240, chart_rect.width - 60),
            max(200, chart_rect.height - caption_offset - 60),
        )

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
