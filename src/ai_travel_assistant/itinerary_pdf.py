"""Stage 10: the checked itinerary as a downloadable PDF (needs the ui extra).

What this file demonstrates:
  1. One more presentation of the same structured output. The PDF is built
     from state["itinerary"] (the validated, grounded Itinerary data), the
     same fields the day cards use. There is no LLM call: the same plan always
     gives the same PDF, it costs no quota, and nothing the model wrote as
     prose can drop a day or an activity on the way.
  2. Fonts are data too. The built-in PDF fonts only know Latin-1, and models
     write characters such as "→", "‑" (U+2011) or "’". The bundled DejaVu Sans
     covers them; anything it cannot draw (emoji, for example) is removed
     instead of breaking the file.

Use: itinerary_pdf(itinerary_dict) -> bytes, pdf_file_name(itinerary_dict).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fpdf.fonts import FontFace

from ai_travel_assistant import web_ui
from ai_travel_assistant.web_ui import _duration, _hours, _money

FONT_DIR = Path(__file__).parent / "fonts"
FONT = "DejaVu"
TIME_ORDER = {"morning": 0, "afternoon": 1, "evening": 2}
DISCLAIMER = (
    "Opening hours, prices and road conditions change. Please check the details "
    "before you travel."
)


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    value = hex_colour.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


class _PlanPDF(FPDF):
    """A4 pages with the app name and a page number in the footer."""

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font(FONT, "", 8)
        self.set_text_color(*_rgb(web_ui.MUTED))
        self.cell(0, 6, f"{web_ui.APP_NAME} · page {self.page_no()}/{{nb}}", align="C")


class _Writer:
    """Small helpers so the layout code below reads top to bottom."""

    def __init__(self) -> None:
        pdf = _PlanPDF(format="A4")
        pdf.set_margins(18, 16, 18)
        pdf.set_auto_page_break(auto=True, margin=18)
        pdf.add_font(FONT, "", str(FONT_DIR / "DejaVuSans.ttf"))
        pdf.add_font(FONT, "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
        pdf.set_title("Trip plan")
        pdf.set_author(web_ui.APP_NAME)
        pdf.add_page()
        self.pdf = pdf
        self._glyphs = pdf.fonts[FONT.lower()].cmap

    def clean(self, value: object) -> str:
        """Text the font can draw: unknown characters removed, spaces tidied."""
        text = "" if value is None else str(value)
        text = "".join(ch for ch in text if ch in "\n" or ord(ch) in self._glyphs)
        return re.sub(r"[ \t]+", " ", text).strip()

    def text(
        self,
        value: object,
        size: float = 10,
        bold: bool = False,
        colour: str = web_ui.INK,
        height: float = 5.2,
        indent: float = 0,
    ) -> None:
        pdf = self.pdf
        pdf.set_font(FONT, "B" if bold else "", size)
        pdf.set_text_color(*_rgb(colour))
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(
            pdf.epw - indent,
            height,
            self.clean(value),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

    def heading(self, value: str) -> None:
        pdf = self.pdf
        if pdf.get_y() > pdf.h - 45:  # keep a heading with its first lines
            pdf.add_page()
        pdf.ln(4)
        self.text(value, size=13, bold=True, colour=web_ui.OCEAN, height=7)
        pdf.set_draw_color(*_rgb(web_ui.GOLD))
        pdf.set_line_width(0.5)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + 30, pdf.get_y())
        pdf.ln(2.5)

    def bullets(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.text(f"•  {line}", indent=2)
            self.pdf.ln(0.8)


# --- Sections -------------------------------------------------------------------------


def _title_band(w: _Writer, plan: Mapping[str, Any]) -> None:
    pdf = w.pdf
    days = plan.get("day_plans") or []
    title = w.clean(f"{len(days)}-day trip to {plan.get('destination', '')}")
    pdf.set_font(FONT, "B", 20)
    title_lines = len(pdf.multi_cell(pdf.epw, 9, title, dry_run=True, output="LINES"))
    band = 21 + 9 * title_lines  # a long destination wraps; the band grows with it
    top = pdf.get_y()
    pdf.set_fill_color(*_rgb(web_ui.OCEAN_DEEP))
    pdf.rect(0, 0, pdf.w, top + band, style="F")
    pdf.set_fill_color(255, 255, 255)  # later fills (the costs table) stay white
    pdf.set_y(top + 4)
    w.text(title, size=20, bold=True, colour="#FFFFFF", height=9)
    start = str(plan.get("start_from", ""))
    if plan.get("start_assumed"):
        start += " (assumed)"
    activities = sum(len(d.get("activities") or []) for d in days)
    w.text(
        f"{web_ui.APP_NAME} · starts from {start} · {activities} activities",
        size=9.5,
        colour=web_ui.GOLD,
    )
    pdf.set_y(top + band + 4)
    if plan.get("summary"):
        w.text(plan["summary"], size=10.5)


def _day(w: _Writer, day: Mapping[str, Any]) -> None:
    w.heading(f"Day {day.get('day', '?')} · {day.get('theme', '')}")
    if day.get("forecast_date"):
        w.text(f"Date: {day['forecast_date']}", size=9.5, colour=web_ui.MUTED)
    if day.get("weather"):
        w.text(f"Weather: {day['weather']}", size=9.5, colour=web_ui.MUTED)
    for leg in day.get("travel") or []:
        parts = [f"{leg.get('from_place', '')} → {leg.get('to_place', '')}"]
        if leg.get("road_distance_km") is not None:
            parts.append(f"{leg['road_distance_km']:g} km")
        if leg.get("driving_minutes") is not None:
            parts.append(f"about {_duration(leg['driving_minutes'])} by road")
        w.text("Travel: " + " · ".join(parts), size=9.5, colour=web_ui.MUTED)
    w.pdf.ln(1.5)

    activities = sorted(
        day.get("activities") or [],
        key=lambda a: TIME_ORDER.get(str(a.get("time_of_day")), 3),
    )
    for act in activities:
        if w.pdf.get_y() > w.pdf.h - 40:  # keep an activity's lines on one page
            w.pdf.add_page()
        w.text(
            f"{str(act.get('time_of_day', '')).capitalize()}  ·  {act.get('name', '')}",
            size=10.5,
            bold=True,
        )
        if act.get("description"):
            w.text(act["description"], indent=4)
        tags = []
        if act.get("estimated_hours"):
            tags.append(f"about {_hours(act['estimated_hours'])}")
        if act.get("distance_km") is not None:
            tags.append(f"{act['distance_km']:g} km away")
        if act.get("source"):
            tags.append(f"source: {act['source']}")
        if tags:
            w.text(" · ".join(tags), size=8.5, colour=web_ui.MUTED, indent=4)
        w.pdf.ln(2)


def _costs(w: _Writer, costs: list[Mapping[str, Any]]) -> None:
    w.heading("Costs")
    pdf = w.pdf
    pdf.set_font(FONT, "", 9.5)
    pdf.set_text_color(*_rgb(web_ui.INK))
    with pdf.table(
        col_widths=(3, 1.3, 1.3),
        text_align=("LEFT", "RIGHT", "RIGHT"),
        headings_style=FontFace(emphasis="BOLD", color=_rgb(web_ui.OCEAN)),
        line_height=6,
        borders_layout="HORIZONTAL_LINES",
        cell_fill_mode="NONE",
    ) as table:
        header = table.row()
        for label in ("Item", "LKR", "USD"):
            header.cell(label)
        for cost in costs:
            row = table.row()
            row.cell(w.clean(cost.get("item")))
            row.cell(w.clean(_money(cost.get("amount_lkr"), "LKR")))
            row.cell(w.clean(_money(cost.get("amount_usd"), "USD")))


def _budget(w: _Writer, plan: Mapping[str, Any]) -> bool:
    """Entrance fees, other costs and the budget; False if the plan has none."""
    from ai_travel_assistant.itinerary_budget import budget_sections

    sections = budget_sections(plan)
    for title, items in sections:
        w.heading(title)
        for item in items:
            w.text(item.text, size=9.5, bold=bool(item.details))
            for detail in item.details:
                w.text(f"•  {detail}", size=8.5, indent=4)
            w.pdf.ln(1)
    return bool(sections)


def _notes(w: _Writer, plan: Mapping[str, Any]) -> None:
    facts = plan.get("facts") or []
    tips = plan.get("tips") or []
    if facts or tips:
        w.heading("Good to know")
        w.bullets(f"{f.get('text', '')} ({f.get('source', '')})" for f in facts)
        w.bullets(f"{tip} (suggestion, not checked)" for tip in tips)

    checks = [f"Assumed: {x}" for x in plan.get("assumptions") or []]
    checks += [f"Check: {x}" for x in plan.get("warnings") or []]
    checks += [f"Not available: {x}" for x in plan.get("not_available") or []]
    if checks:
        w.heading("Checks, assumptions and gaps")
        w.bullets(checks)


# --- Public API -----------------------------------------------------------------------


def itinerary_pdf(plan: Mapping[str, Any], today: date | None = None) -> bytes:
    """The whole checked itinerary as a PDF file's bytes."""
    w = _Writer()
    _title_band(w, plan)
    for day in plan.get("day_plans") or []:
        _day(w, day)
    if not _budget(w, plan) and plan.get("costs"):
        _costs(w, plan["costs"])  # itineraries saved before the budget existed
    _notes(w, plan)

    w.pdf.ln(6)
    made = (today or date.today()).isoformat()
    w.text(
        f"Made on {made} from: {', '.join(web_ui.DATA_SOURCES)}.",
        size=8.5,
        colour=web_ui.MUTED,
    )
    w.text(DISCLAIMER, size=8.5, colour=web_ui.MUTED)
    return bytes(w.pdf.output())


def pdf_file_name(plan: Mapping[str, Any]) -> str:
    """'kandy-2-day-plan.pdf': only letters, digits and hyphens."""
    destination = str(plan.get("destination", "")).lower()
    slug = re.sub(r"[^a-z0-9]+", "-", destination).strip("-") or "trip"
    days = len(plan.get("day_plans") or [])
    return f"{slug}-{days}-day-plan.pdf" if days else f"{slug}-plan.pdf"
