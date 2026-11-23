"""Stage 10 polish: the look of the web page (styling, header, itinerary cards).

web_app.py decides WHAT the page shows; this module decides HOW it looks, so
the agent code never changes for a visual tweak.

What this file demonstrates:
  1. Rendering from structured output. The itinerary is drawn from the
     checked Itinerary data (state["itinerary"]), not by re-parsing the
     Markdown: day tabs, a timeline per day, weather and travel chips, costs.
     This is why Stage 8 made the plan a schema: a UI can use its fields.
  2. Never trusting model text inside HTML. Every value from the model or a
     tool goes through html.escape() before it is placed in an HTML string,
     so a place name such as "<script>" is shown as text, never run.
  3. Theme in two layers: .streamlit/config.toml sets the colours Streamlit
     itself uses; the CSS here styles the custom header and cards.
"""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from typing import Any

# --- Palette (keep in step with .streamlit/config.toml) ---------------------------

OCEAN_DEEP = "#0B3B47"
OCEAN = "#0F766E"
SEA = "#14B8A6"
GOLD = "#C9A227"
SAND = "#F3EEE5"
PAGE = "#FBF9F5"
INK = "#1E293B"
MUTED = "#64748B"

APP_NAME = "Ceylon Concierge"
TAGLINE = "Your AI travel planner for Sri Lanka"
DATA_SOURCES = (
    "Open-Meteo",
    "OpenStreetMap",
    "Wikivoyage",
    "OSRM routing",
    "Central Cultural Fund (ticket prices)",
)

USER_AVATAR = "🧳"
ASSISTANT_AVATAR = "🌴"

# A card whose "message" is this opens the province/district picker instead.
WEATHER_PICKER = ""

# (title, subtitle, the message it sends)
SUGGESTIONS: tuple[tuple[str, str, str], ...] = (
    # First: the easiest start. The agent then asks each detail with buttons.
    ("Plan a trip for me", "Answer a few quick questions", "Plan me a trip."),
    ("Kandy in 2 days", "Temples, the lake and culture", "Plan 2 days in Kandy"),
    (
        "Ella for hikers",
        "Peaks, bridges and tea country",
        "Plan 3 days in Ella; I like hiking and scenic places",
    ),
    (
        "Weather check",
        "Pick a province, then a district",
        WEATHER_PICKER,
    ),
)

TIME_ICONS = {"morning": "🌅", "afternoon": "☀️", "evening": "🌙"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,650&family=Inter:wght@400;500;600&display=swap');

/* Body text in Inter. Not a blanket rule on every Streamlit class: the
   icons (expander arrows, sidebar toggle) are drawn with an icon font. */
html, body, p, li, label, input, textarea, button, td, th,
.stMarkdown, [data-testid="stCaptionContainer"] {{
  font-family: 'Inter', system-ui, sans-serif;
}}
[data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}
h1, h2, h3, .cc-display {{ font-family: 'Fraunces', Georgia, serif; }}
.block-container {{ padding-top: 2rem; max-width: 900px; }}

/* Header */
.cc-hero {{
  background: linear-gradient(135deg, {OCEAN_DEEP} 0%, {OCEAN} 60%, {SEA} 100%);
  border-radius: 20px; padding: 28px 32px 24px; color: #fff;
  box-shadow: 0 12px 32px rgba(11, 59, 71, 0.18); position: relative;
  overflow: hidden; margin-bottom: 1.25rem;
}}
.cc-hero::after {{
  content: ""; position: absolute; right: -60px; top: -60px; width: 220px;
  height: 220px; border-radius: 50%;
  background: radial-gradient(circle, rgba(201,162,39,0.45), transparent 70%);
}}
.cc-hero h1 {{
  color: #fff; font-size: 2.1rem; margin: 0; padding: 0; letter-spacing: -0.01em;
  font-family: 'Fraunces', Georgia, serif !important; font-weight: 650;
}}
.cc-hero .cc-tagline {{ color: rgba(255,255,255,0.85); margin: 4px 0 16px; }}
.cc-chip {{
  display: inline-block; padding: 3px 10px; margin: 0 6px 6px 0;
  border-radius: 999px; font-size: 0.78rem; font-weight: 500;
  background: rgba(255,255,255,0.14); color: #fff;
  border: 1px solid rgba(255,255,255,0.25);
}}
.cc-badge {{
  position: absolute; right: 24px; top: 22px; z-index: 1; padding: 4px 12px;
  border-radius: 999px; font-size: 0.75rem; font-weight: 600;
  background: {GOLD}; color: {OCEAN_DEEP};
}}
@media (max-width: 640px) {{
  .cc-hero {{ padding: 22px 20px 18px; }}
  .cc-hero h1 {{ font-size: 1.7rem; }}
  .cc-badge {{ position: static; display: inline-block; margin-bottom: 10px; }}
}}

/* Chat */
[data-testid="stChatMessage"] {{
  background: #fff; border: 1px solid #EAE4D8; border-radius: 16px;
  padding: 14px 18px; box-shadow: 0 2px 10px rgba(30, 41, 59, 0.05);
}}
[data-testid="stChatInput"] {{ border-radius: 14px; }}

/* Suggestion buttons on the empty page */
div[data-testid="stHorizontalBlock"] .stButton > button {{
  height: 100%; text-align: left; justify-content: flex-start;
  border-radius: 14px; border: 1px solid #E6DECF; background: #fff;
  padding: 14px 16px; white-space: normal; line-height: 1.35;
  box-shadow: 0 2px 8px rgba(30, 41, 59, 0.04);
  transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
}}
div[data-testid="stHorizontalBlock"] .stButton > button > div,
div[data-testid="stHorizontalBlock"] .stButton > button > div > span,
div[data-testid="stHorizontalBlock"] .stButton [data-testid="stMarkdownContainer"],
div[data-testid="stHorizontalBlock"] .stButton > button p {{
  display: block; width: 100%; text-align: left; margin: 0;
}}
div[data-testid="stHorizontalBlock"] .stButton > button strong {{
  color: {OCEAN_DEEP}; font-size: 1rem;
}}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {{
  transform: translateY(-2px); border-color: {GOLD};
  box-shadow: 0 8px 18px rgba(11, 59, 71, 0.10);
}}

/* Sidebar */
[data-testid="stSidebar"] {{ background: {SAND}; }}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
  color: {OCEAN_DEEP}; font-size: 1.05rem;
}}

/* Itinerary */
.cc-plan-head {{ margin: 2px 0 10px; }}
.cc-plan-head .cc-display {{ font-size: 1.55rem; color: {OCEAN_DEEP}; margin: 0; }}
.cc-plan-head p {{ color: {MUTED}; margin: 4px 0 10px; }}
.cc-stat {{
  display: inline-block; padding: 4px 12px; margin: 0 8px 8px 0;
  border-radius: 10px; background: {SAND}; color: {INK}; font-size: 0.85rem;
}}
.cc-stat b {{ color: {OCEAN}; }}
.cc-meta {{
  display: inline-block; padding: 4px 10px; margin: 0 8px 8px 0;
  border-radius: 999px; font-size: 0.82rem; background: #EEF7F6;
  color: {OCEAN_DEEP}; border: 1px solid #CDE8E4;
}}
.cc-timeline {{ border-left: 2px solid #E6DECF; margin: 8px 0 4px 10px; }}
.cc-act {{ position: relative; padding: 2px 0 14px 22px; }}
.cc-act::before {{
  content: ""; position: absolute; left: -7px; top: 6px; width: 12px;
  height: 12px; border-radius: 50%; background: {GOLD};
  box-shadow: 0 0 0 3px {PAGE};
}}
.cc-when {{
  font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.06em; color: {MUTED};
}}
.cc-name {{ font-weight: 600; color: {INK}; font-size: 1.02rem; }}
.cc-desc {{ color: #334155; margin: 2px 0 4px; }}
.cc-tag {{
  display: inline-block; font-size: 0.74rem; color: {MUTED};
  margin-right: 10px;
}}
.cc-table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
.cc-table th {{
  text-align: left; color: {MUTED}; font-weight: 600; font-size: 0.78rem;
  text-transform: uppercase; letter-spacing: 0.05em;
  border-bottom: 1px solid #E6DECF; padding: 6px 4px;
}}
.cc-table td {{ border-bottom: 1px solid #F0EBE1; padding: 8px 4px; }}
.cc-table td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
</style>
"""


# --- Small formatters ---------------------------------------------------------------


def _duration(minutes: float) -> str:
    hours, mins = divmod(round(minutes), 60)
    return f"{hours} h {mins} min" if hours else f"{mins} min"


def _hours(value: float) -> str:
    return f"{value:g} h"


def _money(value: float | None, currency: str) -> str:
    if value is None:
        return "–"
    return (
        f"{currency} {value:,.0f}" if currency == "LKR" else f"{currency} {value:,.2f}"
    )


# --- HTML pieces (pure: unit-tested without Streamlit) -----------------------------


def hero_html(model_label: str | None) -> str:
    """The page header: name, tagline, data sources and the model badge."""
    chips = "".join(f'<span class="cc-chip">{escape(s)}</span>' for s in DATA_SOURCES)
    badge = (
        f'<span class="cc-badge">{escape(model_label)}</span>' if model_label else ""
    )
    return (
        f'<div class="cc-hero">{badge}<h1>{escape(APP_NAME)}</h1>'
        f'<div class="cc-tagline">{escape(TAGLINE)}</div>{chips}</div>'
    )


def plan_header_html(itinerary: Mapping[str, Any]) -> str:
    """Destination, summary and the headline numbers of a plan."""
    days = itinerary.get("day_plans") or []
    activities = sum(len(day.get("activities") or []) for day in days)
    start = escape(str(itinerary.get("start_from", "")))
    if itinerary.get("start_assumed"):
        start += " (assumed)"
    stats = [
        f'<span class="cc-stat"><b>{len(days)}</b> days</span>',
        f'<span class="cc-stat"><b>{activities}</b> activities</span>',
        f'<span class="cc-stat">Starts from <b>{start}</b></span>',
    ]
    title = f"{len(days)}-day trip to {itinerary.get('destination', '')}"
    return (
        f'<div class="cc-plan-head"><div class="cc-display">{escape(title)}</div>'
        f"<p>{escape(str(itinerary.get('summary', '')))}</p>{''.join(stats)}</div>"
    )


def day_meta_html(day: Mapping[str, Any]) -> str:
    """The weather and travel chips at the top of a day."""
    chips = []
    if day.get("weather"):
        chips.append(f'<span class="cc-meta">🌤️ {escape(str(day["weather"]))}</span>')
    for leg in day.get("travel") or []:
        parts = [f"{leg.get('from_place', '')} → {leg.get('to_place', '')}"]
        if leg.get("road_distance_km") is not None:
            parts.append(f"{leg['road_distance_km']:g} km")
        if leg.get("driving_minutes") is not None:
            parts.append(_duration(leg["driving_minutes"]))
        chips.append(f'<span class="cc-meta">🚗 {escape(" · ".join(parts))}</span>')
    return "".join(chips)


def timeline_html(activities: list[Mapping[str, Any]]) -> str:
    """A day's activities as a vertical timeline."""
    items = []
    for act in activities:
        when = str(act.get("time_of_day", ""))
        tags = [f"📚 {act.get('source', '')}"]
        if act.get("estimated_hours"):
            tags.insert(0, f"⏱ about {_hours(act['estimated_hours'])}")
        if act.get("distance_km") is not None:
            tags.insert(0, f"📍 {act['distance_km']:g} km away")
        tag_html = "".join(f'<span class="cc-tag">{escape(t)}</span>' for t in tags)
        items.append(
            '<div class="cc-act">'
            f'<div class="cc-when">{TIME_ICONS.get(when, "•")} {escape(when)}</div>'
            f'<div class="cc-name">{escape(str(act.get("name", "")))}</div>'
            f'<div class="cc-desc">{escape(str(act.get("description", "")))}</div>'
            f"{tag_html}</div>"
        )
    return f'<div class="cc-timeline">{"".join(items)}</div>'


def costs_html(costs: list[Mapping[str, Any]]) -> str:
    """Quoted prices with their conversions, as a small table."""
    rows = "".join(
        f"<tr><td>{escape(str(c.get('item', '')))}</td>"
        f'<td class="num">{escape(_money(c.get("amount_lkr"), "LKR"))}</td>'
        f'<td class="num">{escape(_money(c.get("amount_usd"), "USD"))}</td></tr>'
        for c in costs
    )
    return (
        '<table class="cc-table"><tr><th>Item</th><th style="text-align:right">LKR</th>'
        f'<th style="text-align:right">USD</th></tr>{rows}</table>'
    )


def budget_items_markdown(items: list[Any]) -> str:
    """One budget section's items as nested Markdown bullets."""
    lines = []
    for item in items:
        lines.append(f"- **{item.text}**" if item.details else f"- {item.text}")
        lines.extend(f"    - {detail}" for detail in item.details)
    return "\n".join(lines)


def day_tab_label(day: Mapping[str, Any]) -> str:
    return f"Day {day.get('day', '?')} · {day.get('theme', '')}"


# --- Streamlit views ----------------------------------------------------------------


def apply_theme() -> None:
    import streamlit as st

    st.markdown(CSS, unsafe_allow_html=True)


def pdf_download_button(
    itinerary: Mapping[str, Any], key: str, *, sample: bool = False
) -> None:
    """'Download plan (PDF)', or a short note if the PDF cannot be made.

    sample=True: the prepared demo plan; the PDF and its file name say so.
    """
    import streamlit as st

    from ai_travel_assistant.itinerary_pdf import itinerary_pdf, pdf_file_name

    try:
        data = itinerary_pdf(itinerary, sample=sample)
    except Exception:  # a broken PDF must never take the chat page down
        st.caption("The PDF could not be made for this plan.")
        return
    name = pdf_file_name(itinerary)
    st.download_button(
        "⬇ Download sample plan (PDF)" if sample else "⬇ Download plan (PDF)",
        data,
        file_name=f"sample-{name}" if sample else name,
        mime="application/pdf",
        key=key,
        type="primary",
        use_container_width=True,
    )


def render_itinerary(
    itinerary: Mapping[str, Any],
    markdown_text: str,
    key: str,
    *,
    sample: bool = False,
) -> None:
    """Draw a checked itinerary as cards, with a PDF download and the plain text.

    sample=True: the prepared demo plan (sample_trip.py), labelled as such
    above the cards, on the download button and inside the PDF.
    """
    import streamlit as st

    if sample:
        from ai_travel_assistant.sample_trip import SAMPLE_LABEL, SAMPLE_NOTE

        st.warning(f"**{SAMPLE_LABEL}.** {SAMPLE_NOTE}", icon="🧪")
    st.markdown(plan_header_html(itinerary), unsafe_allow_html=True)

    days = itinerary.get("day_plans") or []
    for tab, day in zip(st.tabs([day_tab_label(d) for d in days]), days, strict=True):
        with tab:
            meta = day_meta_html(day)
            if meta:
                st.markdown(meta, unsafe_allow_html=True)
            st.markdown(
                timeline_html(day.get("activities") or []), unsafe_allow_html=True
            )

    # Entrance fees, other costs and the budget: the same sections as the
    # Markdown reply and the PDF (itinerary_budget.budget_sections).
    from ai_travel_assistant.itinerary_budget import budget_sections

    money = budget_sections(itinerary)
    for title, items in money:
        icon = "🧾" if title.startswith("Trip budget") else "💰"
        st.markdown(f"##### {icon} {title}")
        st.markdown(budget_items_markdown(items))
    if not money and itinerary.get("costs"):
        st.markdown("##### 💰 Costs")
        st.markdown(costs_html(itinerary["costs"]), unsafe_allow_html=True)

    facts = itinerary.get("facts") or []
    tips = itinerary.get("tips") or []
    if facts or tips:
        st.markdown("##### 🧭 Good to know")
        for fact in facts:
            st.markdown(f"- {fact.get('text', '')} _({fact.get('source', '')})_")
        for tip in tips:
            st.markdown(f"- {tip} _(suggestion, not checked)_")

    checks = list(itinerary.get("warnings") or [])
    gaps = list(itinerary.get("not_available") or [])
    assumptions = list(itinerary.get("assumptions") or [])
    if checks or gaps or assumptions:
        count = len(checks) + len(gaps) + len(assumptions)
        with st.expander(f"Checks, assumptions and gaps ({count})"):
            for line in assumptions:
                st.markdown(f"- **Assumed:** {line}")
            for line in checks:
                st.markdown(f"- **Check:** {line}")
            for line in gaps:
                st.markdown(f"- **Not available:** {line}")

    pdf_download_button(itinerary, key=f"pdf-{key}", sample=sample)
    with st.expander("Plain-text version"):
        st.markdown(markdown_text)
