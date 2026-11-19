"""Sri Lanka's 9 provinces and 25 districts, for pick-a-place buttons.

Fixed public facts, so they live in code: no API call and no LLM is needed to
show them. The web page uses them for the weather picker (province, then
district); the message it sends names one district, which the weather tool
geocodes like any other place (each district shares its name with its main
town).
"""

from __future__ import annotations

PROVINCES: dict[str, tuple[str, ...]] = {
    "Western": ("Colombo", "Gampaha", "Kalutara"),
    "Central": ("Kandy", "Matale", "Nuwara Eliya"),
    "Southern": ("Galle", "Matara", "Hambantota"),
    "Northern": ("Jaffna", "Kilinochchi", "Mannar", "Vavuniya", "Mullaitivu"),
    "Eastern": ("Trincomalee", "Batticaloa", "Ampara"),
    "North Western": ("Kurunegala", "Puttalam"),
    "North Central": ("Anuradhapura", "Polonnaruwa"),
    "Uva": ("Badulla", "Monaragala"),
    "Sabaragamuwa": ("Ratnapura", "Kegalle"),
}

DISTRICTS: tuple[str, ...] = tuple(d for ds in PROVINCES.values() for d in ds)


def province_of(district: str) -> str | None:
    """The province a district belongs to, or None if it is not a district."""
    wanted = district.strip().casefold()
    for province, districts in PROVINCES.items():
        if any(d.casefold() == wanted for d in districts):
            return province
    return None


def weather_question(district: str) -> str:
    """The message a district button sends: plain enough for any LLM.

    Only the district name: it geocodes on its own (checked for all 25),
    while "Badulla, Uva Province" might be passed to the tool whole.
    """
    return f"What's the weather in {district}?"
