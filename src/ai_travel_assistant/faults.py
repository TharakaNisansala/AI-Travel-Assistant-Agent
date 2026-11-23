"""Stage 7: fault injection, to see how the agent behaves when an API is down.

What this file demonstrates:
  1. Testing failure paths on purpose. Real outages are rare and random; a
     switch that makes one service fail every time lets you watch the whole
     chain react: the service's retry, its typed error, the tool's
     "X unavailable" text, and the agent's honest answer without that data.
  2. Injecting at the lowest level. The failure is an httpx transport that
     never reaches the network, plugged into the service's own make_client().
     Everything above it (retries, error handling, tools) is the real code.

Off unless the FAULT_INJECT environment variable is set, e.g. in PowerShell:
    $env:FAULT_INJECT = "weather"            # weather API unreachable
    $env:FAULT_INJECT = "weather,guide:busy" # also the guide answers 503
    Remove-Item Env:FAULT_INJECT             # back to normal
Services: weather, attractions, guide, travel, currency.
Modes: down (connection refused, the default), busy (HTTP 503), timeout.
An unknown service or mode is an error, so a typo never silently does nothing.
"""

from __future__ import annotations

import os

import httpx

ENV_VAR = "FAULT_INJECT"
SERVICES = ("weather", "attractions", "guide", "travel", "currency", "pricing")
MODES = ("down", "busy", "timeout")
DEFAULT_MODE = "down"


class FaultConfigError(ValueError):
    """FAULT_INJECT names an unknown service or mode."""


def parse_faults(value: str | None) -> dict[str, str]:
    """'weather,guide:busy' -> {'weather': 'down', 'guide': 'busy'}. Empty -> {}."""
    faults: dict[str, str] = {}
    for part in (value or "").split(","):
        part = part.strip().lower()
        if not part:
            continue
        service, _, mode = part.partition(":")
        service, mode = service.strip(), (mode.strip() or DEFAULT_MODE)
        if service not in SERVICES:
            raise FaultConfigError(
                f"{ENV_VAR}: unknown service {service!r}; use {', '.join(SERVICES)}"
            )
        if mode not in MODES:
            raise FaultConfigError(
                f"{ENV_VAR}: unknown mode {mode!r}; use {', '.join(MODES)}"
            )
        faults[service] = mode
    return faults


def active_faults() -> dict[str, str]:
    """The faults switched on right now (read on every call; {} when unset)."""
    return parse_faults(os.environ.get(ENV_VAR))


def describe_active_faults() -> str | None:
    """A warning line for the terminal, or None when nothing is injected."""
    faults = active_faults()
    if not faults:
        return None
    listed = ", ".join(f"{name} ({mode})" for name, mode in faults.items())
    return f"FAULT INJECTION ACTIVE ({ENV_VAR}): {listed}"


def _failing_handler(mode: str):
    def handle(request: httpx.Request) -> httpx.Response:
        if mode == "busy":
            return httpx.Response(503, request=request, text="Injected fault: busy")
        if mode == "timeout":
            raise httpx.ReadTimeout("Injected fault: timeout", request=request)
        raise httpx.ConnectError("Injected fault: service down", request=request)

    return handle


def transport_for(service: str) -> httpx.BaseTransport | None:
    """A failing transport if `service` is injected, otherwise None (normal use)."""
    if service not in SERVICES:
        raise ValueError(f"unknown service {service!r}")
    mode = active_faults().get(service)
    if mode is None:
        return None
    return httpx.MockTransport(_failing_handler(mode))
