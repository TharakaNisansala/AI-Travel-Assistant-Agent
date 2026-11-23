"""Stage 7: keeping the agent standing when Gemini or a tool fails.

What this file demonstrates:
  1. Classifying errors before deciding what to do. A 503 or a timeout is
     TRANSIENT: one more try may work. A 429 is a rate limit or the daily
     quota: retrying at once only burns quota, so it is never retried. A 400
     or 401 fails the same way every time.
  2. ONE retry layer for Gemini. The graph's llm node gets a LangGraph
     RetryPolicy (llm_retry_policy) that retries transient errors once; the
     chat builds its model with max_retries=0, so the client does not retry
     underneath. Stacked layers multiply: 4 client tries x 2 node tries = 8
     calls for one step, on a 20-calls-a-day free tier.
  3. repair_history(): a PURE function that makes a message history valid for
     the model after a failure. A crash between "the LLM asked for a tool" and
     "the tool answered" leaves a tool call without a result, which the Gemini
     API rejects. The repair only changes what the LLM SEES; the saved thread
     is never modified.
  4. A loop cap. route_after_agent counts the tool rounds of the current turn
     and sends a turn that keeps asking for tools to a fallback answer.
  5. Safe messages for the traveller: describe_llm_error() never shows the
     raw error text (it can carry request details), only a short explanation.

No graph here; tool_calling.py wires these into the agent.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import httpx
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolInvocationError
from langgraph.types import RetryPolicy

from ai_travel_assistant.context_window import current_turn_start

# --- Classifying errors ---------------------------------------------------------

# Server-side statuses that usually clear up within seconds.
TRANSIENT_STATUSES = frozenset({500, 502, 503, 504})
RATE_LIMIT_STATUS = 429
TOO_LARGE_STATUS = 413
_RATE_LIMIT_CODE = re.compile(r"rate_limit_exceeded")

_LEADING_STATUS = re.compile(r"^\s*(\d{3})\b")
# How the per-day quota is named in a 429: Google says GenerateRequestsPerDay...,
# Groq says "requests per day (RPD)" or "tokens per day (TPD)".
_DAILY_QUOTA = re.compile(r"per\s*day|perday", re.IGNORECASE)


def error_status(exc: BaseException) -> int | None:
    """The HTTP status behind an error, if it has one.

    Gemini errors (google.genai APIError and LangChain's subclasses) carry it
    as .code; other clients (Groq, OpenAI-style SDKs) use .status_code. As a
    last resort, a message that starts with a status ("503 UNAVAILABLE ...")
    counts too.
    """
    for attribute in ("code", "status_code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    match = _LEADING_STATUS.match(str(exc))
    return int(match.group(1)) if match else None


def is_rate_limited(exc: BaseException) -> bool:
    """True for a 429: too many requests per minute, or the daily quota.

    Also Groq's 413 with code rate_limit_exceeded: ONE request bigger than the
    tokens-per-minute limit (seen live: 8,952 tokens against 8,000). It is the
    provider's limit, not the agent's fault, and retrying cannot fix it.
    """
    status = error_status(exc)
    if status == TOO_LARGE_STATUS:
        return bool(_RATE_LIMIT_CODE.search(str(exc)))
    return status == RATE_LIMIT_STATUS


def is_quota_exhausted(exc: BaseException) -> bool:
    """True for a 429 caused by the DAILY quota (no point trying again today)."""
    return is_rate_limited(exc) and bool(_DAILY_QUOTA.search(str(exc)))


_SDK_CONNECTION_ERRORS = frozenset({"APIConnectionError", "APITimeoutError"})


def is_transient_llm_error(exc: BaseException) -> bool:
    """True if one more try may succeed: a 5xx, a timeout or a dropped connection.

    429 is deliberately NOT transient here: the per-minute limit needs about a
    minute and the daily quota a day, so an immediate retry only wastes calls.
    """
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    # The Groq SDK (like OpenAI's) wraps timeouts and dropped connections in
    # its own APIConnectionError/APITimeoutError, which carry no status code.
    # Matched by name so this module never has to import the Groq SDK.
    if any(cls.__name__ in _SDK_CONNECTION_ERRORS for cls in type(exc).__mro__):
        return True
    return error_status(exc) in TRANSIENT_STATUSES


# --- The one retry layer for the LLM --------------------------------------------

# 2 attempts = the first call plus one retry, as approved for Stage 7.
LLM_MAX_ATTEMPTS = 2
LLM_RETRY_WAIT_S = 2.0


def llm_retry_policy(initial_interval: float = LLM_RETRY_WAIT_S) -> RetryPolicy:
    """The RetryPolicy for the agent's llm node: transient errors, one retry.

    LangGraph re-runs the whole node on a retry. The llm node only reads the
    state and calls the model, so running it again is safe. Tests pass
    initial_interval=0 so they never sleep.
    """
    return RetryPolicy(
        max_attempts=LLM_MAX_ATTEMPTS,
        initial_interval=initial_interval,
        backoff_factor=2.0,
        jitter=initial_interval > 0,
        retry_on=is_transient_llm_error,
    )


# --- Messages for the traveller ------------------------------------------------

# Provider-neutral: the same messages serve Gemini and Groq (LLM_PROVIDER).
QUOTA_MESSAGE = (
    "The model's daily free quota is used up. Try again later (Gemini's resets "
    "at about 13:30 Sri Lanka time; Groq's frees up gradually over the day), "
    "or switch LLM_PROVIDER in .env."
)
RATE_LIMIT_MESSAGE = (
    "The model's rate limit was reached. Please wait a minute and try again."
)
BUSY_MESSAGE = (
    "The model is busy or did not answer in time. Please try again in a minute."
)
REJECTED_MESSAGE = (
    "The model rejected the request. Check LLM_PROVIDER and that provider's "
    "API key and model ID in .env."
)


def describe_llm_error(exc: BaseException) -> str:
    """A short, safe explanation of a failed turn.

    Never includes the error's own text: it can carry request details or
    (in a badly formed error) a key. Only the exception's type name is shown
    for unknown errors, to help debugging.
    """
    # Imported here: provider_pool imports this module.
    from ai_travel_assistant.provider_pool import AllProvidersUnavailable

    if isinstance(exc, AllProvidersUnavailable):
        return {
            "quota": QUOTA_MESSAGE,
            "rate_limit": RATE_LIMIT_MESSAGE,
            "busy": BUSY_MESSAGE,
        }.get(exc.kind, REJECTED_MESSAGE)
    if is_quota_exhausted(exc):
        return QUOTA_MESSAGE
    if is_rate_limited(exc):
        return RATE_LIMIT_MESSAGE
    if is_transient_llm_error(exc):
        return BUSY_MESSAGE
    if error_status(exc) in (400, 401, 403, 404):
        return REJECTED_MESSAGE
    return f"Something went wrong ({type(exc).__name__}). Please try again."


# --- Tool errors ------------------------------------------------------------------


def tool_error_message(exc: Exception) -> str:
    """What the LLM reads when a tool raises (ToolNode's handle_tool_errors).

    The services already turn expected failures (timeouts, 503s) into
    "X unavailable" text, so anything reaching here is unexpected, e.g. a bug
    in a formatter. Bad arguments from the LLM keep LangGraph's own message,
    which tells the LLM how to fix the call. ToolNode runs every call of a
    parallel step separately, so one failing tool never hides the others.
    """
    if isinstance(exc, ToolInvocationError):
        return exc.message
    return (
        f"Tool failed with an internal error ({type(exc).__name__}). Do not call "
        "it again in this turn; tell the traveller this data is unavailable."
    )


# --- Repairing the history the LLM sees ------------------------------------------

NOT_RUN_RESULT = (
    "Not run: the earlier request was interrupted before this tool answered."
)
NO_ANSWER_NOTE = "(No answer: the previous request failed before I could reply.)"


def repair_history(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """A valid history for the model, built without changing the input.

    - A tool call without a result (the run crashed in between) gets a
      placeholder result right after its AIMessage.
    - A tool result whose call is not open just before it is dropped.
    - Two traveller messages in a row (the turn between them failed) get a
      short note between them, so the model knows the first went unanswered.
    A history without these problems comes back unchanged (as a new list).
    The input messages are never mutated; only new placeholders are created.
    """
    repaired: list[AnyMessage] = []
    pending: dict[str, str] = {}  # open tool_call_id -> tool name

    def close_pending() -> None:
        for call_id, name in pending.items():
            repaired.append(
                ToolMessage(
                    NOT_RUN_RESULT, tool_call_id=call_id, name=name, status="error"
                )
            )
        pending.clear()

    for message in messages:
        if isinstance(message, ToolMessage):
            if message.tool_call_id in pending:
                del pending[message.tool_call_id]
                repaired.append(message)
            continue  # a result without an open call: drop it
        close_pending()
        if (
            isinstance(message, HumanMessage)
            and repaired
            and isinstance(repaired[-1], HumanMessage)
        ):
            repaired.append(AIMessage(NO_ANSWER_NOTE))
        repaired.append(message)
        if isinstance(message, AIMessage) and message.tool_calls:
            pending = {call["id"]: call["name"] for call in message.tool_calls}
    close_pending()
    return repaired


# --- The loop cap -------------------------------------------------------------------

MAX_TOOL_ROUNDS = 4

AgentRoute = Literal["tools", "fallback", "done", "compose"]
TO_TOOLS: AgentRoute = "tools"
TO_FALLBACK: AgentRoute = "fallback"
TO_DONE: AgentRoute = "done"
TO_COMPOSE: AgentRoute = "compose"  # Stage 8: the agent submitted its final answer


def tool_rounds_this_turn(messages: Sequence[AnyMessage]) -> int:
    """How many times the LLM asked for tools since the latest traveller message.

    Read from the messages, so there is no counter to reset, and it stays
    right after a Stage 6 resume (the reply starts the new turn).
    """
    start = current_turn_start(messages)
    return sum(
        1
        for message in messages[start:]
        if isinstance(message, AIMessage) and message.tool_calls
    )


def make_route_after_agent(
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    final_tool: str | None = None,
) -> Callable[[Mapping], AgentRoute]:
    """Build the router that replaces tools_condition, with a loop cap.

    The LLM may run tools max_tool_rounds times in one turn; when it asks a
    (max_tool_rounds + 1)-th time, the turn goes to the fallback answer instead.

    final_tool (Stage 8, e.g. "submit_itinerary") names a tool that FINISHES
    the turn: a call to it goes to compose, checked before the cap, because
    finishing must never be blocked by the limit on gathering data.
    """

    def route_after_agent(state: Mapping) -> AgentRoute:
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        if not (isinstance(last, AIMessage) and last.tool_calls):
            return TO_DONE
        if final_tool and any(call["name"] == final_tool for call in last.tool_calls):
            return TO_COMPOSE
        if tool_rounds_this_turn(messages) > max_tool_rounds:
            return TO_FALLBACK
        return TO_TOOLS

    return route_after_agent


LOOP_CAP_RESULT = "Not run: the tool limit for this request was reached."
WRAP_UP_INSTRUCTION = (
    "The tool limit for this request has been reached; no more tools can be "
    "called. Answer the traveller now using only the tool results above. Say "
    "briefly which information you could not gather."
)
FALLBACK_REPLY = (
    "Sorry, I could not finish this request: it needed more lookups than I am "
    "allowed in one turn. Please ask again with a narrower question, for "
    "example one place or one topic at a time."
)


def close_pending_calls(last: AIMessage) -> list[ToolMessage]:
    """Results for the calls the loop cap stopped, so no tool call is orphaned."""
    return [
        ToolMessage(
            LOOP_CAP_RESULT, tool_call_id=call["id"], name=call["name"], status="error"
        )
        for call in last.tool_calls
    ]
