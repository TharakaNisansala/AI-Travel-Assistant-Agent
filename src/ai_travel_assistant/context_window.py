"""Stage 5: keeping long conversations inside a token budget.

What this file demonstrates:
  1. Why trimming matters: every LLM call sends the whole history, so a long
     chat costs more tokens (and time) on every turn. Old tool results are the
     bulkiest part: one itinerary turn holds guide, attractions and weather
     text worth several thousand tokens.
  2. Compacting before trimming. From EARLIER turns we keep only what the
     traveller said and the agent's final answers; their tool calls and tool
     results are dropped, because the final answer already holds what mattered.
     This also means a tool call can never be separated from its result, which
     the model API would reject.
  3. trim_messages(strategy="last", start_on="human"): keep the newest messages
     that fit max_tokens, starting at a traveller message so the history never
     opens with an orphan AI reply.
  4. The CURRENT turn (from the latest traveller message on) is never touched:
     its tool results are what the LLM is working from right now.

Only what the LLM SEES is trimmed. The checkpointer still saves the complete
thread, so nothing is lost and a later, bigger budget could show more.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately

# Budget for EARLIER turns, in approximate tokens (about 4 characters each).
# Roughly ten question-and-answer pairs of itinerary size; the current turn
# comes on top of this.
MAX_HISTORY_TOKENS = 8_000


def current_turn_start(messages: Sequence[AnyMessage]) -> int:
    """Index of the latest traveller message (0 if there is none)."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return index
    return 0


def compact_earlier_turns(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """Keep traveller messages and final answers; drop tool calls and results."""
    return [
        message
        for message in messages
        if isinstance(message, HumanMessage)
        or (isinstance(message, AIMessage) and not message.tool_calls)
    ]


def history_for_llm(
    messages: Sequence[AnyMessage], max_tokens: int | None = MAX_HISTORY_TOKENS
) -> list[BaseMessage]:
    """The messages the LLM should see: compacted, trimmed history + this turn.

    max_tokens=None turns trimming off and returns the messages unchanged.
    """
    if max_tokens is None:
        return list(messages)
    start = current_turn_start(messages)
    earlier = compact_earlier_turns(messages[:start])
    kept = trim_messages(
        earlier,
        max_tokens=max_tokens,
        token_counter=count_tokens_approximately,
        strategy="last",
        start_on="human",
    )
    return [*kept, *messages[start:]]
