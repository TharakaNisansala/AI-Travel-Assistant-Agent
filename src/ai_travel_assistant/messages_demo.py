"""Stage 1 - Lesson 2: LangChain message types and prompting.

What this file demonstrates:
  1. A plain string prompt.
  2. The same idea as a list of messages: SystemMessage + HumanMessage.
  3. A Sri Lanka travel assistant system instruction, built as a reusable prompt.
  4. Conversation history: an earlier HumanMessage/AIMessage pair, then a follow-up.
  5. The same question with and without a system instruction.

Run from the project root:
    uv run python -m ai_travel_assistant.messages_demo             # all examples
    uv run python -m ai_travel_assistant.messages_demo 4           # one example
    uv run python -m ai_travel_assistant.messages_demo --dry-run   # no API calls

--dry-run prints the messages each example would send, without calling Gemini.

No LangGraph, tools or memory here. The "history" in Example 4 is a Python list
that we write by hand.
"""

from __future__ import annotations

import argparse
import sys

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ai_travel_assistant.llm_basics import (
    ConfigError,
    build_llm,
    content_to_text,
    load_settings,
    safe_error_message,
)

# --- The system instruction ---------------------------------------------------
# The system message sets the assistant's role and rules. The model reads it
# before every user turn, so the rules apply to the whole conversation.

TRAVEL_ASSISTANT_INSTRUCTIONS = """\
You are a knowledgeable travel assistant for Sri Lanka.

How you work:
- Recommend Sri Lankan destinations and activities that match the traveller's
  stated interests (for example beaches, hiking, wildlife, culture, food).
- If essential information is missing, ask one to three short clarification
  questions before recommending. Essential information: trip length, interests,
  and travel month when the season matters (the monsoons affect each coast
  differently).
- Do not invent facts. If you are unsure about prices, opening hours, timetables
  or entry rules, say so and suggest checking an official source.
- Keep answers short: at most 5 bullet points or 120 words.
"""

# --- A reusable prompt --------------------------------------------------------
# A ChatPromptTemplate is a recipe for a list of messages. We write it once and
# fill in the blanks for each request:
#   ("system", ...)                  -> becomes a SystemMessage
#   MessagesPlaceholder("history")   -> earlier messages are inserted here
#   ("human", "{question}")          -> becomes a HumanMessage; {question} is filled in

TRAVEL_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", TRAVEL_ASSISTANT_INSTRUCTIONS),
        MessagesPlaceholder("history", optional=True),
        ("human", "{question}"),
    ]
)


def build_travel_messages(
    question: str, history: list[BaseMessage] | None = None
) -> list[BaseMessage]:
    """Fill TRAVEL_PROMPT and return the concrete list of messages."""
    prompt_value = TRAVEL_PROMPT.invoke(
        {"question": question, "history": history or []}
    )
    return prompt_value.to_messages()


# --- The examples -------------------------------------------------------------
# Each build_example_N() returns exactly what .invoke() will receive. Keeping the
# input separate from the API call lets us print it (--dry-run) and test it offline.


def build_example_1() -> str:
    # A plain string. LangChain turns it into [HumanMessage(content=...)] for us.
    return "Suggest one place to visit in Sri Lanka, in one sentence."


def build_example_2() -> list[BaseMessage]:
    # The same kind of request, but we choose the roles ourselves.
    return [
        SystemMessage("You are a concise travel guide. Answer in one sentence."),
        HumanMessage("Suggest one place to visit in Sri Lanka."),
    ]


def build_example_3() -> list[BaseMessage]:
    # A vague request. The instructions say to ask clarification questions.
    return build_travel_messages("I want to visit Sri Lanka. Where should I go?")


# Example 4's earlier turns. The AIMessage is a reply we write ourselves, as if
# the model had said it in a previous call. The model has no memory between calls:
# it knows about earlier turns only because we send them again.
EXAMPLE_4_HISTORY: list[BaseMessage] = [
    HumanMessage("I have 3 days, I love hiking and tea plantations."),
    AIMessage(
        "Ella suits you well: hike Little Adam's Peak and Ella Rock, and visit a "
        "tea estate. Which month are you travelling?"
    ),
]


def build_example_4() -> list[BaseMessage]:
    # "there" only makes sense if the model can see the earlier turns.
    return build_travel_messages(
        "In March. How do I get there from Colombo?", history=EXAMPLE_4_HISTORY
    )


EXAMPLE_5_QUESTION = "Where should I go in Sri Lanka?"


def build_example_5() -> dict[str, list[BaseMessage]]:
    # The same question twice: once on its own, once with our system instruction.
    return {
        "without system instruction": [HumanMessage(EXAMPLE_5_QUESTION)],
        "with system instruction": build_travel_messages(EXAMPLE_5_QUESTION),
    }


EXAMPLES = {
    "1": ("Simple string prompt", build_example_1),
    "2": ("SystemMessage + HumanMessage", build_example_2),
    "3": ("Sri Lanka travel assistant instruction", build_example_3),
    "4": ("Conversation history", build_example_4),
    "5": ("With vs without a system instruction", build_example_5),
}

# --- Running and printing -----------------------------------------------------


def show_input(prompt: str | list[BaseMessage]) -> None:
    """Print what we are about to send, so the message structure is visible."""
    if isinstance(prompt, str):
        print(f"[plain string] {prompt!r}")
        return
    for message in prompt:
        # message.type is "system", "human" or "ai".
        print(f"[{message.type}] {message.content}")


def ask(llm: BaseChatModel, prompt: str | list[BaseMessage]) -> AIMessage:
    """Send a string or a list of messages and return the model's AIMessage."""
    return llm.invoke(prompt)


def run_example(llm: BaseChatModel | None, key: str) -> None:
    """Print one example's input and, unless llm is None (dry run), its reply."""
    title, build = EXAMPLES[key]
    print(f"\n{'=' * 70}\nExample {key}: {title}\n{'=' * 70}")

    prompt = build()
    # Example 5 holds two prompts; the others hold one. Treat them the same way.
    variants = prompt if isinstance(prompt, dict) else {"": prompt}

    for label, variant in variants.items():
        if label:
            print(f"\n--- {label} ---")
        show_input(variant)
        if llm is None:
            continue
        reply = ask(llm, variant)
        print(f"\n[ai] {content_to_text(reply.content).strip()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LangChain message types demo.")
    parser.add_argument("example", nargs="?", default="all", choices=[*EXAMPLES, "all"])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the messages without calling Gemini",
    )
    args = parser.parse_args(argv)
    keys = list(EXAMPLES) if args.example == "all" else [args.example]

    if args.dry_run:
        for key in keys:
            run_example(None, key)
        return 0

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    llm = build_llm(settings)  # default temperature 1.0, as Google recommends
    print(f"Model: {settings.model}")
    for key in keys:
        try:
            run_example(llm, key)
        except Exception as exc:  # noqa: BLE001 - report any API failure safely
            print(
                f"\n[example {key}] Gemini call failed: "
                f"{safe_error_message(exc, settings.api_key)}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
