# ADR-0002: An understanding step before the agent, and a compose step after it

- **Status:** Accepted, implemented (Stages 6 and 8).
  - Understanding and clarification: live-verified 2026-11-23 on Groq (deliverable test and 6/6 routing prompts).
  - Compose: verified offline; in the live eval so far, 5 of 6 plan scenarios ended in a structured, schema-valid plan (run `2026-11-23T1156`, incomplete).
- **Date:** Stages 6 and 8; recorded 2026-11-23.
- **Related:** [architecture.md](../architecture.md), [ADR-0004](0004-llm-provider-switch.md).

## Context

A plain ReAct loop (`llm` ⇄ `tools`) can answer anything, but:

- Whether it asks a clarifying question or guesses depends on the model's mood, so clarification cannot be tested reliably.
- Its final answer is free text: nothing guarantees the day count, and nothing stops it naming places or prices that no tool returned.

The free Gemini tier allows about 20 calls a day, so every extra call per message matters.

## Decision

### Understand: one structured call, then pure-Python routing

- `understand_request` (`request_understanding.py`) makes one `with_structured_output(RequestUnderstanding)` call at temperature 0 for each traveller message. It returns the intent (`plan`, `modify`, `question`, `off_topic`) and the trip details stated in that message.
- Code, not the LLM, merges those details with what is already known (a new destination starts a new trip) and lists what is still missing (`missing_critical`).
- `route_after_understanding` is a pure function. For a `plan`, it asks about every missing `PLAN_FIELDS` value (destination, days, travellers, then citizenship / party / children only when needed, origin, hotels, and budget only when hotels are wanted), one at a time, up to `len(PLAN_FIELDS)` questions in a row. After that the agent answers and states its assumptions. Questions and off-topic messages are not clarified.
- `ask_clarification` writes the question from fixed templates (no LLM). `wait_for_reply` only calls `interrupt()`, so re-running it on resume is safe.
- Quick-reply buttons (web page) and bare numbers ("3" after "How many days?") are read without an LLM call.
- If the understanding call fails, the message goes to the agent as before; a daily-quota 429 ends the turn instead.
- `--no-clarify` removes the step to save quota. Without an understander the graph is the Stage 5 one.

### Compose: a final-answer tool, checked by code

The blueprint planned `compose_answer` as a separate LLM call. **As built, it has no LLM:**

- With `structured_itinerary=True`, the agent also gets `submit_itinerary`, whose arguments are the `ItineraryDraft` schema. It fills the structure in the call that would have written prose, so there is no extra call.
- The tools node never runs `submit_itinerary`. `route_after_agent` sends it to `compose_answer`, which:
  1. validates the draft; on errors, sends the field paths and messages (never values) back to the agent **once**;
  2. on a second failure, salvages every day, activity and item that is valid on its own, with a warning;
  3. grounds it: each place, travel figure, price, forecast and fact must come from that destination's tool result, or it is removed, set to null or flagged;
  4. renders Markdown with a fixed template and stores the checked plan in `state["itinerary"]`.

## Alternatives considered

| Option | Why not |
|---|---|
| Plain `agent ⇄ tools` loop | Cheaper by one call per message, but clarification and output shape cannot be tested reliably |
| Understanding inside the agent prompt ("ask if anything is missing") | Same model call, but no structured intent to route on or to test |
| LLM writes the clarification question | Varied wording, one more call; fixed templates are free and testable |
| `compose_answer` as a second LLM call (blueprint) | One more call per plan on a 20-a-day quota |
| Grounding by an LLM judge | Costs a call and can itself be wrong; code checks are deterministic |

## Consequences

**Benefits**

- Routing and clarification are unit-tested without an LLM; the eval measures them (clarify-when-needed and false-clarify rates).
- Plans have a guaranteed shape (days, activities, costs, sources), which the web page draws as day cards and exports as PDF.
- Grounding is enforced by code: unconfirmed data is removed or listed as "Not available".

**Costs and limitations**

- One extra LLM call per message for understanding (fewer with quick replies; none with `--no-clarify`).
- The understander's schema must suit every provider. Seen live on 2026-11-23: Groq rejected `null` for the list fields `interests` and `party`, so the understanding failed quietly; fixed by allowing `null` and turning it into `[]`.
- Extraction is only as good as the model: "two Indian adults" left `children` empty and triggered a needless question (eval `itin-galle-from-kandy`); the field description was clarified on 2026-11-23, not yet re-checked live.
- The model may still answer an itinerary in prose instead of calling `submit_itinerary`; the eval tracks this rate (`structured`).
- Grounding applies to itineraries only. Prose answers to questions are not checked.
- Strict grounding can remove useful content that is true but not in a tool result (in the live eval, generic activities such as "Evening stroll" counted as unconfirmed places).

## Future considerations (not implemented)

- Ground or flag clock times and opening hours in prose answers too.
- Let the understander ask about several missing fields in one question.
- Scorer refinement: separate named places from generic activities in the grounding metric.
