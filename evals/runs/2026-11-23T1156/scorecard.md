# Scorecard: run 2026-11-23T1156

**Model:** openai/gpt-oss-120b  
**Scenarios scored:** 15 (15 still missing)  
**Gemini calls in the run:** 15

| Metric | Result | Counted | Target | Met |
|---|---|---|---|---|
| scenario pass rate | 93% | 14/15 scenarios | >= 85% | yes |
| extraction | 97% | 32/33 fields | >= 90% | yes |
| route | 100% | 7/7 scenarios | - | tracked |
| clarify_when_needed | 100% | 2/2 turns | >= 90% | yes |
| no_false_clarify | 100% | 3/3 turns | >= 95% | yes |
| required_tools | n/a | 0/0 scenarios | >= 90% | tracked |
| crash_free | 100% | 15/15 scenarios | >= 100% | yes |
| structured | n/a | 0/0 scenarios | - | tracked |
| schema_valid | n/a | 0/0 scenarios | >= 100% | tracked |
| days | n/a | 0/0 scenarios | >= 100% | tracked |
| destination | n/a | 0/0 scenarios | - | tracked |
| grounding | n/a | 0/0 places | >= 95% | tracked |
| failures_reported | n/a | 0/0 scenarios | >= 100% | tracked |
| plan_replaced | n/a | 0/0 scenarios | - | tracked |
| median latency (agent turns) | n/a | | <= 30s | tracked |
| LLM calls per plan | n/a | | - | tracked |

## Verdict

Incomplete: 15 scenarios have no record yet, so this is not a sign-off.
All targets met.

## Failures

- `und-nuwara-eliya-midrange` extraction: 4/5; wrong: travel_dates: expected 'next weekend', got None

## Missing

`itin-ella-3d`, `itin-kandy-2d`, `itin-galle-from-kandy`, `itin-nuwara-eliya-3d`, `itin-sigiriya-2d`, `itin-mirissa-beach`, `itin-anuradhapura-history`, `itin-trincomalee-budget`, `fault-weather-ella`, `fault-attractions-kandy`, `fault-travel-galle`, `fault-guide-busy-nuwara-eliya`, `multi-ella-easier-day`, `multi-clarify-then-plan`, `multi-kandy-longer`
