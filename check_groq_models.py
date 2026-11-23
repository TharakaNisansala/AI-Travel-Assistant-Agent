"""List the Groq models this API key can use, to choose GROQ_MODEL in .env.

A development utility: it calls Groq's model list (no chat request, so it uses
no tokens) and prints each model's ID and context window, sorted by ID. Pick a
model that supports tool use; the list does not say which ones do, so check
the Groq console docs. The context window is the most a single request can
hold, but on the free tier the tokens-per-minute limit can be smaller: one
request over it fails with 413 (seen with openai/gpt-oss-120b at 8,000 TPM).

Run from the project root (needs GROQ_API_KEY in .env or the shell):
    uv run python check_groq_models.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from groq import Groq


def main() -> int:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print(
            "GROQ_API_KEY is not set. Add it to .env (see .env.example) or set it "
            "in the shell.",
            file=sys.stderr,
        )
        return 1

    models = Groq(api_key=api_key).models.list().data
    print(f"\n{len(models)} Groq models available:\n")
    print(f"{'Model ID':<50} {'Context window':>15}")
    print("-" * 66)
    for model in sorted(models, key=lambda m: m.id):
        # Not in every SDK version's Model type, so read it defensively.
        window = getattr(model, "context_window", None)
        shown = f"{window:,}" if isinstance(window, int) else "n/a"
        print(f"{model.id:<50} {shown:>15}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
