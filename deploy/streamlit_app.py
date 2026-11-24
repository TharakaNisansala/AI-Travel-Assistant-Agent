"""Streamlit Community Cloud entry point for the public demo.

Set "Main file path" to deploy/streamlit_app.py when you deploy
(docs/DEPLOYMENT.md). Locally, the usual command still works:
    uv run --extra ui streamlit run src/ai_travel_assistant/web_app.py

Why this file and this folder exist:
  - Community Cloud uses the FIRST dependency file it finds, looking in the
    entry point's folder before the repository root, and uv.lock comes
    before requirements.txt. Keeping the entry point here lets
    deploy/requirements.txt (pinned from uv.lock, web extra, no dev tools)
    win over the root uv.lock, so the install is the same every time.
  - The requirements install only third-party packages, not this project,
    so the src/ folder is put on the import path here instead.
  - Community Cloud runs the app from the repository root, so
    .streamlit/config.toml (the theme) stays at the root.

Public mode (provider fallback, limits, private sessions, the sample trip)
is switched on by TRAVEL_PUBLIC = "1" in the app's Secrets, not here.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_travel_assistant import web_app  # noqa: E402 - needs SRC on the path

web_app.run()
