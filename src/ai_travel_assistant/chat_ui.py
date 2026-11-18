"""Stage 10: the terminal chat with rich (the default for --chat).

What this file demonstrates:
  1. Rendering an event stream. TurnView gets the Step / Token / TurnDone
     events from streaming.stream_turn and draws them; it never touches the
     graph. The plain chat (tool_calling.run_agent_chat, --plain) and the web
     page render the same turns their own way.
  2. Feedback while waiting. A spinner runs from the moment you press Enter;
     each step is printed as it happens; the reply is typed out as Markdown
     while Gemini writes it. A checked itinerary arrives in one piece and is
     shown in a panel, a clarification question in its own panel.
  3. The same commands as the plain chat (tool_calling.chat_command).

rich falls back to plain text when the output is not a terminal (a pipe or a
test), so the same code is tested by printing into a StringIO console.
"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status

from ai_travel_assistant.streaming import Event, Step, Token, TurnDone, stream_turn
from ai_travel_assistant.tool_calling import DEFAULT_USER, chat_command, open_chat

THINKING = "Thinking…"


class TurnView:
    """Draws one turn's events: spinner and steps, typed reply, final panels."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.status: Status | None = None
        self.live: Live | None = None
        self.typed = ""  # the reply typed so far in the current Live block

    # A turn starts with the spinner, so there is feedback before any event.
    def start(self) -> None:
        self._spin(THINKING)

    def _spin(self, text: str) -> None:
        if self.status is None:
            self.status = self.console.status(text, spinner="dots")
            self.status.start()
        else:
            self.status.update(text)

    def _stop_spinner(self) -> None:
        if self.status is not None:
            self.status.stop()
            self.status = None

    def _stop_typing(self) -> None:
        if self.live is not None:
            self.live.stop()
            self.live = None

    def handle(self, event: Event) -> None:
        if isinstance(event, Step):
            self._stop_typing()
            self.typed = ""
            self._stop_spinner()  # print above the spinner, then restart it
            self.console.print(f"  [dim]{event.text}[/dim]")
            self._spin(THINKING)
        elif isinstance(event, Token):
            self._stop_spinner()
            if self.live is None:
                self.live = Live(console=self.console, refresh_per_second=12)
                self.live.start()
            self.typed += event.text
            self.live.update(Markdown(self.typed))
        elif isinstance(event, TurnDone):
            self.finish(event)

    def finish(self, done: TurnDone) -> None:
        self._stop_spinner()
        self._stop_typing()
        if done.kind == "itinerary":
            self.console.print(
                Panel(
                    Markdown(done.answer), title="Your itinerary", border_style="green"
                )
            )
            self.console.print("[dim]Type /itinerary for the JSON.[/dim]")
        elif done.kind == "clarification":
            self.console.print(
                Panel(done.answer, title="A quick question", border_style="yellow")
            )
        elif done.kind == "off_topic":
            self.console.print(Panel(done.answer, border_style="magenta"))
        elif self.typed.strip() != done.answer.strip():
            # Not (fully) typed out, e.g. the model answered without streaming.
            self.console.print(Markdown(done.answer))
        self.typed = ""

    def abort(self) -> None:
        """Clean up after an error in the middle of a turn."""
        self._stop_spinner()
        self._stop_typing()
        self.typed = ""


def run_rich_chat(
    graph: CompiledStateGraph,
    console: Console | None = None,
    read: Callable[[str], str] | None = None,
    *,
    thread_id: str | None = None,
    user_id: str = DEFAULT_USER,
    describe_error: Callable[[Exception], str] = lambda exc: type(exc).__name__,
) -> None:
    """Chat in the terminal with live progress and streamed replies.

    Same threads, commands and errors as run_agent_chat; only the drawing
    differs. read defaults to console.input; tests pass a fake.
    """
    console = console or Console()
    read = read or (lambda prompt: console.input(prompt))
    thread_id, lines = open_chat(graph, thread_id)
    console.print(Panel.fit("Sri Lanka travel assistant", border_style="cyan"))
    for line in lines:
        console.print(f"[dim]{line}[/dim]")
    while True:
        text = read("[bold cyan]You[/bold cyan]: ").strip()
        if not text:
            continue
        command = chat_command(graph, thread_id, user_id, text)
        if command is not None:
            console.print(command.reply, markup=False, highlight=False)
            if command.quit:
                return
            thread_id = command.thread_id or thread_id
            continue
        view = TurnView(console)
        view.start()
        try:
            for event in stream_turn(graph, thread_id, text, user_id=user_id):
                view.handle(event)
        except Exception as exc:  # keep chatting after one failed turn
            view.abort()
            console.print(f"[red]Request failed:[/red] {describe_error(exc)}")
