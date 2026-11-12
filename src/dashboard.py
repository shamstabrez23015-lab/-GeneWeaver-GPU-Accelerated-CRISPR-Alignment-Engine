"""
GeneWeaver: GPU-Accelerated CRISPR Alignment Engine
Terminal Dashboard (Day 2)

A clean, modular terminal interface built with Textual with a structured
layout including a Header, Sidebar (system and engine status), Main Content
area (processing info, progress bar at 0%, mismatch counts, alignment results),
and Footer.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, Label, ProgressBar


class ProjectStatusPanel(Container):
    """Sidebar panel displaying project information and engine readiness."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = "Project Status"

    def compose(self) -> ComposeResult:
        yield Label("[b]Project status:[/b] Ready", id="status-project", classes="stat-line")
        yield Label("[b]Genome file:[/b] Not loaded", id="status-genome", classes="stat-line")
        yield Label("[b]Scan status:[/b] Not started", id="status-scan", classes="stat-line")
        yield Label("[b]CPU engine:[/b] Not started", id="status-cpu-engine", classes="stat-line")
        yield Label("[b]GPU engine:[/b] Not connected", id="status-gpu-engine", classes="stat-line")
        yield Label("[b]GPU memory:[/b] Not available yet", id="status-gpu-memory", classes="stat-line")


class ProcessingInfoPanel(Container):
    """Main content panel displaying genome processing and alignment progress."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = "Processing Information"

    def compose(self) -> ComposeResult:
        yield Label("[b]Genome batch:[/b] None", id="status-genome-batch", classes="stat-line")
        yield Label("[b]Sequences scanned:[/b] 0", id="status-sequences-scanned", classes="stat-line")
        yield Label("[b]Targets identified:[/b] 0", id="status-targets-identified", classes="stat-line")
        yield Label("[b]Alignment progress:[/b] 0%", id="label-progress", classes="stat-line")
        yield ProgressBar(total=100, show_eta=False, id="alignment-progress-bar")


class ResultsPanel(Container):
    """Main content panel displaying mismatch summary and alignment results."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = "Alignment Results"

    def compose(self) -> ComposeResult:
        yield Label("[b]Mismatch counts:[/b] None recorded", id="status-mismatch-counts", classes="stat-line")
        yield Label("[b]Alignment results:[/b] No results available", id="status-alignment-results", classes="stat-line")
        yield Label("Detailed alignment matches will appear here once scanning begins.", id="results-placeholder", classes="results-empty")


class GeneWeaverApp(App):
    """Main Textual application for GeneWeaver."""

    TITLE = "GeneWeaver"
    SUB_TITLE = "GPU-Accelerated CRISPR Alignment Engine"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("d", "toggle_dark", "Toggle Dark Mode", show=True),
        Binding("h", "help", "Help", show=True),
    ]

    CSS = """
    Screen {
        background: $surface;
        color: $text;
    }

    #main-container {
        width: 100%;
        height: 1fr;
        layout: horizontal;
        padding: 0 1;
    }

    #sidebar {
        width: 36;
        min-width: 30;
        max-width: 42;
        height: 100%;
        margin-right: 1;
        layout: vertical;
        overflow-y: auto;
    }

    #main-content {
        width: 1fr;
        height: 100%;
        layout: vertical;
        overflow-y: auto;
    }

    ProjectStatusPanel {
        width: 100%;
        height: 100%;
        min-height: 10;
        border: round $primary;
        background: $panel;
        padding: 1 1;
    }

    ProcessingInfoPanel {
        width: 100%;
        height: auto;
        border: round $accent;
        background: $panel;
        padding: 1 1;
        margin-bottom: 1;
    }

    ResultsPanel {
        width: 100%;
        height: 1fr;
        min-height: 8;
        border: round $secondary;
        background: $panel;
        padding: 1 1;
    }

    .stat-line {
        margin-bottom: 1;
    }

    #alignment-progress-bar {
        width: 100%;
        margin-top: 0;
        margin-bottom: 0;
    }

    .results-empty {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def action_help(self) -> None:
        """Display a brief help notification."""
        self.notify(
            "GeneWeaver v0.1 | Keys: Q = Quit, D = Toggle Theme, H = Help",
            title="Help & Controls",
        )

    def compose(self) -> ComposeResult:
        """Compose the user interface elements."""
        yield Header(show_clock=True)
        yield Container(
            Container(
                ProjectStatusPanel(id="project-status-panel"),
                id="sidebar",
            ),
            Container(
                ProcessingInfoPanel(id="processing-info-panel"),
                ResultsPanel(id="results-panel"),
                id="main-content",
            ),
            id="main-container",
        )
        yield Footer()


def main() -> None:
    """Entry point to launch the GeneWeaver dashboard."""
    app = GeneWeaverApp()
    app.run()


if __name__ == "__main__":
    main()
