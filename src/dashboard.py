"""
GeneWeaver: GPU-Accelerated CRISPR Alignment Engine
Terminal Dashboard (Day 1)

A clean, modular terminal interface built with Textual for monitoring
CRISPR genomic alignment status, processing metrics, and results.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.widgets import Footer, Header, Label, ProgressBar


class ProjectStatusPanel(Container):
    """Panel displaying overall project and engine readiness."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = "Project Status"

    def compose(self) -> ComposeResult:
        yield Label("[b]Project status:[/b] Ready", id="status-project", classes="stat-line")
        yield Label("[b]Genome file:[/b] No FASTA file loaded", id="status-genome", classes="stat-line")
        yield Label("[b]GPU engine status:[/b] Not connected yet", id="status-gpu-engine", classes="stat-line")
        yield Label("[b]GPU memory:[/b] Waiting for Week 2 integration", id="status-gpu-memory", classes="stat-line")


class ProcessingInfoPanel(Container):
    """Panel displaying CRISPR scanning and CPU/GPU processing status."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = "Processing Information"

    def compose(self) -> ComposeResult:
        yield Label("[b]Scan status:[/b] Not started", id="status-scan", classes="stat-line")
        yield Label("[b]CPU alignment status:[/b] Not started", id="status-cpu-align", classes="stat-line")
        yield Label("[b]Alignment progress:[/b]", id="label-progress", classes="stat-line")
        yield ProgressBar(total=100, show_eta=False, id="alignment-progress-bar")


class ResultsPanel(Container):
    """Panel displaying alignment and match results."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = "Alignment Results"

    def compose(self) -> ComposeResult:
        yield Label("No results available yet.", id="results-placeholder", classes="results-empty")


class GeneWeaverApp(App):
    """Main Textual application for GeneWeaver."""

    TITLE = "GeneWeaver"
    SUB_TITLE = "GPU-Accelerated CRISPR Alignment Engine"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("d", "toggle_dark", "Toggle Dark Mode", show=True),
    ]

    CSS = """
    Screen {
        background: $surface;
        color: $text;
    }

    #main-container {
        width: 100%;
        height: 100%;
        padding: 1 2;
        layout: vertical;
    }

    #panels-top-row {
        height: auto;
        width: 100%;
        margin-bottom: 1;
    }

    ProjectStatusPanel {
        width: 1fr;
        height: auto;
        border: round $primary;
        background: $panel;
        padding: 1 2;
        margin-right: 1;
    }

    ProcessingInfoPanel {
        width: 1fr;
        height: auto;
        border: round $accent;
        background: $panel;
        padding: 1 2;
        margin-left: 1;
    }

    ResultsPanel {
        width: 100%;
        height: 1fr;
        min-height: 8;
        border: round $secondary;
        background: $panel;
        padding: 1 2;
    }

    .stat-line {
        margin-bottom: 1;
    }

    #alignment-progress-bar {
        width: 100%;
        margin-top: 0;
        margin-bottom: 1;
    }

    .results-empty {
        color: $text-muted;
        padding: 1 0;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the user interface elements."""
        yield Header(show_clock=True)
        yield Container(
            Horizontal(
                ProjectStatusPanel(id="project-status-panel"),
                ProcessingInfoPanel(id="processing-info-panel"),
                id="panels-top-row",
            ),
            ResultsPanel(id="results-panel"),
            id="main-container",
        )
        yield Footer()


def main() -> None:
    """Entry point to launch the GeneWeaver dashboard."""
    app = GeneWeaverApp()
    app.run()


if __name__ == "__main__":
    main()
