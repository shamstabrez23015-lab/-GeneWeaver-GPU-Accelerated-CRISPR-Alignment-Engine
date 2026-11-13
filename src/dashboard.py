"""
GeneWeaver: GPU-Accelerated CRISPR Alignment Engine
Terminal Dashboard (Day 3)

A clean, modular terminal interface built with Textual with a structured
layout including a Header, Sidebar (system and engine status), Main Content
area (processing info, progress bar at 0%, mismatch counts, alignment results),
and Footer. Provides dynamic update capabilities for scan status and progress.
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
        self._current_status: str = "Not started"
        self.scan_status_widget = Label(
            f"[b]Scan status:[/b] {self._current_status}",
            id="scan-status",
            classes="stat-line",
        )

    def compose(self) -> ComposeResult:
        yield Label("[b]Project status:[/b] Ready", id="status-project", classes="stat-line")
        yield Label("[b]Genome file:[/b] Not loaded", id="status-genome", classes="stat-line")
        yield self.scan_status_widget
        yield Label("[b]CPU engine:[/b] Not started", id="status-cpu-engine", classes="stat-line")
        yield Label("[b]GPU engine:[/b] Not connected", id="status-gpu-engine", classes="stat-line")
        yield Label("[b]GPU memory:[/b] Not available yet", id="status-gpu-memory", classes="stat-line")

    def update_status(self, status: str) -> None:
        """Update the scan status label display."""
        self._current_status = status
        self.scan_status_widget.update(f"[b]Scan status:[/b] {status}")


class ProcessingInfoPanel(Container):
    """Main content panel displaying genome processing and alignment progress."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = "Processing Information"
        self._current_progress: int = 0
        self.progress_label = Label(
            f"[b]Alignment progress:[/b] {self._current_progress}%",
            id="label-progress",
            classes="stat-line",
        )
        self.progress_bar = ProgressBar(
            total=100,
            show_eta=False,
            id="alignment-progress",
        )

    def compose(self) -> ComposeResult:
        yield Label("[b]Genome batch:[/b] None", id="status-genome-batch", classes="stat-line")
        yield Label("[b]Sequences scanned:[/b] 0", id="status-sequences-scanned", classes="stat-line")
        yield Label("[b]Targets identified:[/b] 0", id="status-targets-identified", classes="stat-line")
        yield self.progress_label
        yield self.progress_bar

    def update_progress(self, progress: int) -> None:
        """Update the progress bar and percentage label display."""
        self._current_progress = progress
        self.progress_bar.update(progress=progress)
        self.progress_label.update(f"[b]Alignment progress:[/b] {progress}%")


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

    SUPPORTED_SCAN_STATUSES = {
        "Not started",
        "Ready",
        "Scanning",
        "Completed",
        "Failed",
    }

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

    #alignment-progress,
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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scan_status: str = "Not started"
        self.scan_progress: int = 0
        self.project_status_panel = ProjectStatusPanel(id="project-status-panel")
        self.processing_info_panel = ProcessingInfoPanel(id="processing-info-panel")
        self.results_panel = ResultsPanel(id="results-panel")

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
                self.project_status_panel,
                id="sidebar",
            ),
            Container(
                self.processing_info_panel,
                self.results_panel,
                id="main-content",
            ),
            id="main-container",
        )
        yield Footer()

    def query_one(self, selector, *args, **kwargs):
        """Query for a child widget with legacy alias mapping and fallback."""
        if selector == "#status-scan":
            selector = "#scan-status"
        elif selector == "#alignment-progress-bar":
            selector = "#alignment-progress"
        try:
            return super().query_one(selector, *args, **kwargs)
        except Exception:
            if selector == "#scan-status" and hasattr(self, "project_status_panel"):
                return self.project_status_panel.scan_status_widget
            if selector == "#alignment-progress" and hasattr(self, "processing_info_panel"):
                return self.processing_info_panel.progress_bar
            if selector == "#label-progress" and hasattr(self, "processing_info_panel"):
                return self.processing_info_panel.progress_label
            if selector == "#project-status-panel" and hasattr(self, "project_status_panel"):
                return self.project_status_panel
            if selector == "#processing-info-panel" and hasattr(self, "processing_info_panel"):
                return self.processing_info_panel
            if selector == "#results-panel" and hasattr(self, "results_panel"):
                return self.results_panel
            raise

    def query(self, selector=None, *args, **kwargs):
        """Query for child widgets with legacy alias mapping."""
        if selector == "#status-scan":
            selector = "#scan-status"
        elif selector == "#alignment-progress-bar":
            selector = "#alignment-progress"
        return super().query(selector, *args, **kwargs)

    def update_scan_status(self, status: str) -> None:
        """Update the scan status component.

        Accepts only supported statuses:
        - Not started
        - Ready
        - Scanning
        - Completed
        - Failed

        Raises:
            ValueError: If status is unsupported or not a string.
        """
        if not isinstance(status, str) or status not in self.SUPPORTED_SCAN_STATUSES:
            raise ValueError(
                f"Unsupported scan status: {status!r}. "
                f"Supported statuses: {', '.join(sorted(self.SUPPORTED_SCAN_STATUSES))}"
            )

        self.scan_status = status
        self.project_status_panel.update_status(status)

        try:
            widget = super().query_one("#scan-status", Label)
            widget.update(f"[b]Scan status:[/b] {status}")
        except Exception:
            pass

    def update_scan_progress(self, progress: int) -> None:
        """Update the scan progress component.

        Accepts integer progress from 0 to 100.

        Raises:
            ValueError: If progress is not an integer or outside [0, 100].
        """
        if isinstance(progress, bool) or not isinstance(progress, int):
            raise ValueError(
                f"Progress must be an integer, got {type(progress).__name__}: {progress!r}"
            )
        if not (0 <= progress <= 100):
            raise ValueError(
                f"Progress must be between 0 and 100, got: {progress}"
            )

        self.scan_progress = progress
        self.processing_info_panel.update_progress(progress)

        try:
            pb = super().query_one("#alignment-progress", ProgressBar)
            pb.update(progress=progress)
        except Exception:
            pass

        try:
            lbl = super().query_one("#label-progress", Label)
            lbl.update(f"[b]Alignment progress:[/b] {progress}%")
        except Exception:
            pass


def main() -> None:
    """Entry point to launch the GeneWeaver dashboard."""
    app = GeneWeaverApp()
    app.run()


if __name__ == "__main__":
    main()
