"""
GeneWeaver: GPU-Accelerated CRISPR Alignment Engine
Terminal Dashboard (Day 8)

A clean, modular terminal interface built with Textual with a structured
layout including a Header, Sidebar (system, GPU status, and GPU memory), Main Content
area (processing info, progress bar at 0%, mismatch counts, alignment results,
benchmark results), and Footer. Provides dynamic update capabilities for scan
status, progress, alignment results DataTable, benchmark results, and GPU memory.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Label, ProgressBar

# Required fields for update_alignment_results
_REQUIRED_RESULT_FIELDS = frozenset(
    {"record_id", "start", "end", "matched_sequence", "mismatch_count"}
)

# Required fields for update_benchmark_results
_REQUIRED_BENCHMARK_FIELDS = frozenset(
    {
        "cpu_average_seconds",
        "gpu_average_seconds",
        "speedup",
        "cpu_match_count",
        "gpu_match_count",
        "results_match",
        "gpu_available",
        "message",
    }
)

# Demo label disclaimer
DEMO_LABEL_TEXT: str = "DEMO DATA — sample only; no real genome scan performed"

# Sample alignment results for demo purposes
DEMO_ALIGNMENT_RESULTS: list[dict] = [
    {
        "record_id": "DEMO-SITE-001",
        "start": 10500,
        "end": 10523,
        "matched_sequence": "GAGTCCGAGCAGAAGAAGAATGG",
        "mismatch_count": 0,
        "mismatch_positions": [],
    },
    {
        "record_id": "DEMO-SITE-002",
        "start": 42100,
        "end": 42123,
        "matched_sequence": "GAGTCCGATCAGAAGAAGAATGG",
        "mismatch_count": 1,
        "mismatch_positions": [8],
    },
    {
        "record_id": "DEMO-SITE-003",
        "start": 89450,
        "end": 89473,
        "matched_sequence": "GAGTCCGATCAGAAGAAGAAGGG",
        "mismatch_count": 2,
        "mismatch_positions": [8, 20],
    },
    {
        "record_id": "DEMO-SITE-004",
        "start": 120300,
        "end": 120323,
        "matched_sequence": "GCGTCCGATCAGAAGAAGAAGGG",
        "mismatch_count": 3,
        "mismatch_positions": [1, 8, 20],
    },
]


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


class GpuStatusPanel(Container):
    """Sidebar panel displaying GPU accelerator status and device information."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = "GPU Status"
        self._gpu_status: str = "Not checked"
        self._gpu_device: str = "Not detected"
        self.gpu_status_widget = Label(
            f"[b]GPU Status:[/b] {self._gpu_status}",
            id="gpu-status",
            classes="stat-line",
        )
        self.gpu_device_widget = Label(
            f"[b]GPU Device:[/b] {self._gpu_device}",
            id="gpu-device",
            classes="stat-line",
        )
        self.status_widget = self.gpu_status_widget
        self.device_widget = self.gpu_device_widget

    def compose(self) -> ComposeResult:
        yield self.gpu_status_widget
        yield self.gpu_device_widget

    def update_status(self, status: str, device_name: str | None = None) -> None:
        """Update GPU status and device name label displays."""
        self._gpu_status = status
        self.gpu_status_widget.update(f"[b]GPU Status:[/b] {status}")
        if device_name is not None:
            self._gpu_device = device_name
            self.gpu_device_widget.update(f"[b]GPU Device:[/b] {device_name}")

    def update_gpu_status(self, status: str, device_name: str | None = None) -> None:
        """Alias for update_status."""
        self.update_status(status, device_name)

    def query_one(self, selector, *args, **kwargs):
        """Query for a child widget with fallback for unmounted tests."""
        try:
            return super().query_one(selector, *args, **kwargs)
        except Exception:
            if selector == "#gpu-status" and hasattr(self, "gpu_status_widget"):
                return self.gpu_status_widget
            if selector == "#gpu-device" and hasattr(self, "gpu_device_widget"):
                return self.gpu_device_widget
            raise


def format_bytes(num_bytes: int) -> str:
    """Format byte counts into human-readable strings with appropriate units (B, KB, MB, GB, TB).

    Uses binary prefixes (1024 bytes = 1 KB).
    """
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["KB", "MB", "GB", "TB", "PB"]
    val = float(num_bytes)
    for unit in units:
        val /= 1024.0
        if val < 1024.0 or unit == units[-1]:
            return f"{val:.2f} {unit}"
    return f"{val:.2f} PB"


class GpuMemoryPanel(Container):
    """Sidebar panel displaying GPU memory utilization and capacity."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = "GPU Memory"
        self._memory_status: str = "Not checked"
        self._total_memory: str = "Not available"
        self._used_memory: str = "Not available"
        self._free_memory: str = "Not available"

        self.memory_status_widget = Label(
            f"[b]GPU Memory Status:[/b] {self._memory_status}",
            id="gpu-memory-status",
            classes="stat-line",
        )
        self.total_memory_widget = Label(
            f"[b]Total GPU Memory:[/b] {self._total_memory}",
            id="gpu-memory-total",
            classes="stat-line",
        )
        self.used_memory_widget = Label(
            f"[b]Used GPU Memory:[/b] {self._used_memory}",
            id="gpu-memory-used",
            classes="stat-line",
        )
        self.free_memory_widget = Label(
            f"[b]Free GPU Memory:[/b] {self._free_memory}",
            id="gpu-memory-free",
            classes="stat-line",
        )

    def compose(self) -> ComposeResult:
        yield self.memory_status_widget
        yield self.total_memory_widget
        yield self.used_memory_widget
        yield self.free_memory_widget

    def update_memory(
        self,
        status: str,
        total: str = "Not available",
        used: str = "Not available",
        free: str = "Not available",
    ) -> None:
        """Update GPU memory labels."""
        self._memory_status = status
        self._total_memory = total
        self._used_memory = used
        self._free_memory = free

        self.memory_status_widget.update(f"[b]GPU Memory Status:[/b] {status}")
        self.total_memory_widget.update(f"[b]Total GPU Memory:[/b] {total}")
        self.used_memory_widget.update(f"[b]Used GPU Memory:[/b] {used}")
        self.free_memory_widget.update(f"[b]Free GPU Memory:[/b] {free}")

    def query_one(self, selector, *args, **kwargs):
        """Query for a child widget with fallback for unmounted tests."""
        try:
            return super().query_one(selector, *args, **kwargs)
        except Exception:
            mapping = {
                "#gpu-memory-status": self.memory_status_widget,
                "#gpu-memory-total": self.total_memory_widget,
                "#gpu-memory-used": self.used_memory_widget,
                "#gpu-memory-free": self.free_memory_widget,
            }
            if selector in mapping:
                return mapping[selector]
            raise


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

    # Column definitions: (label, width) — widths tuned for a normal 80-col terminal
    _COLUMNS: list[tuple[str, int]] = [
        ("Record ID", 14),
        ("Start Position", 16),
        ("End Position", 14),
        ("Matched Sequence", 20),
        ("Mismatch Count", 15),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = "Alignment Results"
        self.demo_label = Label(
            "",
            id="demo-data-label",
            classes="demo-label demo-data-label",
        )
        self.demo_label.display = False

    def compose(self) -> ComposeResult:
        yield Label("[b]Mismatch counts:[/b] None recorded", id="status-mismatch-counts", classes="stat-line")
        yield Label("[b]Alignment results:[/b] No results available", id="status-alignment-results", classes="stat-line")
        yield self.demo_label
        yield Label("Detailed alignment matches will appear here once scanning begins.", id="results-placeholder", classes="results-empty")
        yield DataTable(id="alignment-results-table", show_cursor=False)

    def on_mount(self) -> None:
        """Add columns to the DataTable once the app context is live."""
        table: DataTable = self.query_one("#alignment-results-table", DataTable)
        if len(table.columns) == 0:
            for label, width in self._COLUMNS:
                table.add_column(label, width=width)

    def set_demo_label_visible(self, visible: bool = True) -> None:
        """Toggle visibility and text of the demo data label."""
        if visible:
            self.demo_label.update(DEMO_LABEL_TEXT)
            self.demo_label.display = True
        else:
            self.demo_label.update("")
            self.demo_label.display = False

    def query_one(self, selector, *args, **kwargs):
        """Query for a child widget with alias mapping and fallback."""
        if selector in ("#demo-label", "#demo-data-label", "#demo-banner", "#status-demo"):
            selector = "#demo-data-label"
        try:
            return super().query_one(selector, *args, **kwargs)
        except Exception:
            if selector == "#demo-data-label" and hasattr(self, "demo_label"):
                return self.demo_label
            raise


class BenchmarkResultsPanel(Container):
    """Main content panel displaying CPU vs GPU benchmark timing results."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = "Benchmark Results"
        self._cpu_time_widget = Label(
            "[b]CPU Average Time:[/b] Not run",
            id="cpu-benchmark-time",
            classes="stat-line",
        )
        self._gpu_time_widget = Label(
            "[b]GPU Average Time:[/b] Not run",
            id="gpu-benchmark-time",
            classes="stat-line",
        )
        self._speedup_widget = Label(
            "[b]GPU Speedup:[/b] Not available",
            id="gpu-benchmark-speedup",
            classes="stat-line",
        )
        self._status_widget = Label(
            "[b]Benchmark Status:[/b] Not run",
            id="benchmark-status",
            classes="stat-line",
        )

    def compose(self) -> ComposeResult:
        yield self._cpu_time_widget
        yield self._gpu_time_widget
        yield self._speedup_widget
        yield self._status_widget

    def query_one(self, selector, *args, **kwargs):
        """Query for a child widget with fallback for unmounted tests."""
        try:
            return super().query_one(selector, *args, **kwargs)
        except Exception:
            mapping = {
                "#cpu-benchmark-time": self._cpu_time_widget,
                "#gpu-benchmark-time": self._gpu_time_widget,
                "#gpu-benchmark-speedup": self._speedup_widget,
                "#benchmark-status": self._status_widget,
            }
            if selector in mapping:
                return mapping[selector]
            raise


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

    SUPPORTED_GPU_STATUSES = {
        "Not checked",
        "Checking",
        "Available",
        "Unavailable",
        "Error",
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
        height: auto;
        min-height: 10;
        border: round $primary;
        background: $panel;
        padding: 1 1;
        margin-bottom: 1;
    }

    GpuStatusPanel {
        width: 100%;
        height: auto;
        min-height: 5;
        border: round $success;
        background: $panel;
        padding: 1 1;
        margin-bottom: 1;
    }

    GpuMemoryPanel {
        width: 100%;
        height: auto;
        min-height: 8;
        border: round $accent;
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

    BenchmarkResultsPanel {
        width: 100%;
        height: auto;
        min-height: 8;
        border: round $warning;
        background: $panel;
        padding: 1 1;
        margin-top: 1;
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

    #demo-data-label,
    #demo-label,
    .demo-data-label {
        color: $warning;
        text-style: bold;
        margin-top: 1;
        margin-bottom: 0;
    }

    #alignment-results-table {
        width: 100%;
        height: 1fr;
        margin-top: 1;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scan_status: str = "Not started"
        self.scan_progress: int = 0
        self.is_demo_mode: bool = False
        self.gpu_memory_status: str = "Not checked"
        self.gpu_memory_total: int | None = None
        self.gpu_memory_used: int | None = None
        self.gpu_memory_free: int | None = None
        self.project_status_panel = ProjectStatusPanel(id="project-status-panel")
        self.gpu_status_panel = GpuStatusPanel(id="gpu-status-panel")
        self.gpu_memory_panel = GpuMemoryPanel(id="gpu-memory-panel")
        self.processing_info_panel = ProcessingInfoPanel(id="processing-info-panel")
        self.results_panel = ResultsPanel(id="results-panel")
        self.benchmark_results_panel = BenchmarkResultsPanel(id="benchmark-results-panel")

    @property
    def demo_label(self) -> Label:
        """Return the demo data warning label widget."""
        return self.results_panel.demo_label

    @property
    def demo_mode(self) -> bool:
        """Return True if demo results have been loaded."""
        return self.is_demo_mode

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
                self.gpu_status_panel,
                self.gpu_memory_panel,
                id="sidebar",
            ),
            Container(
                self.processing_info_panel,
                self.results_panel,
                self.benchmark_results_panel,
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
        elif selector in ("#demo-label", "#demo-data-label", "#demo-banner", "#status-demo"):
            selector = "#demo-data-label"
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
            if selector == "#demo-data-label" and hasattr(self, "results_panel"):
                return self.results_panel.demo_label
            if selector == "#gpu-status-panel" and hasattr(self, "gpu_status_panel"):
                return self.gpu_status_panel
            if selector == "#gpu-status" and hasattr(self, "gpu_status_panel"):
                return self.gpu_status_panel.gpu_status_widget
            if selector == "#gpu-device" and hasattr(self, "gpu_status_panel"):
                return self.gpu_status_panel.gpu_device_widget
            if selector == "#gpu-memory-panel" and hasattr(self, "gpu_memory_panel"):
                return self.gpu_memory_panel
            if selector == "#gpu-memory-status" and hasattr(self, "gpu_memory_panel"):
                return self.gpu_memory_panel.memory_status_widget
            if selector == "#gpu-memory-total" and hasattr(self, "gpu_memory_panel"):
                return self.gpu_memory_panel.total_memory_widget
            if selector == "#gpu-memory-used" and hasattr(self, "gpu_memory_panel"):
                return self.gpu_memory_panel.used_memory_widget
            if selector == "#gpu-memory-free" and hasattr(self, "gpu_memory_panel"):
                return self.gpu_memory_panel.free_memory_widget
            if selector == "#benchmark-results-panel" and hasattr(self, "benchmark_results_panel"):
                return self.benchmark_results_panel
            if selector == "#cpu-benchmark-time" and hasattr(self, "benchmark_results_panel"):
                return self.benchmark_results_panel._cpu_time_widget
            if selector == "#gpu-benchmark-time" and hasattr(self, "benchmark_results_panel"):
                return self.benchmark_results_panel._gpu_time_widget
            if selector == "#gpu-benchmark-speedup" and hasattr(self, "benchmark_results_panel"):
                return self.benchmark_results_panel._speedup_widget
            if selector == "#benchmark-status" and hasattr(self, "benchmark_results_panel"):
                return self.benchmark_results_panel._status_widget
            raise

    def query(self, selector=None, *args, **kwargs):
        """Query for child widgets with legacy alias mapping."""
        if selector == "#status-scan":
            selector = "#scan-status"
        elif selector == "#alignment-progress-bar":
            selector = "#alignment-progress"
        elif selector in ("#demo-label", "#demo-data-label", "#demo-banner", "#status-demo"):
            selector = "#demo-data-label"
        return super().query(selector, *args, **kwargs)

    def update_gpu_status(self, status: str, device_name: str | None = None) -> None:
        """Update the GPU status and device name displayed in the GPU Status panel.

        Accepts only supported statuses:
        - Not checked
        - Checking
        - Available   (requires a non-empty device_name)
        - Unavailable
        - Error

        Args:
            status:      One of the supported GPU status strings.
            device_name: The detected GPU device name. Required when status is
                         'Available'. Ignored (and cleared) when status is
                         'Unavailable' or 'Error'. Optional otherwise.

        Raises:
            ValueError: If status is unsupported.
            ValueError: If status is 'Available' but device_name is None or empty.
        """
        if not isinstance(status, str) or status not in self.SUPPORTED_GPU_STATUSES:
            raise ValueError(
                f"Unsupported GPU status: {status!r}. "
                f"Supported statuses: {', '.join(sorted(self.SUPPORTED_GPU_STATUSES))}"
            )

        if status == "Available":
            if not device_name:
                raise ValueError(
                    "device_name must be provided and non-empty when status is 'Available'."
                )

        # Determine the device string to display
        if status in ("Unavailable", "Error"):
            display_device = "Not detected"
        elif device_name is not None:
            display_device = device_name
        else:
            display_device = None  # keep existing device label unchanged

        self.gpu_status_panel.update_status(status, display_device)

        try:
            status_widget = super().query_one("#gpu-status", Label)
            status_widget.update(f"[b]GPU Status:[/b] {status}")
        except Exception:
            pass

        if display_device is not None:
            try:
                device_widget = super().query_one("#gpu-device", Label)
                device_widget.update(f"[b]GPU Device:[/b] {display_device}")
            except Exception:
                pass

    def update_gpu_memory(
        self,
        free_bytes: int | None,
        total_bytes: int | None,
    ) -> None:
        """Update GPU memory utilization and capacity information.

        Args:
            free_bytes:  Free GPU memory in bytes, or None if unavailable.
            total_bytes: Total GPU memory in bytes, or None if unavailable.

        Raises:
            ValueError: If only one of free_bytes or total_bytes is None.
            ValueError: If byte values are negative.
            ValueError: If total_bytes is zero.
            ValueError: If free_bytes is greater than total_bytes.
            ValueError: If byte values are invalid types (e.g. bool, str, float).
        """
        if free_bytes is None and total_bytes is None:
            status = "Unavailable"
            total_str = "Not available"
            used_str = "Not available"
            free_str = "Not available"
            self.gpu_memory_status = status
            self.gpu_memory_total = None
            self.gpu_memory_used = None
            self.gpu_memory_free = None
        elif free_bytes is None or total_bytes is None:
            raise ValueError(
                "Both free_bytes and total_bytes must be provided, or both must be None."
            )
        else:
            if isinstance(free_bytes, bool) or isinstance(total_bytes, bool):
                raise ValueError("Memory byte values cannot be boolean.")
            if not isinstance(free_bytes, int) or not isinstance(total_bytes, int):
                raise ValueError(
                    f"Memory byte values must be integers, got {type(free_bytes).__name__} and {type(total_bytes).__name__}."
                )
            if free_bytes < 0 or total_bytes < 0:
                raise ValueError("Memory values cannot be negative.")
            if total_bytes == 0:
                raise ValueError("Total GPU memory cannot be zero.")
            if free_bytes > total_bytes:
                raise ValueError(
                    f"Free GPU memory ({free_bytes} bytes) cannot be greater than total GPU memory ({total_bytes} bytes)."
                )

            used_bytes = total_bytes - free_bytes
            status = "Available"
            total_str = format_bytes(total_bytes)
            used_str = format_bytes(used_bytes)
            free_str = format_bytes(free_bytes)

            self.gpu_memory_status = status
            self.gpu_memory_total = total_bytes
            self.gpu_memory_used = used_bytes
            self.gpu_memory_free = free_bytes

        self.gpu_memory_panel.update_memory(
            status=status,
            total=total_str,
            used=used_str,
            free=free_str,
        )

        try:
            self.query_one("#gpu-memory-status", Label).update(
                f"[b]GPU Memory Status:[/b] {status}"
            )
        except Exception:
            pass
        try:
            self.query_one("#gpu-memory-total", Label).update(
                f"[b]Total GPU Memory:[/b] {total_str}"
            )
        except Exception:
            pass
        try:
            self.query_one("#gpu-memory-used", Label).update(
                f"[b]Used GPU Memory:[/b] {used_str}"
            )
        except Exception:
            pass
        try:
            self.query_one("#gpu-memory-free", Label).update(
                f"[b]Free GPU Memory:[/b] {free_str}"
            )
        except Exception:
            pass

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

    def update_alignment_results(self, results: list[dict]) -> None:
        """Populate the alignment results table from a list of flattened result dicts.

        Each dict must contain these keys:
            - record_id         : identifier for the genomic record
            - start             : genomic start position (int)
            - end               : genomic end position (int)
            - matched_sequence  : the matched DNA sequence string
            - mismatch_count    : number of mismatches (int)
            - mismatch_positions: (accepted but not displayed as a column today)

        Args:
            results: list of result dicts.  An empty list clears the table.

        Raises:
            ValueError: if any dict is missing a required field.
        """
        # Locate the DataTable widget — may live inside ResultsPanel
        try:
            table: DataTable = super().query_one("#alignment-results-table", DataTable)
        except Exception:
            table = self.results_panel.query_one("#alignment-results-table", DataTable)

        # Ensure table columns exist if called before app mount
        if len(table.columns) == 0 and hasattr(self, "results_panel"):
            for label, width in self.results_panel._COLUMNS:
                table.add_column(label, width=width)

        # Validate all rows up-front before touching the table
        for idx, row in enumerate(results):
            missing = _REQUIRED_RESULT_FIELDS - row.keys()
            if missing:
                raise ValueError(
                    f"Result at index {idx} is missing required fields: "
                    f"{', '.join(sorted(missing))}"
                )

        # Clear previous rows, then insert new ones
        table.clear()
        for row in results:
            table.add_row(
                str(row["record_id"]),
                str(row["start"]),
                str(row["end"]),
                str(row["matched_sequence"]),
                str(row["mismatch_count"]),
            )

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

    def update_benchmark_results(self, results: dict) -> None:
        """Update the benchmark results panel from a benchmark data dictionary.

        Args:
            results: dict containing benchmark data with the following keys:
                - cpu_average_seconds  : CPU average alignment time (float)
                - gpu_average_seconds  : GPU average alignment time (float or None)
                - speedup              : measured GPU speedup factor (float or None)
                - cpu_match_count      : number of CPU alignment matches (int)
                - gpu_match_count      : number of GPU alignment matches (int or None)
                - results_match        : whether CPU/GPU results agree (bool or None)
                - gpu_available        : whether GPU execution was performed (bool)
                - message              : benchmark summary message (str)

        Raises:
            ValueError: if results is not a dict or is missing required keys.
        """
        if not isinstance(results, dict):
            raise ValueError(
                f"update_benchmark_results expects a dict, got {type(results).__name__!r}"
            )
        missing = _REQUIRED_BENCHMARK_FIELDS - results.keys()
        if missing:
            raise ValueError(
                f"Benchmark results dict is missing required fields: "
                f"{', '.join(sorted(missing))}"
            )

        # --- CPU average time ---
        cpu_secs = results["cpu_average_seconds"]
        if cpu_secs is not None:
            cpu_ms = float(cpu_secs) * 1000.0
            cpu_display = f"{cpu_ms:.3f} ms"
        else:
            cpu_display = "Not run"

        # --- GPU average time ---
        gpu_available: bool = bool(results["gpu_available"])
        gpu_secs = results["gpu_average_seconds"]
        if not gpu_available:
            gpu_display = "GPU Unavailable"
        elif gpu_secs is not None:
            gpu_ms = float(gpu_secs) * 1000.0
            gpu_display = f"{gpu_ms:.3f} ms"
        else:
            gpu_display = "Not run"

        # --- GPU speedup ---
        speedup = results["speedup"]
        if not gpu_available:
            speedup_display = "GPU Unavailable"
        elif speedup is not None:
            speedup_display = f"{float(speedup):.2f}x"
        else:
            speedup_display = "Not available"

        # --- Benchmark status / message ---
        message: str = str(results["message"]) if results["message"] is not None else ""
        results_match = results["results_match"]
        if results_match is True:
            match_note = " | Results match: Yes ✓"
        elif results_match is False:
            match_note = " | Results match: No ✗"
        else:
            match_note = ""
        status_display = f"{message}{match_note}" if message else f"Not run{match_note}"

        # Update widgets
        panel = self.benchmark_results_panel
        panel._cpu_time_widget.update(f"[b]CPU Average Time:[/b] {cpu_display}")
        panel._gpu_time_widget.update(f"[b]GPU Average Time:[/b] {gpu_display}")
        panel._speedup_widget.update(f"[b]GPU Speedup:[/b] {speedup_display}")
        panel._status_widget.update(f"[b]Benchmark Status:[/b] {status_display}")

        # Also attempt live DOM updates (no-op when not mounted)
        try:
            self.query_one("#cpu-benchmark-time", Label).update(
                f"[b]CPU Average Time:[/b] {cpu_display}"
            )
        except Exception:
            pass
        try:
            self.query_one("#gpu-benchmark-time", Label).update(
                f"[b]GPU Average Time:[/b] {gpu_display}"
            )
        except Exception:
            pass
        try:
            self.query_one("#gpu-benchmark-speedup", Label).update(
                f"[b]GPU Speedup:[/b] {speedup_display}"
            )
        except Exception:
            pass
        try:
            self.query_one("#benchmark-status", Label).update(
                f"[b]Benchmark Status:[/b] {status_display}"
            )
        except Exception:
            pass

    def load_demo_results(self, results: list[dict] | None = None) -> None:
        """Load sample CRISPR alignment results for demonstration.

        Updates scan status to 'Completed', scan progress to 100%,
        populates the existing alignment results table using illustrative
        sample data, and displays a clear visible label:
        'DEMO DATA — sample only; no real genome scan performed'

        Args:
            results: Optional custom list of sample result dicts.
                     Defaults to DEMO_ALIGNMENT_RESULTS.
        """
        self.is_demo_mode = True
        demo_data = [
            dict(r) for r in (results if results is not None else DEMO_ALIGNMENT_RESULTS)
        ]

        # Use existing update methods wherever possible
        self.update_scan_status("Completed")
        self.update_scan_progress(100)
        self.update_alignment_results(demo_data)

        # Display the demo data disclaimer label
        self.results_panel.set_demo_label_visible(True)
        try:
            lbl = super().query_one("#demo-data-label", Label)
            lbl.update(DEMO_LABEL_TEXT)
            lbl.display = True
        except Exception:
            pass


def main() -> None:
    """Entry point to launch the GeneWeaver dashboard."""
    app = GeneWeaverApp()
    app.run()


if __name__ == "__main__":
    main()
