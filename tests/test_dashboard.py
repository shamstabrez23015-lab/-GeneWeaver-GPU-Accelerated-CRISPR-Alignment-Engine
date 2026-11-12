"""
Tests for the GeneWeaver terminal dashboard (Day 1 & Day 2).
"""

from __future__ import annotations

import pytest
from textual.containers import Container
from textual.widgets import Footer, Header, Label, ProgressBar

from src.dashboard import (
    GeneWeaverApp,
    ProcessingInfoPanel,
    ProjectStatusPanel,
    ResultsPanel,
)


# =============================================================================
# Preserved Day 1 Tests
# =============================================================================


def test_import_gene_weaver_app() -> None:
    """Verify that GeneWeaverApp imports successfully."""
    from src.dashboard import GeneWeaverApp as ImportedApp

    assert ImportedApp is not None
    assert issubclass(ImportedApp, object)


def test_instantiate_gene_weaver_app() -> None:
    """Verify that GeneWeaverApp can be instantiated with correct metadata."""
    app = GeneWeaverApp()
    assert app is not None
    assert isinstance(app, GeneWeaverApp)
    assert app.TITLE == "GeneWeaver"
    assert app.SUB_TITLE == "GPU-Accelerated CRISPR Alignment Engine"


def test_instantiate_dashboard_panels() -> None:
    """Verify that each dashboard panel component can be instantiated."""
    project_panel = ProjectStatusPanel()
    assert project_panel is not None
    assert project_panel.border_title == "Project Status"

    proc_panel = ProcessingInfoPanel()
    assert proc_panel is not None
    assert proc_panel.border_title == "Processing Information"

    results_panel = ResultsPanel()
    assert results_panel is not None
    assert results_panel.border_title == "Alignment Results"


def test_dashboard_composition() -> None:
    """Verify that compose yields the required top-level widgets."""
    app = GeneWeaverApp()
    composed_widgets = list(app.compose())

    # Should contain Header, Main Container, and Footer
    widget_types = [type(w) for w in composed_widgets]
    assert Header in widget_types
    assert Footer in widget_types


# =============================================================================
# Day 2 Tests: Layout Structure, Widgets, and Identifiers
# =============================================================================


def test_application_initialization() -> None:
    """Verify application metadata and keyboard bindings on initialization."""
    app = GeneWeaverApp()
    assert app.TITLE == "GeneWeaver"
    assert app.SUB_TITLE == "GPU-Accelerated CRISPR Alignment Engine"

    # Verify standard keyboard bindings
    binding_keys = {b.key: b.action for b in app.BINDINGS}
    assert "q" in binding_keys
    assert binding_keys["q"] == "quit"
    assert "d" in binding_keys
    assert binding_keys["d"] == "toggle_dark"
    assert "h" in binding_keys
    assert binding_keys["h"] == "help"


def test_header_and_footer_availability() -> None:
    """Verify Header and Footer widgets are configured and returned by compose."""
    app = GeneWeaverApp()
    composed = list(app.compose())

    headers = [w for w in composed if isinstance(w, Header)]
    footers = [w for w in composed if isinstance(w, Footer)]

    assert len(headers) == 1, "Header widget must be yielded once"
    assert len(footers) == 1, "Footer widget must be yielded once"
    assert getattr(headers[0], "_show_clock", True) is True, "Header should enable clock display"


def test_sidebar_and_main_content_composition() -> None:
    """Verify separate container structure for sidebar and main content."""
    app = GeneWeaverApp()
    composed = list(app.compose())

    main_container = next(
        (w for w in composed if isinstance(w, Container) and w.id == "main-container"),
        None,
    )
    assert main_container is not None, "Top-level main-container must exist"

    # Inspect pending child containers of main-container
    children = main_container._pending_children
    sidebar = next((c for c in children if getattr(c, "id", None) == "sidebar"), None)
    main_content = next((c for c in children if getattr(c, "id", None) == "main-content"), None)

    assert sidebar is not None, "Sidebar container with id='sidebar' must exist"
    assert main_content is not None, "Main content container with id='main-content' must exist"

    # Check that sidebar contains ProjectStatusPanel
    sidebar_panels = [c for c in sidebar._pending_children if isinstance(c, ProjectStatusPanel)]
    assert len(sidebar_panels) == 1
    assert sidebar_panels[0].id == "project-status-panel"

    # Check that main content contains ProcessingInfoPanel and ResultsPanel
    main_panels = [c for c in main_content._pending_children]
    proc_panels = [c for c in main_panels if isinstance(c, ProcessingInfoPanel)]
    results_panels = [c for c in main_panels if isinstance(c, ResultsPanel)]
    assert len(proc_panels) == 1
    assert proc_panels[0].id == "processing-info-panel"
    assert len(results_panels) == 1
    assert results_panels[0].id == "results-panel"


def test_presence_of_progress_bar() -> None:
    """Verify that the progress bar widget exists and is initially at 0%."""
    proc_panel = ProcessingInfoPanel()
    composed = list(proc_panel.compose())

    progress_bars = [w for w in composed if isinstance(w, ProgressBar)]
    assert len(progress_bars) == 1, "ProcessingInfoPanel must contain a ProgressBar"

    pb = progress_bars[0]
    assert pb.id == "alignment-progress-bar"
    assert pb.total == 100
    assert pb.progress == 0, "Progress bar should initially start at 0%"


def test_presence_of_genome_and_scan_status_widgets() -> None:
    """Verify that genome, scan, and engine status indicators display correct initial states."""
    project_panel = ProjectStatusPanel()
    project_widgets = {w.id: w for w in project_panel.compose() if getattr(w, "id", None)}

    # Project information
    assert "status-project" in project_widgets
    assert "Ready" in str(project_widgets["status-project"].render())

    # Genome file status
    assert "status-genome" in project_widgets
    assert "Not loaded" in str(project_widgets["status-genome"].render())

    # Scan status
    assert "status-scan" in project_widgets
    assert "Not started" in str(project_widgets["status-scan"].render())

    # CPU engine status
    assert "status-cpu-engine" in project_widgets
    assert "Not started" in str(project_widgets["status-cpu-engine"].render())

    # GPU engine status
    assert "status-gpu-engine" in project_widgets
    assert "Not connected" in str(project_widgets["status-gpu-engine"].render())

    # GPU memory status
    assert "status-gpu-memory" in project_widgets
    assert "Not available yet" in str(project_widgets["status-gpu-memory"].render())

    # Alignment results and mismatch display
    results_panel = ResultsPanel()
    results_widgets = {w.id: w for w in results_panel.compose() if getattr(w, "id", None)}

    assert "status-alignment-results" in results_widgets
    assert "No results available" in str(results_widgets["status-alignment-results"].render())

    assert "status-mismatch-counts" in results_widgets
    assert "None recorded" in str(results_widgets["status-mismatch-counts"].render())


def test_dashboard_widgets_unique_ids() -> None:
    """Verify all dashboard widgets have unique, non-colliding IDs."""
    app = GeneWeaverApp()
    top_level = list(app.compose())

    all_ids: list[str] = []

    def collect_ids(widget) -> None:
        if getattr(widget, "id", None):
            all_ids.append(widget.id)
        # Traverse pending children
        for child in getattr(widget, "_pending_children", []):
            collect_ids(child)
        # If widget has compose method and is a custom panel, collect composed widgets
        if isinstance(widget, (ProjectStatusPanel, ProcessingInfoPanel, ResultsPanel)):
            for composed_child in widget.compose():
                collect_ids(composed_child)

    for w in top_level:
        collect_ids(w)

    assert len(all_ids) > 0, "Widgets with IDs must be present"
    assert len(all_ids) == len(set(all_ids)), f"Duplicate widget IDs detected: {all_ids}"

    # Verify presence of essential IDs
    expected_ids = {
        "main-container",
        "sidebar",
        "main-content",
        "project-status-panel",
        "processing-info-panel",
        "results-panel",
        "status-project",
        "status-genome",
        "status-scan",
        "status-cpu-engine",
        "status-gpu-engine",
        "status-gpu-memory",
        "label-progress",
        "alignment-progress-bar",
        "status-mismatch-counts",
        "status-alignment-results",
        "results-placeholder",
    }
    missing_ids = expected_ids - set(all_ids)
    assert not missing_ids, f"Missing expected widget IDs: {missing_ids}"
