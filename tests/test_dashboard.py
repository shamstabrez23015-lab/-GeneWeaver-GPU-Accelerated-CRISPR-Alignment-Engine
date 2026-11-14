"""
Tests for the GeneWeaver terminal dashboard (Day 1, Day 2, Day 3, and Day 4).
"""

from __future__ import annotations

import asyncio as _asyncio

import pytest
from textual.containers import Container
from textual.widgets import DataTable, Footer, Header, Label, ProgressBar

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
# Preserved Day 2 Tests: Layout Structure, Widgets, and Identifiers
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
    assert pb.id in ("alignment-progress", "alignment-progress-bar")
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

    # Scan status (supports stable 'scan-status' or legacy 'status-scan')
    scan_id = "scan-status" if "scan-status" in project_widgets else "status-scan"
    assert scan_id in project_widgets
    assert "Not started" in str(project_widgets[scan_id].render())

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
        "status-cpu-engine",
        "status-gpu-engine",
        "status-gpu-memory",
        "label-progress",
        "status-mismatch-counts",
        "status-alignment-results",
        "results-placeholder",
        "alignment-results-table",
    }
    scan_id = "scan-status" if "scan-status" in all_ids else "status-scan"
    expected_ids.add(scan_id)
    progress_id = "alignment-progress" if "alignment-progress" in all_ids else "alignment-progress-bar"
    expected_ids.add(progress_id)

    missing_ids = expected_ids - set(all_ids)
    assert not missing_ids, f"Missing expected widget IDs: {missing_ids}"


# =============================================================================
# Day 3 Tests: Dynamic Scan Status and Progress Updates
# =============================================================================


def test_initial_scan_status_is_not_started() -> None:
    """Verify that initial scan status is 'Not started'."""
    app = GeneWeaverApp()
    assert app.scan_status == "Not started"
    status_widget = app.query_one("#scan-status")
    assert "Not started" in str(status_widget.render())

    # Verify panel component initial display
    panel = ProjectStatusPanel()
    widgets = {w.id: w for w in panel.compose() if getattr(w, "id", None)}
    assert "scan-status" in widgets
    assert "Not started" in str(widgets["scan-status"].render())


def test_initial_progress_is_zero() -> None:
    """Verify that initial progress is 0%."""
    app = GeneWeaverApp()
    assert app.scan_progress == 0
    pb = app.query_one("#alignment-progress", ProgressBar)
    assert pb.progress == 0
    assert pb.total == 100
    label = app.query_one("#label-progress")
    assert "0%" in str(label.render())

    # Verify panel component initial display
    proc_panel = ProcessingInfoPanel()
    progress_bars = [w for w in proc_panel.compose() if isinstance(w, ProgressBar)]
    assert len(progress_bars) == 1
    assert progress_bars[0].id == "alignment-progress"
    assert progress_bars[0].progress == 0


def test_update_scan_status_ready() -> None:
    """Verify updating scan status to 'Ready'."""
    app = GeneWeaverApp()
    app.update_scan_status("Ready")
    assert app.scan_status == "Ready"
    status_widget = app.query_one("#scan-status")
    assert "Ready" in str(status_widget.render())


def test_update_scan_status_scanning() -> None:
    """Verify updating scan status to 'Scanning'."""
    app = GeneWeaverApp()
    app.update_scan_status("Scanning")
    assert app.scan_status == "Scanning"
    status_widget = app.query_one("#scan-status")
    assert "Scanning" in str(status_widget.render())


def test_update_scan_status_completed() -> None:
    """Verify updating scan status to 'Completed'."""
    app = GeneWeaverApp()
    app.update_scan_status("Completed")
    assert app.scan_status == "Completed"
    status_widget = app.query_one("#scan-status")
    assert "Completed" in str(status_widget.render())


def test_update_scan_status_failed() -> None:
    """Verify updating scan status to 'Failed'."""
    app = GeneWeaverApp()
    app.update_scan_status("Failed")
    assert app.scan_status == "Failed"
    status_widget = app.query_one("#scan-status")
    assert "Failed" in str(status_widget.render())


@pytest.mark.parametrize(
    "invalid_status",
    [
        "In progress",
        "pending",
        "Complete",
        "ERROR",
        "finished",
        "Unknown",
        "",
        "ready",
        123,
        None,
    ],
)
def test_reject_unsupported_scan_status(invalid_status) -> None:
    """Verify that unsupported scan statuses raise ValueError."""
    app = GeneWeaverApp()
    with pytest.raises(ValueError):
        app.update_scan_status(invalid_status)  # type: ignore[arg-type]


@pytest.mark.parametrize("progress_val", [25, 50, 100])
def test_update_scan_progress(progress_val: int) -> None:
    """Verify updating progress to 25, 50, and 100."""
    app = GeneWeaverApp()
    app.update_scan_progress(progress_val)
    assert app.scan_progress == progress_val
    pb = app.query_one("#alignment-progress", ProgressBar)
    assert pb.progress == progress_val
    label = app.query_one("#label-progress")
    assert f"{progress_val}%" in str(label.render())


@pytest.mark.parametrize("below_zero", [-1, -5, -50, -100])
def test_reject_progress_below_zero(below_zero: int) -> None:
    """Verify that progress values below 0 raise ValueError."""
    app = GeneWeaverApp()
    with pytest.raises(ValueError):
        app.update_scan_progress(below_zero)


@pytest.mark.parametrize("above_hundred", [101, 105, 150, 200, 1000])
def test_reject_progress_above_hundred(above_hundred: int) -> None:
    """Verify that progress values above 100 raise ValueError."""
    app = GeneWeaverApp()
    with pytest.raises(ValueError):
        app.update_scan_progress(above_hundred)


@pytest.mark.parametrize("non_integer", [True, False, 25.5, "50", None, [50]])
def test_reject_non_integer_progress(non_integer) -> None:
    """Verify that non-integer progress values raise ValueError."""
    app = GeneWeaverApp()
    with pytest.raises(ValueError):
        app.update_scan_progress(non_integer)  # type: ignore[arg-type]


# =============================================================================
# Day 4 Tests: Alignment Results DataTable
# =============================================================================


def _run(coro):
    """Run an async coroutine synchronously (no external async plugin needed)."""
    return _asyncio.run(coro)


# ---------------------------------------------------------------------------
# Sample test data — manually crafted for testing only.
# These do NOT represent real genomic scan output.
# ---------------------------------------------------------------------------
_SAMPLE_RESULT_1 = {
    "record_id": "REC-001",
    "start": 100,
    "end": 123,
    "matched_sequence": "ACGTACGTACGT",
    "mismatch_count": 0,
    "mismatch_positions": [],
}
_SAMPLE_RESULT_2 = {
    "record_id": "REC-002",
    "start": 500,
    "end": 523,
    "matched_sequence": "TTGCAATTGCAA",
    "mismatch_count": 2,
    "mismatch_positions": [3, 9],
}
_SAMPLE_RESULT_3 = {
    "record_id": "REC-003",
    "start": 1050,
    "end": 1073,
    "matched_sequence": "GCGCGCGCGCGC",
    "mismatch_count": 1,
    "mismatch_positions": [6],
}


# ------------------------------------------------------------------
# Structure tests — no live app needed; inspect compose() output
# ------------------------------------------------------------------


def test_results_panel_contains_data_table() -> None:
    """Verify ResultsPanel composes a DataTable with the correct stable ID."""
    panel = ResultsPanel()
    composed = list(panel.compose())
    tables = [w for w in composed if isinstance(w, DataTable)]
    assert len(tables) == 1, "ResultsPanel must contain exactly one DataTable"
    assert tables[0].id == "alignment-results-table"


def test_results_panel_column_definitions() -> None:
    """Verify ResultsPanel declares exactly five columns with expected labels."""
    expected_labels = {
        "Record ID",
        "Start Position",
        "End Position",
        "Matched Sequence",
        "Mismatch Count",
    }
    actual_labels = {label for label, _ in ResultsPanel._COLUMNS}
    assert actual_labels == expected_labels, (
        f"Column labels mismatch.\nExpected: {expected_labels}\nGot: {actual_labels}"
    )
    assert len(ResultsPanel._COLUMNS) == 5, "ResultsPanel must define exactly 5 columns"


def test_update_alignment_results_method_exists() -> None:
    """Verify GeneWeaverApp exposes update_alignment_results as a callable."""
    app = GeneWeaverApp()
    assert hasattr(app, "update_alignment_results")
    assert callable(app.update_alignment_results)


# ------------------------------------------------------------------
# Runtime tests — driven via asyncio.run() + Textual's run_test()
# ------------------------------------------------------------------


def test_table_exists_in_running_app() -> None:
    """Verify the DataTable widget is queryable once the app is running."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            table = app.query_one("#alignment-results-table", DataTable)
            assert table is not None
    _run(_inner())


def test_table_has_five_columns() -> None:
    """Verify the alignment results table has exactly five columns."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            table = app.query_one("#alignment-results-table", DataTable)
            assert len(table.columns) == 5
    _run(_inner())


def test_table_column_labels() -> None:
    """Verify all five column labels are present and correct."""
    expected = {
        "Record ID",
        "Start Position",
        "End Position",
        "Matched Sequence",
        "Mismatch Count",
    }

    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            table = app.query_one("#alignment-results-table", DataTable)
            actual = {str(col.label) for col in table.columns.values()}
            assert actual == expected
    _run(_inner())


def test_table_initially_empty() -> None:
    """Verify the alignment results table starts with zero rows."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            table = app.query_one("#alignment-results-table", DataTable)
            assert table.row_count == 0
    _run(_inner())


def test_add_single_alignment_result() -> None:
    """Verify adding one result produces exactly one table row."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            app.update_alignment_results([_SAMPLE_RESULT_1])
            table = app.query_one("#alignment-results-table", DataTable)
            assert table.row_count == 1
    _run(_inner())


def test_add_multiple_alignment_results() -> None:
    """Verify adding three results produces exactly three table rows."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            app.update_alignment_results(
                [_SAMPLE_RESULT_1, _SAMPLE_RESULT_2, _SAMPLE_RESULT_3]
            )
            table = app.query_one("#alignment-results-table", DataTable)
            assert table.row_count == 3
    _run(_inner())


def test_correct_record_ids_and_positions() -> None:
    """Verify record_id and genomic positions are stored correctly in the table."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            app.update_alignment_results([_SAMPLE_RESULT_1, _SAMPLE_RESULT_2])
            table = app.query_one("#alignment-results-table", DataTable)

            row_keys = list(table.rows.keys())
            assert len(row_keys) == 2

            # Row 0 — REC-001
            row0 = [
                str(table.get_cell(row_keys[0], col_key))
                for col_key in table.columns
            ]
            assert "REC-001" in row0
            assert "100" in row0   # start
            assert "123" in row0   # end

            # Row 1 — REC-002
            row1 = [
                str(table.get_cell(row_keys[1], col_key))
                for col_key in table.columns
            ]
            assert "REC-002" in row1
            assert "500" in row1
            assert "523" in row1
    _run(_inner())


def test_correct_mismatch_counts() -> None:
    """Verify mismatch_count values are stored correctly in the table."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            app.update_alignment_results([_SAMPLE_RESULT_1, _SAMPLE_RESULT_2])
            table = app.query_one("#alignment-results-table", DataTable)

            row_keys = list(table.rows.keys())
            row0_vals = [
                str(table.get_cell(row_keys[0], col_key))
                for col_key in table.columns
            ]
            row1_vals = [
                str(table.get_cell(row_keys[1], col_key))
                for col_key in table.columns
            ]
            assert "0" in row0_vals    # REC-001: 0 mismatches
            assert "2" in row1_vals    # REC-002: 2 mismatches
    _run(_inner())


def test_updating_results_clears_previous_rows() -> None:
    """Verify calling update_alignment_results replaces, not appends, existing rows."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            # First load: 3 rows
            app.update_alignment_results(
                [_SAMPLE_RESULT_1, _SAMPLE_RESULT_2, _SAMPLE_RESULT_3]
            )
            table = app.query_one("#alignment-results-table", DataTable)
            assert table.row_count == 3

            # Second load: 1 row — must replace, not append
            app.update_alignment_results([_SAMPLE_RESULT_2])
            assert table.row_count == 1
    _run(_inner())


def test_empty_results_list_leaves_table_empty() -> None:
    """Verify passing an empty list clears the table."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            # Pre-populate, then clear with []
            app.update_alignment_results([_SAMPLE_RESULT_1, _SAMPLE_RESULT_2])
            table = app.query_one("#alignment-results-table", DataTable)
            assert table.row_count == 2

            app.update_alignment_results([])
            assert table.row_count == 0
    _run(_inner())


def test_missing_required_field_raises_value_error() -> None:
    """Verify a result dict missing a required field raises ValueError clearly."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            bad_result = {
                "record_id": "REC-BAD",
                "start": 10,
                # "end" is intentionally missing
                "matched_sequence": "ACGT",
                "mismatch_count": 0,
                "mismatch_positions": [],
            }
            with pytest.raises(ValueError, match="end"):
                app.update_alignment_results([bad_result])
    _run(_inner())


def test_missing_multiple_fields_raises_value_error() -> None:
    """Verify a result dict missing multiple required fields raises ValueError."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            very_bad = {"record_id": "REC-EMPTY"}
            with pytest.raises(ValueError):
                app.update_alignment_results([very_bad])
    _run(_inner())


def test_mismatch_positions_not_a_column() -> None:
    """Verify mismatch_positions is accepted as input but is NOT a table column."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            table = app.query_one("#alignment-results-table", DataTable)
            col_labels = {str(col.label) for col in table.columns.values()}
            assert "Mismatch Positions" not in col_labels
            assert "mismatch_positions" not in col_labels

            # Field must be accepted without error
            app.update_alignment_results([_SAMPLE_RESULT_2])
            assert table.row_count == 1
    _run(_inner())


def test_scan_status_and_progress_still_work_after_day4_changes() -> None:
    """Verify Day 3 dynamic update methods still function alongside Day 4 additions."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            app.update_scan_status("Scanning")
            assert app.scan_status == "Scanning"

            app.update_scan_progress(42)
            assert app.scan_progress == 42

            # Confirm no conflicts with the results table
            app.update_alignment_results([_SAMPLE_RESULT_1])
            table = app.query_one("#alignment-results-table", DataTable)
            assert table.row_count == 1
    _run(_inner())
