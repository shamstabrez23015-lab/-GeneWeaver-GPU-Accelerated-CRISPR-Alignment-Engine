"""
Tests for the GeneWeaver terminal dashboard (Day 1, Day 2, Day 3, Day 4, Day 5, Day 6, Day 7, and Day 8).
"""

from __future__ import annotations

import asyncio as _asyncio

import pytest
from textual.containers import Container
from textual.widgets import DataTable, Footer, Header, Label, ProgressBar

from src.dashboard import (
    DEMO_ALIGNMENT_RESULTS,
    DEMO_LABEL_TEXT,
    BenchmarkResultsPanel,
    GeneWeaverApp,
    GpuMemoryPanel,
    GpuStatusPanel,
    ProcessingInfoPanel,
    ProjectStatusPanel,
    ResultsPanel,
    format_bytes,
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


# =============================================================================
# Day 5 Tests: Dashboard Sample Alignment Results Integration
# =============================================================================


def test_load_demo_results_method_exists() -> None:
    """Verify GeneWeaverApp provides load_demo_results as a callable."""
    app = GeneWeaverApp()
    assert hasattr(app, "load_demo_results")
    assert callable(app.load_demo_results)


def test_dashboard_initially_has_no_alignment_results() -> None:
    """Verify dashboard initially starts in an empty state with no results."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            table = app.query_one("#alignment-results-table", DataTable)
            assert table.row_count == 0
            assert app.scan_status == "Not started"
            assert app.scan_progress == 0
            assert app.is_demo_mode is False

            demo_label = app.query_one("#demo-data-label", Label)
            assert demo_label.display is False
            assert DEMO_LABEL_TEXT not in str(demo_label.render())
    _run(_inner())


def test_calling_load_demo_results_updates_status() -> None:
    """Verify calling load_demo_results updates scan status to Completed."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            app.load_demo_results()
            assert app.scan_status == "Completed"
            status_widget = app.query_one("#scan-status", Label)
            assert "Completed" in str(status_widget.render())
    _run(_inner())


def test_calling_load_demo_results_progress_becomes_100() -> None:
    """Verify calling load_demo_results sets progress to 100%."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            app.load_demo_results()
            assert app.scan_progress == 100
            pb = app.query_one("#alignment-progress", ProgressBar)
            assert pb.progress == 100
            lbl = app.query_one("#label-progress", Label)
            assert "100%" in str(lbl.render())
    _run(_inner())


def test_sample_results_appear_in_table() -> None:
    """Verify sample alignment records appear properly formatted in DataTable."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            app.load_demo_results()
            table = app.query_one("#alignment-results-table", DataTable)
            assert table.row_count == len(DEMO_ALIGNMENT_RESULTS)
            assert table.row_count > 0

            row_keys = list(table.rows.keys())
            for idx, expected_record in enumerate(DEMO_ALIGNMENT_RESULTS):
                row_vals = [
                    str(table.get_cell(row_keys[idx], col_key))
                    for col_key in table.columns
                ]
                assert str(expected_record["record_id"]) in row_vals
                assert str(expected_record["start"]) in row_vals
                assert str(expected_record["end"]) in row_vals
                assert str(expected_record["matched_sequence"]) in row_vals
                assert str(expected_record["mismatch_count"]) in row_vals
    _run(_inner())


def test_demo_label_is_visible() -> None:
    """Verify that the demo disclaimer label is clearly visible after loading demo results."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            demo_label = app.query_one("#demo-data-label", Label)
            assert demo_label.display is False

            app.load_demo_results()

            assert demo_label.display is True
            assert DEMO_LABEL_TEXT in str(demo_label.render())
            assert "DEMO DATA — sample only; no real genome scan performed" in str(demo_label.render())
            assert app.is_demo_mode is True
    _run(_inner())


def test_calling_load_demo_results_again_does_not_duplicate_rows() -> None:
    """Verify calling load_demo_results repeatedly clears old rows and prevents duplication."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            app.load_demo_results()
            table = app.query_one("#alignment-results-table", DataTable)
            initial_count = table.row_count
            assert initial_count == len(DEMO_ALIGNMENT_RESULTS)

            # Second call
            app.load_demo_results()
            assert table.row_count == initial_count

            # Third call
            app.load_demo_results()
            assert table.row_count == initial_count
    _run(_inner())


def test_sample_data_format_and_required_fields() -> None:
    """Verify sample records follow the required alignment results format."""
    required_keys = {
        "record_id",
        "start",
        "end",
        "matched_sequence",
        "mismatch_count",
        "mismatch_positions",
    }
    assert len(DEMO_ALIGNMENT_RESULTS) >= 3, "Should provide a small illustrative batch of records"
    for record in DEMO_ALIGNMENT_RESULTS:
        assert required_keys.issubset(record.keys())
        assert isinstance(record["record_id"], str)
        assert isinstance(record["start"], int)
        assert isinstance(record["end"], int)
        assert record["start"] < record["end"]
        assert isinstance(record["matched_sequence"], str)
        assert len(record["matched_sequence"]) > 0
        assert isinstance(record["mismatch_count"], int)
        assert record["mismatch_count"] >= 0
        assert isinstance(record["mismatch_positions"], list)
        assert len(record["mismatch_positions"]) == record["mismatch_count"]


def test_demo_results_do_not_fabricate_gpu_metrics() -> None:
    """Verify demo results do not display fabricated GPU or hardware metrics."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            app.load_demo_results()

            cpu_label = app.query_one("#status-cpu-engine", Label)
            assert "Not started" in str(cpu_label.render())

            gpu_engine_label = app.query_one("#status-gpu-engine", Label)
            assert "Not connected" in str(gpu_engine_label.render())

            gpu_mem_label = app.query_one("#status-gpu-memory", Label)
            assert "Not available yet" in str(gpu_mem_label.render())

            seq_scanned = app.query_one("#status-sequences-scanned", Label)
            assert "0" in str(seq_scanned.render())
    _run(_inner())


def test_demo_label_widget_accessors_and_aliases() -> None:
    """Verify demo label can be accessed via property and alias selectors."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            app.load_demo_results()

            lbl_data = app.query_one("#demo-data-label", Label)
            lbl_alias = app.query_one("#demo-label", Label)
            assert lbl_data is lbl_alias
            assert app.demo_label is lbl_data
            assert app.demo_mode is True
    _run(_inner())


# =============================================================================
# Day 6 Tests: GPU Status Dashboard Integration
# =============================================================================


def test_gpu_status_panel_can_be_instantiated() -> None:
    """Verify GpuStatusPanel can be instantiated independently."""
    panel = GpuStatusPanel()
    assert panel is not None
    assert panel.border_title == "GPU Status"


def test_gpu_status_panel_initial_status() -> None:
    """Verify GPU status widget initially reads 'Not checked'."""
    panel = GpuStatusPanel()
    assert "Not checked" in str(panel.gpu_status_widget.render())


def test_gpu_status_panel_initial_device() -> None:
    """Verify GPU device widget initially reads 'Not detected'."""
    panel = GpuStatusPanel()
    assert "Not detected" in str(panel.gpu_device_widget.render())


def test_gpu_status_panel_compose_has_stable_ids() -> None:
    """Verify GpuStatusPanel yields widgets with stable IDs gpu-status and gpu-device."""
    panel = GpuStatusPanel()
    composed = {w.id: w for w in panel.compose() if getattr(w, "id", None)}
    assert "gpu-status" in composed, "gpu-status widget must be present"
    assert "gpu-device" in composed, "gpu-device widget must be present"


def test_app_has_gpu_status_panel() -> None:
    """Verify GeneWeaverApp creates a GpuStatusPanel instance."""
    app = GeneWeaverApp()
    assert hasattr(app, "gpu_status_panel")
    assert isinstance(app.gpu_status_panel, GpuStatusPanel)


def test_app_initial_gpu_status_is_not_checked() -> None:
    """Verify app-level GPU status widget initially shows 'Not checked'."""
    app = GeneWeaverApp()
    widget = app.query_one("#gpu-status")
    assert "Not checked" in str(widget.render())


def test_app_initial_gpu_device_is_not_detected() -> None:
    """Verify app-level GPU device widget initially shows 'Not detected'."""
    app = GeneWeaverApp()
    widget = app.query_one("#gpu-device")
    assert "Not detected" in str(widget.render())


def test_update_gpu_status_method_exists() -> None:
    """Verify GeneWeaverApp exposes update_gpu_status as a callable."""
    app = GeneWeaverApp()
    assert hasattr(app, "update_gpu_status")
    assert callable(app.update_gpu_status)


def test_update_gpu_status_to_checking() -> None:
    """Verify updating GPU status to 'Checking' is reflected in the widget."""
    app = GeneWeaverApp()
    app.update_gpu_status("Checking")
    widget = app.query_one("#gpu-status")
    assert "Checking" in str(widget.render())


def test_update_gpu_status_to_available_with_device() -> None:
    """Verify updating to 'Available' shows the device name."""
    app = GeneWeaverApp()
    app.update_gpu_status("Available", device_name="NVIDIA GeForce RTX 3080")
    status_widget = app.query_one("#gpu-status")
    device_widget = app.query_one("#gpu-device")
    assert "Available" in str(status_widget.render())
    assert "NVIDIA GeForce RTX 3080" in str(device_widget.render())


def test_update_gpu_status_to_unavailable() -> None:
    """Verify updating to 'Unavailable' sets status and shows 'Not detected' for device."""
    app = GeneWeaverApp()
    # First set a device name
    app.update_gpu_status("Available", device_name="NVIDIA GeForce RTX 3080")
    # Then go Unavailable — device should clear
    app.update_gpu_status("Unavailable")
    status_widget = app.query_one("#gpu-status")
    device_widget = app.query_one("#gpu-device")
    assert "Unavailable" in str(status_widget.render())
    assert "Not detected" in str(device_widget.render())
    assert "NVIDIA GeForce RTX 3080" not in str(device_widget.render())


def test_update_gpu_status_to_error() -> None:
    """Verify updating to 'Error' sets status and shows 'Not detected' for device."""
    app = GeneWeaverApp()
    app.update_gpu_status("Available", device_name="NVIDIA GeForce RTX 3080")
    app.update_gpu_status("Error")
    status_widget = app.query_one("#gpu-status")
    device_widget = app.query_one("#gpu-device")
    assert "Error" in str(status_widget.render())
    assert "Not detected" in str(device_widget.render())
    assert "NVIDIA GeForce RTX 3080" not in str(device_widget.render())


def test_update_gpu_status_clears_old_device_on_unavailable() -> None:
    """Verify old device name is cleared when switching to Unavailable."""
    app = GeneWeaverApp()
    app.update_gpu_status("Available", device_name="AMD Radeon RX 7900 XTX")
    device_widget = app.query_one("#gpu-device")
    assert "AMD Radeon RX 7900 XTX" in str(device_widget.render())

    app.update_gpu_status("Unavailable")
    assert "AMD Radeon RX 7900 XTX" not in str(device_widget.render())
    assert "Not detected" in str(device_widget.render())


def test_update_gpu_status_clears_old_device_on_error() -> None:
    """Verify old device name is cleared when switching to Error."""
    app = GeneWeaverApp()
    app.update_gpu_status("Available", device_name="AMD Radeon RX 7900 XTX")
    app.update_gpu_status("Error")
    device_widget = app.query_one("#gpu-device")
    assert "AMD Radeon RX 7900 XTX" not in str(device_widget.render())
    assert "Not detected" in str(device_widget.render())


@pytest.mark.parametrize(
    "bad_status",
    [
        "running",
        "active",
        "GPU Ready",
        "disabled",
        "online",
        "",
        123,
        None,
        True,
    ],
)
def test_reject_unsupported_gpu_status(bad_status) -> None:
    """Verify unsupported GPU statuses raise ValueError."""
    app = GeneWeaverApp()
    with pytest.raises(ValueError):
        app.update_gpu_status(bad_status)  # type: ignore[arg-type]


def test_reject_available_without_device_name() -> None:
    """Verify 'Available' status raises ValueError when device_name is None."""
    app = GeneWeaverApp()
    with pytest.raises(ValueError):
        app.update_gpu_status("Available")


def test_reject_available_with_empty_device_name() -> None:
    """Verify 'Available' status raises ValueError when device_name is an empty string."""
    app = GeneWeaverApp()
    with pytest.raises(ValueError):
        app.update_gpu_status("Available", device_name="")


def test_gpu_status_does_not_auto_change_on_load_demo() -> None:
    """Verify GPU status is not automatically set to Available by load_demo_results."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            app.load_demo_results()
            gpu_status_widget = app.query_one("#gpu-status", Label)
            assert "Available" not in str(gpu_status_widget.render()), (
                "GPU status must not be automatically set to 'Available' by load_demo_results"
            )
            assert "Not checked" in str(gpu_status_widget.render())
    _run(_inner())


def test_gpu_status_in_running_app() -> None:
    """Verify GPU status widgets are queryable in a live running app."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            status_widget = app.query_one("#gpu-status", Label)
            device_widget = app.query_one("#gpu-device", Label)
            assert status_widget is not None
            assert device_widget is not None
            assert "Not checked" in str(status_widget.render())
            assert "Not detected" in str(device_widget.render())
    _run(_inner())


def test_update_gpu_status_in_running_app() -> None:
    """Verify update_gpu_status works correctly in a live running app."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            app.update_gpu_status("Checking")
            status_widget = app.query_one("#gpu-status", Label)
            assert "Checking" in str(status_widget.render())

            app.update_gpu_status("Available", device_name="NVIDIA Tesla T4")
            status_widget = app.query_one("#gpu-status", Label)
            device_widget = app.query_one("#gpu-device", Label)
            assert "Available" in str(status_widget.render())
            assert "NVIDIA Tesla T4" in str(device_widget.render())

            app.update_gpu_status("Unavailable")
            assert "Unavailable" in str(status_widget.render())
            assert "Not detected" in str(device_widget.render())
    _run(_inner())


def test_gpu_status_widgets_have_unique_ids_no_collision() -> None:
    """Verify gpu-status and gpu-device IDs do not collide with existing widget IDs."""
    app = GeneWeaverApp()
    top_level = list(app.compose())

    all_ids: list[str] = []

    def collect_ids(widget) -> None:
        if getattr(widget, "id", None):
            all_ids.append(widget.id)
        for child in getattr(widget, "_pending_children", []):
            collect_ids(child)
        if isinstance(widget, (ProjectStatusPanel, GpuStatusPanel, ProcessingInfoPanel, ResultsPanel)):
            for composed_child in widget.compose():
                collect_ids(composed_child)

    for w in top_level:
        collect_ids(w)

    assert "gpu-status" in all_ids, "gpu-status widget ID must be present"
    assert "gpu-device" in all_ids, "gpu-device widget ID must be present"
    assert len(all_ids) == len(set(all_ids)), f"Duplicate widget IDs detected: {all_ids}"


def test_existing_dashboard_features_unaffected_by_gpu_section() -> None:
    """Verify all existing dashboard features still work after GPU section was added."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            # Scan status
            app.update_scan_status("Scanning")
            assert app.scan_status == "Scanning"

            # Progress
            app.update_scan_progress(75)
            assert app.scan_progress == 75

            # Results table
            app.update_alignment_results([_SAMPLE_RESULT_1, _SAMPLE_RESULT_2])
            table = app.query_one("#alignment-results-table", DataTable)
            assert table.row_count == 2

            # Demo mode
            app.load_demo_results()
            assert app.is_demo_mode is True
            demo_lbl = app.query_one("#demo-data-label", Label)
            assert demo_lbl.display is True

            # GPU section is independent
            app.update_gpu_status("Available", device_name="NVIDIA GeForce GTX 1660")
            gpu_lbl = app.query_one("#gpu-status", Label)
            assert "Available" in str(gpu_lbl.render())
    _run(_inner())


# =============================================================================
# Day 7 Tests: Benchmark Results Panel
# =============================================================================

# ---------------------------------------------------------------------------
# Sample benchmark dictionaries — manually crafted for testing purposes only.
# These do NOT represent real benchmark measurements.
# ---------------------------------------------------------------------------

_SAMPLE_BENCH_CPU_ONLY = {
    "cpu_average_seconds": 0.250,       # 250 ms — test value only
    "gpu_average_seconds": None,        # GPU not measured
    "speedup": None,
    "cpu_match_count": 10,
    "gpu_match_count": None,
    "results_match": None,
    "gpu_available": False,
    "message": "CPU-only benchmark complete",
}

_SAMPLE_BENCH_GPU_AVAILABLE = {
    "cpu_average_seconds": 0.200,       # 200 ms — test value only
    "gpu_average_seconds": 0.040,       # 40 ms — test value only
    "speedup": 5.0,                     # test speedup value
    "cpu_match_count": 8,
    "gpu_match_count": 8,
    "results_match": True,
    "gpu_available": True,
    "message": "GPU benchmark complete",
}

_SAMPLE_BENCH_GPU_UNAVAILABLE = {
    "cpu_average_seconds": 0.180,       # 180 ms — test value only
    "gpu_average_seconds": None,
    "speedup": None,
    "cpu_match_count": 6,
    "gpu_match_count": None,
    "results_match": None,
    "gpu_available": False,
    "message": "GPU not available on this machine",
}

_SAMPLE_BENCH_RESULTS_MISMATCH = {
    "cpu_average_seconds": 0.300,       # 300 ms — test value only
    "gpu_average_seconds": 0.060,       # 60 ms — test value only
    "speedup": 5.0,                     # test speedup value
    "cpu_match_count": 10,
    "gpu_match_count": 9,
    "results_match": False,
    "gpu_available": True,
    "message": "Result mismatch detected",
}

_SAMPLE_BENCH_MISSING_GPU_VALUES = {
    "cpu_average_seconds": 0.150,       # 150 ms — test value only
    "gpu_average_seconds": None,
    "speedup": None,
    "cpu_match_count": 5,
    "gpu_match_count": None,
    "results_match": None,
    "gpu_available": True,              # GPU attempted but values missing
    "message": "GPU execution incomplete",
}


# ---------------------------------------------------------------------------
# Structure tests — no live app needed
# ---------------------------------------------------------------------------


def test_benchmark_results_panel_can_be_instantiated() -> None:
    """Verify BenchmarkResultsPanel can be instantiated independently."""
    panel = BenchmarkResultsPanel()
    assert panel is not None
    assert panel.border_title == "Benchmark Results"


def test_benchmark_results_panel_initial_cpu_time() -> None:
    """Verify CPU Average Time initially reads 'Not run'."""
    panel = BenchmarkResultsPanel()
    assert "Not run" in str(panel._cpu_time_widget.render())


def test_benchmark_results_panel_initial_gpu_time() -> None:
    """Verify GPU Average Time initially reads 'Not run'."""
    panel = BenchmarkResultsPanel()
    assert "Not run" in str(panel._gpu_time_widget.render())


def test_benchmark_results_panel_initial_speedup() -> None:
    """Verify GPU Speedup initially reads 'Not available'."""
    panel = BenchmarkResultsPanel()
    assert "Not available" in str(panel._speedup_widget.render())


def test_benchmark_results_panel_initial_status() -> None:
    """Verify Benchmark Status initially reads 'Not run'."""
    panel = BenchmarkResultsPanel()
    assert "Not run" in str(panel._status_widget.render())


def test_benchmark_results_panel_compose_stable_ids() -> None:
    """Verify BenchmarkResultsPanel yields widgets with the required stable IDs."""
    panel = BenchmarkResultsPanel()
    composed = {w.id: w for w in panel.compose() if getattr(w, "id", None)}
    assert "cpu-benchmark-time" in composed
    assert "gpu-benchmark-time" in composed
    assert "gpu-benchmark-speedup" in composed
    assert "benchmark-status" in composed


def test_app_has_benchmark_results_panel() -> None:
    """Verify GeneWeaverApp creates a BenchmarkResultsPanel instance."""
    app = GeneWeaverApp()
    assert hasattr(app, "benchmark_results_panel")
    assert isinstance(app.benchmark_results_panel, BenchmarkResultsPanel)


def test_update_benchmark_results_method_exists() -> None:
    """Verify GeneWeaverApp exposes update_benchmark_results as a callable."""
    app = GeneWeaverApp()
    assert hasattr(app, "update_benchmark_results")
    assert callable(app.update_benchmark_results)


# ---------------------------------------------------------------------------
# Initial display tests (via app query_one fallback)
# ---------------------------------------------------------------------------


def test_app_initial_cpu_benchmark_time_not_run() -> None:
    """Verify app-level CPU Average Time widget initially shows 'Not run'."""
    app = GeneWeaverApp()
    widget = app.query_one("#cpu-benchmark-time")
    assert "Not run" in str(widget.render())


def test_app_initial_gpu_benchmark_time_not_run() -> None:
    """Verify app-level GPU Average Time widget initially shows 'Not run'."""
    app = GeneWeaverApp()
    widget = app.query_one("#gpu-benchmark-time")
    assert "Not run" in str(widget.render())


def test_app_initial_gpu_benchmark_speedup_not_available() -> None:
    """Verify app-level GPU Speedup widget initially shows 'Not available'."""
    app = GeneWeaverApp()
    widget = app.query_one("#gpu-benchmark-speedup")
    assert "Not available" in str(widget.render())


def test_app_initial_benchmark_status_not_run() -> None:
    """Verify app-level Benchmark Status widget initially shows 'Not run'."""
    app = GeneWeaverApp()
    widget = app.query_one("#benchmark-status")
    assert "Not run" in str(widget.render())


# ---------------------------------------------------------------------------
# update_benchmark_results tests (CPU-only / GPU available / GPU unavailable)
# ---------------------------------------------------------------------------


def test_update_benchmark_cpu_time_displayed_in_ms() -> None:
    """Verify CPU average time is formatted in milliseconds after update."""
    app = GeneWeaverApp()
    # _SAMPLE_BENCH_CPU_ONLY has cpu_average_seconds=0.250 → 250.000 ms
    app.update_benchmark_results(_SAMPLE_BENCH_CPU_ONLY)
    widget = app.query_one("#cpu-benchmark-time")
    rendered = str(widget.render())
    assert "250.000 ms" in rendered


def test_update_benchmark_gpu_time_when_available() -> None:
    """Verify GPU average time is shown in ms when GPU benchmark ran successfully."""
    app = GeneWeaverApp()
    # _SAMPLE_BENCH_GPU_AVAILABLE has gpu_average_seconds=0.040 → 40.000 ms
    app.update_benchmark_results(_SAMPLE_BENCH_GPU_AVAILABLE)
    widget = app.query_one("#gpu-benchmark-time")
    rendered = str(widget.render())
    assert "40.000 ms" in rendered


def test_update_benchmark_gpu_time_not_shown_when_gpu_unavailable() -> None:
    """Verify GPU Average Time shows 'GPU Unavailable' when gpu_available is False."""
    app = GeneWeaverApp()
    app.update_benchmark_results(_SAMPLE_BENCH_GPU_UNAVAILABLE)
    widget = app.query_one("#gpu-benchmark-time")
    rendered = str(widget.render())
    assert "GPU Unavailable" in rendered
    # Must NOT display a fabricated time value
    assert "ms" not in rendered


def test_update_benchmark_gpu_speedup_when_available() -> None:
    """Verify GPU Speedup is shown when a valid measured speedup is provided."""
    app = GeneWeaverApp()
    # _SAMPLE_BENCH_GPU_AVAILABLE has speedup=5.0
    app.update_benchmark_results(_SAMPLE_BENCH_GPU_AVAILABLE)
    widget = app.query_one("#gpu-benchmark-speedup")
    rendered = str(widget.render())
    assert "5.00x" in rendered


def test_update_benchmark_gpu_speedup_unavailable_when_gpu_not_run() -> None:
    """Verify GPU Speedup shows 'GPU Unavailable' when gpu_available is False."""
    app = GeneWeaverApp()
    app.update_benchmark_results(_SAMPLE_BENCH_GPU_UNAVAILABLE)
    widget = app.query_one("#gpu-benchmark-speedup")
    rendered = str(widget.render())
    assert "GPU Unavailable" in rendered


def test_update_benchmark_gpu_time_not_run_when_missing_values() -> None:
    """Verify GPU Average Time shows 'Not run' when gpu_available but values missing."""
    app = GeneWeaverApp()
    app.update_benchmark_results(_SAMPLE_BENCH_MISSING_GPU_VALUES)
    widget = app.query_one("#gpu-benchmark-time")
    rendered = str(widget.render())
    # gpu_available=True but gpu_average_seconds=None → "Not run"
    assert "Not run" in rendered
    # Must NOT fabricate a timing value
    assert "ms" not in rendered


def test_update_benchmark_speedup_not_available_when_missing_values() -> None:
    """Verify GPU Speedup shows 'Not available' when gpu_available but speedup is None."""
    app = GeneWeaverApp()
    app.update_benchmark_results(_SAMPLE_BENCH_MISSING_GPU_VALUES)
    widget = app.query_one("#gpu-benchmark-speedup")
    rendered = str(widget.render())
    assert "Not available" in rendered


def test_update_benchmark_status_message_shown() -> None:
    """Verify benchmark message is reflected in the status widget."""
    app = GeneWeaverApp()
    app.update_benchmark_results(_SAMPLE_BENCH_GPU_AVAILABLE)
    widget = app.query_one("#benchmark-status")
    rendered = str(widget.render())
    assert "GPU benchmark complete" in rendered


def test_update_benchmark_results_match_indicated() -> None:
    """Verify results_match=True is clearly indicated in the status widget."""
    app = GeneWeaverApp()
    app.update_benchmark_results(_SAMPLE_BENCH_GPU_AVAILABLE)
    widget = app.query_one("#benchmark-status")
    rendered = str(widget.render())
    assert "Yes" in rendered or "match" in rendered.lower()


def test_update_benchmark_results_mismatch_indicated() -> None:
    """Verify results_match=False is clearly indicated in the status widget."""
    app = GeneWeaverApp()
    app.update_benchmark_results(_SAMPLE_BENCH_RESULTS_MISMATCH)
    widget = app.query_one("#benchmark-status")
    rendered = str(widget.render())
    assert "No" in rendered or "mismatch" in rendered.lower() or "✗" in rendered


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_update_benchmark_results_rejects_non_dict() -> None:
    """Verify update_benchmark_results raises ValueError for non-dict input."""
    app = GeneWeaverApp()
    with pytest.raises(ValueError):
        app.update_benchmark_results("not a dict")  # type: ignore[arg-type]


def test_update_benchmark_results_rejects_list() -> None:
    """Verify update_benchmark_results raises ValueError for a list."""
    app = GeneWeaverApp()
    with pytest.raises(ValueError):
        app.update_benchmark_results([])  # type: ignore[arg-type]


def test_update_benchmark_results_rejects_missing_key() -> None:
    """Verify update_benchmark_results raises ValueError when required key is missing."""
    app = GeneWeaverApp()
    incomplete = dict(_SAMPLE_BENCH_GPU_AVAILABLE)
    del incomplete["cpu_average_seconds"]
    with pytest.raises(ValueError, match="cpu_average_seconds"):
        app.update_benchmark_results(incomplete)


def test_update_benchmark_results_rejects_multiple_missing_keys() -> None:
    """Verify ValueError is raised when several required keys are missing."""
    app = GeneWeaverApp()
    with pytest.raises(ValueError):
        app.update_benchmark_results({"message": "incomplete"})


# ---------------------------------------------------------------------------
# No duplicate widgets / repeated updates
# ---------------------------------------------------------------------------


def test_update_benchmark_results_multiple_times_no_duplicates() -> None:
    """Verify calling update_benchmark_results multiple times replaces, not duplicates, display."""
    # Test without live app — verify widget count stays at 4 on the panel object
    app = GeneWeaverApp()
    panel = app.benchmark_results_panel

    # Compose always returns exactly 4 benchmark widgets
    initial_count = len(list(panel.compose()))
    assert initial_count == 4

    # First update
    app.update_benchmark_results(_SAMPLE_BENCH_CPU_ONLY)
    assert len(list(panel.compose())) == initial_count

    # Second update with different data
    app.update_benchmark_results(_SAMPLE_BENCH_GPU_AVAILABLE)
    assert len(list(panel.compose())) == initial_count

    # Third update
    app.update_benchmark_results(_SAMPLE_BENCH_GPU_UNAVAILABLE)
    assert len(list(panel.compose())) == initial_count

    # Widget IDs must remain unique across calls
    all_ids = [w.id for w in panel.compose() if getattr(w, "id", None)]
    assert len(all_ids) == len(set(all_ids))
    assert set(all_ids) == {
        "cpu-benchmark-time",
        "gpu-benchmark-time",
        "gpu-benchmark-speedup",
        "benchmark-status",
    }


# ---------------------------------------------------------------------------
# Widget IDs uniqueness (benchmark panel among all dashboard widgets)
# ---------------------------------------------------------------------------


def test_benchmark_widget_ids_do_not_collide() -> None:
    """Verify benchmark widget IDs do not collide with existing dashboard IDs."""
    app = GeneWeaverApp()
    top_level = list(app.compose())

    all_ids: list[str] = []

    def collect_ids(widget) -> None:
        if getattr(widget, "id", None):
            all_ids.append(widget.id)
        for child in getattr(widget, "_pending_children", []):
            collect_ids(child)
        if isinstance(
            widget,
            (ProjectStatusPanel, GpuStatusPanel, ProcessingInfoPanel, ResultsPanel, BenchmarkResultsPanel),
        ):
            for composed_child in widget.compose():
                collect_ids(composed_child)

    for w in top_level:
        collect_ids(w)

    assert "cpu-benchmark-time" in all_ids
    assert "gpu-benchmark-time" in all_ids
    assert "gpu-benchmark-speedup" in all_ids
    assert "benchmark-status" in all_ids
    assert len(all_ids) == len(set(all_ids)), f"Duplicate widget IDs: {all_ids}"


# ---------------------------------------------------------------------------
# Existing dashboard features unaffected
# ---------------------------------------------------------------------------


def test_existing_dashboard_features_unaffected_by_benchmark_panel() -> None:
    """Verify all pre-existing dashboard features still work after adding benchmark panel."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            # Scan status
            app.update_scan_status("Scanning")
            assert app.scan_status == "Scanning"

            # Progress
            app.update_scan_progress(60)
            assert app.scan_progress == 60

            # Results table
            app.update_alignment_results([_SAMPLE_RESULT_1, _SAMPLE_RESULT_2])
            table = app.query_one("#alignment-results-table", DataTable)
            assert table.row_count == 2

            # Demo mode
            app.load_demo_results()
            assert app.is_demo_mode is True
            demo_lbl = app.query_one("#demo-data-label", Label)
            assert demo_lbl.display is True

            # GPU status panel
            app.update_gpu_status("Available", device_name="NVIDIA GeForce RTX 4090")
            gpu_lbl = app.query_one("#gpu-status", Label)
            assert "Available" in str(gpu_lbl.render())

            # Benchmark panel (using test data, not real measurements)
            app.update_benchmark_results(_SAMPLE_BENCH_GPU_AVAILABLE)
            cpu_lbl = app.query_one("#cpu-benchmark-time", Label)
            assert "200.000 ms" in str(cpu_lbl.render())

    _run(_inner())


def test_benchmark_does_not_auto_populate_on_startup() -> None:
    """Verify benchmark fields are Not run / Not available on fresh startup."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            cpu_w = app.query_one("#cpu-benchmark-time", Label)
            gpu_w = app.query_one("#gpu-benchmark-time", Label)
            spd_w = app.query_one("#gpu-benchmark-speedup", Label)
            sta_w = app.query_one("#benchmark-status", Label)

            assert "Not run" in str(cpu_w.render())
            assert "Not run" in str(gpu_w.render())
            assert "Not available" in str(spd_w.render())
            assert "Not run" in str(sta_w.render())

    _run(_inner())


def test_load_demo_results_does_not_populate_benchmark_panel() -> None:
    """Verify calling load_demo_results does NOT auto-populate benchmark fields."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            app.load_demo_results()

            cpu_w = app.query_one("#cpu-benchmark-time", Label)
            gpu_w = app.query_one("#gpu-benchmark-time", Label)
            spd_w = app.query_one("#gpu-benchmark-speedup", Label)
            sta_w = app.query_one("#benchmark-status", Label)

            assert "Not run" in str(cpu_w.render())
            assert "Not run" in str(gpu_w.render())
            assert "Not available" in str(spd_w.render())
            assert "Not run" in str(sta_w.render())

    _run(_inner())


# =============================================================================
# Day 8 Tests: GPU Memory Dashboard Integration
# =============================================================================

# ---------------------------------------------------------------------------
# Sample mock values — manually crafted for testing purposes only.
# These do NOT represent real GPU hardware measurements.
# ---------------------------------------------------------------------------
_MOCK_TOTAL_BYTES_8GB = 8 * 1024 * 1024 * 1024       # 8 GB mock test value
_MOCK_FREE_BYTES_6GB = 6 * 1024 * 1024 * 1024        # 6 GB mock test value
_MOCK_USED_BYTES_2GB = 2 * 1024 * 1024 * 1024        # 2 GB mock expected used value

_MOCK_TOTAL_BYTES_16GB = 16 * 1024 * 1024 * 1024     # 16 GB mock test value
_MOCK_FREE_BYTES_10GB = 10 * 1024 * 1024 * 1024      # 10 GB mock test value
_MOCK_USED_BYTES_6GB = 6 * 1024 * 1024 * 1024        # 6 GB mock expected used value

_MOCK_TOTAL_BYTES_500MB = 500 * 1024 * 1024          # 500 MB mock test value
_MOCK_FREE_BYTES_200MB = 200 * 1024 * 1024           # 200 MB mock test value
_MOCK_USED_BYTES_300MB = 300 * 1024 * 1024           # 300 MB mock expected used value


# ---------------------------------------------------------------------------
# Structure and panel tests — no live app needed
# ---------------------------------------------------------------------------


def test_gpu_memory_panel_can_be_instantiated() -> None:
    """Verify GpuMemoryPanel can be instantiated independently."""
    panel = GpuMemoryPanel()
    assert panel is not None
    assert panel.border_title == "GPU Memory"


def test_gpu_memory_panel_initial_display() -> None:
    """Verify GPU memory panel widgets initially read 'Not checked' and 'Not available'."""
    panel = GpuMemoryPanel()
    assert "Not checked" in str(panel.memory_status_widget.render())
    assert "Not available" in str(panel.total_memory_widget.render())
    assert "Not available" in str(panel.used_memory_widget.render())
    assert "Not available" in str(panel.free_memory_widget.render())


def test_gpu_memory_panel_compose_has_stable_ids() -> None:
    """Verify GpuMemoryPanel yields widgets with the required stable IDs."""
    panel = GpuMemoryPanel()
    composed = {w.id: w for w in panel.compose() if getattr(w, "id", None)}
    assert "gpu-memory-status" in composed
    assert "gpu-memory-total" in composed
    assert "gpu-memory-used" in composed
    assert "gpu-memory-free" in composed


def test_app_has_gpu_memory_panel() -> None:
    """Verify GeneWeaverApp creates a GpuMemoryPanel instance."""
    app = GeneWeaverApp()
    assert hasattr(app, "gpu_memory_panel")
    assert isinstance(app.gpu_memory_panel, GpuMemoryPanel)


def test_update_gpu_memory_method_exists() -> None:
    """Verify GeneWeaverApp exposes update_gpu_memory as a callable."""
    app = GeneWeaverApp()
    assert hasattr(app, "update_gpu_memory")
    assert callable(app.update_gpu_memory)


# ---------------------------------------------------------------------------
# Initial display tests (app-level query_one)
# ---------------------------------------------------------------------------


def test_initial_gpu_memory_display() -> None:
    """Verify all GPU memory widgets initially display 'Not checked' and 'Not available'."""
    app = GeneWeaverApp()
    status_w = app.query_one("#gpu-memory-status")
    total_w = app.query_one("#gpu-memory-total")
    used_w = app.query_one("#gpu-memory-used")
    free_w = app.query_one("#gpu-memory-free")

    assert "Not checked" in str(status_w.render())
    assert "Not available" in str(total_w.render())
    assert "Not available" in str(used_w.render())
    assert "Not available" in str(free_w.render())

    assert app.gpu_memory_status == "Not checked"
    assert app.gpu_memory_total is None
    assert app.gpu_memory_used is None
    assert app.gpu_memory_free is None


# ---------------------------------------------------------------------------
# Memory updates, calculations, and formatting
# ---------------------------------------------------------------------------


def test_updating_total_gpu_memory() -> None:
    """Verify updating total GPU memory formats and displays correctly."""
    app = GeneWeaverApp()
    app.update_gpu_memory(_MOCK_FREE_BYTES_6GB, _MOCK_TOTAL_BYTES_8GB)

    total_w = app.query_one("#gpu-memory-total")
    assert "8.00 GB" in str(total_w.render())
    assert app.gpu_memory_total == _MOCK_TOTAL_BYTES_8GB


def test_updating_free_gpu_memory() -> None:
    """Verify updating free GPU memory formats and displays correctly."""
    app = GeneWeaverApp()
    app.update_gpu_memory(_MOCK_FREE_BYTES_6GB, _MOCK_TOTAL_BYTES_8GB)

    free_w = app.query_one("#gpu-memory-free")
    assert "6.00 GB" in str(free_w.render())
    assert app.gpu_memory_free == _MOCK_FREE_BYTES_6GB


def test_correct_used_memory_calculation() -> None:
    """Verify used GPU memory is calculated correctly as total minus free."""
    app = GeneWeaverApp()

    # Case 1: 8 GB total - 6 GB free = 2 GB used
    app.update_gpu_memory(_MOCK_FREE_BYTES_6GB, _MOCK_TOTAL_BYTES_8GB)
    used_w = app.query_one("#gpu-memory-used")
    assert "2.00 GB" in str(used_w.render())
    assert app.gpu_memory_used == _MOCK_USED_BYTES_2GB

    # Case 2: 500 MB total - 200 MB free = 300 MB used
    app.update_gpu_memory(_MOCK_FREE_BYTES_200MB, _MOCK_TOTAL_BYTES_500MB)
    assert "300.00 MB" in str(used_w.render())
    assert app.gpu_memory_used == _MOCK_USED_BYTES_300MB

    # Case 3: 8 GB total - 8 GB free = 0 bytes used
    app.update_gpu_memory(_MOCK_TOTAL_BYTES_8GB, _MOCK_TOTAL_BYTES_8GB)
    assert "0 B" in str(used_w.render())
    assert app.gpu_memory_used == 0


def test_update_gpu_memory_sets_status_available() -> None:
    """Verify updating GPU memory sets GPU memory status to 'Available'."""
    app = GeneWeaverApp()
    app.update_gpu_memory(_MOCK_FREE_BYTES_6GB, _MOCK_TOTAL_BYTES_8GB)

    status_w = app.query_one("#gpu-memory-status")
    assert "Available" in str(status_w.render())
    assert app.gpu_memory_status == "Available"


def test_human_readable_memory_formatting() -> None:
    """Verify format_bytes utility and widget display use appropriate units such as MB or GB."""
    # Direct utility test across unit scales
    assert format_bytes(0) == "0 B"
    assert format_bytes(512) == "512 B"
    assert format_bytes(1024) == "1.00 KB"
    assert format_bytes(1536) == "1.50 KB"
    assert format_bytes(1024 * 1024) == "1.00 MB"
    assert format_bytes(500 * 1024 * 1024) == "500.00 MB"
    assert format_bytes(1024 * 1024 * 1024) == "1.00 GB"
    assert format_bytes(8 * 1024 * 1024 * 1024) == "8.00 GB"
    assert format_bytes(24 * 1024 * 1024 * 1024) == "24.00 GB"
    assert format_bytes(1024 * 1024 * 1024 * 1024) == "1.00 TB"

    # In-dashboard formatting check
    app = GeneWeaverApp()
    app.update_gpu_memory(_MOCK_FREE_BYTES_200MB, _MOCK_TOTAL_BYTES_500MB)
    assert "MB" in str(app.query_one("#gpu-memory-total").render())
    assert "MB" in str(app.query_one("#gpu-memory-free").render())
    assert "MB" in str(app.query_one("#gpu-memory-used").render())

    app.update_gpu_memory(_MOCK_FREE_BYTES_6GB, _MOCK_TOTAL_BYTES_8GB)
    assert "GB" in str(app.query_one("#gpu-memory-total").render())
    assert "GB" in str(app.query_one("#gpu-memory-free").render())
    assert "GB" in str(app.query_one("#gpu-memory-used").render())


# ---------------------------------------------------------------------------
# GPU unavailable case
# ---------------------------------------------------------------------------


def test_gpu_unavailable_case() -> None:
    """Verify update_gpu_memory(None, None) displays 'Not available' and status 'Unavailable'."""
    app = GeneWeaverApp()

    # Pre-populate with valid mock values first to ensure it actually updates to unavailable
    app.update_gpu_memory(_MOCK_FREE_BYTES_6GB, _MOCK_TOTAL_BYTES_8GB)
    assert app.gpu_memory_status == "Available"

    # Now transition to unavailable
    app.update_gpu_memory(None, None)

    status_w = app.query_one("#gpu-memory-status")
    total_w = app.query_one("#gpu-memory-total")
    used_w = app.query_one("#gpu-memory-used")
    free_w = app.query_one("#gpu-memory-free")

    assert "Unavailable" in str(status_w.render())
    assert "Not available" in str(total_w.render())
    assert "Not available" in str(used_w.render())
    assert "Not available" in str(free_w.render())

    assert app.gpu_memory_status == "Unavailable"
    assert app.gpu_memory_total is None
    assert app.gpu_memory_used is None
    assert app.gpu_memory_free is None

    # Must NOT fabricate numerical byte strings
    assert "GB" not in str(total_w.render())
    assert "GB" not in str(free_w.render())
    assert "GB" not in str(used_w.render())


# ---------------------------------------------------------------------------
# Input validation tests (ValueError cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("neg_free", [-1, -1024, -1048576, -8589934592])
def test_reject_negative_free_memory(neg_free: int) -> None:
    """Verify negative free memory values raise ValueError."""
    app = GeneWeaverApp()
    with pytest.raises(ValueError, match="negative"):
        app.update_gpu_memory(neg_free, _MOCK_TOTAL_BYTES_8GB)


@pytest.mark.parametrize("neg_total", [-1, -1024, -1048576, -8589934592])
def test_reject_negative_total_memory(neg_total: int) -> None:
    """Verify negative total memory values raise ValueError."""
    app = GeneWeaverApp()
    with pytest.raises(ValueError, match="negative"):
        app.update_gpu_memory(1024, neg_total)


def test_reject_both_negative_memory_values() -> None:
    """Verify both memory values negative raises ValueError."""
    app = GeneWeaverApp()
    with pytest.raises(ValueError, match="negative"):
        app.update_gpu_memory(-1024, -2048)


def test_reject_total_memory_equal_to_zero() -> None:
    """Verify total memory equal to zero raises ValueError."""
    app = GeneWeaverApp()
    with pytest.raises(ValueError, match="zero"):
        app.update_gpu_memory(0, 0)


def test_reject_free_memory_greater_than_total() -> None:
    """Verify free memory greater than total memory raises ValueError."""
    app = GeneWeaverApp()
    with pytest.raises(ValueError, match="greater than total"):
        app.update_gpu_memory(_MOCK_TOTAL_BYTES_8GB + 1024, _MOCK_TOTAL_BYTES_8GB)


def test_reject_only_one_missing_memory_value() -> None:
    """Verify providing only one None value raises ValueError."""
    app = GeneWeaverApp()
    with pytest.raises(ValueError, match="Both free_bytes and total_bytes"):
        app.update_gpu_memory(None, _MOCK_TOTAL_BYTES_8GB)

    with pytest.raises(ValueError, match="Both free_bytes and total_bytes"):
        app.update_gpu_memory(_MOCK_FREE_BYTES_6GB, None)


@pytest.mark.parametrize(
    "invalid_free, invalid_total",
    [
        ("6000000000", _MOCK_TOTAL_BYTES_8GB),
        (_MOCK_FREE_BYTES_6GB, "8000000000"),
        (True, _MOCK_TOTAL_BYTES_8GB),
        (_MOCK_FREE_BYTES_6GB, False),
        (6442450944.5, _MOCK_TOTAL_BYTES_8GB),
        (_MOCK_FREE_BYTES_6GB, 8589934592.0),
        ([1024], _MOCK_TOTAL_BYTES_8GB),
        (_MOCK_FREE_BYTES_6GB, {"total": 8589934592}),
    ],
)
def test_reject_invalid_memory_types(invalid_free, invalid_total) -> None:
    """Verify non-integer/boolean memory types raise ValueError."""
    app = GeneWeaverApp()
    with pytest.raises(ValueError):
        app.update_gpu_memory(invalid_free, invalid_total)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Multiple updates and lifecycle
# ---------------------------------------------------------------------------


def test_updating_memory_information_multiple_times() -> None:
    """Verify update_gpu_memory can be called multiple times correctly replacing values."""
    app = GeneWeaverApp()

    # 1. First update: 8GB total, 6GB free -> 2GB used
    app.update_gpu_memory(_MOCK_FREE_BYTES_6GB, _MOCK_TOTAL_BYTES_8GB)
    assert app.gpu_memory_status == "Available"
    assert "8.00 GB" in str(app.query_one("#gpu-memory-total").render())
    assert "2.00 GB" in str(app.query_one("#gpu-memory-used").render())
    assert "6.00 GB" in str(app.query_one("#gpu-memory-free").render())

    # 2. Second update: 16GB total, 10GB free -> 6GB used
    app.update_gpu_memory(_MOCK_FREE_BYTES_10GB, _MOCK_TOTAL_BYTES_16GB)
    assert app.gpu_memory_status == "Available"
    assert "16.00 GB" in str(app.query_one("#gpu-memory-total").render())
    assert "6.00 GB" in str(app.query_one("#gpu-memory-used").render())
    assert "10.00 GB" in str(app.query_one("#gpu-memory-free").render())

    # 3. Third update: Unavailable
    app.update_gpu_memory(None, None)
    assert app.gpu_memory_status == "Unavailable"
    assert "Not available" in str(app.query_one("#gpu-memory-total").render())
    assert "Not available" in str(app.query_one("#gpu-memory-used").render())
    assert "Not available" in str(app.query_one("#gpu-memory-free").render())

    # 4. Fourth update: 500MB total, 200MB free -> 300MB used
    app.update_gpu_memory(_MOCK_FREE_BYTES_200MB, _MOCK_TOTAL_BYTES_500MB)
    assert app.gpu_memory_status == "Available"
    assert "500.00 MB" in str(app.query_one("#gpu-memory-total").render())
    assert "300.00 MB" in str(app.query_one("#gpu-memory-used").render())
    assert "200.00 MB" in str(app.query_one("#gpu-memory-free").render())


# ---------------------------------------------------------------------------
# Dashboard behavior / Isolation / Anti-fabrication tests
# ---------------------------------------------------------------------------


def test_gpu_memory_not_auto_generated_on_startup() -> None:
    """Verify GPU memory is never fabricated or auto-generated on startup."""
    app = GeneWeaverApp()
    assert app.gpu_memory_status == "Not checked"
    status_w = app.query_one("#gpu-memory-status")
    assert "Not checked" in str(status_w.render())
    assert "Available" not in str(status_w.render())

    for widget_id in ("#gpu-memory-total", "#gpu-memory-used", "#gpu-memory-free"):
        w = app.query_one(widget_id)
        assert "Not available" in str(w.render())


def test_load_demo_results_does_not_affect_gpu_memory() -> None:
    """Verify calling load_demo_results does NOT modify GPU memory values."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            app.load_demo_results()

            status_w = app.query_one("#gpu-memory-status", Label)
            total_w = app.query_one("#gpu-memory-total", Label)
            used_w = app.query_one("#gpu-memory-used", Label)
            free_w = app.query_one("#gpu-memory-free", Label)

            assert "Not checked" in str(status_w.render())
            assert "Not available" in str(total_w.render())
            assert "Not available" in str(used_w.render())
            assert "Not available" in str(free_w.render())
    _run(_inner())


def test_update_gpu_status_does_not_affect_gpu_memory() -> None:
    """Verify update_gpu_status is completely independent of GPU memory section."""
    app = GeneWeaverApp()
    app.update_gpu_status("Available", device_name="NVIDIA GeForce RTX 4080")

    gpu_status_w = app.query_one("#gpu-status")
    assert "Available" in str(gpu_status_w.render())

    # GPU memory section must remain untouched
    mem_status_w = app.query_one("#gpu-memory-status")
    assert "Not checked" in str(mem_status_w.render())
    assert app.gpu_memory_status == "Not checked"


def test_gpu_memory_widget_ids_do_not_collide() -> None:
    """Verify GPU memory widget IDs do not collide with any existing dashboard IDs."""
    app = GeneWeaverApp()
    top_level = list(app.compose())

    all_ids: list[str] = []

    def collect_ids(widget) -> None:
        if getattr(widget, "id", None):
            all_ids.append(widget.id)
        for child in getattr(widget, "_pending_children", []):
            collect_ids(child)
        if isinstance(
            widget,
            (
                ProjectStatusPanel,
                GpuStatusPanel,
                GpuMemoryPanel,
                ProcessingInfoPanel,
                ResultsPanel,
                BenchmarkResultsPanel,
            ),
        ):
            for composed_child in widget.compose():
                collect_ids(composed_child)

    for w in top_level:
        collect_ids(w)

    assert "gpu-memory-status" in all_ids
    assert "gpu-memory-total" in all_ids
    assert "gpu-memory-used" in all_ids
    assert "gpu-memory-free" in all_ids
    assert len(all_ids) == len(set(all_ids)), f"Duplicate widget IDs: {all_ids}"


# ---------------------------------------------------------------------------
# Live running app test
# ---------------------------------------------------------------------------


def test_gpu_memory_in_running_app() -> None:
    """Verify GPU memory updates and display behave correctly in a live running app."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            # Check initial state
            status_w = app.query_one("#gpu-memory-status", Label)
            total_w = app.query_one("#gpu-memory-total", Label)
            used_w = app.query_one("#gpu-memory-used", Label)
            free_w = app.query_one("#gpu-memory-free", Label)

            assert "Not checked" in str(status_w.render())
            assert "Not available" in str(total_w.render())
            assert "Not available" in str(used_w.render())
            assert "Not available" in str(free_w.render())

            # Update with mock measurements
            app.update_gpu_memory(_MOCK_FREE_BYTES_6GB, _MOCK_TOTAL_BYTES_8GB)
            assert "Available" in str(status_w.render())
            assert "8.00 GB" in str(total_w.render())
            assert "2.00 GB" in str(used_w.render())
            assert "6.00 GB" in str(free_w.render())

            # Transition to unavailable
            app.update_gpu_memory(None, None)
            assert "Unavailable" in str(status_w.render())
            assert "Not available" in str(total_w.render())
            assert "Not available" in str(used_w.render())
            assert "Not available" in str(free_w.render())

    _run(_inner())


def test_existing_dashboard_features_unaffected_by_gpu_memory_section() -> None:
    """Verify all existing dashboard features work together with GPU memory section."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            # 1. Scan status
            app.update_scan_status("Scanning")
            assert app.scan_status == "Scanning"

            # 2. Progress
            app.update_scan_progress(80)
            assert app.scan_progress == 80

            # 3. Results table
            app.update_alignment_results([_SAMPLE_RESULT_1])
            table = app.query_one("#alignment-results-table", DataTable)
            assert table.row_count == 1

            # 4. GPU status panel
            app.update_gpu_status("Available", device_name="NVIDIA RTX 4090")
            gpu_w = app.query_one("#gpu-status", Label)
            assert "Available" in str(gpu_w.render())

            # 5. Benchmark results panel
            app.update_benchmark_results(_SAMPLE_BENCH_GPU_AVAILABLE)
            cpu_bench = app.query_one("#cpu-benchmark-time", Label)
            assert "200.000 ms" in str(cpu_bench.render())

            # 6. GPU memory section
            app.update_gpu_memory(_MOCK_FREE_BYTES_10GB, _MOCK_TOTAL_BYTES_16GB)
            mem_status_w = app.query_one("#gpu-memory-status", Label)
            mem_total_w = app.query_one("#gpu-memory-total", Label)
            mem_used_w = app.query_one("#gpu-memory-used", Label)
            mem_free_w = app.query_one("#gpu-memory-free", Label)

            assert "Available" in str(mem_status_w.render())
            assert "16.00 GB" in str(mem_total_w.render())
            assert "6.00 GB" in str(mem_used_w.render())
            assert "10.00 GB" in str(mem_free_w.render())

            # 7. Demo mode
            app.load_demo_results()
            assert app.is_demo_mode is True
            demo_lbl = app.query_one("#demo-data-label", Label)
            assert demo_lbl.display is True

            # Check that GPU memory was not overwritten by demo mode
            assert "16.00 GB" in str(mem_total_w.render())

    _run(_inner())

