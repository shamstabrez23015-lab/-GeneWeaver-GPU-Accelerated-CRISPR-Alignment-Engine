"""
GeneWeaver: GPU-Accelerated CRISPR Alignment Engine
Day 10 - Dashboard Integration Tests

Comprehensive integration test suite verifying that all dashboard sections
and dynamic update mechanisms work together correctly as a unified application:
- Header and Footer
- Sidebar and Main Content
- Scan Status and Progress Bar
- Alignment Results Table
- GPU Status
- GPU Memory Information
- Benchmark Results
- Consolidated GPU Diagnostics

Integration Scenarios Tested:
- Scenario A: Scan progress lifecycle (Not started -> Scanning -> Progress updates -> Completed)
- Scenario B: Alignment results lifecycle (Multiple results -> Update with new -> Verify old cleared -> Empty set)
- Scenario C: GPU information lifecycle (Available with device & memory -> Unavailable -> Cleared info)
- Scenario D: Benchmark information (CPU only -> GPU available -> GPU unavailable -> Anti-fabrication check)
- Scenario E: Combined dashboard (Simultaneous & sequential updates across all panels retain correct values)

IMPORTANT RULES & CONSTRAINTS:
- All test values, dictionaries, and hardware metrics used in this suite are synthetic sample data
  created solely for software testing. They do NOT represent real GPU measurements or actual genomic scans.
- No hardware metrics or GPU utilization values are fabricated.
- Independent from external branches (no imports from Albert's or Faizan's modules).
- Preserves all existing dashboard functionality and interfaces.
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


def _run(coro):
    """Run an async coroutine synchronously for Textual pilot integration tests."""
    return _asyncio.run(coro)


# =============================================================================
# Synthetic Test Data Fixtures
# Note: Sample test values only; not actual GPU hardware measurements.
# =============================================================================

# Sample alignment batches for Scenario B and Scenario E
SAMPLE_ALIGNMENT_BATCH_A = [
    {
        "record_id": "CRISPR-TARGET-001",
        "start": 12000,
        "end": 12023,
        "matched_sequence": "ATGCGATCGATCGATCGATCAGG",
        "mismatch_count": 0,
        "mismatch_positions": [],
    },
    {
        "record_id": "CRISPR-TARGET-002",
        "start": 34500,
        "end": 34523,
        "matched_sequence": "ATGCGATCGATCGATCGATCAGG",
        "mismatch_count": 1,
        "mismatch_positions": [4],
    },
    {
        "record_id": "CRISPR-TARGET-003",
        "start": 78900,
        "end": 78923,
        "matched_sequence": "ATGCGATCGATCGATCGATCAGG",
        "mismatch_count": 2,
        "mismatch_positions": [4, 12],
    },
]

SAMPLE_ALIGNMENT_BATCH_B = [
    {
        "record_id": "CRISPR-TARGET-004",
        "start": 91000,
        "end": 91023,
        "matched_sequence": "TTGCAATTCGATCGATCGATCGG",
        "mismatch_count": 1,
        "mismatch_positions": [2],
    },
    {
        "record_id": "CRISPR-TARGET-005",
        "start": 142000,
        "end": 142023,
        "matched_sequence": "GGCCAATTCGATCGATCGATCGG",
        "mismatch_count": 3,
        "mismatch_positions": [0, 1, 15],
    },
]

# Sample GPU diagnostic dictionaries (for Scenario C and Scenario E)
# Note: Test dictionaries containing illustrative values only.
SAMPLE_GPU_DIAG_AVAILABLE = {
    "available": True,
    "device_name": "Test NVIDIA RTX 4090",
    "total_bytes": 24 * 1024 * 1024 * 1024,      # 24 GB test total
    "free_bytes": 18 * 1024 * 1024 * 1024,       # 18 GB test free
    "used_bytes": 6 * 1024 * 1024 * 1024,        # 6 GB test used
    "message": "GPU detected and ready for acceleration (test sample)",
}

SAMPLE_GPU_DIAG_UNAVAILABLE = {
    "available": False,
    "device_name": None,
    "total_bytes": None,
    "free_bytes": None,
    "used_bytes": None,
    "message": "GPU accelerator unavailable (test sample)",
}

SAMPLE_GPU_DIAG_ALT_AVAILABLE = {
    "available": True,
    "device_name": "Test AMD Radeon Pro VII",
    "total_bytes": 16 * 1024 * 1024 * 1024,      # 16 GB test total
    "free_bytes": 12 * 1024 * 1024 * 1024,       # 12 GB test free
    "used_bytes": 4 * 1024 * 1024 * 1024,        # 4 GB test used
    "message": "Alternate GPU ready (test sample)",
}

# Sample benchmark dictionaries (for Scenario D and Scenario E)
SAMPLE_BENCH_CPU_ONLY = {
    "cpu_average_seconds": 0.450,       # 450 ms test sample
    "gpu_average_seconds": None,
    "speedup": None,
    "cpu_match_count": 15,
    "gpu_match_count": None,
    "results_match": None,
    "gpu_available": False,
    "message": "CPU benchmark complete",
}

SAMPLE_BENCH_GPU_AVAILABLE = {
    "cpu_average_seconds": 0.320,       # 320 ms test sample
    "gpu_average_seconds": 0.040,       # 40 ms test sample
    "speedup": 8.0,                     # 8.00x test speedup factor
    "cpu_match_count": 12,
    "gpu_match_count": 12,
    "results_match": True,
    "gpu_available": True,
    "message": "Comparative benchmark complete",
}

SAMPLE_BENCH_GPU_UNAVAILABLE = {
    "cpu_average_seconds": 0.280,       # 280 ms test sample
    "gpu_average_seconds": None,
    "speedup": None,
    "cpu_match_count": 10,
    "gpu_match_count": None,
    "results_match": None,
    "gpu_available": False,
    "message": "GPU execution skipped - hardware unavailable",
}

SAMPLE_BENCH_MISSING_GPU_VALUES = {
    "cpu_average_seconds": 0.210,
    "gpu_average_seconds": None,
    "speedup": None,
    "cpu_match_count": 5,
    "gpu_match_count": None,
    "results_match": None,
    "gpu_available": True,              # GPU was marked available but no run metrics produced
    "message": "GPU benchmarking was not completed",
}


# =============================================================================
# 1. Complete Application Section Integration Tests
# =============================================================================

def test_integration_header_and_footer() -> None:
    """Verify that Header and Footer work together in the integrated application."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            header = app.query_one(Header)
            footer = app.query_one(Footer)
            assert header is not None
            assert footer is not None

            # Verify application metadata wired to header
            assert app.TITLE == "GeneWeaver"
            assert app.SUB_TITLE == "GPU-Accelerated CRISPR Alignment Engine"

            # Verify keybindings present in footer
            keys = [b.key for b in app.BINDINGS]
            assert "q" in keys
            assert "d" in keys
            assert "h" in keys
    _run(_inner())


def test_integration_sidebar_and_main_content_structure() -> None:
    """Verify that the sidebar and main content containers properly integrate all panels."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            main_container = app.query_one("#main-container", Container)
            sidebar = app.query_one("#sidebar", Container)
            main_content = app.query_one("#main-content", Container)

            assert main_container is not None
            assert sidebar is not None
            assert main_content is not None

            # Verify sidebar contains the 3 expected panels
            assert app.query_one("#project-status-panel", ProjectStatusPanel) is not None
            assert app.query_one("#gpu-status-panel", GpuStatusPanel) is not None
            assert app.query_one("#gpu-memory-panel", GpuMemoryPanel) is not None

            # Verify main content contains the 3 expected panels
            assert app.query_one("#processing-info-panel", ProcessingInfoPanel) is not None
            assert app.query_one("#results-panel", ResultsPanel) is not None
            assert app.query_one("#benchmark-results-panel", BenchmarkResultsPanel) is not None
    _run(_inner())


def test_integration_scan_status_and_progress_bar() -> None:
    """Verify that scan status and progress bar work together cleanly in the application."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            status_lbl = app.query_one("#scan-status", Label)
            prog_lbl = app.query_one("#label-progress", Label)
            prog_bar = app.query_one("#alignment-progress", ProgressBar)

            # Initially Not started and 0%
            assert "Not started" in str(status_lbl.render())
            assert "0%" in str(prog_lbl.render())
            assert prog_bar.progress == 0

            # Scan start
            app.update_scan_status("Ready")
            assert "Ready" in str(status_lbl.render())

            app.update_scan_status("Scanning")
            assert "Scanning" in str(status_lbl.render())

            # Progress increments
            app.update_scan_progress(45)
            assert "45%" in str(prog_lbl.render())
            assert prog_bar.progress == 45
            assert "Scanning" in str(status_lbl.render())

            # Completed
            app.update_scan_status("Completed")
            app.update_scan_progress(100)
            assert "Completed" in str(status_lbl.render())
            assert "100%" in str(prog_lbl.render())
            assert prog_bar.progress == 100
    _run(_inner())


def test_integration_alignment_results_table() -> None:
    """Verify alignment results DataTable integration (columns, rows, clearing)."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            table = app.query_one("#alignment-results-table", DataTable)
            assert table is not None
            assert len(table.columns) == 5
            expected_headers = {
                "Record ID",
                "Start Position",
                "End Position",
                "Matched Sequence",
                "Mismatch Count",
            }
            actual_headers = {str(col.label) for col in table.columns.values()}
            assert actual_headers == expected_headers

            # Populate table
            app.update_alignment_results(SAMPLE_ALIGNMENT_BATCH_A)
            assert table.row_count == len(SAMPLE_ALIGNMENT_BATCH_A)

            # Clear table
            app.update_alignment_results([])
            assert table.row_count == 0
    _run(_inner())


def test_integration_gpu_status() -> None:
    """Verify GPU status panel integration (checking, available with device, unavailable, error)."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            status_w = app.query_one("#gpu-status", Label)
            device_w = app.query_one("#gpu-device", Label)

            # Checking
            app.update_gpu_status("Checking")
            assert "Checking" in str(status_w.render())

            # Available with device
            app.update_gpu_status("Available", device_name="Test NVIDIA GPU")
            assert "Available" in str(status_w.render())
            assert "Test NVIDIA GPU" in str(device_w.render())

            # Unavailable
            app.update_gpu_status("Unavailable")
            assert "Unavailable" in str(status_w.render())
            assert "Not detected" in str(device_w.render())

            # Error
            app.update_gpu_status("Error")
            assert "Error" in str(status_w.render())
            assert "Not detected" in str(device_w.render())
    _run(_inner())


def test_integration_gpu_memory() -> None:
    """Verify GPU memory panel integration (available, used calculation, unit formatting, unavailable)."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            status_w = app.query_one("#gpu-memory-status", Label)
            total_w = app.query_one("#gpu-memory-total", Label)
            used_w = app.query_one("#gpu-memory-used", Label)
            free_w = app.query_one("#gpu-memory-free", Label)

            # Mock 8 GB total, 6 GB free -> 2 GB used
            app.update_gpu_memory(6 * 1024**3, 8 * 1024**3)
            assert "Available" in str(status_w.render())
            assert "8.00 GB" in str(total_w.render())
            assert "2.00 GB" in str(used_w.render())
            assert "6.00 GB" in str(free_w.render())

            # Mock unavailable
            app.update_gpu_memory(None, None)
            assert "Unavailable" in str(status_w.render())
            assert "Not available" in str(total_w.render())
    _run(_inner())


def test_integration_benchmark_results() -> None:
    """Verify benchmark results panel integration (CPU, GPU, speedup, status)."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            cpu_w = app.query_one("#cpu-benchmark-time", Label)
            gpu_w = app.query_one("#gpu-benchmark-time", Label)
            spd_w = app.query_one("#gpu-benchmark-speedup", Label)
            sta_w = app.query_one("#benchmark-status", Label)

            # CPU only
            app.update_benchmark_results(SAMPLE_BENCH_CPU_ONLY)
            assert "450.000 ms" in str(cpu_w.render())
            assert "GPU Unavailable" in str(gpu_w.render())
            assert "GPU Unavailable" in str(spd_w.render())

            # With GPU
            app.update_benchmark_results(SAMPLE_BENCH_GPU_AVAILABLE)
            assert "320.000 ms" in str(cpu_w.render())
            assert "40.000 ms" in str(gpu_w.render())
            assert "8.00x" in str(spd_w.render())
    _run(_inner())


def test_integration_consolidated_gpu_diagnostics() -> None:
    """Verify consolidated GPU diagnostics integration updates both status and memory sections."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            # Set available diagnostics
            app.update_gpu_diagnostics(SAMPLE_GPU_DIAG_AVAILABLE)
            assert "Available" in str(app.query_one("#gpu-status", Label).render())
            assert "Test NVIDIA RTX 4090" in str(app.query_one("#gpu-device", Label).render())
            assert "Available" in str(app.query_one("#gpu-memory-status", Label).render())
            assert "24.00 GB" in str(app.query_one("#gpu-memory-total", Label).render())
            assert "18.00 GB" in str(app.query_one("#gpu-memory-free", Label).render())
            assert "6.00 GB" in str(app.query_one("#gpu-memory-used", Label).render())

            # Transition to unavailable diagnostics
            app.update_gpu_diagnostics(SAMPLE_GPU_DIAG_UNAVAILABLE)
            assert "Unavailable" in str(app.query_one("#gpu-status", Label).render())
            assert "Not detected" in str(app.query_one("#gpu-device", Label).render())
            assert "Unavailable" in str(app.query_one("#gpu-memory-status", Label).render())
            assert "Not available" in str(app.query_one("#gpu-memory-total", Label).render())
    _run(_inner())


def test_integration_all_sections_queryable_without_collision() -> None:
    """Verify that every dashboard section's child widgets are queryable without ID collisions."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            # Project status widgets
            assert app.query_one("#scan-status", Label) is not None
            assert app.query_one("#status-project", Label) is not None

            # Processing info widgets
            assert app.query_one("#label-progress", Label) is not None
            assert app.query_one("#alignment-progress", ProgressBar) is not None

            # Alignment results widgets
            assert app.query_one("#alignment-results-table", DataTable) is not None
            assert app.query_one("#demo-data-label", Label) is not None

            # GPU status widgets
            assert app.query_one("#gpu-status", Label) is not None
            assert app.query_one("#gpu-device", Label) is not None

            # GPU memory widgets
            assert app.query_one("#gpu-memory-status", Label) is not None
            assert app.query_one("#gpu-memory-total", Label) is not None
            assert app.query_one("#gpu-memory-used", Label) is not None
            assert app.query_one("#gpu-memory-free", Label) is not None

            # Benchmark results widgets
            assert app.query_one("#cpu-benchmark-time", Label) is not None
            assert app.query_one("#gpu-benchmark-time", Label) is not None
            assert app.query_one("#gpu-benchmark-speedup", Label) is not None
            assert app.query_one("#benchmark-status", Label) is not None
    _run(_inner())


# =============================================================================
# 2. Integration Scenario A – Scan Progress Lifecycle
# Sequence: Not started -> Scanning -> Update progress -> Completed
# =============================================================================

def test_scenario_a_scan_progress_lifecycle_unmounted() -> None:
    """Scenario A: Test scan progress transitions on unmounted app."""
    app = GeneWeaverApp()

    # Step 1: Initial state - Not started, 0%
    assert app.scan_status == "Not started"
    assert app.scan_progress == 0
    assert "Not started" in str(app.query_one("#scan-status").render())
    assert "0%" in str(app.query_one("#label-progress").render())
    assert app.query_one("#alignment-progress", ProgressBar).progress == 0

    # Step 2: Transition to Scanning
    app.update_scan_status("Scanning")
    assert app.scan_status == "Scanning"
    assert "Scanning" in str(app.query_one("#scan-status").render())
    assert app.scan_progress == 0

    # Step 3: Incremental progress updates
    for pct in [10, 35, 70, 95]:
        app.update_scan_progress(pct)
        assert app.scan_progress == pct
        assert f"{pct}%" in str(app.query_one("#label-progress").render())
        assert app.query_one("#alignment-progress", ProgressBar).progress == pct
        assert app.scan_status == "Scanning"

    # Step 4: Transition to Completed
    app.update_scan_status("Completed")
    app.update_scan_progress(100)
    assert app.scan_status == "Completed"
    assert app.scan_progress == 100
    assert "Completed" in str(app.query_one("#scan-status").render())
    assert "100%" in str(app.query_one("#label-progress").render())
    assert app.query_one("#alignment-progress", ProgressBar).progress == 100


def test_scenario_a_scan_progress_lifecycle_running_app() -> None:
    """Scenario A: Test scan progress transitions in a live running Textual application."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            status_w = app.query_one("#scan-status", Label)
            progress_lbl = app.query_one("#label-progress", Label)
            progress_bar = app.query_one("#alignment-progress", ProgressBar)

            # Initial
            assert "Not started" in str(status_w.render())
            assert progress_bar.progress == 0

            # Scanning
            app.update_scan_status("Scanning")
            assert "Scanning" in str(status_w.render())

            # Progress update
            app.update_scan_progress(50)
            assert app.scan_progress == 50
            assert "50%" in str(progress_lbl.render())
            assert progress_bar.progress == 50

            # Completed
            app.update_scan_status("Completed")
            app.update_scan_progress(100)
            assert "Completed" in str(status_w.render())
            assert "100%" in str(progress_lbl.render())
            assert progress_bar.progress == 100
    _run(_inner())


# =============================================================================
# 3. Integration Scenario B – Alignment Results Lifecycle
# Sequence: Multiple results -> Update with new -> Verify old cleared -> Empty set
# =============================================================================

def test_scenario_b_alignment_results_lifecycle() -> None:
    """Scenario B: Display results, replace with new, verify old cleared, verify empty set."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            table = app.query_one("#alignment-results-table", DataTable)

            # Step 1: Initial state - table is empty
            assert table.row_count == 0

            # Step 2: Display multiple alignment results (Batch A, 3 rows)
            app.update_alignment_results(SAMPLE_ALIGNMENT_BATCH_A)
            assert table.row_count == len(SAMPLE_ALIGNMENT_BATCH_A)

            row_keys_a = list(table.rows.keys())
            for idx, expected in enumerate(SAMPLE_ALIGNMENT_BATCH_A):
                row_vals = [str(table.get_cell(row_keys_a[idx], col)) for col in table.columns]
                assert expected["record_id"] in row_vals
                assert str(expected["start"]) in row_vals
                assert str(expected["end"]) in row_vals
                assert expected["matched_sequence"] in row_vals
                assert str(expected["mismatch_count"]) in row_vals

            # Step 3: Update table with new results (Batch B, 2 rows)
            app.update_alignment_results(SAMPLE_ALIGNMENT_BATCH_B)
            assert table.row_count == len(SAMPLE_ALIGNMENT_BATCH_B)

            # Step 4: Verify old rows are cleared (no Batch A IDs remain)
            row_keys_b = list(table.rows.keys())
            all_cells_b = [
                str(table.get_cell(rk, ck))
                for rk in row_keys_b
                for ck in table.columns
            ]
            for old_rec in SAMPLE_ALIGNMENT_BATCH_A:
                assert old_rec["record_id"] not in all_cells_b

            # Verify Batch B records are present
            for idx, expected in enumerate(SAMPLE_ALIGNMENT_BATCH_B):
                row_vals = [str(table.get_cell(row_keys_b[idx], col)) for col in table.columns]
                assert expected["record_id"] in row_vals
                assert str(expected["start"]) in row_vals

            # Step 5: Verify an empty result set clears the table correctly
            app.update_alignment_results([])
            assert table.row_count == 0
    _run(_inner())


# =============================================================================
# 4. Integration Scenario C – GPU Information Lifecycle
# Sequence: Available -> Display device & memory -> Unavailable -> Cleared info
# =============================================================================

def test_scenario_c_gpu_information_lifecycle_direct_methods() -> None:
    """Scenario C: Test GPU status and memory update methods directly."""
    app = GeneWeaverApp()

    # Step 1: Initial state - Not checked / Not available
    assert "Not checked" in str(app.query_one("#gpu-status").render())
    assert "Not detected" in str(app.query_one("#gpu-device").render())
    assert "Not checked" in str(app.query_one("#gpu-memory-status").render())
    assert "Not available" in str(app.query_one("#gpu-memory-total").render())

    # Step 2: Display GPU Available with device name and memory information
    total_bytes = 16 * 1024 * 1024 * 1024   # 16 GB test sample
    free_bytes = 12 * 1024 * 1024 * 1024    # 12 GB test sample
    app.update_gpu_status("Available", device_name="Test NVIDIA A100")
    app.update_gpu_memory(free_bytes, total_bytes)

    # Verify Available state
    assert "Available" in str(app.query_one("#gpu-status").render())
    assert "Test NVIDIA A100" in str(app.query_one("#gpu-device").render())
    assert "Available" in str(app.query_one("#gpu-memory-status").render())
    assert "16.00 GB" in str(app.query_one("#gpu-memory-total").render())
    assert "12.00 GB" in str(app.query_one("#gpu-memory-free").render())
    assert "4.00 GB" in str(app.query_one("#gpu-memory-used").render())

    # Step 3: Update status to Unavailable
    app.update_gpu_status("Unavailable")
    app.update_gpu_memory(None, None)

    # Step 4: Verify previous device and memory values are cleared
    assert "Unavailable" in str(app.query_one("#gpu-status").render())
    assert "Not detected" in str(app.query_one("#gpu-device").render())
    assert "Test NVIDIA A100" not in str(app.query_one("#gpu-device").render())

    assert "Unavailable" in str(app.query_one("#gpu-memory-status").render())
    assert "Not available" in str(app.query_one("#gpu-memory-total").render())
    assert "Not available" in str(app.query_one("#gpu-memory-used").render())
    assert "Not available" in str(app.query_one("#gpu-memory-free").render())
    assert "GB" not in str(app.query_one("#gpu-memory-total").render())


def test_scenario_c_gpu_information_lifecycle_via_diagnostics() -> None:
    """Scenario C: Test GPU status and memory update via consolidated update_gpu_diagnostics."""
    app = GeneWeaverApp()

    # Step 1: Available via diagnostics dictionary
    app.update_gpu_diagnostics(SAMPLE_GPU_DIAG_AVAILABLE)

    assert "Available" in str(app.query_one("#gpu-status").render())
    assert "Test NVIDIA RTX 4090" in str(app.query_one("#gpu-device").render())
    assert "24.00 GB" in str(app.query_one("#gpu-memory-total").render())
    assert "18.00 GB" in str(app.query_one("#gpu-memory-free").render())
    assert "6.00 GB" in str(app.query_one("#gpu-memory-used").render())

    # Step 2: Unavailable via diagnostics dictionary
    app.update_gpu_diagnostics(SAMPLE_GPU_DIAG_UNAVAILABLE)

    # Step 3: Verify all previous values cleared
    assert "Unavailable" in str(app.query_one("#gpu-status").render())
    assert "Not detected" in str(app.query_one("#gpu-device").render())
    assert "Test NVIDIA RTX 4090" not in str(app.query_one("#gpu-device").render())

    assert "Not available" in str(app.query_one("#gpu-memory-total").render())
    assert "Not available" in str(app.query_one("#gpu-memory-used").render())
    assert "Not available" in str(app.query_one("#gpu-memory-free").render())


# =============================================================================
# 5. Integration Scenario D – Benchmark Information
# Sequence: CPU only -> GPU available -> GPU unavailable -> Anti-fabrication check
# =============================================================================

def test_scenario_d_benchmark_information() -> None:
    """Scenario D: CPU benchmark, GPU available benchmark, GPU unavailable benchmark, no fake speedup."""
    app = GeneWeaverApp()

    # Step 1: Initial state - Not run / Not available
    assert "Not run" in str(app.query_one("#cpu-benchmark-time").render())
    assert "Not run" in str(app.query_one("#gpu-benchmark-time").render())
    assert "Not available" in str(app.query_one("#gpu-benchmark-speedup").render())
    assert "Not run" in str(app.query_one("#benchmark-status").render())

    # Step 2: Display CPU benchmark information (CPU only)
    app.update_benchmark_results(SAMPLE_BENCH_CPU_ONLY)
    cpu_render = str(app.query_one("#cpu-benchmark-time").render())
    gpu_render = str(app.query_one("#gpu-benchmark-time").render())
    spd_render = str(app.query_one("#gpu-benchmark-speedup").render())

    assert "450.000 ms" in cpu_render
    assert "GPU Unavailable" in gpu_render
    assert "GPU Unavailable" in spd_render

    # Step 3: Display GPU benchmark information when available
    app.update_benchmark_results(SAMPLE_BENCH_GPU_AVAILABLE)
    cpu_render2 = str(app.query_one("#cpu-benchmark-time").render())
    gpu_render2 = str(app.query_one("#gpu-benchmark-time").render())
    spd_render2 = str(app.query_one("#gpu-benchmark-speedup").render())
    status_render2 = str(app.query_one("#benchmark-status").render())

    assert "320.000 ms" in cpu_render2
    assert "40.000 ms" in gpu_render2
    assert "8.00x" in spd_render2
    assert "Comparative benchmark complete" in status_render2
    assert "Yes" in status_render2

    # Step 4: Display GPU unavailable when benchmarking was not performed on GPU
    app.update_benchmark_results(SAMPLE_BENCH_GPU_UNAVAILABLE)
    gpu_render3 = str(app.query_one("#gpu-benchmark-time").render())
    spd_render3 = str(app.query_one("#gpu-benchmark-speedup").render())

    assert "GPU Unavailable" in gpu_render3
    assert "GPU Unavailable" in spd_render3
    # Verify no fake numerical speedup or timing is displayed
    assert "ms" not in gpu_render3
    assert "x" not in spd_render3

    # Step 5: Verify missing GPU metrics when gpu_available is True still shows no fake speedup
    app.update_benchmark_results(SAMPLE_BENCH_MISSING_GPU_VALUES)
    gpu_render4 = str(app.query_one("#gpu-benchmark-time").render())
    spd_render4 = str(app.query_one("#gpu-benchmark-speedup").render())

    assert "Not run" in gpu_render4
    assert "Not available" in spd_render4
    assert "1.00x" not in spd_render4
    assert "0.00x" not in spd_render4


# =============================================================================
# 6. Integration Scenario E – Combined Dashboard Complete Integration
# Sequence: Full dashboard workflow updating all sections simultaneously and sequentially.
# =============================================================================

def test_scenario_e_combined_dashboard_workflow() -> None:
    """Scenario E: Multi-phase workflow updating all sections and verifying retention across updates."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            # Query all section widgets
            scan_status_w = app.query_one("#scan-status", Label)
            progress_lbl = app.query_one("#label-progress", Label)
            progress_bar = app.query_one("#alignment-progress", ProgressBar)
            table = app.query_one("#alignment-results-table", DataTable)
            gpu_status_w = app.query_one("#gpu-status", Label)
            gpu_device_w = app.query_one("#gpu-device", Label)
            gpu_mem_status_w = app.query_one("#gpu-memory-status", Label)
            gpu_mem_total_w = app.query_one("#gpu-memory-total", Label)
            gpu_mem_used_w = app.query_one("#gpu-memory-used", Label)
            gpu_mem_free_w = app.query_one("#gpu-memory-free", Label)
            cpu_bench_w = app.query_one("#cpu-benchmark-time", Label)
            gpu_bench_w = app.query_one("#gpu-benchmark-time", Label)
            gpu_speedup_w = app.query_one("#gpu-benchmark-speedup", Label)
            bench_status_w = app.query_one("#benchmark-status", Label)

            # Phase 1: Verify Initial Clean State across all sections
            assert "Not started" in str(scan_status_w.render())
            assert progress_bar.progress == 0
            assert table.row_count == 0
            assert "Not checked" in str(gpu_status_w.render())
            assert "Not checked" in str(gpu_mem_status_w.render())
            assert "Not run" in str(cpu_bench_w.render())

            # Phase 2: Start Scan
            app.update_scan_status("Scanning")
            app.update_scan_progress(20)
            assert "Scanning" in str(scan_status_w.render())
            assert progress_bar.progress == 20
            assert "20%" in str(progress_lbl.render())

            # Phase 3: Populate Alignment Results (Batch A)
            app.update_alignment_results(SAMPLE_ALIGNMENT_BATCH_A)
            assert table.row_count == 3

            # Phase 4: Configure GPU Diagnostics (Available)
            app.update_gpu_diagnostics(SAMPLE_GPU_DIAG_AVAILABLE)
            assert "Available" in str(gpu_status_w.render())
            assert "Test NVIDIA RTX 4090" in str(gpu_device_w.render())
            assert "24.00 GB" in str(gpu_mem_total_w.render())
            assert "6.00 GB" in str(gpu_mem_used_w.render())
            assert "18.00 GB" in str(gpu_mem_free_w.render())

            # Phase 5: Update Benchmark Results
            app.update_benchmark_results(SAMPLE_BENCH_GPU_AVAILABLE)
            assert "320.000 ms" in str(cpu_bench_w.render())
            assert "40.000 ms" in str(gpu_bench_w.render())
            assert "8.00x" in str(gpu_speedup_w.render())

            # Phase 6: Verify ALL sections retain their correct values simultaneously
            assert "Scanning" in str(scan_status_w.render())
            assert progress_bar.progress == 20
            assert table.row_count == 3
            assert "Available" in str(gpu_status_w.render())
            assert "Test NVIDIA RTX 4090" in str(gpu_device_w.render())
            assert "24.00 GB" in str(gpu_mem_total_w.render())
            assert "8.00x" in str(gpu_speedup_w.render())

            # Phase 7: Advance Scan Progress to 65% and refresh results with Batch B
            app.update_scan_progress(65)
            app.update_alignment_results(SAMPLE_ALIGNMENT_BATCH_B)

            assert progress_bar.progress == 65
            assert "65%" in str(progress_lbl.render())
            assert table.row_count == 2
            # Verify other panels remained unchanged during scan advance
            assert "Scanning" in str(scan_status_w.render())
            assert "Test NVIDIA RTX 4090" in str(gpu_device_w.render())
            assert "24.00 GB" in str(gpu_mem_total_w.render())
            assert "8.00x" in str(gpu_speedup_w.render())

            # Phase 8: Complete Scan and transition GPU to Unavailable
            app.update_scan_status("Completed")
            app.update_scan_progress(100)
            app.update_gpu_diagnostics(SAMPLE_GPU_DIAG_UNAVAILABLE)
            app.update_benchmark_results(SAMPLE_BENCH_GPU_UNAVAILABLE)

            # Phase 9: Verify Final Combined State
            assert "Completed" in str(scan_status_w.render())
            assert progress_bar.progress == 100
            assert "100%" in str(progress_lbl.render())
            assert table.row_count == 2
            assert "Unavailable" in str(gpu_status_w.render())
            assert "Not detected" in str(gpu_device_w.render())
            assert "Test NVIDIA RTX 4090" not in str(gpu_device_w.render())
            assert "Not available" in str(gpu_mem_total_w.render())
            assert "GPU Unavailable" in str(gpu_speedup_w.render())

    _run(_inner())


# =============================================================================
# 7. Additional Robustness & Isolation Integration Tests
# =============================================================================

def test_demo_mode_integration_preserves_independent_sections() -> None:
    """Verify loading demo results updates scan & table while leaving GPU & benchmark sections intact."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            # Set custom GPU and benchmark states prior to demo mode
            app.update_gpu_diagnostics(SAMPLE_GPU_DIAG_AVAILABLE)
            app.update_benchmark_results(SAMPLE_BENCH_GPU_AVAILABLE)

            # Load demo results
            app.load_demo_results()

            # Demo results updates scan and table
            assert app.scan_status == "Completed"
            assert app.scan_progress == 100
            assert app.is_demo_mode is True
            table = app.query_one("#alignment-results-table", DataTable)
            assert table.row_count == len(DEMO_ALIGNMENT_RESULTS)

            # Demo disclaimer label is visible
            demo_lbl = app.query_one("#demo-data-label", Label)
            assert demo_lbl.display is True
            assert DEMO_LABEL_TEXT in str(demo_lbl.render())

            # GPU diagnostics must be preserved (not overridden by demo mode)
            assert "Available" in str(app.query_one("#gpu-status").render())
            assert "Test NVIDIA RTX 4090" in str(app.query_one("#gpu-device").render())
            assert "24.00 GB" in str(app.query_one("#gpu-memory-total").render())

            # Benchmark results must be preserved
            assert "8.00x" in str(app.query_one("#gpu-benchmark-speedup").render())
    _run(_inner())


def test_order_independence_of_panel_updates() -> None:
    """Verify panel updates can occur in any arbitrary sequence without dependency errors."""
    async def _inner():
        app = GeneWeaverApp()
        async with app.run_test() as pilot:
            # Sequence: Benchmark -> GPU Diagnostics -> Table -> Progress -> Status
            app.update_benchmark_results(SAMPLE_BENCH_CPU_ONLY)
            app.update_gpu_diagnostics(SAMPLE_GPU_DIAG_AVAILABLE)
            app.update_alignment_results(SAMPLE_ALIGNMENT_BATCH_A)
            app.update_scan_progress(40)
            app.update_scan_status("Scanning")

            assert app.scan_status == "Scanning"
            assert app.scan_progress == 40
            assert "450.000 ms" in str(app.query_one("#cpu-benchmark-time").render())
            assert "Available" in str(app.query_one("#gpu-status").render())
            assert "Test NVIDIA RTX 4090" in str(app.query_one("#gpu-device").render())
            table = app.query_one("#alignment-results-table", DataTable)
            assert table.row_count == len(SAMPLE_ALIGNMENT_BATCH_A)
    _run(_inner())


def test_invalid_updates_do_not_corrupt_existing_valid_state() -> None:
    """Verify that a rejected update in one section does not corrupt state in other sections."""
    app = GeneWeaverApp()

    # Valid initial updates
    app.update_scan_status("Scanning")
    app.update_scan_progress(50)
    app.update_gpu_diagnostics(SAMPLE_GPU_DIAG_AVAILABLE)

    # Attempt invalid scan status update -> raises ValueError
    with pytest.raises(ValueError):
        app.update_scan_status("INVALID_STATUS")

    # Attempt invalid progress update -> raises ValueError
    with pytest.raises(ValueError):
        app.update_scan_progress(150)

    # Attempt invalid diagnostics update -> raises ValueError
    with pytest.raises(ValueError):
        app.update_gpu_diagnostics({"invalid": "dict"})

    # Verify previous valid state is completely preserved
    assert app.scan_status == "Scanning"
    assert app.scan_progress == 50
    assert "Available" in str(app.query_one("#gpu-status").render())
    assert "Test NVIDIA RTX 4090" in str(app.query_one("#gpu-device").render())
    assert "24.00 GB" in str(app.query_one("#gpu-memory-total").render())
