"""Unit tests for system reporting and integration status in GeneWeaver."""

import copy
import sys
from pathlib import Path

# Ensure project root is in sys.path when running pytest directly without python -m
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from src.system_report import generate_system_report


@pytest.fixture
def sample_benchmark_results():
    return {
        "cpu_average_seconds": 0.05,
        "gpu_average_seconds": 0.01,
        "speedup": 5.0,
        "cpu_match_count": 2,
        "gpu_match_count": 2,
        "results_match": True,
        "gpu_available": True,
        "message": "Benchmark succeeded.",
    }


@pytest.fixture
def sample_memory_estimate():
    return {
        "genome_length": 100_000,
        "guide_length": 20,
        "genome_input_bytes": 100_000,
        "guide_input_bytes": 20,
        "chunk_buffer_bytes": 100_019,
        "estimated_peak_device_input_bytes": 100_039,
        "note": "Theoretical estimate based on sequence lengths.",
    }


@pytest.fixture
def sample_gpu_diagnostics():
    return {
        "available": True,
        "device_name": "NVIDIA GeForce RTX 4090",
        "free_bytes": 20_000_000_000,
        "total_bytes": 24_000_000_000,
        "used_bytes": 4_000_000_000,
        "message": "Device operational.",
    }


def test_report_all_sections_supplied(sample_benchmark_results, sample_memory_estimate, sample_gpu_diagnostics):
    """Test report generation when all three subsystem dictionaries are provided."""
    report = generate_system_report(
        benchmark_results=sample_benchmark_results,
        memory_estimate=sample_memory_estimate,
        gpu_diagnostics=sample_gpu_diagnostics,
    )

    assert report["benchmark_summary"] is not None
    assert report["memory_estimation_summary"] is not None
    assert report["gpu_diagnostics_summary"] is not None

    status = report["overall_system_status"]
    assert status["ready"] is True
    assert status["status"] == "ready"


def test_report_benchmark_only(sample_benchmark_results):
    """Test report when only benchmark results are supplied."""
    report = generate_system_report(benchmark_results=sample_benchmark_results)

    assert report["benchmark_summary"] is not None
    assert report["memory_estimation_summary"] is None
    assert report["gpu_diagnostics_summary"] is None
    assert report["overall_system_status"]["ready"] is False
    assert "GPU hardware diagnostics are not available" in report["overall_system_status"]["description"]


def test_report_memory_only(sample_memory_estimate):
    """Test report when only memory estimation is supplied."""
    report = generate_system_report(memory_estimate=sample_memory_estimate)

    assert report["benchmark_summary"] is None
    assert report["memory_estimation_summary"] is not None
    assert report["gpu_diagnostics_summary"] is None
    assert report["overall_system_status"]["ready"] is False


def test_report_gpu_diagnostics_only(sample_gpu_diagnostics):
    """Test report when only GPU diagnostics are supplied."""
    report = generate_system_report(gpu_diagnostics=sample_gpu_diagnostics)

    assert report["benchmark_summary"] is None
    assert report["memory_estimation_summary"] is None
    assert report["gpu_diagnostics_summary"] is not None
    assert report["overall_system_status"]["ready"] is False


def test_missing_optional_sections():
    """Test calling generate_system_report without arguments (all sections None)."""
    report = generate_system_report()

    assert report["benchmark_summary"] is None
    assert report["memory_estimation_summary"] is None
    assert report["gpu_diagnostics_summary"] is None
    assert report["overall_system_status"]["ready"] is False
    assert report["overall_system_status"]["status"] == "unconfigured"


def test_gpu_unavailable_case():
    """Test report handling when GPU is unavailable."""
    gpu_diag_unavail = {
        "available": False,
        "device_name": None,
        "free_bytes": None,
        "total_bytes": None,
        "used_bytes": None,
        "message": "No CUDA device detected.",
    }
    bench_unavail = {
        "cpu_average_seconds": 0.05,
        "gpu_average_seconds": None,
        "speedup": None,
        "cpu_match_count": 1,
        "gpu_match_count": None,
        "results_match": None,
        "gpu_available": False,
        "message": "Only CPU benchmarked.",
    }

    report = generate_system_report(
        benchmark_results=bench_unavail,
        gpu_diagnostics=gpu_diag_unavail,
    )

    assert report["gpu_diagnostics_summary"]["available"] is False
    assert report["gpu_diagnostics_summary"]["device_name"] is None
    assert report["benchmark_summary"]["speedup"] is None
    assert report["overall_system_status"]["ready"] is False
    assert report["overall_system_status"]["status"] == "cpu_only"
    assert "CPU-only" in report["overall_system_status"]["description"]


def test_correct_benchmark_speedup_reporting():
    """Test that speedup is included only when valid GPU execution occurred."""
    bench_with_speedup = {
        "cpu_average_seconds": 0.10,
        "gpu_average_seconds": 0.02,
        "speedup": 5.0,
        "cpu_match_count": 3,
        "gpu_match_count": 3,
        "results_match": True,
        "gpu_available": True,
    }
    rep_speedup = generate_system_report(benchmark_results=bench_with_speedup)
    assert rep_speedup["benchmark_summary"]["speedup"] == 5.0

    # If speedup is present in input but gpu_available is False, speedup must be suppressed to None
    bench_false_speedup = {
        "cpu_average_seconds": 0.10,
        "gpu_average_seconds": None,
        "speedup": 5.0,
        "cpu_match_count": 3,
        "gpu_match_count": None,
        "results_match": None,
        "gpu_available": False,
    }
    rep_suppressed = generate_system_report(benchmark_results=bench_false_speedup)
    assert rep_suppressed["benchmark_summary"]["speedup"] is None


def test_correct_memory_estimate_labelling(sample_memory_estimate):
    """Test that memory estimation is explicitly labelled as an estimate."""
    report = generate_system_report(memory_estimate=sample_memory_estimate)
    mem_summary = report["memory_estimation_summary"]

    assert mem_summary["is_estimate"] is True
    assert "estimate" in mem_summary["note"].lower()


def test_cpu_gpu_result_mismatch():
    """Test that a mismatch between CPU and GPU alignment results is flagged and unreadied."""
    bench_mismatch = {
        "cpu_average_seconds": 0.05,
        "gpu_average_seconds": 0.01,
        "speedup": 5.0,
        "cpu_match_count": 2,
        "gpu_match_count": 1,
        "results_match": False,
        "gpu_available": True,
    }
    gpu_diag = {"available": True, "device_name": "RTX 4090"}

    report = generate_system_report(benchmark_results=bench_mismatch, gpu_diagnostics=gpu_diag)

    assert report["benchmark_summary"]["results_match"] is False
    assert report["overall_system_status"]["ready"] is False
    assert report["overall_system_status"]["status"] == "mismatch_detected"
    assert "mismatch" in report["overall_system_status"]["description"].lower()


def test_invalid_input_dictionaries():
    """Test that non-dictionary inputs raise a clear ValueError."""
    with pytest.raises(ValueError) as exc:
        generate_system_report(benchmark_results="invalid_string")
    assert "benchmark_results must be a dict" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        generate_system_report(memory_estimate=[1, 2, 3])
    assert "memory_estimate must be a dict" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        generate_system_report(gpu_diagnostics=42)
    assert "gpu_diagnostics must be a dict" in str(exc.value)


def test_negative_timing_or_memory_values():
    """Test that negative timing, count, or memory values raise ValueError."""
    with pytest.raises(ValueError) as exc:
        generate_system_report(benchmark_results={"cpu_average_seconds": -0.01, "cpu_match_count": 0})
    assert "non-negative" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        generate_system_report(benchmark_results={"cpu_average_seconds": 0.01, "cpu_match_count": -5})
    assert "non-negative" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        generate_system_report(memory_estimate={"genome_length": -100})
    assert "non-negative" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        generate_system_report(gpu_diagnostics={"available": True, "free_bytes": -1000})
    assert "non-negative" in str(exc.value)


def test_original_input_not_modified(sample_benchmark_results, sample_memory_estimate, sample_gpu_diagnostics):
    """Test that input dictionaries are not modified by report generation."""
    bench_copy = copy.deepcopy(sample_benchmark_results)
    mem_copy = copy.deepcopy(sample_memory_estimate)
    gpu_copy = copy.deepcopy(sample_gpu_diagnostics)

    generate_system_report(
        benchmark_results=sample_benchmark_results,
        memory_estimate=sample_memory_estimate,
        gpu_diagnostics=sample_gpu_diagnostics,
    )

    assert sample_benchmark_results == bench_copy
    assert sample_memory_estimate == mem_copy
    assert sample_gpu_diagnostics == gpu_copy
