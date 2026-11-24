"""Unit tests for CPU vs GPU alignment benchmark comparison in GeneWeaver."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path when running pytest directly without python -m
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from src.benchmark_comparison import compare_alignment_performance
from src.cpu_alignment import find_off_targets
from src.models import AlignmentMatch


def test_cpu_benchmark_execution():
    """Test standard benchmark execution with CPU function alone."""
    with patch("src.benchmark_comparison.get_gpu_info", return_value={"available": False, "message": "No GPU"}):
        res = compare_alignment_performance(
            genome_sequence="ATGCATGC",
            guide_sequence="ATGC",
            cpu_function=find_off_targets,
            gpu_function=None,
            max_mismatches=0,
            repeats=2,
        )

    assert res["cpu_average_seconds"] >= 0.0
    assert res["cpu_match_count"] == 2
    assert res["gpu_available"] is False
    assert res["gpu_average_seconds"] is None
    assert res["speedup"] is None
    assert res["gpu_match_count"] is None
    assert res["results_match"] is None
    assert "GPU is unavailable" in res["message"]


def test_gpu_benchmark_execution_mocked():
    """Test successful benchmark comparison when GPU is available and matches match."""
    expected_matches = [
        AlignmentMatch(start=0, end=4, matched_sequence="ATGC", mismatch_count=0, mismatch_positions=[])
    ]
    mock_cpu = MagicMock(return_value=expected_matches)
    mock_gpu = MagicMock(return_value=expected_matches)

    mock_gpu_info = {"available": True, "device_name": "Mock RTX 4090", "message": "CUDA available"}

    with patch("src.benchmark_comparison.get_gpu_info", return_value=mock_gpu_info):
        res = compare_alignment_performance(
            genome_sequence="ATGC",
            guide_sequence="ATGC",
            cpu_function=mock_cpu,
            gpu_function=mock_gpu,
            max_mismatches=0,
            repeats=2,
        )

    assert res["gpu_available"] is True
    assert res["cpu_match_count"] == 1
    assert res["gpu_match_count"] == 1
    assert res["results_match"] is True
    assert res["speedup"] is not None
    assert res["gpu_average_seconds"] >= 0.0
    assert "matching results" in res["message"]


def test_gpu_unavailable_case():
    """Test that when GPU is unavailable, GPU benchmarking is skipped and speedup is None."""
    mock_gpu_func = MagicMock()
    mock_gpu_info = {"available": False, "message": "CUDA driver not found"}

    with patch("src.benchmark_comparison.get_gpu_info", return_value=mock_gpu_info):
        res = compare_alignment_performance(
            genome_sequence="ATGC",
            guide_sequence="ATGC",
            cpu_function=find_off_targets,
            gpu_function=mock_gpu_func,
            max_mismatches=0,
            repeats=1,
        )

    assert res["gpu_available"] is False
    assert res["speedup"] is None
    assert res["gpu_average_seconds"] is None
    mock_gpu_func.assert_not_called()


def test_correct_speedup_calculation():
    """Test that speedup is accurately calculated as cpu_average / gpu_average."""
    match = [AlignmentMatch(start=0, end=4, matched_sequence="ATGC", mismatch_count=0, mismatch_positions=[])]
    mock_cpu = MagicMock(return_value=match)
    mock_gpu = MagicMock(return_value=match)

    mock_gpu_info = {"available": True, "message": "GPU OK"}

    # Simulate CPU taking 0.10s and GPU taking 0.02s (speedup = 5.0)
    # 1 repeat: perf_counter calls: cpu_start, cpu_end, gpu_start, gpu_end
    timing_sequence = [0.0, 0.10, 0.20, 0.22]
    with patch("src.benchmark_comparison.get_gpu_info", return_value=mock_gpu_info):
        with patch("time.perf_counter", side_effect=timing_sequence):
            res = compare_alignment_performance(
                genome_sequence="ATGC",
                guide_sequence="ATGC",
                cpu_function=mock_cpu,
                gpu_function=mock_gpu,
                repeats=1,
            )

    assert pytest.approx(res["cpu_average_seconds"], rel=1e-3) == 0.10
    assert pytest.approx(res["gpu_average_seconds"], rel=1e-3) == 0.02
    assert pytest.approx(res["speedup"], rel=1e-3) == 5.0


def test_cpu_and_gpu_result_comparison_matching():
    """Test results_match is True when both functions return identical matches."""
    matches = [
        AlignmentMatch(start=0, end=4, matched_sequence="ATGC", mismatch_count=0, mismatch_positions=[]),
        AlignmentMatch(start=4, end=8, matched_sequence="ACGC", mismatch_count=1, mismatch_positions=[1]),
    ]
    mock_cpu = MagicMock(return_value=matches)
    mock_gpu = MagicMock(return_value=list(matches))

    with patch("src.benchmark_comparison.get_gpu_info", return_value={"available": True}):
        res = compare_alignment_performance(
            genome_sequence="ATGCACGC",
            guide_sequence="ATGC",
            cpu_function=mock_cpu,
            gpu_function=mock_gpu,
            max_mismatches=1,
            repeats=1,
        )

    assert res["results_match"] is True
    assert res["cpu_match_count"] == 2
    assert res["gpu_match_count"] == 2


def test_mismatched_cpu_and_gpu_results():
    """Test detection and warning when GPU returns different results from CPU."""
    cpu_matches = [
        AlignmentMatch(start=0, end=4, matched_sequence="ATGC", mismatch_count=0, mismatch_positions=[])
    ]
    gpu_matches = []  # Discrepancy

    mock_cpu = MagicMock(return_value=cpu_matches)
    mock_gpu = MagicMock(return_value=gpu_matches)

    with patch("src.benchmark_comparison.get_gpu_info", return_value={"available": True}):
        res = compare_alignment_performance(
            genome_sequence="ATGC",
            guide_sequence="ATGC",
            cpu_function=mock_cpu,
            gpu_function=mock_gpu,
            repeats=1,
        )

    assert res["results_match"] is False
    assert res["cpu_match_count"] == 1
    assert res["gpu_match_count"] == 0
    assert "mismatched" in res["message"]


def test_invalid_repeat_count():
    """Test rejection of invalid repeat count arguments."""
    with pytest.raises(ValueError) as exc:
        compare_alignment_performance("ATGC", "ATGC", cpu_function=find_off_targets, repeats=0)
    assert "repeats must be at least 1" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        compare_alignment_performance("ATGC", "ATGC", cpu_function=find_off_targets, repeats="3")
    assert "repeats must be an integer" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        compare_alignment_performance("ATGC", "ATGC", cpu_function=find_off_targets, repeats=True)
    assert "repeats must be an integer" in str(exc.value)


def test_missing_cpu_function():
    """Test that missing or non-callable cpu_function raises ValueError."""
    with pytest.raises(ValueError) as exc:
        compare_alignment_performance("ATGC", "ATGC", cpu_function=None)
    assert "cpu_function must be provided" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        compare_alignment_performance("ATGC", "ATGC", cpu_function="not_a_function")
    assert "cpu_function must be provided" in str(exc.value)


def test_required_dictionary_fields():
    """Test that the returned dictionary contains all required fields."""
    expected_fields = {
        "cpu_average_seconds",
        "gpu_average_seconds",
        "speedup",
        "cpu_match_count",
        "gpu_match_count",
        "results_match",
        "gpu_available",
        "message",
    }

    # Test GPU unavailable return keys
    with patch("src.benchmark_comparison.get_gpu_info", return_value={"available": False}):
        res_unavail = compare_alignment_performance("ATGC", "ATGC", cpu_function=find_off_targets, repeats=1)
    assert expected_fields.issubset(res_unavail.keys())
    assert len(res_unavail) == len(expected_fields)

    # Test GPU available return keys
    mock_gpu_info = {"available": True}
    with patch("src.benchmark_comparison.get_gpu_info", return_value=mock_gpu_info):
        res_avail = compare_alignment_performance(
            "ATGC", "ATGC", cpu_function=find_off_targets, gpu_function=find_off_targets, repeats=1
        )
    assert expected_fields.issubset(res_avail.keys())
    assert len(res_avail) == len(expected_fields)


def test_timing_values_non_negative():
    """Test that timing values are non-negative."""
    mock_gpu_info = {"available": True}
    with patch("src.benchmark_comparison.get_gpu_info", return_value=mock_gpu_info):
        res = compare_alignment_performance(
            "ATGCATGC", "ATGC", cpu_function=find_off_targets, gpu_function=find_off_targets, repeats=2
        )

    assert res["cpu_average_seconds"] >= 0.0
    assert res["gpu_average_seconds"] >= 0.0
    assert res["speedup"] >= 0.0


def test_gpu_execution_error_handling():
    """Test that an error inside gpu_function does not crash the comparison or report false speedup."""
    def faulty_gpu(*args, **kwargs):
        raise RuntimeError("CUDA out of memory error")

    mock_gpu_info = {"available": True}
    with patch("src.benchmark_comparison.get_gpu_info", return_value=mock_gpu_info):
        res = compare_alignment_performance(
            "ATGC", "ATGC", cpu_function=find_off_targets, gpu_function=faulty_gpu, repeats=1
        )

    assert res["gpu_available"] is True
    assert res["speedup"] is None
    assert res["gpu_average_seconds"] is None
    assert res["results_match"] is None
    assert "GPU execution error encountered" in res["message"]


def test_genome_shorter_than_guide():
    """Test handling when genome sequence is shorter than guide sequence."""
    res = compare_alignment_performance(
        "ATG", "ATGC", cpu_function=find_off_targets, repeats=1
    )
    assert res["cpu_match_count"] == 0


def test_invalid_dna_sequence_rejected():
    """Test that invalid DNA sequences are rejected with ValueError."""
    with pytest.raises(ValueError) as exc:
        compare_alignment_performance("ATGCX", "ATGC", cpu_function=find_off_targets)
    assert "Invalid DNA character" in str(exc.value)
