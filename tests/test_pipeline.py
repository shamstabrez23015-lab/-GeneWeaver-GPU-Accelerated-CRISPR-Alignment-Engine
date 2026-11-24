"""Unit tests for the GeneWeaver end-to-end pipeline."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path when running pytest directly without python -m
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from src.models import AlignmentMatch
from src.pipeline import run_geneweaver_pipeline


@pytest.fixture
def sample_fasta_file(tmp_path):
    """Create a temporary valid FASTA file."""
    fasta = tmp_path / "sample.fasta"
    fasta.write_text(">seq1 Test chromosome 1\nATGCATGCATGC\n>seq2 Test chromosome 2\nACGCACGC\n")
    return str(fasta)


@pytest.fixture
def empty_fasta_file(tmp_path):
    """Create an empty FASTA file."""
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("")
    return str(fasta)


def test_pipeline_cpu_only(sample_fasta_file):
    """Test running the pipeline with use_gpu=False (CPU-only execution)."""
    result = run_geneweaver_pipeline(
        fasta_path=sample_fasta_file,
        guide_sequence="ATGC",
        max_mismatches=0,
        use_gpu=False,
    )

    assert result["status"] == "Completed"
    assert len(result["records"]) == 2
    assert result["gpu_results"] is None
    assert result["benchmark_results"] is None
    assert "GPU execution skipped by user request" in result["message"]

    # Verify CPU results
    assert len(result["cpu_results"]) == 2
    assert result["cpu_results"][0]["record_id"] == "seq1"
    assert len(result["cpu_results"][0]["matches"]) == 3


def test_pipeline_mocked_gpu_success(sample_fasta_file):
    """Test successful pipeline execution when GPU is mocked and available."""
    mock_matches = [
        AlignmentMatch(start=0, end=4, matched_sequence="ATGC", mismatch_count=0, mismatch_positions=[])
    ]
    mock_gpu_diag = {
        "available": True,
        "device_name": "NVIDIA GeForce RTX 4090",
        "free_bytes": 20_000_000_000,
        "total_bytes": 24_000_000_000,
        "used_bytes": 4_000_000_000,
        "message": "Device operational.",
    }
    mock_bench = {
        "cpu_average_seconds": 0.05,
        "gpu_average_seconds": 0.005,
        "speedup": 10.0,
        "cpu_match_count": 3,
        "gpu_match_count": 3,
        "results_match": True,
        "gpu_available": True,
        "message": "Benchmark succeeded.",
    }

    with patch("src.pipeline.get_cuda_diagnostics", return_value=mock_gpu_diag), \
         patch("src.pipeline.is_cuda_available", return_value=True), \
         patch("src.pipeline.find_off_targets_gpu_chunked", return_value=mock_matches), \
         patch("src.pipeline.compare_alignment_performance", return_value=mock_bench):

        result = run_geneweaver_pipeline(
            fasta_path=sample_fasta_file,
            guide_sequence="ATGC",
            max_mismatches=0,
            use_gpu=True,
        )

    assert result["status"] == "Completed"
    assert result["gpu_results"] is not None
    assert len(result["gpu_results"]) == 2
    assert result["benchmark_results"] is not None
    assert result["benchmark_results"]["speedup"] == 10.0
    assert "GPU alignment completed successfully" in result["message"]


def test_pipeline_gpu_unavailable(sample_fasta_file):
    """Test pipeline execution when GPU is requested but unavailable on the host."""
    mock_gpu_diag = {
        "available": False,
        "device_name": None,
        "free_bytes": None,
        "total_bytes": None,
        "used_bytes": None,
        "message": "No CUDA device detected.",
    }

    with patch("src.pipeline.get_cuda_diagnostics", return_value=mock_gpu_diag), \
         patch("src.pipeline.is_cuda_available", return_value=False):

        result = run_geneweaver_pipeline(
            fasta_path=sample_fasta_file,
            guide_sequence="ATGC",
            use_gpu=True,
        )

    assert result["status"] == "Completed"
    assert result["gpu_results"] is None
    assert result["benchmark_results"] is None
    assert "GPU execution unavailable" in result["message"]


def test_pipeline_invalid_guide_sequence(sample_fasta_file):
    """Test that invalid DNA guide sequences raise ValueError."""
    with pytest.raises(ValueError) as exc:
        run_geneweaver_pipeline(sample_fasta_file, "ATGCX")
    assert "Invalid DNA character" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        run_geneweaver_pipeline(sample_fasta_file, "")
    assert "Guide sequence cannot be empty" in str(exc.value)


def test_pipeline_invalid_fasta_input():
    """Test that non-existent FASTA paths raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        run_geneweaver_pipeline("non_existent_file.fasta", "ATGC")


def test_pipeline_empty_fasta_file(empty_fasta_file):
    """Test that an empty FASTA file raises ValueError consistently with genomic_parser."""
    with pytest.raises(ValueError) as exc:
        run_geneweaver_pipeline(empty_fasta_file, "ATGC")
    assert "No FASTA records found" in str(exc.value)


def test_pipeline_memory_estimate_included(sample_fasta_file):
    """Test that memory audit estimates are calculated and included in pipeline results."""
    result = run_geneweaver_pipeline(
        fasta_path=sample_fasta_file,
        guide_sequence="ATGC",
        chunk_size=10_000,
        use_gpu=False,
    )

    mem = result["memory_estimate"]
    assert mem["genome_length"] == 20  # 12 + 8 bases
    assert mem["guide_length"] == 4
    assert mem["chunk_size"] == 10_000
    assert mem["genome_input_bytes"] == 20
    assert mem["guide_input_bytes"] == 4
    assert mem["unit"] == "bytes"
    assert "estimates" in mem["note"].lower()


def test_pipeline_system_report_included(sample_fasta_file):
    """Test that the consolidated system report is generated in the pipeline output."""
    result = run_geneweaver_pipeline(
        fasta_path=sample_fasta_file,
        guide_sequence="ATGC",
        use_gpu=False,
    )

    sys_rep = result["system_report"]
    assert "overall_system_status" in sys_rep
    assert "memory_estimation_summary" in sys_rep
    assert "gpu_diagnostics_summary" in sys_rep


def test_pipeline_benchmark_skipped_when_gpu_unavailable(sample_fasta_file):
    """Test that benchmark comparison is skipped when GPU execution is not performed."""
    with patch("src.pipeline.get_cuda_diagnostics", return_value={"available": False}), \
         patch("src.pipeline.is_cuda_available", return_value=False), \
         patch("src.pipeline.compare_alignment_performance") as mock_bench_fn:

        result = run_geneweaver_pipeline(
            fasta_path=sample_fasta_file,
            guide_sequence="ATGC",
            use_gpu=True,
        )

        assert result["benchmark_results"] is None
        mock_bench_fn.assert_not_called()
        assert "Benchmark comparison skipped" in result["message"]


def test_pipeline_gpu_execution_error_handling(sample_fasta_file):
    """Test that an unexpected GPU execution error falls back to CPU without crashing."""
    mock_gpu_diag = {"available": True, "device_name": "Tesla V100"}

    def _faulty_gpu_kernel(*args, **kwargs):
        raise RuntimeError("CUDA kernel execution failed with launch failure")

    with patch("src.pipeline.get_cuda_diagnostics", return_value=mock_gpu_diag), \
         patch("src.pipeline.is_cuda_available", return_value=True), \
         patch("src.pipeline.find_off_targets_gpu_chunked", side_effect=_faulty_gpu_kernel):

        result = run_geneweaver_pipeline(
            fasta_path=sample_fasta_file,
            guide_sequence="ATGC",
            use_gpu=True,
        )

    assert result["status"] == "Completed"
    assert result["gpu_results"] is None
    assert result["benchmark_results"] is None
    assert "fell back to CPU results" in result["message"]
    # CPU results should still be valid
    assert len(result["cpu_results"]) == 2


def test_pipeline_required_result_fields(sample_fasta_file):
    """Test that the pipeline return dictionary conforms to all expected contracts."""
    required_keys = {
        "records",
        "cpu_results",
        "gpu_results",
        "benchmark_results",
        "memory_estimate",
        "gpu_diagnostics",
        "system_report",
        "status",
        "message",
    }
    result = run_geneweaver_pipeline(
        fasta_path=sample_fasta_file,
        guide_sequence="ATGC",
        use_gpu=False,
    )
    assert required_keys.issubset(result.keys())
