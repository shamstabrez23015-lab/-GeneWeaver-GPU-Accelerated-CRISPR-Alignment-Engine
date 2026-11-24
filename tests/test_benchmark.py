"""Unit tests for CPU alignment performance benchmarking in GeneWeaver."""

import sys
from pathlib import Path

# Ensure project root is in sys.path when running pytest directly without python -m
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from src.benchmark import benchmark_cpu_alignment


def test_benchmark_valid_sequence():
    """Test benchmarking a valid DNA sequence alignment."""
    genome = "ATGCATGCATGC"
    guide = "ATGC"
    result = benchmark_cpu_alignment(genome, guide, max_mismatches=0, repeats=2)

    assert isinstance(result, dict)
    assert result["genome_length"] == 12
    assert result["guide_length"] == 4
    assert result["repeats"] == 2


def test_correct_genome_and_guide_lengths():
    """Test that cleaned sequence lengths are accurately reported."""
    genome = "  a t g c   t g c a   a t g c  "  # 12 base pairs after cleaning
    guide = " a t g c "                     # 4 base pairs after cleaning
    result = benchmark_cpu_alignment(genome, guide, max_mismatches=0, repeats=1)

    assert result["genome_length"] == 12
    assert result["guide_length"] == 4


def test_correct_number_of_windows():
    """Test that the number of scanned windows is computed correctly."""
    genome = "ATGCATGCAT"  # length 10
    guide = "ATGC"        # length 4
    # Windows: 10 - 4 + 1 = 7
    result = benchmark_cpu_alignment(genome, guide, max_mismatches=0, repeats=1)
    assert result["windows_checked"] == 7


def test_correct_match_count():
    """Test that match_count accurately reports the number of matches found."""
    genome = "ATGCGGATGC"  # Matches at 0 and 6
    guide = "ATGC"
    result = benchmark_cpu_alignment(genome, guide, max_mismatches=0, repeats=2)
    assert result["match_count"] == 2

    # With mismatch threshold filtering
    result_mismatch = benchmark_cpu_alignment("ACGCGGATGC", "ATGC", max_mismatches=1, repeats=1)
    assert result_mismatch["match_count"] == 2


def test_multiple_repetitions():
    """Test benchmarking across multiple repetitions and verifying timing relationships."""
    genome = "ATGCATGC" * 10
    guide = "ATGC"
    result = benchmark_cpu_alignment(genome, guide, max_mismatches=1, repeats=4)

    assert result["repeats"] == 4
    assert result["min_seconds"] <= result["average_seconds"] <= result["max_seconds"]


def test_invalid_repeat_values():
    """Test that invalid repeats values raise a descriptive ValueError."""
    genome = "ATGCATGC"
    guide = "ATGC"

    # Zero or negative
    with pytest.raises(ValueError) as exc:
        benchmark_cpu_alignment(genome, guide, repeats=0)
    assert "repeats must be at least 1" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        benchmark_cpu_alignment(genome, guide, repeats=-3)
    assert "repeats must be at least 1" in str(exc.value)

    # Non-integer types
    with pytest.raises(ValueError) as exc:
        benchmark_cpu_alignment(genome, guide, repeats="3")
    assert "repeats must be an integer" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        benchmark_cpu_alignment(genome, guide, repeats=2.5)
    assert "repeats must be an integer" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        benchmark_cpu_alignment(genome, guide, repeats=True)
    assert "repeats must be an integer" in str(exc.value)


def test_genome_shorter_than_guide():
    """Test handling of genome sequences that are shorter than the guide."""
    genome = "ATG"
    guide = "ATGC"
    result = benchmark_cpu_alignment(genome, guide, max_mismatches=1, repeats=2)

    assert result["genome_length"] == 3
    assert result["guide_length"] == 4
    assert result["windows_checked"] == 0
    assert result["match_count"] == 0
    assert result["comparisons_per_second"] == 0.0


def test_returned_dictionary_fields():
    """Test that the returned dictionary contains all required benchmark fields."""
    expected_fields = {
        "genome_length",
        "guide_length",
        "windows_checked",
        "repeats",
        "average_seconds",
        "min_seconds",
        "max_seconds",
        "comparisons_per_second",
        "match_count",
    }
    result = benchmark_cpu_alignment("ATGCATGC", "ATGC", repeats=1)
    assert expected_fields.issubset(result.keys())
    assert len(result) == len(expected_fields)


def test_timing_values_non_negative():
    """Test that all timing and throughput metrics are non-negative."""
    result = benchmark_cpu_alignment("ATGCATGC", "ATGC", repeats=2)

    assert result["average_seconds"] >= 0.0
    assert result["min_seconds"] >= 0.0
    assert result["max_seconds"] >= 0.0
    assert result["comparisons_per_second"] >= 0.0


def test_invalid_dna_sequences_reused():
    """Test that existing DNA validation is reused to reject invalid sequence characters."""
    with pytest.raises(ValueError) as exc:
        benchmark_cpu_alignment("ATGCX", "ATGC")
    assert "Invalid DNA character" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        benchmark_cpu_alignment("ATGC", "ATG1")
    assert "Invalid DNA character" in str(exc.value)


def test_empty_guide_sequence_rejection():
    """Test that an empty guide sequence raises ValueError."""
    with pytest.raises(ValueError) as exc:
        benchmark_cpu_alignment("ATGC", "")
    assert "Guide sequence cannot be empty" in str(exc.value)
