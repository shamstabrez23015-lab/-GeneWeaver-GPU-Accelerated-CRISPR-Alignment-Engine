"""Unit tests for alignment reporting and summarization in GeneWeaver."""

import copy
import sys
from pathlib import Path

# Ensure project root is in sys.path when running pytest directly without python -m
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from src.alignment_report import (
    flatten_alignment_results,
    summarize_alignment_results,
)
from src.models import AlignmentMatch


# ============================================================================
# Flattening Tests
# ============================================================================

def test_flatten_one_record():
    """Test flattening results from a single record with matches."""
    match1 = AlignmentMatch(
        start=0,
        end=4,
        matched_sequence="ATGC",
        mismatch_count=0,
        mismatch_positions=[],
    )
    match2 = AlignmentMatch(
        start=6,
        end=10,
        matched_sequence="ACGC",
        mismatch_count=1,
        mismatch_positions=[1],
    )
    results = [
        {
            "record_id": "chr1",
            "matches": [match1, match2],
        }
    ]

    flat = flatten_alignment_results(results)
    assert len(flat) == 2

    assert flat[0] == {
        "record_id": "chr1",
        "start": 0,
        "end": 4,
        "matched_sequence": "ATGC",
        "mismatch_count": 0,
        "mismatch_positions": [],
    }
    assert flat[1] == {
        "record_id": "chr1",
        "start": 6,
        "end": 10,
        "matched_sequence": "ACGC",
        "mismatch_count": 1,
        "mismatch_positions": [1],
    }


def test_flatten_multiple_records():
    """Test flattening multiple records into a combined flat list."""
    match_chr1 = AlignmentMatch(
        start=0,
        end=4,
        matched_sequence="ATGC",
        mismatch_count=0,
        mismatch_positions=[],
    )
    match_chr2 = AlignmentMatch(
        start=2,
        end=6,
        matched_sequence="AAGC",
        mismatch_count=1,
        mismatch_positions=[2],
    )
    results = [
        {"record_id": "chr1", "matches": [match_chr1]},
        {"record_id": "chr2", "matches": [match_chr2]},
    ]

    flat = flatten_alignment_results(results)
    assert len(flat) == 2
    assert flat[0]["record_id"] == "chr1"
    assert flat[0]["matched_sequence"] == "ATGC"
    assert flat[1]["record_id"] == "chr2"
    assert flat[1]["matched_sequence"] == "AAGC"


def test_flatten_preserve_order():
    """Test that the order of records and their respective matches is strictly preserved."""
    match_a1 = AlignmentMatch(start=0, end=4, matched_sequence="ATGC", mismatch_count=0, mismatch_positions=[])
    match_a2 = AlignmentMatch(start=8, end=12, matched_sequence="ATGC", mismatch_count=0, mismatch_positions=[])
    match_b1 = AlignmentMatch(start=1, end=5, matched_sequence="TTGC", mismatch_count=1, mismatch_positions=[0])

    results = [
        {"record_id": "chr_alpha", "matches": [match_a1, match_a2]},
        {"record_id": "chr_beta", "matches": [match_b1]},
    ]

    flat = flatten_alignment_results(results)
    assert len(flat) == 3
    assert (flat[0]["record_id"], flat[0]["start"]) == ("chr_alpha", 0)
    assert (flat[1]["record_id"], flat[1]["start"]) == ("chr_alpha", 8)
    assert (flat[2]["record_id"], flat[2]["start"]) == ("chr_beta", 1)


def test_flatten_zero_matches():
    """Test flattening records that have no alignment matches."""
    results_empty = [
        {"record_id": "chr_empty", "matches": []}
    ]
    assert flatten_alignment_results(results_empty) == []

    # Mixed records: one with matches, one without
    match1 = AlignmentMatch(start=0, end=4, matched_sequence="ATGC", mismatch_count=0, mismatch_positions=[])
    results_mixed = [
        {"record_id": "chr_active", "matches": [match1]},
        {"record_id": "chr_silent", "matches": []},
    ]
    flat_mixed = flatten_alignment_results(results_mixed)
    assert len(flat_mixed) == 1
    assert flat_mixed[0]["record_id"] == "chr_active"

    # Completely empty results list
    assert flatten_alignment_results([]) == []


# ============================================================================
# Summary Statistics Tests
# ============================================================================

def test_summarize_total_records():
    """Test accurate counting of total records processed."""
    results = [
        {"record_id": "chr1", "matches": []},
        {"record_id": "chr2", "matches": []},
        {"record_id": "chr3", "matches": []},
    ]
    summary = summarize_alignment_results(results)
    assert summary["total_records"] == 3

    assert summarize_alignment_results([])["total_records"] == 0


def test_summarize_records_with_matches():
    """Test counting only records that contain at least one match."""
    match1 = AlignmentMatch(start=0, end=4, matched_sequence="ATGC", mismatch_count=0, mismatch_positions=[])
    results = [
        {"record_id": "chr1", "matches": [match1]},
        {"record_id": "chr2", "matches": []},
        {"record_id": "chr3", "matches": [match1]},
    ]
    summary = summarize_alignment_results(results)
    assert summary["records_with_matches"] == 2
    assert summary["total_records"] == 3


def test_summarize_total_matches():
    """Test total match count summation across all records."""
    m1 = AlignmentMatch(start=0, end=4, matched_sequence="ATGC", mismatch_count=0, mismatch_positions=[])
    m2 = AlignmentMatch(start=5, end=9, matched_sequence="ATGC", mismatch_count=0, mismatch_positions=[])
    m3 = AlignmentMatch(start=10, end=14, matched_sequence="ATGC", mismatch_count=0, mismatch_positions=[])

    results = [
        {"record_id": "chr1", "matches": [m1, m2]},
        {"record_id": "chr2", "matches": []},
        {"record_id": "chr3", "matches": [m3]},
    ]
    summary = summarize_alignment_results(results)
    assert summary["total_matches"] == 3


def test_summarize_total_mismatches():
    """Test cumulative sum of mismatches across all alignment matches."""
    m1 = AlignmentMatch(start=0, end=4, matched_sequence="ATGC", mismatch_count=0, mismatch_positions=[])
    m2 = AlignmentMatch(start=5, end=9, matched_sequence="ACGC", mismatch_count=1, mismatch_positions=[1])
    m3 = AlignmentMatch(start=10, end=14, matched_sequence="ACTC", mismatch_count=2, mismatch_positions=[1, 2])

    results = [
        {"record_id": "chr1", "matches": [m1, m2]},
        {"record_id": "chr2", "matches": [m3]},
    ]
    summary = summarize_alignment_results(results)
    assert summary["total_mismatches"] == 3  # 0 + 1 + 2


def test_summarize_average_mismatches():
    """Test calculation of average mismatches per match."""
    m1 = AlignmentMatch(start=0, end=4, matched_sequence="ACGC", mismatch_count=1, mismatch_positions=[1])
    m2 = AlignmentMatch(start=5, end=9, matched_sequence="ACTC", mismatch_count=2, mismatch_positions=[1, 2])

    results = [
        {"record_id": "chr1", "matches": [m1, m2]},
    ]
    summary = summarize_alignment_results(results)
    # 3 total mismatches / 2 total matches = 1.5
    assert summary["average_mismatches_per_match"] == 1.5


def test_summarize_zero_match_average():
    """Test that average_mismatches_per_match returns 0.0 when there are no matches."""
    results_empty_records = [
        {"record_id": "chr1", "matches": []},
        {"record_id": "chr2", "matches": []},
    ]
    summary = summarize_alignment_results(results_empty_records)
    assert summary["total_matches"] == 0
    assert summary["total_mismatches"] == 0
    assert summary["average_mismatches_per_match"] == 0.0

    summary_empty_list = summarize_alignment_results([])
    assert summary_empty_list["total_records"] == 0
    assert summary_empty_list["total_matches"] == 0
    assert summary_empty_list["average_mismatches_per_match"] == 0.0


# ============================================================================
# Validation and Immutability Tests
# ============================================================================

def test_invalid_or_malformed_input():
    """Test handling of invalid input structures and malformed match objects."""
    # Non-list input
    with pytest.raises(ValueError) as exc:
        flatten_alignment_results("invalid_input")
    assert "must be a list" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        summarize_alignment_results("invalid_input")
    assert "must be a list" in str(exc.value)

    # Missing record_id
    with pytest.raises(ValueError) as exc:
        flatten_alignment_results([{"matches": []}])
    assert "record_id" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        summarize_alignment_results([{"matches": []}])
    assert "record_id" in str(exc.value)

    # Missing matches field
    with pytest.raises(ValueError) as exc:
        flatten_alignment_results([{"record_id": "chr1"}])
    assert "matches" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        summarize_alignment_results([{"record_id": "chr1"}])
    assert "matches" in str(exc.value)

    # Malformed match object (e.g., None, dict, string)
    with pytest.raises(ValueError) as exc:
        flatten_alignment_results([{"record_id": "chr1", "matches": [None]}])
    assert "Malformed match object" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        summarize_alignment_results([{"record_id": "chr1", "matches": ["not_a_match"]}])
    assert "Malformed match object" in str(exc.value)

    # Malformed object missing required fields
    class IncompleteMatch:
        start = 0

    with pytest.raises(ValueError) as exc:
        flatten_alignment_results([{"record_id": "chr1", "matches": [IncompleteMatch()]}])
    assert "Malformed match object" in str(exc.value)


def test_original_input_not_modified():
    """Test that flattening and summarizing do not mutate or alter the original input data."""
    match = AlignmentMatch(
        start=0,
        end=4,
        matched_sequence="ATGC",
        mismatch_count=1,
        mismatch_positions=[2],
    )
    original_results = [
        {"record_id": "chr1", "matches": [match]},
        {"record_id": "chr2", "matches": []},
    ]

    snapshot = copy.deepcopy(original_results)

    flat = flatten_alignment_results(original_results)
    assert original_results == snapshot

    # Modifying flat output must not affect original results or matches
    flat[0]["record_id"] = "modified_id"
    flat[0]["mismatch_positions"].append(99)
    assert original_results[0]["record_id"] == "chr1"
    assert original_results[0]["matches"][0].mismatch_positions == [2]

    # Summarize should also not modify input
    summarize_alignment_results(original_results)
    assert original_results == snapshot

