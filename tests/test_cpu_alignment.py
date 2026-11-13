"""Unit tests for CPU alignment sequence validation and off-target search in GeneWeaver."""

import sys
from pathlib import Path

# Ensure project root is in sys.path when running pytest directly without python -m
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from src.cpu_alignment import (
    find_off_targets,
    find_off_targets_in_records,
    validate_dna_sequence,
)
from src.models import AlignmentMatch


# ============================================================================
# Day 1 Tests: Sequence Validation
# ============================================================================

def test_valid_dna_sequences():
    """Test that valid DNA sequences with canonical and ambiguous bases pass."""
    assert validate_dna_sequence("ATGC") == "ATGC"
    assert validate_dna_sequence("ATGCN") == "ATGCN"
    assert validate_dna_sequence("NNNN") == "NNNN"
    assert validate_dna_sequence("AAGTCGATCGTAGCTA") == "AAGTCGATCGTAGCTA"


def test_lowercase_dna_input():
    """Test that lowercase and mixed-case DNA sequences are converted to uppercase."""
    assert validate_dna_sequence("atgc") == "ATGC"
    assert validate_dna_sequence("atgcn") == "ATGCN"
    assert validate_dna_sequence("aTgCn") == "ATGCN"


def test_whitespace_removal():
    """Test that all whitespace (spaces, tabs, newlines) is stripped from input."""
    assert validate_dna_sequence("A T G C") == "ATGC"
    assert validate_dna_sequence("  ATGC  ") == "ATGC"
    assert validate_dna_sequence("ATGC\n\tNN\r\n") == "ATGCNN"
    assert validate_dna_sequence(" a t \n g \t c ") == "ATGC"


def test_invalid_dna_characters():
    """Test that invalid characters raise a descriptive ValueError."""
    with pytest.raises(ValueError) as exc_info:
        validate_dna_sequence("ATGCX")
    assert "Invalid DNA character(s) found" in str(exc_info.value)
    assert "'X'" in str(exc_info.value)

    # Test numbers and symbols
    with pytest.raises(ValueError) as exc_info:
        validate_dna_sequence("AT12GC")
    assert "Invalid DNA character(s) found" in str(exc_info.value)

    # Test invalid ambiguous IUPAC codes not in ATGCN (e.g., R, Y, S, W)
    with pytest.raises(ValueError) as exc_info:
        validate_dna_sequence("ATGCR")
    assert "Invalid DNA character(s) found" in str(exc_info.value)


def test_empty_sequence_handling():
    """Test that empty sequences or whitespace-only inputs are handled gracefully."""
    assert validate_dna_sequence("") == ""
    assert validate_dna_sequence("   ") == ""
    assert validate_dna_sequence("\n\t  \r\n") == ""


# ============================================================================
# Day 2 Tests: CPU Off-Target Alignment Search
# ============================================================================

def test_exact_match():
    """Test finding an exact match with 0 mismatches."""
    genome = "ATGC"
    guide = "ATGC"
    matches = find_off_targets(genome, guide, max_mismatches=0)
    assert len(matches) == 1
    assert matches[0].start == 0
    assert matches[0].end == 4
    assert matches[0].matched_sequence == "ATGC"
    assert matches[0].mismatch_count == 0
    assert matches[0].mismatch_positions == []


def test_one_mismatch():
    """Test finding a match with exactly one base mismatch."""
    genome = "ACGC"
    guide = "ATGC"
    matches = find_off_targets(genome, guide, max_mismatches=1)
    assert len(matches) == 1
    assert matches[0].start == 0
    assert matches[0].end == 4
    assert matches[0].matched_sequence == "ACGC"
    assert matches[0].mismatch_count == 1
    assert matches[0].mismatch_positions == [1]


def test_multiple_mismatches():
    """Test finding a match with multiple base mismatches."""
    genome = "ACGA"
    guide = "ATGC"
    matches = find_off_targets(genome, guide, max_mismatches=2)
    assert len(matches) == 1
    assert matches[0].start == 0
    assert matches[0].end == 4
    assert matches[0].matched_sequence == "ACGA"
    assert matches[0].mismatch_count == 2
    assert matches[0].mismatch_positions == [1, 3]


def test_max_allowed_mismatch_threshold():
    """Test that matches equal to max_mismatches are included, but higher are excluded."""
    genome = "CCGC"  # mismatches at pos 0 ('C'!='A') and pos 1 ('C'!='T') = 2 mismatches against "ATGC"
    guide = "ATGC"
    # When threshold is 2, CCGC should be included
    matches_allowed = find_off_targets(genome, guide, max_mismatches=2)
    assert len(matches_allowed) == 1
    assert matches_allowed[0].mismatch_count == 2

    # When threshold is 1, CCGC should be excluded
    matches_filtered = find_off_targets(genome, guide, max_mismatches=1)
    assert len(matches_filtered) == 0


def test_match_exceeding_threshold():
    """Test that windows exceeding the mismatch threshold are omitted."""
    genome = "CCCC"
    guide = "ATGC"  # 3 mismatches: pos 0, 1, 2
    matches = find_off_targets(genome, guide, max_mismatches=2)
    assert len(matches) == 0


def test_multiple_matching_locations():
    """Test finding matches across multiple genomic positions in genomic order."""
    genome = "ATGCGGATGC"
    guide = "ATGC"
    matches = find_off_targets(genome, guide, max_mismatches=0)
    assert len(matches) == 2
    assert matches[0].start == 0
    assert matches[0].end == 4
    assert matches[0].matched_sequence == "ATGC"
    assert matches[1].start == 6
    assert matches[1].end == 10
    assert matches[1].matched_sequence == "ATGC"


def test_correct_zero_based_coordinates():
    """Test that zero-based start and exclusive end coordinates are accurate."""
    genome = "TTATGCGG"  # "ATGC" starts at index 2, ends at index 6
    guide = "ATGC"
    matches = find_off_targets(genome, guide, max_mismatches=0)
    assert len(matches) == 1
    assert matches[0].start == 2
    assert matches[0].end == 6
    assert genome[matches[0].start : matches[0].end] == matches[0].matched_sequence


def test_correct_mismatch_positions():
    """Test that zero-based positions within the window are correctly recorded."""
    genome = "GTTT"
    guide = "ATGA"  # pos 0 ('G'!='A'), pos 2 ('T'!='G'), pos 3 ('T'!='A')
    matches = find_off_targets(genome, guide, max_mismatches=3)
    assert len(matches) == 1
    assert matches[0].mismatch_positions == [0, 2, 3]
    assert matches[0].mismatch_count == 3


def test_genome_shorter_than_guide():
    """Test that genome shorter than guide returns an empty list without error."""
    assert find_off_targets("ATG", "ATGC", max_mismatches=1) == []
    assert find_off_targets("", "ATGC", max_mismatches=1) == []


def test_empty_guide_sequence():
    """Test that an empty or whitespace-only guide sequence raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        find_off_targets("ATGC", "")
    assert "Guide sequence cannot be empty" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        find_off_targets("ATGC", "   \t\n ")
    assert "Guide sequence cannot be empty" in str(exc_info.value)


def test_negative_mismatch_threshold():
    """Test that negative max_mismatches raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        find_off_targets("ATGC", "ATGC", max_mismatches=-1)
    assert "non-negative" in str(exc_info.value)


def test_ambiguous_n_mismatch():
    """Test that ambiguous N bases are treated as mismatches when differing from guide."""
    genome = "ATNC"
    guide = "ATGC"
    matches = find_off_targets(genome, guide, max_mismatches=1)
    assert len(matches) == 1
    assert matches[0].mismatch_positions == [2]
    assert matches[0].mismatch_count == 1


def test_find_off_targets_formatting():
    """Test that find_off_targets handles lowercase input and whitespace cleanly."""
    genome = "  a t g c   t g c a  "
    guide = "atgc"
    matches = find_off_targets(genome, guide, max_mismatches=0)
    assert len(matches) == 1
    assert matches[0].start == 0
    assert matches[0].end == 4
    assert matches[0].matched_sequence == "ATGC"


# ============================================================================
# Day 3 Tests: Multiple FASTA Records Alignment Search
# ============================================================================

def test_search_single_record():
    """Test searching a single FASTA record dictionary."""
    records = [
        {
            "record_id": "chr1",
            "description": "Chromosome 1",
            "sequence": "ATGCATGC",
        }
    ]
    results = find_off_targets_in_records(records, "ATGC", max_mismatches=0)
    assert len(results) == 1
    assert results[0]["record_id"] == "chr1"
    assert len(results[0]["matches"]) == 2
    assert isinstance(results[0]["matches"][0], AlignmentMatch)
    assert results[0]["matches"][0].start == 0
    assert results[0]["matches"][0].end == 4
    assert results[0]["matches"][0].matched_sequence == "ATGC"
    assert results[0]["matches"][1].start == 4
    assert results[0]["matches"][1].end == 8


def test_search_multiple_records():
    """Test searching across multiple FASTA record dictionaries."""
    records = [
        {"record_id": "chr1", "sequence": "ATGC"},
        {"record_id": "chr2", "sequence": "ACGC"},
        {"record_id": "chr3", "sequence": "CCGC"},
    ]
    results = find_off_targets_in_records(records, "ATGC", max_mismatches=1)
    assert len(results) == 3
    # chr1 has 0 mismatches
    assert len(results[0]["matches"]) == 1
    assert results[0]["matches"][0].mismatch_count == 0
    # chr2 has 1 mismatch
    assert len(results[1]["matches"]) == 1
    assert results[1]["matches"][0].mismatch_count == 1
    # chr3 has 2 mismatches (> 1 threshold)
    assert len(results[2]["matches"]) == 0


def test_preserve_record_order():
    """Test that the order of input records is strictly preserved in the results."""
    records = [
        {"record_id": "chr_gamma", "sequence": "ATGC"},
        {"record_id": "chr_alpha", "sequence": "ATGC"},
        {"record_id": "chr_beta", "sequence": "ATGC"},
    ]
    results = find_off_targets_in_records(records, "ATGC", max_mismatches=0)
    assert [r["record_id"] for r in results] == ["chr_gamma", "chr_alpha", "chr_beta"]


def test_correct_record_ids_in_results():
    """Test that output record_id values accurately match input records."""
    records = [
        {"record_id": "scaffold_001", "sequence": "ATGC"},
        {"record_id": "scaffold_002", "sequence": "GGGG"},
    ]
    results = find_off_targets_in_records(records, "ATGC", max_mismatches=0)
    assert results[0]["record_id"] == "scaffold_001"
    assert results[1]["record_id"] == "scaffold_002"


def test_returning_records_with_no_matches():
    """Test that records with zero matches are included in the result list."""
    records = [
        {"record_id": "matching_chr", "sequence": "ATGC"},
        {"record_id": "non_matching_chr", "sequence": "GGGG"},
    ]
    results = find_off_targets_in_records(records, "ATGC", max_mismatches=0)
    assert len(results) == 2
    assert results[0]["record_id"] == "matching_chr"
    assert len(results[0]["matches"]) == 1
    assert results[1]["record_id"] == "non_matching_chr"
    assert results[1]["matches"] == []


def test_correct_match_positions_within_each_record():
    """Test that match start and end coordinates are accurate within each record's sequence."""
    records = [
        {"record_id": "chr1", "sequence": "CCATGCC"},
        {"record_id": "chr2", "sequence": "CATGCCC"},
    ]
    results = find_off_targets_in_records(records, "ATGC", max_mismatches=0)
    # chr1: ATGC starts at index 2
    assert len(results[0]["matches"]) == 1
    assert results[0]["matches"][0].start == 2
    assert results[0]["matches"][0].end == 6
    assert results[0]["matches"][0].matched_sequence == "ATGC"
    # chr2: ATGC starts at index 1
    assert len(results[1]["matches"]) == 1
    assert results[1]["matches"][0].start == 1
    assert results[1]["matches"][0].end == 5
    assert results[1]["matches"][0].matched_sequence == "ATGC"


def test_handling_missing_record_id():
    """Test that a record missing 'record_id' raises a descriptive ValueError."""
    records = [
        {"sequence": "ATGC"}
    ]
    with pytest.raises(ValueError) as exc_info:
        find_off_targets_in_records(records, "ATGC")
    assert "record_id" in str(exc_info.value)


def test_handling_missing_sequence():
    """Test that a record missing 'sequence' raises a descriptive ValueError."""
    records = [
        {"record_id": "chr1"}
    ]
    with pytest.raises(ValueError) as exc_info:
        find_off_targets_in_records(records, "ATGC")
    assert "sequence" in str(exc_info.value)


def test_handling_empty_records_list():
    """Test that an empty list of records returns an empty result list without error."""
    results = find_off_targets_in_records([], "ATGC", max_mismatches=0)
    assert results == []


