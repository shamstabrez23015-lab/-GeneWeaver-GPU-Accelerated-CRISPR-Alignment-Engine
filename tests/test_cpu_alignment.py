"""Unit tests for CPU alignment sequence validation in GeneWeaver."""

import sys
from pathlib import Path

# Ensure project root is in sys.path when running pytest directly without python -m
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from src.cpu_alignment import validate_dna_sequence


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
