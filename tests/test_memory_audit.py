"""Unit tests for DNA sequence alignment memory estimation in GeneWeaver."""

import sys
from pathlib import Path

# Ensure project root is in sys.path when running pytest directly without python -m
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from src.memory_audit import estimate_alignment_memory


def test_normal_genome_and_guide():
    """Test memory estimation for a normal genome and guide sequence."""
    genome = "A" * 250_000
    guide = "G" * 20
    res = estimate_alignment_memory(genome, guide, chunk_size=100_000, bytes_per_base=1)

    assert res["genome_length"] == 250_000
    assert res["guide_length"] == 20
    assert res["chunk_size"] == 100_000
    assert res["bytes_per_base"] == 1
    assert res["genome_input_bytes"] == 250_000
    assert res["guide_input_bytes"] == 20
    # Overlap = 19, chunk_buffer = 100_019
    assert res["chunk_buffer_bytes"] == 100_019
    assert res["estimated_peak_device_input_bytes"] == 100_019 + 20
    assert res["unit"] == "bytes"
    assert "estimates" in res["note"].lower()


def test_genome_input_bytes():
    """Test accurate calculation of total genome input bytes."""
    genome = "ATGCATGC"  # 8 bases
    guide = "ATGC"      # 4 bases
    res_1 = estimate_alignment_memory(genome, guide, bytes_per_base=1)
    assert res_1["genome_input_bytes"] == 8

    res_2 = estimate_alignment_memory(genome, guide, bytes_per_base=2)
    assert res_2["genome_input_bytes"] == 16


def test_guide_input_bytes():
    """Test accurate calculation of guide input bytes."""
    genome = "ATGCATGC"
    guide = "ATGCA"  # 5 bases
    res_1 = estimate_alignment_memory(genome, guide, bytes_per_base=1)
    assert res_1["guide_input_bytes"] == 5

    res_4 = estimate_alignment_memory(genome, guide, bytes_per_base=4)
    assert res_4["guide_input_bytes"] == 20


def test_chunk_buffer_calculation():
    """Test chunk buffer calculation including required overlap."""
    genome = "A" * 200
    guide = "G" * 10  # overlap = 9
    # chunk_size = 50 -> max_chunk_length = min(200, 50 + 9) = 59
    res = estimate_alignment_memory(genome, guide, chunk_size=50, bytes_per_base=1)
    assert res["chunk_buffer_bytes"] == 59


def test_estimated_peak_device_memory():
    """Test peak device memory equals chunk_buffer_bytes + guide_input_bytes."""
    genome = "A" * 500
    guide = "C" * 25  # overlap = 24
    res = estimate_alignment_memory(genome, guide, chunk_size=100, bytes_per_base=1)

    expected_chunk_buffer = 100 + 24
    expected_guide_bytes = 25
    assert res["chunk_buffer_bytes"] == expected_chunk_buffer
    assert res["guide_input_bytes"] == expected_guide_bytes
    assert res["estimated_peak_device_input_bytes"] == expected_chunk_buffer + expected_guide_bytes


def test_chunk_overlap_calculation():
    """Test overlap of guide_length - 1 across various guide lengths."""
    # guide length 1 -> overlap = 0
    res_1 = estimate_alignment_memory("A" * 100, "A", chunk_size=20)
    assert res_1["chunk_buffer_bytes"] == 20  # 20 + 0

    # guide length 30 -> overlap = 29
    res_30 = estimate_alignment_memory("A" * 100, "A" * 30, chunk_size=20)
    assert res_30["chunk_buffer_bytes"] == 20 + 29


def test_genome_shorter_than_guide():
    """Test handling when genome sequence is shorter than guide sequence."""
    genome = "ATG"     # length 3
    guide = "ATGCAT"   # length 6, overlap = 5
    res = estimate_alignment_memory(genome, guide, chunk_size=100, bytes_per_base=1)

    assert res["genome_length"] == 3
    assert res["guide_length"] == 6
    assert res["chunk_buffer_bytes"] == 3
    assert res["guide_input_bytes"] == 6
    assert res["estimated_peak_device_input_bytes"] == 9


def test_small_final_chunk():
    """Test boundary where genome length is smaller than chunk_size + overlap."""
    genome = "A" * 105
    guide = "C" * 10   # overlap = 9, chunk_size + overlap = 109
    res = estimate_alignment_memory(genome, guide, chunk_size=100, bytes_per_base=1)

    # Since genome is 105, chunk buffer is capped at genome length 105
    assert res["chunk_buffer_bytes"] == 105
    assert res["genome_length"] == 105


def test_empty_genome_handling():
    """Test that empty genome is handled gracefully without error."""
    res = estimate_alignment_memory("", "ATGC", chunk_size=100, bytes_per_base=1)
    assert res["genome_length"] == 0
    assert res["genome_input_bytes"] == 0
    assert res["chunk_buffer_bytes"] == 0
    assert res["guide_input_bytes"] == 4
    assert res["estimated_peak_device_input_bytes"] == 4


def test_invalid_dna_sequence():
    """Test rejection of invalid DNA sequence characters."""
    with pytest.raises(ValueError) as exc:
        estimate_alignment_memory("ATGCX", "ATGC")
    assert "Invalid DNA character" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        estimate_alignment_memory("ATGC", "ATG1")
    assert "Invalid DNA character" in str(exc.value)


def test_empty_guide():
    """Test rejection of empty or whitespace-only guide sequences."""
    with pytest.raises(ValueError) as exc:
        estimate_alignment_memory("ATGC", "")
    assert "Guide sequence cannot be empty" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        estimate_alignment_memory("ATGC", "   \t\n  ")
    assert "Guide sequence cannot be empty" in str(exc.value)


def test_invalid_chunk_size():
    """Test rejection of chunk sizes less than 1 or non-integer values."""
    with pytest.raises(ValueError) as exc:
        estimate_alignment_memory("ATGC", "ATGC", chunk_size=0)
    assert "chunk_size must be an integer greater than or equal to 1" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        estimate_alignment_memory("ATGC", "ATGC", chunk_size=-10)
    assert "chunk_size must be an integer greater than or equal to 1" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        estimate_alignment_memory("ATGC", "ATGC", chunk_size="1000")
    assert "chunk_size must be an integer" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        estimate_alignment_memory("ATGC", "ATGC", chunk_size=True)
    assert "chunk_size must be an integer" in str(exc.value)


def test_invalid_bytes_per_base():
    """Test rejection of bytes_per_base less than 1 or non-integer values."""
    with pytest.raises(ValueError) as exc:
        estimate_alignment_memory("ATGC", "ATGC", bytes_per_base=0)
    assert "bytes_per_base must be an integer greater than or equal to 1" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        estimate_alignment_memory("ATGC", "ATGC", bytes_per_base=-2)
    assert "bytes_per_base must be an integer greater than or equal to 1" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        estimate_alignment_memory("ATGC", "ATGC", bytes_per_base=1.5)
    assert "bytes_per_base must be an integer" in str(exc.value)


def test_required_dictionary_fields():
    """Test that all required fields are present in the returned dictionary."""
    expected_keys = {
        "genome_length",
        "guide_length",
        "chunk_size",
        "bytes_per_base",
        "genome_input_bytes",
        "guide_input_bytes",
        "chunk_buffer_bytes",
        "estimated_peak_device_input_bytes",
        "unit",
        "note",
    }
    res = estimate_alignment_memory("ATGC", "ATGC")
    assert expected_keys.issubset(res.keys())
    assert len(res) == len(expected_keys)


def test_note_clarifies_estimates():
    """Test that the note explicitly mentions that memory values are estimates."""
    res = estimate_alignment_memory("ATGC", "ATGC")
    note = res["note"].lower()
    assert "estimate" in note
    assert "gpu" in note
