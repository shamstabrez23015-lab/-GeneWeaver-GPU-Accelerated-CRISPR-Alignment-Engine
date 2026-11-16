"""Unit tests for CUDA-based DNA sequence alignment module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# Ensure project root is in sys.path when running pytest
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.cuda_alignment import (
    BASE_TO_INT,
    decode_sequence,
    encode_sequence,
    find_off_targets_gpu,
    is_cuda_available,
)
from src.models import AlignmentMatch


# ---------------------------------------------------------------------------
# Hardware-Independent Validation Tests
# ---------------------------------------------------------------------------


def test_dna_encoding():
    """Test standard DNA base encoding and decoding to/from NumPy uint8 arrays."""
    seq = "ACGTN"
    encoded = encode_sequence(seq)

    assert isinstance(encoded, np.ndarray)
    assert encoded.dtype == np.uint8
    assert list(encoded) == [0, 1, 2, 3, 4]

    # Verify decoding round-trip
    decoded = decode_sequence(encoded)
    assert decoded == seq


def test_dna_encoding_whitespace_and_lowercase():
    """Test encoding handles lowercase characters and strips whitespace."""
    seq = "  a  c \t g \n t \r n  "
    encoded = encode_sequence(seq)
    assert list(encoded) == [0, 1, 2, 3, 4]
    assert decode_sequence(encoded) == "ACGTN"


def test_dna_decoding_invalid_integer():
    """Test decoding raises ValueError on unrecognized integer base codes."""
    invalid_encoded = np.array([0, 1, 99], dtype=np.uint8)
    with pytest.raises(ValueError, match="Invalid encoded DNA base"):
        decode_sequence(invalid_encoded)


def test_invalid_dna_input():
    """Test that invalid DNA characters raise ValueError."""
    with pytest.raises(ValueError, match="Invalid DNA character"):
        encode_sequence("ACGTX")

    with pytest.raises(ValueError, match="Invalid DNA character"):
        find_off_targets_gpu("ACGTACGT", "ACGTZ")

    with pytest.raises(ValueError, match="Invalid DNA character"):
        find_off_targets_gpu("ACGT123", "ACGT")


def test_empty_guide():
    """Test that an empty guide sequence raises ValueError."""
    with pytest.raises(ValueError, match="Guide sequence cannot be empty"):
        find_off_targets_gpu("ACGTACGT", "")

    with pytest.raises(ValueError, match="Guide sequence cannot be empty"):
        find_off_targets_gpu("ACGTACGT", "   \t\n  ")


def test_negative_mismatch_threshold():
    """Test that negative max_mismatches raises ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        find_off_targets_gpu("ACGTACGT", "ACGT", max_mismatches=-1)

    with pytest.raises(ValueError, match="non-negative"):
        find_off_targets_gpu("ACGTACGT", "ACGT", max_mismatches=-10)


def test_genome_shorter_than_guide():
    """Test that a genome shorter than the guide returns an empty list without error."""
    # Shorter genome
    result = find_off_targets_gpu("ACGT", "ACGTACGT", max_mismatches=2)
    assert result == []

    # Empty genome with non-empty guide
    result_empty = find_off_targets_gpu("", "ACGT", max_mismatches=1)
    assert result_empty == []


def test_cuda_unavailable_error():
    """Test that find_off_targets_gpu raises RuntimeError when CUDA is not available."""
    with patch("src.cuda_alignment.is_cuda_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            find_off_targets_gpu("ACGTACGT", "ACGT", max_mismatches=1)


# ---------------------------------------------------------------------------
# Simulated CUDA Device Execution Tests (Hardware-Independent)
# ---------------------------------------------------------------------------


class _MockCUDADeviceArray:
    """Mock CUDA device array that stores data in a NumPy array."""

    def __init__(self, arr: np.ndarray):
        self._arr = np.array(arr)
        self.shape = self._arr.shape

    def copy_to_host(self) -> np.ndarray:
        return self._arr.copy()


def _run_mocked_gpu_alignment(genome: str, guide: str, max_mismatches: int = 3):
    """Execute find_off_targets_gpu using simulated CUDA memory and kernel execution."""
    cleaned_genome = "".join(genome.split()).upper()
    cleaned_guide = "".join(guide.split()).upper()
    genome_len = len(cleaned_genome)
    guide_len = len(cleaned_guide)
    num_windows = genome_len - guide_len + 1

    encoded_genome = encode_sequence(cleaned_genome)
    encoded_guide = encode_sequence(cleaned_guide)

    # Compute matches using the exact kernel logic
    h_is_match = np.zeros(num_windows, dtype=np.bool_)
    h_mismatch_counts = np.zeros(num_windows, dtype=np.int32)
    h_mismatch_positions = np.zeros((num_windows, guide_len), dtype=np.int32)

    for idx in range(num_windows):
        mismatches = 0
        for j in range(guide_len):
            if encoded_genome[idx + j] != encoded_guide[j]:
                if mismatches < guide_len:
                    h_mismatch_positions[idx, mismatches] = j
                mismatches += 1
        h_mismatch_counts[idx] = mismatches
        h_is_match[idx] = mismatches <= max_mismatches

    mock_cuda = MagicMock()
    mock_cuda.to_device.side_effect = lambda arr: _MockCUDADeviceArray(arr)
    mock_cuda.device_array.side_effect = lambda shape, dtype: _MockCUDADeviceArray(
        h_is_match if dtype == np.bool_ else (
            h_mismatch_counts if shape == num_windows else h_mismatch_positions
        )
    )

    mock_kernel = MagicMock()

    with patch("src.cuda_alignment.is_cuda_available", return_value=True), \
         patch("src.cuda_alignment.cuda", mock_cuda), \
         patch("src.cuda_alignment.dna_alignment_kernel", mock_kernel):
        return find_off_targets_gpu(genome, guide, max_mismatches=max_mismatches)


def test_exact_match():
    """Test alignment with a 0-mismatch exact match."""
    matches = _run_mocked_gpu_alignment("ACGTACGT", "ACGT", max_mismatches=0)

    assert len(matches) == 2
    assert matches[0] == AlignmentMatch(
        start=0,
        end=4,
        matched_sequence="ACGT",
        mismatch_count=0,
        mismatch_positions=[],
    )
    assert matches[1] == AlignmentMatch(
        start=4,
        end=8,
        matched_sequence="ACGT",
        mismatch_count=0,
        mismatch_positions=[],
    )


def test_mismatches_within_threshold():
    """Test alignment when mismatches are within max_mismatches threshold."""
    # Guide: ACGT, Window 0: AGGT (1 mismatch at index 1: C != G)
    matches = _run_mocked_gpu_alignment("AGGT", "ACGT", max_mismatches=1)

    assert len(matches) == 1
    assert matches[0].start == 0
    assert matches[0].end == 4
    assert matches[0].matched_sequence == "AGGT"
    assert matches[0].mismatch_count == 1
    assert matches[0].mismatch_positions == [1]


def test_mismatches_above_threshold():
    """Test alignment filters out windows where mismatches exceed max_mismatches."""
    # Guide: ACGT, Target: AAAA (3 mismatches: C->A, G->A, T->A)
    # With max_mismatches=1, it should not be returned
    matches = _run_mocked_gpu_alignment("AAAA", "ACGT", max_mismatches=1)
    assert matches == []

    # With max_mismatches=3, it should be returned
    matches_allowed = _run_mocked_gpu_alignment("AAAA", "ACGT", max_mismatches=3)
    assert len(matches_allowed) == 1
    assert matches_allowed[0].mismatch_count == 3
    assert matches_allowed[0].mismatch_positions == [1, 2, 3]


def test_correct_start_and_end_positions():
    """Test that returned AlignmentMatch objects have accurate zero-based coordinates."""
    # Genome: TTTTACGTTTTT (12 bp), Guide: ACGT (4 bp)
    # Match at index 4 (end 8)
    matches = _run_mocked_gpu_alignment("TTTTACGTTTTT", "ACGT", max_mismatches=0)

    assert len(matches) == 1
    assert matches[0].start == 4
    assert matches[0].end == 8
    assert matches[0].matched_sequence == "ACGT"


def test_correct_mismatch_positions():
    """Test that mismatch positions match 0-based indices within the guide sequence."""
    # Genome: ACAT, Guide: ACGT (mismatch at index 2: A vs G)
    matches = _run_mocked_gpu_alignment("ACAT", "ACGT", max_mismatches=1)

    assert len(matches) == 1
    assert matches[0].mismatch_positions == [2]

    # Multiple mismatches: Genome: CCGA, Guide: ACGT
    # Index 0: C != A, Index 3: A != T
    multi_matches = _run_mocked_gpu_alignment("CCGA", "ACGT", max_mismatches=2)
    assert len(multi_matches) == 1
    assert multi_matches[0].mismatch_count == 2
    assert multi_matches[0].mismatch_positions == [0, 3]


# ---------------------------------------------------------------------------
# Real Hardware CUDA Execution Tests (Skipped when GPU is unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_cuda_available(), reason="NVIDIA CUDA GPU not available")
def test_real_gpu_exact_match():
    """Test real GPU kernel execution for an exact match."""
    matches = find_off_targets_gpu("ACGTACGT", "ACGT", max_mismatches=0)
    assert len(matches) == 2
    assert matches[0].start == 0
    assert matches[0].end == 4
    assert matches[0].mismatch_count == 0


@pytest.mark.skipif(not is_cuda_available(), reason="NVIDIA CUDA GPU not available")
def test_real_gpu_mismatch_detection():
    """Test real GPU kernel execution for mismatch detection."""
    matches = find_off_targets_gpu("ACAT", "ACGT", max_mismatches=1)
    assert len(matches) == 1
    assert matches[0].mismatch_count == 1
    assert matches[0].mismatch_positions == [2]

