"""CUDA-based sequence alignment module for GeneWeaver."""

from __future__ import annotations

from typing import Any
import numpy as np

try:
    from src.config import DEFAULT_MAX_MISMATCHES, VALID_DNA_BASES
    from src.models import AlignmentMatch
except (ImportError, ModuleNotFoundError):
    from config import DEFAULT_MAX_MISMATCHES, VALID_DNA_BASES
    from models import AlignmentMatch

try:
    from src.cpu_alignment import validate_dna_sequence
except (ImportError, ModuleNotFoundError):
    try:
        from cpu_alignment import validate_dna_sequence
    except (ImportError, ModuleNotFoundError):
        def validate_dna_sequence(sequence: str) -> str:
            cleaned = "".join(sequence.split()).upper()
            invalid_chars = set(cleaned) - VALID_DNA_BASES
            if invalid_chars:
                sorted_invalid = sorted(invalid_chars)
                raise ValueError(
                    f"Invalid DNA character(s) found: {sorted_invalid}. "
                    f"Allowed bases are: {sorted(VALID_DNA_BASES)}"
                )
            return cleaned

# Numeric integer encoding for DNA nucleotides
BASE_TO_INT: dict[str, int] = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
}
INT_TO_BASE: dict[int, str] = {v: k for k, v in BASE_TO_INT.items()}

# Pre-computed lookup table for ASCII byte conversion to uint8
_ENCODE_LOOKUP = np.zeros(256, dtype=np.uint8)
for _char, _val in BASE_TO_INT.items():
    _ENCODE_LOOKUP[ord(_char)] = _val


def encode_sequence(sequence: str) -> np.ndarray:
    """Encode a DNA sequence string into a 1D NumPy array of uint8 values.

    Parameters
    ----------
    sequence:
        The raw DNA sequence string. Will be validated and cleaned.

    Returns
    -------
    np.ndarray
        A 1D array of dtype uint8 representing encoded nucleotides:
        A=0, C=1, G=2, T=3, N=4.

    Raises
    ------
    ValueError
        If sequence contains unsupported characters.
    """
    cleaned = validate_dna_sequence(sequence)
    if not cleaned:
        return np.empty(0, dtype=np.uint8)

    ascii_bytes = np.frombuffer(cleaned.encode("ascii"), dtype=np.uint8)
    return _ENCODE_LOOKUP[ascii_bytes].copy()


def decode_sequence(encoded: np.ndarray) -> str:
    """Decode a 1D NumPy array of uint8 values back into a DNA sequence string.

    Parameters
    ----------
    encoded:
        A 1D array of integer encoded nucleotides (0=A, 1=C, 2=G, 3=T, 4=N).

    Returns
    -------
    str
        The decoded uppercase DNA sequence string.

    Raises
    ------
    ValueError
        If any element in encoded is not a recognized base code.
    """
    chars: list[str] = []
    for val in encoded:
        int_val = int(val)
        if int_val not in INT_TO_BASE:
            raise ValueError(f"Invalid encoded DNA base integer: {int_val}.")
        chars.append(INT_TO_BASE[int_val])
    return "".join(chars)


# Safely attempt to import numba.cuda without failing on machines without CUDA drivers
try:
    from numba import cuda
    _CUDA_AVAILABLE = bool(cuda.is_available())
except Exception:
    cuda = None
    _CUDA_AVAILABLE = False


def is_cuda_available() -> bool:
    """Check whether a functional CUDA GPU is available for computation.

    Returns
    -------
    bool
        True if Numba CUDA is accessible and a device is detected, False otherwise.
    """
    global cuda, _CUDA_AVAILABLE
    if cuda is None:
        try:
            from numba import cuda as _cuda
            cuda = _cuda
            _CUDA_AVAILABLE = bool(cuda.is_available())
        except Exception:
            return False
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


# Define CUDA kernel if cuda module is importable
if cuda is not None:
    try:
        @cuda.jit
        def dna_alignment_kernel(
            d_genome,
            d_guide,
            d_is_match,
            d_mismatch_counts,
            d_mismatch_positions,
            num_windows,
            guide_len,
            max_mismatches,
        ):
            """CUDA kernel to align guide sequence against sliding windows of a genome.

            Assigns one thread per genome window to count mismatches and record
            their 0-based positions within the window.
            """
            idx = cuda.grid(1)
            if idx >= num_windows:
                return

            mismatches = 0
            for j in range(guide_len):
                if d_genome[idx + j] != d_guide[j]:
                    if mismatches < d_mismatch_positions.shape[1]:
                        d_mismatch_positions[idx, mismatches] = j
                    mismatches += 1

            d_mismatch_counts[idx] = mismatches
            d_is_match[idx] = mismatches <= max_mismatches
    except Exception:
        dna_alignment_kernel = None
else:
    dna_alignment_kernel = None


def find_off_targets_gpu(
    genome_sequence: str,
    guide_sequence: str,
    max_mismatches: int = DEFAULT_MAX_MISMATCHES,
) -> list[AlignmentMatch]:
    """Find potential CRISPR off-target alignment sites using a CUDA GPU.

    Scans the genome sequence using a sliding window of the same length as the
    guide sequence and identifies all windows with a number of mismatched bases
    less than or equal to max_mismatches.

    Parameters
    ----------
    genome_sequence:
        Target genomic DNA sequence string.
    guide_sequence:
        Query CRISPR guide RNA / DNA sequence string.
    max_mismatches:
        Maximum allowable mismatches (default: 3).

    Returns
    -------
    list[AlignmentMatch]
        A list of AlignmentMatch objects in genomic order.

    Raises
    ------
    ValueError
        If max_mismatches is negative, guide_sequence is empty,
        or either sequence contains invalid DNA characters.
    RuntimeError
        If CUDA is unavailable or no functional CUDA device is detected.
    """
    if max_mismatches < 0:
        raise ValueError(
            f"max_mismatches must be non-negative, got {max_mismatches}."
        )

    cleaned_genome = validate_dna_sequence(genome_sequence)
    cleaned_guide = validate_dna_sequence(guide_sequence)

    if not cleaned_guide:
        raise ValueError("Guide sequence cannot be empty.")

    genome_len = len(cleaned_genome)
    guide_len = len(cleaned_guide)

    if genome_len < guide_len:
        return []

    if not is_cuda_available() or dna_alignment_kernel is None:
        raise RuntimeError(
            "CUDA is not available. GPU alignment cannot be executed."
        )

    # Encode sequences to uint8 NumPy arrays
    h_genome = encode_sequence(cleaned_genome)
    h_guide = encode_sequence(cleaned_guide)

    num_windows = genome_len - guide_len + 1

    # Transfer encoded inputs to GPU device
    d_genome = cuda.to_device(h_genome)
    d_guide = cuda.to_device(h_guide)

    # Allocate device output buffers
    d_is_match = cuda.device_array(num_windows, dtype=np.bool_)
    d_mismatch_counts = cuda.device_array(num_windows, dtype=np.int32)
    d_mismatch_positions = cuda.device_array((num_windows, guide_len), dtype=np.int32)

    # Configure grid and block dimensions
    threads_per_block = 256
    blocks_per_grid = (num_windows + threads_per_block - 1) // threads_per_block

    # Execute CUDA kernel
    dna_alignment_kernel[blocks_per_grid, threads_per_block](
        d_genome,
        d_guide,
        d_is_match,
        d_mismatch_counts,
        d_mismatch_positions,
        num_windows,
        guide_len,
        max_mismatches,
    )
    cuda.synchronize()

    # Copy output buffers back to host
    h_is_match = d_is_match.copy_to_host()
    h_mismatch_counts = d_mismatch_counts.copy_to_host()
    h_mismatch_positions = d_mismatch_positions.copy_to_host()

    # Convert matches to AlignmentMatch objects
    matches: list[AlignmentMatch] = []
    for start in range(num_windows):
        if h_is_match[start]:
            count = int(h_mismatch_counts[start])
            positions = [int(p) for p in h_mismatch_positions[start, :count]]
            end = start + guide_len
            matches.append(
                AlignmentMatch(
                    start=start,
                    end=end,
                    matched_sequence=cleaned_genome[start:end],
                    mismatch_count=count,
                    mismatch_positions=positions,
                )
            )

    return matches

