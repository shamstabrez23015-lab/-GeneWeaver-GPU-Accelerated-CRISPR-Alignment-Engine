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


def find_off_targets_gpu_chunked(
    genome_sequence: str,
    guide_sequence: str,
    max_mismatches: int = DEFAULT_MAX_MISMATCHES,
    chunk_size: int = 100000,
) -> list[AlignmentMatch]:
    """Find CRISPR off-target sites by processing the genome in overlapping chunks via GPU.

    Large genomes are divided into overlapping segments so that every possible
    genome window is evaluated by the CUDA kernel exactly once.  An overlap of
    ``guide_length - 1`` bases is added to the right of each chunk to ensure
    matches that cross a chunk boundary are not missed.

    Parameters
    ----------
    genome_sequence:
        Target genomic DNA sequence string.
    guide_sequence:
        Query CRISPR guide RNA / DNA sequence string.
    max_mismatches:
        Maximum allowable mismatches (default: 3).
    chunk_size:
        Number of new (non-overlapping) genome bases processed per GPU call.
        Must be a positive integer (default: 100 000).

    Returns
    -------
    list[AlignmentMatch]
        Combined results from all chunks in genome order, using global
        0-based coordinates.  No duplicates are returned.

    Raises
    ------
    ValueError
        If ``max_mismatches`` is negative, ``chunk_size`` is not a positive
        integer, ``guide_sequence`` is empty, or either sequence contains
        invalid DNA characters.
    RuntimeError
        If CUDA is unavailable (propagated from :func:`find_off_targets_gpu`).
    """
    # --- Input validation ---------------------------------------------------
    if max_mismatches < 0:
        raise ValueError(
            f"max_mismatches must be non-negative, got {max_mismatches}."
        )
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(
            f"chunk_size must be a positive integer, got {chunk_size!r}."
        )

    cleaned_genome = validate_dna_sequence(genome_sequence)
    cleaned_guide = validate_dna_sequence(guide_sequence)

    if not cleaned_guide:
        raise ValueError("Guide sequence cannot be empty.")

    genome_len = len(cleaned_genome)
    guide_len = len(cleaned_guide)

    if genome_len < guide_len:
        return []

    # --- Chunked alignment --------------------------------------------------
    # Overlap = guide_len - 1 ensures every window that crosses a boundary is
    # fully contained in at least one chunk.
    overlap = guide_len - 1
    all_matches: list[AlignmentMatch] = []
    chunk_start = 0

    while chunk_start < genome_len:
        chunk_end = min(chunk_start + chunk_size + overlap, genome_len)
        chunk = cleaned_genome[chunk_start:chunk_end]

        # find_off_targets_gpu raises RuntimeError when CUDA is unavailable —
        # let that propagate; never fall back to CPU silently.
        chunk_matches = find_off_targets_gpu(chunk, cleaned_guide, max_mismatches)

        for match in chunk_matches:
            global_start = chunk_start + match.start

            # Deduplication rule:
            # A match whose global start position falls inside the overlap zone
            # of a non-final chunk will be found again at the beginning of the
            # next chunk — skip it here to avoid duplicates.
            # Exception: always include all matches from the final chunk region.
            is_last_chunk = chunk_end == genome_len
            if global_start < chunk_start + chunk_size or is_last_chunk:
                global_end = chunk_start + match.end
                all_matches.append(
                    AlignmentMatch(
                        start=global_start,
                        end=global_end,
                        matched_sequence=match.matched_sequence,
                        mismatch_count=match.mismatch_count,
                        mismatch_positions=match.mismatch_positions,
                    )
                )

        chunk_start += chunk_size

    return all_matches


def get_gpu_memory_info() -> dict:
    """Retrieve current GPU memory information from the active CUDA device.

    Uses the Numba CUDA context API (``cuda.current_context().get_memory_info()``)
    to query the actual free and total GPU memory.  Used memory is derived as
    ``total - free`` — no values are fabricated or hardcoded.

    Returns
    -------
    dict
        A dictionary with the following keys:

        ``available`` : bool
            True when CUDA is reachable and memory info was retrieved.
        ``free_bytes`` : int or None
            Free GPU memory in bytes, or None when unavailable.
        ``total_bytes`` : int or None
            Total GPU memory in bytes, or None when unavailable.
        ``used_bytes`` : int or None
            Used GPU memory in bytes (total − free), or None when unavailable.
        ``message`` : str
            Human-readable status or error description.

    Notes
    -----
    This function never raises — all error paths are caught and surfaced
    through the returned dictionary so that callers can inspect status
    without wrapping in a try/except.
    """
    _unavailable = {
        "available": False,
        "free_bytes": None,
        "total_bytes": None,
        "used_bytes": None,
    }

    # --- CUDA not importable ------------------------------------------------
    if cuda is None:
        return {
            **_unavailable,
            "message": (
                "GPU memory information is unavailable: "
                "the Numba CUDA package could not be imported."
            ),
        }

    # --- CUDA present but no device -----------------------------------------
    try:
        cuda_present = cuda.is_available()
    except Exception as exc:
        return {
            **_unavailable,
            "message": (
                f"GPU memory information is unavailable: "
                f"CUDA availability check failed ({exc})."
            ),
        }

    if not cuda_present:
        return {
            **_unavailable,
            "message": (
                "GPU memory information is unavailable: "
                "no compatible CUDA device was detected."
            ),
        }

    # --- Retrieve memory info from CUDA context -----------------------------
    try:
        context = cuda.current_context()
        mem_info = context.get_memory_info()

        # Numba ≥ 0.54 returns a named tuple (free, total); older versions
        # may return a plain tuple — handle both by index.
        if hasattr(mem_info, "free"):
            free_bytes = int(mem_info.free)
            total_bytes = int(mem_info.total)
        else:
            free_bytes = int(mem_info[0])
            total_bytes = int(mem_info[1])

        used_bytes = total_bytes - free_bytes

        return {
            "available": True,
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "used_bytes": used_bytes,
            "message": (
                f"GPU memory info retrieved successfully: "
                f"{free_bytes:,} bytes free of {total_bytes:,} bytes total."
            ),
        }

    except Exception as exc:
        return {
            **_unavailable,
            "message": (
                f"GPU memory information is unavailable: "
                f"failed to retrieve memory info from CUDA context ({exc})."
            ),
        }

