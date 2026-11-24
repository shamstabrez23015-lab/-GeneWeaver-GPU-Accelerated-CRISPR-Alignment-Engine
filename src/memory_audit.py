"""DNA sequence alignment memory estimation and audit module for GeneWeaver."""

from typing import Any, Dict

try:
    from src.config import DEFAULT_CHUNK_SIZE
    from src.cpu_alignment import validate_dna_sequence
except (ImportError, ModuleNotFoundError):
    from config import DEFAULT_CHUNK_SIZE
    from cpu_alignment import validate_dna_sequence


def estimate_alignment_memory(
    genome_sequence: str,
    guide_sequence: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    bytes_per_base: int = 1,
) -> Dict[str, Any]:
    """Calculate approximate memory requirements for genomic DNA sequence alignment.

    Estimates buffer sizes for the genome, guide sequence, chunk buffer, and peak
    device input memory. For chunked execution, accounts for the required overlap
    between consecutive chunks:
        overlap = guide_length - 1

    Args:
        genome_sequence: Target genomic DNA sequence.
        guide_sequence: Query CRISPR guide RNA / DNA sequence.
        chunk_size: Base chunk size for sequence chunking (default: 100,000).
        bytes_per_base: Encoded memory representation per nucleotide in bytes (default: 1).

    Returns:
        A dictionary containing:
            - genome_length (int): Total number of base pairs in cleaned genome.
            - guide_length (int): Total number of base pairs in cleaned guide.
            - chunk_size (int): Configured chunk size in bases.
            - bytes_per_base (int): Multiplier for memory per base.
            - genome_input_bytes (int): Total raw size of genome sequence in bytes.
            - guide_input_bytes (int): Total raw size of guide sequence in bytes.
            - chunk_buffer_bytes (int): Maximum input chunk buffer size in bytes.
            - estimated_peak_device_input_bytes (int): Peak input buffer memory on device.
            - unit (str): Unit of memory measurement ("bytes").
            - note (str): Disclaimer clarifying that values are theoretical estimates.

    Raises:
        ValueError: If guide sequence is empty, DNA characters are invalid,
                    or chunk_size / bytes_per_base is less than 1.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError(
            f"chunk_size must be an integer greater than or equal to 1, got {chunk_size}."
        )

    if not isinstance(bytes_per_base, int) or isinstance(bytes_per_base, bool) or bytes_per_base < 1:
        raise ValueError(
            f"bytes_per_base must be an integer greater than or equal to 1, got {bytes_per_base}."
        )

    cleaned_genome = validate_dna_sequence(genome_sequence)
    cleaned_guide = validate_dna_sequence(guide_sequence)

    if not cleaned_guide:
        raise ValueError("Guide sequence cannot be empty.")

    genome_len = len(cleaned_genome)
    guide_len = len(cleaned_guide)

    genome_input_bytes = genome_len * bytes_per_base
    guide_input_bytes = guide_len * bytes_per_base

    # Consecutive chunks must overlap by (guide_length - 1) bases so no off-target
    # alignments spanning chunk boundaries are missed.
    overlap = guide_len - 1

    if genome_len == 0:
        max_chunk_length = 0
    else:
        max_chunk_length = min(genome_len, chunk_size + overlap)

    chunk_buffer_bytes = max_chunk_length * bytes_per_base
    estimated_peak_device_input_bytes = chunk_buffer_bytes + guide_input_bytes

    note = (
        "These values are theoretical estimates, not actual measured GPU memory "
        "usage, runtime context, or kernel execution overhead."
    )

    return {
        "genome_length": genome_len,
        "guide_length": guide_len,
        "chunk_size": chunk_size,
        "bytes_per_base": bytes_per_base,
        "genome_input_bytes": genome_input_bytes,
        "guide_input_bytes": guide_input_bytes,
        "chunk_buffer_bytes": chunk_buffer_bytes,
        "estimated_peak_device_input_bytes": estimated_peak_device_input_bytes,
        "unit": "bytes",
        "note": note,
    }
