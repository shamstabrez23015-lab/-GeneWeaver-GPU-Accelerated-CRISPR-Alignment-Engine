"""CPU performance benchmarking module for GeneWeaver."""

import time

try:
    from src.config import DEFAULT_MAX_MISMATCHES
    from src.cpu_alignment import find_off_targets, validate_dna_sequence
except (ImportError, ModuleNotFoundError):
    from config import DEFAULT_MAX_MISMATCHES
    from cpu_alignment import find_off_targets, validate_dna_sequence


def benchmark_cpu_alignment(
    genome_sequence: str,
    guide_sequence: str,
    max_mismatches: int = DEFAULT_MAX_MISMATCHES,
    repeats: int = 3,
) -> dict:
    """Benchmark CPU-based CRISPR sequence alignment performance.

    Executes find_off_targets() over multiple repetitions to measure execution
    time, throughput (comparisons per second), and match statistics.

    Args:
        genome_sequence: Target genomic DNA sequence.
        guide_sequence: Query CRISPR guide RNA / DNA sequence.
        max_mismatches: Maximum allowed base mismatches (default: 3).
        repeats: Number of benchmark iterations to run (default: 3). Must be >= 1.

    Returns:
        A dictionary containing benchmark metrics:
            - genome_length (int): Cleaned genome sequence length.
            - guide_length (int): Cleaned guide sequence length.
            - windows_checked (int): Number of genomic windows evaluated.
            - repeats (int): Number of benchmark repetitions.
            - average_seconds (float): Mean execution time per repetition in seconds.
            - min_seconds (float): Minimum execution time in seconds.
            - max_seconds (float): Maximum execution time in seconds.
            - comparisons_per_second (float): Window comparisons per second.
            - match_count (int): Number of alignment matches discovered.

    Raises:
        ValueError: If repeats is not an integer or is less than 1,
                    or if DNA sequences / max_mismatches are invalid.
    """
    if not isinstance(repeats, int) or isinstance(repeats, bool):
        raise ValueError(
            f"repeats must be an integer, got {type(repeats).__name__}."
        )
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}.")

    cleaned_genome = validate_dna_sequence(genome_sequence)
    cleaned_guide = validate_dna_sequence(guide_sequence)

    if not cleaned_guide:
        raise ValueError("Guide sequence cannot be empty.")

    genome_len = len(cleaned_genome)
    guide_len = len(cleaned_guide)

    if genome_len < guide_len:
        windows_checked = 0
    else:
        windows_checked = genome_len - guide_len + 1

    durations: list[float] = []
    last_matches = []

    for _ in range(repeats):
        start_time = time.perf_counter()
        last_matches = find_off_targets(
            genome_sequence=cleaned_genome,
            guide_sequence=cleaned_guide,
            max_mismatches=max_mismatches,
        )
        end_time = time.perf_counter()
        durations.append(end_time - start_time)

    min_seconds = min(durations)
    max_seconds = max(durations)
    average_seconds = sum(durations) / len(durations)
    match_count = len(last_matches)

    comparisons_per_second = (
        float(windows_checked) / average_seconds if average_seconds > 0 else 0.0
    )

    return {
        "genome_length": genome_len,
        "guide_length": guide_len,
        "windows_checked": windows_checked,
        "repeats": repeats,
        "average_seconds": average_seconds,
        "min_seconds": min_seconds,
        "max_seconds": max_seconds,
        "comparisons_per_second": comparisons_per_second,
        "match_count": match_count,
    }
