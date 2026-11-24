"""CPU vs GPU performance comparison module for GeneWeaver."""

import time
from typing import Any, Callable, Dict, Optional

try:
    from src.config import DEFAULT_MAX_MISMATCHES
    from src.cpu_alignment import validate_dna_sequence
    from src.gpu_environment import get_gpu_info
except (ImportError, ModuleNotFoundError):
    from config import DEFAULT_MAX_MISMATCHES
    from cpu_alignment import validate_dna_sequence
    from gpu_environment import get_gpu_info


def compare_alignment_performance(
    genome_sequence: str,
    guide_sequence: str,
    cpu_function: Callable[..., Any],
    gpu_function: Optional[Callable[..., Any]] = None,
    max_mismatches: int = DEFAULT_MAX_MISMATCHES,
    repeats: int = 3,
) -> Dict[str, Any]:
    """Compare alignment performance and result accuracy between CPU and GPU implementations.

    Benchmarks CPU and (if available and provided) GPU alignment functions over
    multiple repetitions. Validates that both implementations produce identical
    matches and computes the resulting speedup factor.

    Args:
        genome_sequence: Target genomic DNA sequence.
        guide_sequence: Query CRISPR guide RNA / DNA sequence.
        cpu_function: Callable CPU alignment function (e.g., find_off_targets).
        gpu_function: Optional callable GPU alignment function.
        max_mismatches: Maximum allowable mismatches (default: 3).
        repeats: Number of benchmarking repetitions to run (default: 3).

    Returns:
        A dictionary containing:
            - cpu_average_seconds (float): Mean CPU execution time.
            - gpu_average_seconds (float or None): Mean GPU execution time.
            - speedup (float or None): CPU time / GPU time ratio, or None if GPU not run.
            - cpu_match_count (int): Number of matches found by CPU.
            - gpu_match_count (int or None): Number of matches found by GPU.
            - results_match (bool or None): True if CPU and GPU matches match, None if GPU not run.
            - gpu_available (bool): True if GPU environment is detected.
            - message (str): Informative status summary.

    Raises:
        ValueError: If repeats < 1, DNA sequences are invalid, or cpu_function is missing.
    """
    if not isinstance(repeats, int) or isinstance(repeats, bool):
        raise ValueError(
            f"repeats must be an integer, got {type(repeats).__name__}."
        )
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}.")

    if cpu_function is None or not callable(cpu_function):
        raise ValueError("A valid callable cpu_function must be provided.")

    if gpu_function is not None and not callable(gpu_function):
        raise ValueError("gpu_function must be callable if provided.")

    if max_mismatches < 0:
        raise ValueError(f"max_mismatches must be non-negative, got {max_mismatches}.")

    cleaned_genome = validate_dna_sequence(genome_sequence)
    cleaned_guide = validate_dna_sequence(guide_sequence)

    if not cleaned_guide:
        raise ValueError("Guide sequence cannot be empty.")

    # 1. Run and measure CPU alignment
    cpu_durations: list[float] = []
    cpu_matches = []

    for _ in range(repeats):
        start_time = time.perf_counter()
        cpu_matches = cpu_function(cleaned_genome, cleaned_guide, max_mismatches)
        end_time = time.perf_counter()
        cpu_durations.append(end_time - start_time)

    cpu_average_seconds = sum(cpu_durations) / len(cpu_durations)
    cpu_match_count = len(cpu_matches) if cpu_matches is not None else 0

    # 2. Check GPU environment
    gpu_info = get_gpu_info()
    gpu_available = bool(gpu_info.get("available", False))

    gpu_average_seconds = None
    speedup = None
    gpu_match_count = None
    results_match = None

    if not gpu_available:
        message = (
            f"GPU is unavailable ({gpu_info.get('message', 'No compatible CUDA GPU detected')}). "
            "Only CPU alignment benchmarked."
        )
    elif gpu_function is None:
        message = "GPU is available, but no gpu_function was supplied. Only CPU alignment benchmarked."
    else:
        # 3. Run and measure GPU alignment
        try:
            gpu_durations: list[float] = []
            gpu_matches = []

            for _ in range(repeats):
                start_time = time.perf_counter()
                gpu_matches = gpu_function(cleaned_genome, cleaned_guide, max_mismatches)
                end_time = time.perf_counter()
                gpu_durations.append(end_time - start_time)

            gpu_average_seconds = sum(gpu_durations) / len(gpu_durations)
            gpu_match_count = len(gpu_matches) if gpu_matches is not None else 0

            speedup = (
                float(cpu_average_seconds) / gpu_average_seconds
                if gpu_average_seconds > 0
                else 0.0
            )

            results_match = (cpu_matches == gpu_matches)

            if results_match:
                message = (
                    f"CPU and GPU benchmarks completed successfully with matching results "
                    f"(Speedup: {speedup:.2f}x)."
                )
            else:
                message = (
                    f"Warning: CPU and GPU benchmarks completed, but alignment results mismatched "
                    f"(Speedup: {speedup:.2f}x)."
                )
        except Exception as err:
            gpu_average_seconds = None
            speedup = None
            gpu_match_count = None
            results_match = None
            message = f"GPU execution error encountered during benchmark: {err}. No speedup reported."

    return {
        "cpu_average_seconds": cpu_average_seconds,
        "gpu_average_seconds": gpu_average_seconds,
        "speedup": speedup,
        "cpu_match_count": cpu_match_count,
        "gpu_match_count": gpu_match_count,
        "results_match": results_match,
        "gpu_available": gpu_available,
        "message": message,
    }
