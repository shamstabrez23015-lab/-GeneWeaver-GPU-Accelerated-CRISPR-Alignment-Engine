"""GeneWeaver end-to-end execution pipeline module."""

from __future__ import annotations

import logging
from typing import Any

from src.alignment_report import flatten_alignment_results, summarize_alignment_results
from src.benchmark_comparison import compare_alignment_performance
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_MAX_MISMATCHES
from src.cpu_alignment import find_off_targets, find_off_targets_in_records, validate_dna_sequence
from src.cuda_alignment import find_off_targets_gpu_chunked, is_cuda_available
from src.cuda_diagnostics import get_cuda_diagnostics
from src.genomic_parser import parse_fasta
from src.memory_audit import estimate_alignment_memory
from src.system_report import generate_system_report

logger = logging.getLogger(__name__)


def run_geneweaver_pipeline(
    fasta_path: str,
    guide_sequence: str,
    max_mismatches: int = DEFAULT_MAX_MISMATCHES,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    use_gpu: bool = True,
) -> dict[str, Any]:
    """Execute the complete GeneWeaver CRISPR alignment and benchmarking pipeline.

    Orchestrates:
        1. FASTA file parsing via genomic_parser.
        2. Guide sequence cleaning and validation.
        3. CPU sequence alignment across all FASTA records.
        4. CUDA availability check and diagnostics retrieval.
        5. Chunked GPU sequence alignment (if requested and CUDA is available).
        6. Memory audit estimation based on sequence lengths and chunk overlap.
        7. CPU vs GPU benchmark comparison (only when GPU execution was successful).
        8. Consolidated system report generation.

    Parameters
    ----------
    fasta_path:
        Path to the target FASTA genomic reference file.
    guide_sequence:
        Query CRISPR guide RNA / DNA sequence.
    max_mismatches:
        Maximum allowed base mismatches (default: 3).
    chunk_size:
        Genomic chunk size in base pairs for chunked GPU execution (default: 100,000).
    use_gpu:
        Whether to attempt GPU-accelerated alignment (default: True).

    Returns
    -------
    dict[str, Any]
        Structured pipeline results containing:
            - records (list[dict]): Parsed FASTA record dictionaries.
            - cpu_results (list[dict]): Per-record CPU alignment matches.
            - gpu_results (list[dict] or None): Per-record GPU alignment matches.
            - benchmark_results (dict or None): CPU vs GPU benchmark comparison metrics.
            - memory_estimate (dict): Theoretical buffer and memory requirements.
            - gpu_diagnostics (dict): CUDA device presence and VRAM status.
            - system_report (dict): Consolidated system diagnostic report.
            - status (str): Overall pipeline execution status.
            - message (str): Informative summary of pipeline run and GPU status.

    Raises
    ------
    FileNotFoundError:
        If fasta_path does not exist or is invalid.
    ValueError:
        If FASTA file contains no records, guide sequence is invalid/empty,
        or parameters (max_mismatches, chunk_size) are invalid.
    """
    # -------------------------------------------------------------------------
    # 1. Parameter Validation & Guide Cleaning
    # -------------------------------------------------------------------------
    if max_mismatches < 0:
        raise ValueError(f"max_mismatches must be non-negative, got {max_mismatches}.")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}.")

    cleaned_guide = validate_dna_sequence(guide_sequence)
    if not cleaned_guide:
        raise ValueError("Guide sequence cannot be empty.")

    # -------------------------------------------------------------------------
    # 2. Parse FASTA File
    # -------------------------------------------------------------------------
    records = parse_fasta(fasta_path)

    # -------------------------------------------------------------------------
    # 3. CPU Sequence Alignment
    # -------------------------------------------------------------------------
    cpu_results = find_off_targets_in_records(
        records=records,
        guide_sequence=cleaned_guide,
        max_mismatches=max_mismatches,
    )

    # -------------------------------------------------------------------------
    # 4. GPU Diagnostics and Availability
    # -------------------------------------------------------------------------
    gpu_diagnostics = get_cuda_diagnostics()
    cuda_is_supported = bool(gpu_diagnostics.get("available", False)) and is_cuda_available()

    # -------------------------------------------------------------------------
    # 5. GPU Sequence Alignment (if requested & available)
    # -------------------------------------------------------------------------
    gpu_results = None
    gpu_execution_succeeded = False
    status_message_parts = []

    if not use_gpu:
        status_message_parts.append("GPU execution skipped by user request.")
    elif not cuda_is_supported:
        status_message_parts.append(
            f"GPU execution unavailable: {gpu_diagnostics.get('message', 'No compatible CUDA GPU detected')}."
        )
    else:
        try:
            gpu_results = []
            for record in records:
                matches = find_off_targets_gpu_chunked(
                    genome_sequence=record["sequence"],
                    guide_sequence=cleaned_guide,
                    max_mismatches=max_mismatches,
                    chunk_size=chunk_size,
                )
                gpu_results.append({
                    "record_id": record["record_id"],
                    "matches": matches,
                })
            gpu_execution_succeeded = True
            status_message_parts.append("GPU alignment completed successfully.")
        except Exception as err:
            logger.warning("GPU execution error: %s", err)
            gpu_results = None
            gpu_execution_succeeded = False
            status_message_parts.append(f"GPU execution failed ({err}); fell back to CPU results.")

    # -------------------------------------------------------------------------
    # 6. Memory Audit Estimation
    # -------------------------------------------------------------------------
    combined_sequence = "".join(r["sequence"] for r in records)
    memory_estimate = estimate_alignment_memory(
        genome_sequence=combined_sequence,
        guide_sequence=cleaned_guide,
        chunk_size=chunk_size,
        bytes_per_base=1,
    )

    # -------------------------------------------------------------------------
    # 7. CPU vs GPU Benchmark Comparison
    # -------------------------------------------------------------------------
    benchmark_results = None
    if gpu_execution_succeeded and len(combined_sequence) > 0:
        try:
            def _cpu_bench_fn(genome: str, guide: str, mismatches: int) -> list:
                return find_off_targets(genome, guide, mismatches)

            def _gpu_bench_fn(genome: str, guide: str, mismatches: int) -> list:
                return find_off_targets_gpu_chunked(
                    genome, guide, mismatches, chunk_size=chunk_size
                )

            benchmark_results = compare_alignment_performance(
                genome_sequence=combined_sequence,
                guide_sequence=cleaned_guide,
                cpu_function=_cpu_bench_fn,
                gpu_function=_gpu_bench_fn,
                max_mismatches=max_mismatches,
                repeats=3,
            )
            status_message_parts.append(
                f"Benchmark speedup: {benchmark_results.get('speedup', 'N/A')}x."
            )
        except Exception as err:
            logger.warning("Benchmark comparison failed: %s", err)
            benchmark_results = None
            status_message_parts.append(f"Benchmark comparison skipped due to error: {err}.")
    else:
        status_message_parts.append("Benchmark comparison skipped (GPU alignment was not performed).")

    # -------------------------------------------------------------------------
    # 8. Consolidated System Report
    # -------------------------------------------------------------------------
    system_report = generate_system_report(
        benchmark_results=benchmark_results,
        memory_estimate=memory_estimate,
        gpu_diagnostics=gpu_diagnostics,
    )

    overall_status = "Completed" if (cpu_results is not None) else "Failed"
    final_message = " ".join(status_message_parts)

    return {
        "records": records,
        "parsed_records": records,
        "cpu_results": cpu_results,
        "cpu_alignment_results": cpu_results,
        "gpu_results": gpu_results,
        "gpu_alignment_results": gpu_results,
        "benchmark_results": benchmark_results,
        "memory_estimate": memory_estimate,
        "gpu_diagnostics": gpu_diagnostics,
        "system_report": system_report,
        "consolidated_system_report": system_report,
        "status": overall_status,
        "pipeline_status": overall_status,
        "overall_pipeline_status": overall_status,
        "message": final_message,
    }
