"""Consolidated system reporting and integration status module for GeneWeaver."""

import copy
from typing import Any, Dict, Optional


def _validate_benchmark_results(bench: dict) -> dict:
    """Validate and sanitize benchmark results dictionary."""
    if not isinstance(bench, dict):
        raise ValueError(f"benchmark_results must be a dict, got {type(bench).__name__}.")

    # Timing and count validation
    if "cpu_average_seconds" in bench:
        val = bench["cpu_average_seconds"]
        if not isinstance(val, (int, float)) or isinstance(val, bool) or val < 0:
            raise ValueError(f"cpu_average_seconds must be a non-negative number, got {val}.")

    if "gpu_average_seconds" in bench and bench["gpu_average_seconds"] is not None:
        val = bench["gpu_average_seconds"]
        if not isinstance(val, (int, float)) or isinstance(val, bool) or val < 0:
            raise ValueError(f"gpu_average_seconds must be a non-negative number, got {val}.")

    if "speedup" in bench and bench["speedup"] is not None:
        val = bench["speedup"]
        if not isinstance(val, (int, float)) or isinstance(val, bool) or val < 0:
            raise ValueError(f"speedup must be a non-negative number, got {val}.")

    if "cpu_match_count" in bench:
        val = bench["cpu_match_count"]
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            raise ValueError(f"cpu_match_count must be a non-negative integer, got {val}.")

    if "gpu_match_count" in bench and bench["gpu_match_count"] is not None:
        val = bench["gpu_match_count"]
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            raise ValueError(f"gpu_match_count must be a non-negative integer, got {val}.")

    # Build sanitized summary
    summary = {
        "cpu_average_seconds": bench.get("cpu_average_seconds"),
        "gpu_average_seconds": bench.get("gpu_average_seconds"),
        "speedup": bench.get("speedup"),
        "cpu_match_count": bench.get("cpu_match_count"),
        "gpu_match_count": bench.get("gpu_match_count"),
        "results_match": bench.get("results_match"),
        "gpu_available": bool(bench.get("gpu_available", False)),
        "message": str(bench.get("message", "")),
    }

    # Only report speedup if a valid GPU benchmark was performed
    gpu_ran = summary["gpu_available"] and summary["gpu_average_seconds"] is not None
    if not gpu_ran:
        summary["speedup"] = None

    return summary


def _validate_memory_estimate(mem: dict) -> dict:
    """Validate and sanitize memory estimation dictionary."""
    if not isinstance(mem, dict):
        raise ValueError(f"memory_estimate must be a dict, got {type(mem).__name__}.")

    numeric_fields = (
        "genome_length",
        "guide_length",
        "genome_input_bytes",
        "guide_input_bytes",
        "chunk_buffer_bytes",
        "estimated_peak_device_input_bytes",
    )

    for field in numeric_fields:
        if field in mem:
            val = mem[field]
            if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                raise ValueError(f"{field} must be a non-negative integer, got {val}.")

    summary = {
        "genome_length": mem.get("genome_length"),
        "guide_length": mem.get("guide_length"),
        "genome_input_bytes": mem.get("genome_input_bytes"),
        "guide_input_bytes": mem.get("guide_input_bytes"),
        "chunk_buffer_bytes": mem.get("chunk_buffer_bytes"),
        "estimated_peak_device_input_bytes": mem.get("estimated_peak_device_input_bytes"),
        "is_estimate": True,
        "note": str(mem.get("note", "Theoretical estimate based on sequence lengths and chunk overlap.")),
    }

    return summary


def _validate_gpu_diagnostics(diag: dict) -> dict:
    """Validate and sanitize GPU diagnostics dictionary."""
    if not isinstance(diag, dict):
        raise ValueError(f"gpu_diagnostics must be a dict, got {type(diag).__name__}.")

    byte_fields = ("free_bytes", "total_bytes", "used_bytes")
    for field in byte_fields:
        if field in diag and diag[field] is not None:
            val = diag[field]
            if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                raise ValueError(f"{field} must be a non-negative integer, got {val}.")

    is_available = bool(diag.get("available", False))

    if not is_available:
        summary = {
            "available": False,
            "device_name": None,
            "free_bytes": None,
            "total_bytes": None,
            "used_bytes": None,
            "message": str(diag.get("message", "CUDA GPU is unavailable.")),
        }
    else:
        summary = {
            "available": True,
            "device_name": diag.get("device_name"),
            "free_bytes": diag.get("free_bytes"),
            "total_bytes": diag.get("total_bytes"),
            "used_bytes": diag.get("used_bytes"),
            "message": str(diag.get("message", "CUDA GPU is operational.")),
        }

    return summary


def generate_system_report(
    benchmark_results: Optional[Dict[str, Any]] = None,
    memory_estimate: Optional[Dict[str, Any]] = None,
    gpu_diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a consolidated system report from benchmark, memory, and GPU diagnostics.

    Aggregates diagnostic and performance metrics without mutating input data.
    Provides an overall readiness assessment, distinguishing theoretical estimates
    from measurements and verifying parity between CPU and GPU alignment.

    Args:
        benchmark_results: Dictionary containing CPU vs GPU alignment benchmark data.
        memory_estimate: Dictionary containing theoretical buffer and chunk memory estimates.
        gpu_diagnostics: Dictionary containing device presence and memory metrics.

    Returns:
        A structured dictionary containing:
            - benchmark_summary (dict or None): Performance and speedup statistics.
            - memory_estimation_summary (dict or None): Theoretical memory requirements.
            - gpu_diagnostics_summary (dict or None): GPU device status and memory stats.
            - overall_system_status (dict): Overall system readiness state and explanation.

    Raises:
        ValueError: If any supplied dictionary is malformed or contains negative values.
    """
    bench_clean = _validate_benchmark_results(benchmark_results) if benchmark_results is not None else None
    mem_clean = _validate_memory_estimate(memory_estimate) if memory_estimate is not None else None
    gpu_clean = _validate_gpu_diagnostics(gpu_diagnostics) if gpu_diagnostics is not None else None

    # Determine overall system readiness
    is_ready = True
    status = "ready"
    reasons = []

    if bench_clean is None and mem_clean is None and gpu_clean is None:
        is_ready = False
        status = "unconfigured"
        reasons.append("No subsystem diagnostics or benchmarks provided.")
    else:
        # Check GPU availability
        gpu_avail_diag = gpu_clean["available"] if gpu_clean is not None else None
        gpu_avail_bench = bench_clean["gpu_available"] if bench_clean is not None else None

        if gpu_avail_diag is False or gpu_avail_bench is False:
            is_ready = False
            status = "cpu_only"
            reasons.append("GPU is unavailable; system can only operate in CPU-only mode.")

        # Check alignment parity if GPU ran
        if bench_clean is not None:
            if bench_clean["results_match"] is False:
                is_ready = False
                status = "mismatch_detected"
                reasons.append("Alignment results mismatch between CPU and GPU.")
            elif bench_clean["gpu_available"] and bench_clean["gpu_average_seconds"] is None:
                is_ready = False
                status = "gpu_benchmark_incomplete"
                reasons.append("GPU was marked available but GPU benchmark execution was not completed.")

        # Check for missing critical sections
        if gpu_clean is None:
            is_ready = False
            if status == "ready":
                status = "partial_diagnostics"
            reasons.append("GPU hardware diagnostics are not available.")

        if bench_clean is None:
            is_ready = False
            if status == "ready":
                status = "partial_benchmarks"
            reasons.append("Alignment performance benchmark has not been executed.")

    if is_ready:
        description = "System is fully operational with GPU acceleration verified."
    else:
        description = " ".join(reasons)

    return {
        "benchmark_summary": bench_clean,
        "memory_estimation_summary": mem_clean,
        "gpu_diagnostics_summary": gpu_clean,
        "overall_system_status": {
            "ready": is_ready,
            "status": status,
            "description": description,
        },
    }
