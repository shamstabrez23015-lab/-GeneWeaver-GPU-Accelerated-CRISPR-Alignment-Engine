# GeneWeaver Subsystem Integration Guide

This guide describes the core architecture, subsystem data contracts, interface specifications, and connection procedures for **GeneWeaver: GPU-Accelerated CRISPR Alignment Engine**.

> **Important Integration Notice:**  
> The project modules are currently authored and tested across independent Git feature branches (`faizan-branch`, Albert's CUDA branch, and Taniya's dashboard branch). Before the complete end-to-end pipeline can run, all feature branches must be merged into the integration/main branch following the sequence detailed below.

---

## 1. Project Module Responsibilities

| Subsystem Module | Owner / Branch | Core Responsibilities |
| :--- | :--- | :--- |
| **`src.config`** | Faizan (Lead) | Shared constants (`VALID_DNA_BASES`, `DEFAULT_CHUNK_SIZE`, `DEFAULT_MAX_MISMATCHES`, `PROJECT_NAME`). |
| **`src.models`** | Faizan (Lead) | Central data models, notably the [`AlignmentMatch`](file:///c:/Users/shams/OneDrive/Desktop/GeneWeaver%20GPU-Accelerated%20CRISPR%20Alignment%20Engine/src/models.py) dataclass. |
| **`src.cpu_alignment`** | Faizan (Lead) | CPU sequence validation, single-sequence sliding-window alignment, and multi-FASTA record matching. |
| **`src.alignment_report`** | Faizan (Lead) | Result flattening into tabular format and alignment summary statistics. |
| **`src.benchmark`** | Faizan (Lead) | Standalone CPU alignment benchmarking with repetitions, throughput, and timing metrics. |
| **`src.gpu_environment`** | Faizan (Lead) | Safe Numba CUDA detection and hardware capability querying without runtime exceptions. |
| **`src.benchmark_comparison`** | Faizan (Lead) | Side-by-side CPU vs GPU benchmark comparison, match parity verification, and speedup measurement. |
| **`src.memory_audit`** | Faizan (Lead) | Theoretical buffer sizing, chunk overlap calculations (`guide_length - 1`), and device memory estimation. |
| **`src.system_report`** | Faizan (Lead) | Aggregation of benchmark results, memory estimates, and GPU diagnostics into a consolidated status report. |
| **`genomic_parser.py`** | Albert | Streaming genomic FASTA file parsing into structured record dictionaries. |
| **`cuda_alignment.py`** | Albert | GPU parallel sequence comparison using Numba CUDA kernels and device memory management. |
| **`dashboard.py`** | Taniya | Textual TUI dashboard presenting live alignment tables, metrics, memory audit graphs, and GPU telemetry. |

---

## 2. Core Subsystem Interfaces

### 2.1 CPU Alignment Interface (`src.cpu_alignment`)
- **`validate_dna_sequence(sequence: str) -> str`**  
  Strips whitespace, converts to uppercase, validates against `VALID_DNA_BASES = frozenset("ATGCN")`, and raises `ValueError` on invalid characters.
- **`find_off_targets(genome_sequence: str, guide_sequence: str, max_mismatches: int = 3) -> list[AlignmentMatch]`**  
  Scans sliding windows of length $L = \text{len}(\text{guide})$. Returns `AlignmentMatch` instances containing `start`, `end`, `matched_sequence`, `mismatch_count`, and `mismatch_positions`.
- **`find_off_targets_in_records(records: list[dict], guide_sequence: str, max_mismatches: int = 3) -> list[dict]`**  
  Processes a list of record dictionaries `[{"record_id": ..., "sequence": ...}]`. Returns `[{"record_id": ..., "matches": [AlignmentMatch, ...]}]`.

### 2.2 CUDA Alignment Interface (`cuda_alignment.py` on Albert's branch)
To integrate with the benchmarking and dashboard systems, Albert's CUDA alignment function must adhere to the same functional contract as CPU alignment:
```python
def find_off_targets_cuda(
    genome_sequence: str,
    guide_sequence: str,
    max_mismatches: int = 3
) -> list[AlignmentMatch]:
    """Execute GPU-accelerated CRISPR sequence alignment using custom CUDA kernels."""
    ...
```
- Accepts validated DNA strings and mismatch threshold.
- Returns a `list[AlignmentMatch]` identical in structure and ordering to the CPU output.

### 2.3 GPU Diagnostics Data Fields
Albert's CUDA environment telemetry should produce a dictionary conforming to:
```python
{
    "available": True,                        # bool: True if CUDA device is detected and initialized
    "device_name": "NVIDIA GeForce RTX 4090", # str or None
    "free_bytes": 21474836480,                # int or None: Free VRAM in bytes
    "total_bytes": 25769803776,               # int or None: Total VRAM in bytes
    "used_bytes": 4294967296,                 # int or None: Used VRAM in bytes
    "message": "Device operational."          # str: Diagnostic explanation or error summary
}
```

### 2.4 Benchmark Comparison Output Fields (`src.benchmark_comparison`)
`compare_alignment_performance(...)` returns:
```python
{
    "cpu_average_seconds": 0.0452, # float: Mean CPU execution time
    "gpu_average_seconds": 0.0031, # float or None: Mean GPU execution time (None if GPU did not run)
    "speedup": 14.58,              # float or None: cpu_average_seconds / gpu_average_seconds
    "cpu_match_count": 4,          # int: Matches found by CPU
    "gpu_match_count": 4,          # int or None: Matches found by GPU
    "results_match": True,         # bool or None: True if CPU and GPU matches are identical
    "gpu_available": True,         # bool: Whether CUDA environment was detected
    "message": "..."               # str: Benchmark execution summary
}
```

### 2.5 Memory Audit Output Fields (`src.memory_audit`)
`estimate_alignment_memory(...)` returns:
```python
{
    "genome_length": 1000000,                  # int: Base pairs in genome
    "guide_length": 20,                        # int: Base pairs in guide
    "chunk_size": 100000,                      # int: Configured chunk size
    "bytes_per_base": 1,                       # int: Encoding size per nucleotide
    "genome_input_bytes": 1000000,             # int: Total genome sequence size
    "guide_input_bytes": 20,                   # int: Total guide sequence size
    "chunk_buffer_bytes": 100019,              # int: Chunk buffer accounting for overlap (guide_length - 1)
    "estimated_peak_device_input_bytes": 100039, # int: chunk_buffer_bytes + guide_input_bytes
    "unit": "bytes",                           # str: Measurement unit
    "note": "..."                              # str: Disclaimer clarifying theoretical nature
}
```

---

## 3. Dashboard Integration Contract (`dashboard.py` on Taniya's branch)

Taniya's Textual TUI dashboard interacts with the engine via two primary reporting functions:

### 3.1 Tabular Match Feed (`src.alignment_report.flatten_alignment_results`)
Converts nested record-level alignment results into flat row dictionaries for the results data table:
```python
from src.alignment_report import flatten_alignment_results

table_rows = flatten_alignment_results(alignment_results)
# Each item in table_rows has:
# - row["record_id"]: str
# - row["start"]: int (0-based)
# - row["end"]: int (exclusive)
# - row["matched_sequence"]: str
# - row["mismatch_count"]: int
# - row["mismatch_positions"]: list[int]
```

### 3.2 Summary Telemetry Feed (`src.alignment_report.summarize_alignment_results`)
Computes aggregate metrics for dashboard summary cards:
```python
from src.alignment_report import summarize_alignment_results

summary = summarize_alignment_results(alignment_results)
# Returns:
# - summary["total_records"]: int
# - summary["records_with_matches"]: int
# - summary["total_matches"]: int
# - summary["total_mismatches"]: int
# - summary["average_mismatches_per_match"]: float
```

---

## 4. Recommended Integration Order

To integrate the independent branches without merge conflicts or broken dependencies, proceed in the following order:

```
[faizan-branch: Core & Architecture]
               │
               ▼
   [Merge Albert's CUDA Branch]
   ├── Connect FASTA parser to find_off_targets_in_records
   └── Wire find_off_targets_cuda into compare_alignment_performance
               │
               ▼
   [Merge Taniya's Dashboard Branch]
   ├── Populate results DataTable via flatten_alignment_results()
   ├── Display metrics via summarize_alignment_results()
   ├── Render memory bar using estimate_alignment_memory()
   └── Display consolidated status via generate_system_report()
               │
               ▼
   [End-to-End Pipeline Verification]
```

1. **Step 1: Baseline Architecture (Current Branch)**  
   Verify that `src/config.py`, `src/models.py`, `src/cpu_alignment.py`, `src/benchmark.py`, `src/gpu_environment.py`, `src/benchmark_comparison.py`, `src/memory_audit.py`, and `src/system_report.py` are merged into the target branch.
2. **Step 2: Connect Genomic Parser**  
   Replace mock record lists with `genomic_parser.parse_fasta(file_path)` to stream live FASTA records into `find_off_targets_in_records()`.
3. **Step 3: Connect CUDA Alignment Engine**  
   Pass `find_off_targets_cuda` into `compare_alignment_performance(..., gpu_function=find_off_targets_cuda)` to measure actual GPU kernel execution and verify result parity against CPU output.
4. **Step 4: Connect Dashboard Telemetry**  
   Wire the TUI dashboard events to invoke `generate_system_report()`, updating status pills and summary panels.

---

## 5. Running the Project Test Suite

Execute the full suite of unit tests using pytest from the repository root:

```bash
# Run all tests across the repository
python -m pytest -v

# Run tests with short traceback for quick debugging
python -m pytest -q

# Run specific module tests
python -m pytest tests/test_system_report.py -v
python -m pytest tests/test_memory_audit.py -v
python -m pytest tests/test_benchmark_comparison.py -v
python -m pytest tests/test_cpu_alignment.py -v
```

All tests are hardware-independent and utilize mocks for GPU-specific operations, ensuring reliable test runs in environments with or without a physical NVIDIA GPU.
