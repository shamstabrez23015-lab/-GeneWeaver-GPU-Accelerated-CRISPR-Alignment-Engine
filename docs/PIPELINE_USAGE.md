# GeneWeaver Pipeline Usage Guide

The **GeneWeaver Pipeline** (`src.pipeline`) provides a unified programmatic interface that orchestrates FASTA genomic parsing, DNA sequence validation, CPU alignment, GPU-accelerated chunked kernel execution, memory auditing, performance benchmarking, and consolidated diagnostic reporting.

---

## 1. Quickstart

To run the pipeline in Python:

```python
from src.pipeline import run_geneweaver_pipeline

# Execute end-to-end alignment pipeline
result = run_geneweaver_pipeline(
    fasta_path="data/sample.fasta",
    guide_sequence="GAGTCCGAGCAGAAGAAGAA",
    max_mismatches=3,
    chunk_size=100000,
    use_gpu=True,
)

print("Pipeline Status:", result["status"])
print("Summary Message:", result["message"])
```

---

## 2. Providing Input Data

### 2.1 FASTA Genomic File (`fasta_path`)
- Provide a valid path (relative or absolute) to a FASTA-formatted file (e.g., `.fasta`, `.fa`, `.fna`).
- Sequences must contain valid DNA bases (`A`, `T`, `G`, `C`, and ambiguous base `N`).
- Files containing zero records or invalid sequence characters will raise a descriptive `ValueError`.

### 2.2 CRISPR Guide Sequence (`guide_sequence`)
- Supply a string representing the query CRISPR guide RNA/DNA sequence (typically 20–23 nucleotides).
- Whitespace, line breaks, and lowercase letters are automatically normalized and stripped.
- An empty sequence or invalid nucleotides outside `ATGCN` will raise a `ValueError`.

---

## 3. Execution Modes: CPU vs GPU

### 3.1 CPU-Only Mode (`use_gpu=False`)
When `use_gpu=False` is set:
- The pipeline processes all FASTA records exclusively using the modular CPU sliding-window alignment engine (`src.cpu_alignment`).
- No GPU allocations or CUDA checks are made.
- `result["gpu_results"]` and `result["benchmark_results"]` are set to `None`.
- The system operates reliably on any machine without requiring GPU drivers or NVIDIA hardware.

```python
result = run_geneweaver_pipeline(
    fasta_path="data/genome.fasta",
    guide_sequence="ATGCATGCATGCATGCATGC",
    use_gpu=False,
)
```

### 3.2 GPU Acceleration Mode (`use_gpu=True`)
When `use_gpu=True` is enabled:
1. **Hardware Detection**: The pipeline queries CUDA support and VRAM telemetry via `src.cuda_diagnostics`.
2. **Chunked Kernel Execution**: If a compatible CUDA device is detected, large genomic records are divided into overlapping chunks of `chunk_size` bases (with an overlap of `guide_length - 1` bases to ensure zero missed boundary matches) and aligned in parallel on the GPU (`src.cuda_alignment`).
3. **Parity & Benchmarking**: If GPU execution completes successfully, performance benchmarking (`src.benchmark_comparison`) runs to measure the empirical GPU speedup factor ($T_{\text{CPU}} / T_{\text{GPU}}$) and verify that CPU and GPU results are identical.
4. **Graceful Fallback**: If CUDA is not detected or an unexpected device error occurs, the pipeline catches the error without crashing, logs the diagnostic message, sets `gpu_results=None`, skips speedup calculation, and relies on the CPU alignment results.

---

## 4. Understanding Output Fields

The dictionary returned by `run_geneweaver_pipeline` contains:

```python
{
    "records": [...],            # List of parsed FASTA record dictionaries
    "cpu_results": [...],        # Per-record CPU alignment matches (AlignmentMatch objects)
    "gpu_results": [...] | None, # Per-record GPU alignment matches (None if GPU did not run)
    "benchmark_results": {...} | None, # Execution timings, speedup factor, and parity check
    "memory_estimate": {...},    # Theoretical buffer sizes and peak device memory estimates
    "gpu_diagnostics": {...},    # CUDA device availability, device model, and VRAM telemetry
    "system_report": {...},      # Consolidated report from src.system_report
    "status": "Completed",       # Pipeline status: "Completed" or "Failed"
    "message": "...",            # Human-readable summary of pipeline execution and fallback notes
}
```

### 4.1 Interpreting Memory Estimates
`result["memory_estimate"]` provides theoretical memory planning data:
- `chunk_buffer_bytes`: Buffer size needed on the device for each chunk, including boundary overlap.
- `estimated_peak_device_input_bytes`: Total input buffer memory needed on the GPU (`chunk_buffer_bytes + guide_input_bytes`).
- `note`: Clearly identifies that these values are pre-flight theoretical estimates, not runtime measurements.

### 4.2 Interpreting Benchmark Results
When GPU execution succeeds, `result["benchmark_results"]` reports:
- `cpu_average_seconds`: Mean CPU execution time.
- `gpu_average_seconds`: Mean GPU execution time.
- `speedup`: Speedup ratio ($T_{\text{CPU}} / T_{\text{GPU}}$). Speedup is never fabricated or reported if GPU execution was skipped.
- `results_match`: Boolean verifying that CPU and GPU discovered the exact same alignment sites.

---

## 5. Dashboard Integration

Taniya's Textual TUI dashboard (`src.dashboard`) can consume pipeline results directly:

```python
from src.alignment_report import flatten_alignment_results
from src.dashboard import GeneWeaverDashboard

app = GeneWeaverDashboard()

# Update alignment table
app.update_alignment_results(flatten_alignment_results(result["cpu_results"]))

# Update benchmark panel if GPU benchmark was performed
if result["benchmark_results"] is not None:
    app.update_benchmark_results(result["benchmark_results"])

# Update GPU device status and VRAM
app.update_gpu_diagnostics(result["gpu_diagnostics"])
app.update_scan_status(result["status"])
```

---

## 6. Running the Test Suite

Run the pipeline tests and full regression suite from the project root:

```bash
# Run new pipeline tests
python -m pytest tests/test_pipeline.py -v

# Run full project test suite
python -m pytest -v
```
