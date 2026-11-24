# CUDA Validation Guide — GeneWeaver

> [!NOTE]
> This document covers Days 7–10 of the GeneWeaver CUDA alignment pipeline built by **Albert**.
> It describes how GPU availability is checked, how memory info is retrieved, how chunked
> alignment works, how to run the CUDA tests, and which tests require real NVIDIA hardware.

---

## Table of Contents

1. [How CUDA Availability Is Checked](#1-how-cuda-availability-is-checked)
2. [How GPU Memory Information Is Retrieved](#2-how-gpu-memory-information-is-retrieved)
3. [How Chunked GPU Alignment Works](#3-how-chunked-gpu-alignment-works)
4. [How to Run CUDA Tests](#4-how-to-run-cuda-tests)
5. [Which Tests Require Real NVIDIA GPU Hardware](#5-which-tests-require-real-nvidia-gpu-hardware)
6. [Module Reference](#6-module-reference)

---

## 1. How CUDA Availability Is Checked

### Entry point: `is_cuda_available()` — `src/cuda_alignment.py`

CUDA availability is checked in two layers:

**Layer 1 — Module-level import guard**

At module load time, GeneWeaver attempts to import `numba.cuda`:

```python
try:
    from numba import cuda
    _CUDA_AVAILABLE = bool(cuda.is_available())
except Exception:
    cuda = None
    _CUDA_AVAILABLE = False
```

A broad `except Exception` (not just `ImportError`) is used because `numba_cuda`
can raise non-import errors (e.g. DLL-load failures) even when the package is installed.

**Layer 2 — Runtime check in `is_cuda_available()`**

```python
def is_cuda_available() -> bool:
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
```

Calling `cuda.is_available()` returns `True` only when both:
- The Numba CUDA package is importable, **and**
- A compatible CUDA driver and device are detected on the host.

> [!IMPORTANT]
> `find_off_targets_gpu()` raises `RuntimeError` immediately if `is_cuda_available()`
> returns `False`. It **never silently falls back to CPU execution.**

---

## 2. How GPU Memory Information Is Retrieved

### Entry point: `get_gpu_memory_info()` — `src/cuda_alignment.py`

Memory information is queried from the active Numba CUDA context:

```python
context = cuda.current_context()
mem_info = context.get_memory_info()
```

`get_memory_info()` returns either a **named tuple** (`mem_info.free`, `mem_info.total`)
in Numba ≥ 0.54, or a **plain tuple** `(free, total)` in older versions.
Both formats are handled:

```python
if hasattr(mem_info, "free"):
    free_bytes = int(mem_info.free)
    total_bytes = int(mem_info.total)
else:
    free_bytes = int(mem_info[0])
    total_bytes = int(mem_info[1])

used_bytes = total_bytes - free_bytes
```

> [!CAUTION]
> No GPU memory values are fabricated or hardcoded. If the CUDA context cannot be
> reached, all memory fields (`free_bytes`, `total_bytes`, `used_bytes`) are set to `None`.

### Return shape

```python
{
    "available": bool,
    "free_bytes": int | None,
    "total_bytes": int | None,
    "used_bytes": int | None,   # always total_bytes - free_bytes
    "message": str,
}
```

### Consolidated diagnostics: `get_cuda_diagnostics()` — `src/cuda_diagnostics.py`

Combines device name retrieval and memory info into a single snapshot:

```python
{
    "available": bool,
    "device_name": str | None,
    "free_bytes": int | None,
    "total_bytes": int | None,
    "used_bytes": int | None,
    "message": str,
}
```

Device name is obtained via `cuda.get_current_device().name`.
Byte names are decoded from UTF-8 automatically.
Both functions **never raise** — all errors are surfaced through the returned dictionary.

---

## 3. How Chunked GPU Alignment Works

### Entry point: `find_off_targets_gpu_chunked()` — `src/cuda_alignment.py`

For large genomes, CUDA memory constraints make it impractical to transfer the entire
sequence to GPU in one shot. `find_off_targets_gpu_chunked()` splits the genome into
overlapping segments and calls `find_off_targets_gpu()` on each segment.

### Algorithm

```
overlap = guide_length - 1

chunk_start = 0
while chunk_start < genome_length:
    chunk_end   = min(chunk_start + chunk_size + overlap, genome_length)
    chunk       = genome[chunk_start : chunk_end]

    chunk_matches = find_off_targets_gpu(chunk, guide, max_mismatches)

    for match in chunk_matches:
        global_start = chunk_start + match.start
        global_end   = chunk_start + match.end

        # Include only if the match starts before the overlap zone of this chunk
        # (or this is the final chunk)
        if global_start < chunk_start + chunk_size  or  chunk_end == genome_length:
            emit AlignmentMatch(global_start, global_end, ...)

    chunk_start += chunk_size
```

### Why overlap of `guide_length - 1`?

A match window of length `guide_length` can start at most `guide_length - 1` bases
before a chunk boundary and still extend into the next chunk.  Adding this overlap
ensures every possible genome window is fully contained in at least one chunk.

### Deduplication

A match whose **global start** falls at `chunk_start + chunk_size` or later is in the
overlap zone — it will also appear at the **beginning** of the next chunk.  It is
emitted by the next chunk and suppressed here.  The exception: the final chunk always
emits all its matches.

### Coordinate guarantee

All returned `AlignmentMatch` objects use **global genome coordinates**:
- `start` is 0-based inclusive, relative to the original genome.
- `end` is exclusive, relative to the original genome.
- `mismatch_positions` are 0-based within the guide window (not global).

```
flowchart:
   genome  ──────────────────────────────────────────────
   chunk 0 |←── chunk_size ──→|←── overlap ──→|
   chunk 1              |←── chunk_size ──→|←── overlap ──→|
   ...
```

---

## 4. How to Run CUDA Tests

### Prerequisites

```powershell
# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Verify installed packages
python -m pip show numba biopython numpy pytest
```

### Run only Albert's CUDA module tests

```powershell
python -m pytest tests/test_cuda_alignment.py tests/test_cuda_diagnostics.py tests/test_cuda_integration.py -v
```

### Run the targeted command from the Day 10 task

```powershell
python -m pytest tests/test_cuda_integration.py tests/test_cuda_alignment.py tests/test_cuda_diagnostics.py -v
```

### Run the full test suite

```powershell
python -m pytest tests/ -v
```

### Run only hardware-independent tests (skip GPU tests)

```powershell
python -m pytest tests/test_cuda_integration.py tests/test_cuda_alignment.py tests/test_cuda_diagnostics.py -v -k "not real_gpu"
```

### Expected output on a machine without a GPU

```
N passed, M skipped, 1 warning
```

Skipped tests are those decorated with:

```python
@pytest.mark.skipif(not is_cuda_available(), reason="NVIDIA CUDA GPU not available")
```

---

## 5. Which Tests Require Real NVIDIA GPU Hardware

The table below lists every test that is **skipped** when no CUDA device is detected.
All other tests in these files are hardware-independent and use mocks.

| Test file | Test name | Why it needs a GPU |
|---|---|---|
| `test_cuda_alignment.py` | `test_real_gpu_exact_match` | Calls `find_off_targets_gpu()` without mocking |
| `test_cuda_alignment.py` | `test_real_gpu_mismatch_detection` | Calls `find_off_targets_gpu()` without mocking |
| `test_cuda_alignment.py` | `TestChunkedGpuAlignment::test_real_gpu_chunked_exact_match` | Calls `find_off_targets_gpu_chunked()` without mocking |
| `test_cuda_alignment.py` | `TestGetGpuMemoryInfo::test_real_gpu_memory_info` | Calls `get_gpu_memory_info()` with real CUDA context |
| `test_cuda_diagnostics.py` | `test_real_gpu_diagnostics` | Calls `get_cuda_diagnostics()` with real CUDA context |
| `test_cuda_integration.py` | `TestRealGpuIntegration::test_gpu_and_cpu_agree_on_exact_matches` | Runs CUDA kernel on real hardware |
| `test_cuda_integration.py` | `TestRealGpuIntegration::test_gpu_chunked_and_cpu_agree` | Runs CUDA kernel on real hardware |
| `test_cuda_integration.py` | `TestRealGpuIntegration::test_real_gpu_diagnostics_consistent` | Queries real GPU memory |

> [!TIP]
> On a machine **without** NVIDIA hardware, all 8 tests above are automatically skipped
> with the message *"NVIDIA CUDA GPU not available"*. The rest of the suite still passes.

---

## 6. Module Reference

### `src/cuda_alignment.py`

| Symbol | Kind | Description |
|---|---|---|
| `encode_sequence(sequence)` | function | Encodes a DNA string to a `uint8` NumPy array (A=0, C=1, G=2, T=3, N=4) |
| `decode_sequence(encoded)` | function | Decodes a `uint8` NumPy array back to a DNA string |
| `is_cuda_available()` | function | Returns `True` if a usable CUDA device is detected |
| `dna_alignment_kernel` | CUDA kernel | One thread per genome window; counts and records mismatches |
| `find_off_targets_gpu(genome, guide, max_mismatches)` | function | Runs CUDA kernel; returns `list[AlignmentMatch]` |
| `find_off_targets_gpu_chunked(genome, guide, max_mismatches, chunk_size)` | function | Chunked wrapper; calls `find_off_targets_gpu` per segment |
| `get_gpu_memory_info()` | function | Returns `dict` with free/total/used bytes; never raises |

### `src/cuda_diagnostics.py`

| Symbol | Kind | Description |
|---|---|---|
| `get_cuda_diagnostics()` | function | Returns `dict` with device name + memory info; never raises |

### `src/models.py` — `AlignmentMatch`

| Field | Type | Meaning |
|---|---|---|
| `start` | `int` | 0-based inclusive start in the **global** genome |
| `end` | `int` | Exclusive end in the **global** genome |
| `matched_sequence` | `str` | Genome subsequence `genome[start:end]` |
| `mismatch_count` | `int` | Number of mismatched bases |
| `mismatch_positions` | `list[int]` | 0-based positions **within the window** (not global) |
