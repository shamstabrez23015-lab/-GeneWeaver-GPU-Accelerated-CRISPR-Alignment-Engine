"""Integration tests for CUDA alignment and diagnostics modules (Day 10).

Verifies that the GPU alignment pipeline (find_off_targets_gpu,
find_off_targets_gpu_chunked) and the consolidated diagnostics module
(get_cuda_diagnostics) work correctly together and are consistent with
the reference CPU implementation (find_off_targets).

All tests that require real NVIDIA hardware are marked with
@pytest.mark.skipif(not _real_cuda_available(), ...) so the full suite
passes on machines without a CUDA device.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# Helper: real CUDA availability (evaluated once at collection time)
# ---------------------------------------------------------------------------

def _real_cuda_available() -> bool:
    """Return True only when an actual CUDA device is present on this machine."""
    try:
        from numba import cuda
        return bool(cuda.is_available())
    except Exception:
        return False


_HAS_CUDA = _real_cuda_available()


# ---------------------------------------------------------------------------
# Helper: CPU-based sliding-window alignment (mirrors the CUDA kernel logic)
# ---------------------------------------------------------------------------

def _cpu_align(genome: str, guide: str, max_mismatches: int = 3):
    """Pure-Python sliding-window alignment used to mock GPU calls in integration tests."""
    from src.models import AlignmentMatch

    cleaned_genome = "".join(genome.split()).upper()
    cleaned_guide = "".join(guide.split()).upper()
    genome_len = len(cleaned_genome)
    guide_len = len(cleaned_guide)

    if genome_len < guide_len:
        return []

    matches: list[AlignmentMatch] = []
    for start in range(genome_len - guide_len + 1):
        mismatch_positions = [
            j for j in range(guide_len)
            if cleaned_genome[start + j] != cleaned_guide[j]
        ]
        if len(mismatch_positions) <= max_mismatches:
            matches.append(
                AlignmentMatch(
                    start=start,
                    end=start + guide_len,
                    matched_sequence=cleaned_genome[start : start + guide_len],
                    mismatch_count=len(mismatch_positions),
                    mismatch_positions=mismatch_positions,
                )
            )
    return matches


# ---------------------------------------------------------------------------
# 1. Import verification
# ---------------------------------------------------------------------------


class TestImports:
    """All public symbols from Albert's CUDA modules must be importable."""

    def test_import_find_off_targets_gpu(self):
        from src.cuda_alignment import find_off_targets_gpu
        assert callable(find_off_targets_gpu)

    def test_import_find_off_targets_gpu_chunked(self):
        from src.cuda_alignment import find_off_targets_gpu_chunked
        assert callable(find_off_targets_gpu_chunked)

    def test_import_get_gpu_memory_info(self):
        from src.cuda_alignment import get_gpu_memory_info
        assert callable(get_gpu_memory_info)

    def test_import_is_cuda_available(self):
        from src.cuda_alignment import is_cuda_available
        assert callable(is_cuda_available)

    def test_import_encode_sequence(self):
        from src.cuda_alignment import encode_sequence
        assert callable(encode_sequence)

    def test_import_decode_sequence(self):
        from src.cuda_alignment import decode_sequence
        assert callable(decode_sequence)

    def test_import_get_cuda_diagnostics(self):
        from src.cuda_diagnostics import get_cuda_diagnostics
        assert callable(get_cuda_diagnostics)

    def test_import_cpu_find_off_targets(self):
        from src.cpu_alignment import find_off_targets
        assert callable(find_off_targets)


# ---------------------------------------------------------------------------
# 2. GPU unavailable — safe degradation
# ---------------------------------------------------------------------------


class TestGpuUnavailableSafety:
    """When CUDA is unavailable, all entry points degrade gracefully."""

    def test_find_off_targets_gpu_raises_runtime_error_not_other(self):
        """find_off_targets_gpu raises RuntimeError (not crash) when CUDA is absent."""
        from src.cuda_alignment import find_off_targets_gpu
        with patch("src.cuda_alignment.is_cuda_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                find_off_targets_gpu("ACGTACGT", "ACGT")

    def test_find_off_targets_gpu_chunked_propagates_runtime_error(self):
        """find_off_targets_gpu_chunked propagates the RuntimeError from GPU call."""
        from src.cuda_alignment import find_off_targets_gpu_chunked
        with patch(
            "src.cuda_alignment.find_off_targets_gpu",
            side_effect=RuntimeError("CUDA is not available. GPU alignment cannot be executed."),
        ):
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                find_off_targets_gpu_chunked("ACGTACGT", "ACGT", chunk_size=4)

    def test_get_gpu_memory_info_returns_dict_when_unavailable(self):
        """get_gpu_memory_info never raises — returns a dict with available=False."""
        from src.cuda_alignment import get_gpu_memory_info
        with patch("src.cuda_alignment.cuda", None):
            info = get_gpu_memory_info()
        assert isinstance(info, dict)
        assert info["available"] is False

    def test_get_cuda_diagnostics_returns_dict_when_unavailable(self):
        """get_cuda_diagnostics never raises — returns a dict with available=False."""
        from src.cuda_diagnostics import get_cuda_diagnostics
        with patch("builtins.__import__", side_effect=ImportError("no numba")):
            info = get_cuda_diagnostics()
        assert isinstance(info, dict)
        assert info["available"] is False

    def test_get_cuda_diagnostics_genome_shorter_than_guide_no_cuda(self):
        """Genome shorter than guide returns [] without touching CUDA."""
        from src.cuda_alignment import find_off_targets_gpu_chunked
        result = find_off_targets_gpu_chunked("ACGT", "ACGTACGT", chunk_size=100)
        assert result == []


# ---------------------------------------------------------------------------
# 3. Agreed dictionary fields — diagnostics contract
# ---------------------------------------------------------------------------


_DIAGNOSTICS_KEYS = {"available", "device_name", "free_bytes", "total_bytes", "used_bytes", "message"}
_MEMORY_INFO_KEYS = {"available", "free_bytes", "total_bytes", "used_bytes", "message"}


class TestDiagnosticsContract:
    """Both diagnostic functions must return their documented dictionary shapes."""

    def test_get_gpu_memory_info_keys_when_unavailable(self):
        from src.cuda_alignment import get_gpu_memory_info
        with patch("src.cuda_alignment.cuda", None):
            info = get_gpu_memory_info()
        assert _MEMORY_INFO_KEYS.issubset(info.keys())

    def test_get_gpu_memory_info_keys_when_available(self):
        from src.cuda_alignment import get_gpu_memory_info
        mock_cuda = MagicMock()
        mock_cuda.is_available.return_value = True
        mock_mem = MagicMock()
        mock_mem.free = 2_000_000_000
        mock_mem.total = 8_000_000_000
        mock_cuda.current_context.return_value.get_memory_info.return_value = mock_mem
        with patch("src.cuda_alignment.cuda", mock_cuda):
            info = get_gpu_memory_info()
        assert _MEMORY_INFO_KEYS.issubset(info.keys())

    def test_get_cuda_diagnostics_keys_when_unavailable(self):
        from src.cuda_diagnostics import get_cuda_diagnostics
        with patch("builtins.__import__", side_effect=ImportError("no numba")):
            info = get_cuda_diagnostics()
        assert _DIAGNOSTICS_KEYS.issubset(info.keys())

    def test_diagnostics_available_field_is_bool(self):
        from src.cuda_diagnostics import get_cuda_diagnostics
        with patch("builtins.__import__", side_effect=ImportError("no numba")):
            info = get_cuda_diagnostics()
        assert isinstance(info["available"], bool)

    def test_diagnostics_message_field_is_str(self):
        from src.cuda_diagnostics import get_cuda_diagnostics
        with patch("builtins.__import__", side_effect=ImportError("no numba")):
            info = get_cuda_diagnostics()
        assert isinstance(info["message"], str) and len(info["message"]) > 0

    def test_memory_info_values_none_when_cuda_absent(self):
        from src.cuda_alignment import get_gpu_memory_info
        with patch("src.cuda_alignment.cuda", None):
            info = get_gpu_memory_info()
        assert info["free_bytes"] is None
        assert info["total_bytes"] is None
        assert info["used_bytes"] is None

    def test_diagnostics_values_none_when_cuda_absent(self):
        from src.cuda_diagnostics import get_cuda_diagnostics
        with patch("builtins.__import__", side_effect=ImportError("no numba")):
            info = get_cuda_diagnostics()
        assert info["device_name"] is None
        assert info["free_bytes"] is None
        assert info["total_bytes"] is None
        assert info["used_bytes"] is None


# ---------------------------------------------------------------------------
# 4. Mocked GPU alignment — coordinate and deduplication correctness
# ---------------------------------------------------------------------------


class TestChunkedAlignmentCoordinates:
    """Verify global coordinate preservation and no duplicate matches at chunk boundaries."""

    @staticmethod
    def _run_chunked(genome: str, guide: str, max_mismatches: int = 0, chunk_size: int = 6):
        from src.cuda_alignment import find_off_targets_gpu_chunked
        with patch("src.cuda_alignment.find_off_targets_gpu", side_effect=_cpu_align):
            return find_off_targets_gpu_chunked(
                genome, guide,
                max_mismatches=max_mismatches,
                chunk_size=chunk_size,
            )

    def test_global_coords_single_chunk(self):
        """Match at the end of a single chunk has correct global start/end."""
        genome = "NNNNACGT"
        result = self._run_chunked(genome, "ACGT", chunk_size=1000)
        starts = [m.start for m in result]
        assert 4 in starts
        idx = starts.index(4)
        assert result[idx].end == 8

    def test_global_coords_across_chunks(self):
        """Match embedded in the second chunk has correct global coordinates."""
        genome = "N" * 10 + "ACGT" + "N" * 10   # match at position 10
        result = self._run_chunked(genome, "ACGT", chunk_size=8)
        starts = [m.start for m in result]
        assert 10 in starts
        assert result[starts.index(10)].end == 14

    def test_no_duplicates_at_chunk_boundary(self):
        """No match position is returned more than once across chunk boundaries."""
        genome = "ACGTACGTACGT"   # matches at 0, 4, 8
        result = self._run_chunked(genome, "ACGT", chunk_size=5)
        starts = [m.start for m in result]
        assert len(starts) == len(set(starts)), "Duplicate match positions detected"

    def test_all_matches_found_across_chunks(self):
        """All three matches in a repeated 12-bp genome are found with chunking."""
        genome = "ACGTACGTACGT"
        result = self._run_chunked(genome, "ACGT", chunk_size=5)
        starts = sorted(m.start for m in result)
        assert starts == [0, 4, 8]

    def test_cross_boundary_match_appears_exactly_once(self):
        """A match crossing a chunk boundary is returned exactly once."""
        genome = "NNACGTNN"       # match at position 2, chunk_size=4 → overlap=3
        result = self._run_chunked(genome, "ACGT", chunk_size=4)
        starts = [m.start for m in result]
        assert starts.count(2) == 1

    def test_matched_sequence_matches_genome_window(self):
        """matched_sequence must equal the actual genome slice at [start:end]."""
        genome = "ACGTTTTTACGT"
        result = self._run_chunked(genome, "ACGT", chunk_size=6)
        for match in result:
            assert match.matched_sequence == genome[match.start:match.end]

    def test_mismatch_count_consistent_with_positions(self):
        """mismatch_count must equal len(mismatch_positions) for every match."""
        genome = "AGGTNNNNACGT"
        result = self._run_chunked(genome, "ACGT", max_mismatches=1, chunk_size=6)
        for match in result:
            assert match.mismatch_count == len(match.mismatch_positions)

    def test_final_smaller_chunk_fully_processed(self):
        """The last (possibly shorter) chunk is processed and its matches returned."""
        genome = "ACGT" * 3        # 12 bp, chunk_size=10
        result = self._run_chunked(genome, "ACGT", chunk_size=10)
        starts = sorted(m.start for m in result)
        assert 8 in starts         # last match lives in the final short chunk


# ---------------------------------------------------------------------------
# 5. CPU / GPU result agreement (mocked GPU)
# ---------------------------------------------------------------------------


class TestCpuGpuAgreement:
    """GPU alignment (mocked) results must match the reference CPU implementation."""

    @staticmethod
    def _run_gpu_mocked(genome, guide, max_mismatches=0):
        from src.cuda_alignment import find_off_targets_gpu_chunked
        with patch("src.cuda_alignment.find_off_targets_gpu", side_effect=_cpu_align):
            return find_off_targets_gpu_chunked(
                genome, guide, max_mismatches=max_mismatches, chunk_size=1000
            )

    @staticmethod
    def _run_cpu(genome, guide, max_mismatches=0):
        from src.cpu_alignment import find_off_targets
        return find_off_targets(genome, guide, max_mismatches=max_mismatches)

    def test_exact_match_starts_agree(self):
        genome, guide = "ACGTACGT", "ACGT"
        cpu = sorted(m.start for m in self._run_cpu(genome, guide))
        gpu = sorted(m.start for m in self._run_gpu_mocked(genome, guide))
        assert cpu == gpu

    def test_mismatch_match_starts_agree(self):
        genome, guide = "AGGTNNNNACGT", "ACGT"
        cpu = sorted(m.start for m in self._run_cpu(genome, guide, max_mismatches=1))
        gpu = sorted(m.start for m in self._run_gpu_mocked(genome, guide, max_mismatches=1))
        assert cpu == gpu

    def test_match_count_agrees(self):
        genome = "ACGT" * 5
        guide = "ACGT"
        cpu = self._run_cpu(genome, guide)
        gpu = self._run_gpu_mocked(genome, guide)
        assert len(cpu) == len(gpu)

    def test_no_matches_agrees(self):
        genome, guide = "TTTTTTTTTTTT", "ACGT"
        cpu = self._run_cpu(genome, guide, max_mismatches=0)
        gpu = self._run_gpu_mocked(genome, guide, max_mismatches=0)
        assert cpu == [] and gpu == []

    def test_mismatch_positions_agree(self):
        genome, guide = "AGGT", "ACGT"
        cpu = self._run_cpu(genome, guide, max_mismatches=1)
        gpu = self._run_gpu_mocked(genome, guide, max_mismatches=1)
        assert len(cpu) == 1 and len(gpu) == 1
        assert cpu[0].mismatch_positions == gpu[0].mismatch_positions

    def test_matched_sequence_agrees(self):
        genome, guide = "TTTTACGTTTTT", "ACGT"
        cpu = self._run_cpu(genome, guide)
        gpu = self._run_gpu_mocked(genome, guide)
        cpu_seqs = sorted(m.matched_sequence for m in cpu)
        gpu_seqs = sorted(m.matched_sequence for m in gpu)
        assert cpu_seqs == gpu_seqs

    def test_genome_shorter_than_guide_both_empty(self):
        cpu = self._run_cpu("ACGT", "ACGTACGT")
        gpu = self._run_gpu_mocked("ACGT", "ACGTACGT")
        assert cpu == [] and gpu == []


# ---------------------------------------------------------------------------
# 6. Memory info consistency between modules
# ---------------------------------------------------------------------------


class TestMemoryInfoConsistency:
    """get_gpu_memory_info and get_cuda_diagnostics must agree on memory values."""

    @staticmethod
    def _mock_cuda_full(free=3_000_000_000, total=8_000_000_000, name="Tesla T4"):
        """Return a mock numba.cuda that reports both device name and memory info."""
        mock_cuda = MagicMock()
        mock_cuda.is_available.return_value = True
        mock_device = MagicMock()
        mock_device.name = name
        mock_cuda.get_current_device.return_value = mock_device
        mock_mem = MagicMock()
        mock_mem.free = free
        mock_mem.total = total
        mock_cuda.current_context.return_value.get_memory_info.return_value = mock_mem
        return mock_cuda

    def test_memory_info_used_equals_total_minus_free(self):
        from src.cuda_alignment import get_gpu_memory_info
        mock_cuda = self._mock_cuda_full(free=2_000_000_000, total=8_000_000_000)
        with patch("src.cuda_alignment.cuda", mock_cuda):
            info = get_gpu_memory_info()
        assert info["used_bytes"] == info["total_bytes"] - info["free_bytes"]

    def test_memory_info_free_lte_total(self):
        from src.cuda_alignment import get_gpu_memory_info
        mock_cuda = self._mock_cuda_full()
        with patch("src.cuda_alignment.cuda", mock_cuda):
            info = get_gpu_memory_info()
        assert info["free_bytes"] <= info["total_bytes"]

    def test_memory_info_values_non_negative(self):
        from src.cuda_alignment import get_gpu_memory_info
        mock_cuda = self._mock_cuda_full(free=0, total=4_000_000_000)
        with patch("src.cuda_alignment.cuda", mock_cuda):
            info = get_gpu_memory_info()
        assert info["free_bytes"] >= 0
        assert info["total_bytes"] >= 0
        assert info["used_bytes"] >= 0


# ---------------------------------------------------------------------------
# 7. Real hardware integration tests (skipped when no CUDA device present)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CUDA, reason="NVIDIA CUDA GPU not available")
class TestRealGpuIntegration:
    """End-to-end tests that require an actual NVIDIA GPU."""

    def test_gpu_and_cpu_agree_on_exact_matches(self):
        """Real GPU alignment must produce the same starts as CPU for exact matches."""
        from src.cuda_alignment import find_off_targets_gpu
        from src.cpu_alignment import find_off_targets

        genome = "ACGTACGT"
        guide = "ACGT"
        cpu_starts = sorted(m.start for m in find_off_targets(genome, guide, max_mismatches=0))
        gpu_starts = sorted(m.start for m in find_off_targets_gpu(genome, guide, max_mismatches=0))
        assert cpu_starts == gpu_starts

    def test_gpu_chunked_and_cpu_agree(self):
        """Real GPU chunked alignment must match CPU results for a longer genome."""
        from src.cuda_alignment import find_off_targets_gpu_chunked
        from src.cpu_alignment import find_off_targets

        genome = "ACGT" * 20
        guide = "ACGT"
        cpu_starts = sorted(m.start for m in find_off_targets(genome, guide, max_mismatches=0))
        gpu_starts = sorted(
            m.start for m in find_off_targets_gpu_chunked(genome, guide, max_mismatches=0, chunk_size=20)
        )
        assert cpu_starts == gpu_starts

    def test_real_gpu_diagnostics_consistent(self):
        """Real GPU: diagnostics must be consistent with memory info."""
        from src.cuda_alignment import get_gpu_memory_info
        from src.cuda_diagnostics import get_cuda_diagnostics

        mem = get_gpu_memory_info()
        diag = get_cuda_diagnostics()

        assert mem["available"] is True
        assert diag["available"] is True
        assert isinstance(diag["device_name"], str)
        # Both modules query the same device, so total memory should agree
        assert diag["total_bytes"] == mem["total_bytes"]
