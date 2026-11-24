"""Unit tests for the consolidated CUDA diagnostics module (Day 9)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

# Ensure project root is on sys.path when running pytest directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.cuda_diagnostics import get_cuda_diagnostics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {"available", "device_name", "free_bytes", "total_bytes", "used_bytes", "message"}


def _make_mock_cuda(
    *,
    is_available: bool = True,
    device_name: str | bytes = "Tesla T4",
    free: int = 4_000_000_000,
    total: int = 8_000_000_000,
    availability_raises: Exception | None = None,
    device_raises: Exception | None = None,
    mem_raises: Exception | None = None,
    use_plain_tuple: bool = False,
) -> MagicMock:
    """Build a fully-configured MagicMock that mimics numba.cuda."""
    mock_cuda = MagicMock()

    # --- is_available ---
    if availability_raises is not None:
        mock_cuda.is_available.side_effect = availability_raises
    else:
        mock_cuda.is_available.return_value = is_available

    # --- get_current_device ---
    if device_raises is not None:
        mock_cuda.get_current_device.side_effect = device_raises
    else:
        mock_device = MagicMock()
        mock_device.name = device_name
        mock_cuda.get_current_device.return_value = mock_device

    # --- current_context / get_memory_info ---
    if mem_raises is not None:
        mock_cuda.current_context.return_value.get_memory_info.side_effect = mem_raises
    elif use_plain_tuple:
        mock_cuda.current_context.return_value.get_memory_info.return_value = (free, total)
    else:
        mock_mem = MagicMock()
        mock_mem.free = free
        mock_mem.total = total
        mock_cuda.current_context.return_value.get_memory_info.return_value = mock_mem

    return mock_cuda


# ---------------------------------------------------------------------------
# Required dictionary fields
# ---------------------------------------------------------------------------


class TestRequiredFields:
    """get_cuda_diagnostics() must always return all six required keys."""

    def test_all_keys_present_when_available(self):
        mock_cuda = _make_mock_cuda()
        with patch("src.cuda_diagnostics._cuda", mock_cuda, create=True), \
             patch("builtins.__import__", side_effect=_import_patch(mock_cuda)):
            info = get_cuda_diagnostics()
        assert _REQUIRED_KEYS.issubset(info.keys()), f"Missing: {_REQUIRED_KEYS - info.keys()}"

    def test_all_keys_present_when_unavailable(self):
        with patch("builtins.__import__", side_effect=ImportError("no numba")):
            info = get_cuda_diagnostics()
        assert _REQUIRED_KEYS.issubset(info.keys()), f"Missing: {_REQUIRED_KEYS - info.keys()}"


def _import_patch(mock_cuda):
    """Return a side_effect function that substitutes numba.cuda with mock_cuda."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _patched(name, *args, **kwargs):
        if name == "numba":
            mod = MagicMock()
            mod.cuda = mock_cuda
            return mod
        return real_import(name, *args, **kwargs)

    return _patched


# ---------------------------------------------------------------------------
# CUDA unavailable paths
# ---------------------------------------------------------------------------


class TestCudaUnavailable:
    """When CUDA is unreachable, all non-available fields must be None."""

    def _run_no_numba(self):
        """Simulate Numba not being importable."""
        with patch("builtins.__import__", side_effect=ImportError("no numba")):
            return get_cuda_diagnostics()

    def _run_no_device(self):
        mock_cuda = _make_mock_cuda(is_available=False)
        with patch("builtins.__import__", side_effect=_import_patch(mock_cuda)):
            return get_cuda_diagnostics()

    def test_available_false_no_numba(self):
        info = self._run_no_numba()
        assert info["available"] is False

    def test_device_name_none_no_numba(self):
        info = self._run_no_numba()
        assert info["device_name"] is None

    def test_memory_values_none_no_numba(self):
        info = self._run_no_numba()
        assert info["free_bytes"] is None
        assert info["total_bytes"] is None
        assert info["used_bytes"] is None

    def test_message_non_empty_no_numba(self):
        info = self._run_no_numba()
        assert isinstance(info["message"], str) and len(info["message"]) > 0

    def test_available_false_no_device(self):
        info = self._run_no_device()
        assert info["available"] is False

    def test_device_name_none_no_device(self):
        info = self._run_no_device()
        assert info["device_name"] is None

    def test_memory_values_none_no_device(self):
        info = self._run_no_device()
        assert info["free_bytes"] is None
        assert info["total_bytes"] is None
        assert info["used_bytes"] is None

    def test_unavailable_in_message_no_device(self):
        info = self._run_no_device()
        assert "unavailable" in info["message"].lower()

    def test_no_fabricated_values_when_unavailable(self):
        """The function must not invent device names or memory numbers."""
        info = self._run_no_numba()
        assert info["device_name"] is None
        assert info["free_bytes"] is None
        assert info["total_bytes"] is None
        assert info["used_bytes"] is None


# ---------------------------------------------------------------------------
# CUDA available — successful diagnostics
# ---------------------------------------------------------------------------


class TestCudaAvailable:
    """When a CUDA device is present and queryable, all fields must be populated."""

    @staticmethod
    def _run(free=4_000_000_000, total=8_000_000_000, device_name="Tesla T4"):
        mock_cuda = _make_mock_cuda(free=free, total=total, device_name=device_name)
        with patch("builtins.__import__", side_effect=_import_patch(mock_cuda)):
            return get_cuda_diagnostics()

    def test_available_true(self):
        assert self._run()["available"] is True

    def test_device_name_string(self):
        info = self._run(device_name="NVIDIA A100")
        assert info["device_name"] == "NVIDIA A100"

    def test_device_name_bytes_decoded(self):
        """Device names reported as bytes must be decoded to str."""
        mock_cuda = _make_mock_cuda(device_name=b"Tesla V100")
        with patch("builtins.__import__", side_effect=_import_patch(mock_cuda)):
            info = get_cuda_diagnostics()
        assert isinstance(info["device_name"], str)
        assert "Tesla V100" in info["device_name"]

    def test_correct_free_memory(self):
        info = self._run(free=3_000_000_000, total=8_000_000_000)
        assert info["free_bytes"] == 3_000_000_000

    def test_correct_total_memory(self):
        info = self._run(free=3_000_000_000, total=8_000_000_000)
        assert info["total_bytes"] == 8_000_000_000

    def test_correct_used_memory(self):
        free, total = 3_000_000_000, 8_000_000_000
        info = self._run(free=free, total=total)
        assert info["used_bytes"] == total - free

    def test_used_equals_total_minus_free(self):
        free, total = 1_073_741_824, 4_294_967_296
        info = self._run(free=free, total=total)
        assert info["free_bytes"] + info["used_bytes"] == info["total_bytes"]

    def test_free_does_not_exceed_total(self):
        free, total = 2_000_000_000, 8_000_000_000
        info = self._run(free=free, total=total)
        assert info["free_bytes"] <= info["total_bytes"]

    def test_memory_values_non_negative(self):
        info = self._run(free=0, total=8_000_000_000)
        assert info["free_bytes"] >= 0
        assert info["total_bytes"] >= 0
        assert info["used_bytes"] >= 0

    def test_memory_values_are_integers(self):
        info = self._run()
        assert isinstance(info["free_bytes"], int)
        assert isinstance(info["total_bytes"], int)
        assert isinstance(info["used_bytes"], int)

    def test_message_contains_device_name(self):
        info = self._run(device_name="RTX 4090")
        assert "RTX 4090" in info["message"]

    def test_message_non_empty(self):
        info = self._run()
        assert isinstance(info["message"], str) and len(info["message"]) > 0

    def test_plain_tuple_memory_info(self):
        """get_memory_info() returning a plain (free, total) tuple is handled correctly."""
        free, total = 1_000_000_000, 4_000_000_000
        mock_cuda = _make_mock_cuda(free=free, total=total, use_plain_tuple=True)
        with patch("builtins.__import__", side_effect=_import_patch(mock_cuda)):
            info = get_cuda_diagnostics()
        assert info["available"] is True
        assert info["free_bytes"] == free
        assert info["total_bytes"] == total
        assert info["used_bytes"] == total - free


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """get_cuda_diagnostics() must never raise; errors go into the returned dict."""

    def test_function_never_raises_on_import_error(self):
        with patch("builtins.__import__", side_effect=Exception("catastrophic")):
            info = get_cuda_diagnostics()
        assert isinstance(info, dict)

    def test_function_never_raises_on_availability_error(self):
        mock_cuda = _make_mock_cuda(availability_raises=RuntimeError("CUDA driver crash"))
        with patch("builtins.__import__", side_effect=_import_patch(mock_cuda)):
            info = get_cuda_diagnostics()
        assert isinstance(info, dict)
        assert info["available"] is False

    def test_cuda_is_available_raises(self):
        mock_cuda = _make_mock_cuda(availability_raises=RuntimeError("driver error"))
        with patch("builtins.__import__", side_effect=_import_patch(mock_cuda)):
            info = get_cuda_diagnostics()
        assert info["available"] is False
        assert "unavailable" in info["message"].lower()

    def test_device_query_raises(self):
        """When get_current_device() raises, the function should still not crash."""
        mock_cuda = _make_mock_cuda(device_raises=RuntimeError("no active device"))
        with patch("builtins.__import__", side_effect=_import_patch(mock_cuda)):
            info = get_cuda_diagnostics()
        # Both device name and memory may fail — available could be False
        assert isinstance(info, dict)
        assert _REQUIRED_KEYS.issubset(info.keys())

    def test_memory_retrieval_raises(self):
        """When get_memory_info() raises, available may still be True with memory=None."""
        mock_cuda = _make_mock_cuda(mem_raises=RuntimeError("OOM"))
        with patch("builtins.__import__", side_effect=_import_patch(mock_cuda)):
            info = get_cuda_diagnostics()
        assert isinstance(info, dict)
        # Device name was retrieved successfully so available should be True
        assert info["available"] is True
        assert info["free_bytes"] is None
        assert info["total_bytes"] is None
        assert info["used_bytes"] is None

    def test_memory_error_message_explains_failure(self):
        mock_cuda = _make_mock_cuda(mem_raises=RuntimeError("memory query failed"))
        with patch("builtins.__import__", side_effect=_import_patch(mock_cuda)):
            info = get_cuda_diagnostics()
        assert isinstance(info["message"], str)
        assert len(info["message"]) > 0

    def test_missing_device_name_attribute(self):
        """When device.name is missing, fall back to 'Unknown CUDA Device'."""
        mock_cuda = _make_mock_cuda()
        # Remove the name attribute from the device mock
        device_mock = MagicMock(spec=[])  # spec=[] means no attributes
        mock_cuda.get_current_device.return_value = device_mock
        with patch("builtins.__import__", side_effect=_import_patch(mock_cuda)):
            info = get_cuda_diagnostics()
        assert info["device_name"] is not None
        assert isinstance(info["device_name"], str)


# ---------------------------------------------------------------------------
# Dictionary structure invariants
# ---------------------------------------------------------------------------


class TestDictionaryInvariants:
    """Structural guarantees that must hold regardless of CUDA state."""

    def test_available_is_bool(self):
        with patch("builtins.__import__", side_effect=ImportError("no numba")):
            info = get_cuda_diagnostics()
        assert isinstance(info["available"], bool)

    def test_message_is_always_str(self):
        with patch("builtins.__import__", side_effect=ImportError("no numba")):
            info = get_cuda_diagnostics()
        assert isinstance(info["message"], str)

    def test_six_keys_exactly_minimum(self):
        """The returned dict must contain at least the six documented keys."""
        with patch("builtins.__import__", side_effect=ImportError("no numba")):
            info = get_cuda_diagnostics()
        assert _REQUIRED_KEYS.issubset(info.keys())

    def test_no_fabricated_device_name_when_unavailable(self):
        with patch("builtins.__import__", side_effect=ImportError("no numba")):
            info = get_cuda_diagnostics()
        assert info["device_name"] is None

    def test_no_fabricated_memory_when_unavailable(self):
        with patch("builtins.__import__", side_effect=ImportError("no numba")):
            info = get_cuda_diagnostics()
        assert info["free_bytes"] is None
        assert info["total_bytes"] is None
        assert info["used_bytes"] is None


# ---------------------------------------------------------------------------
# Real hardware test (skipped when no CUDA device is present)
# ---------------------------------------------------------------------------


def _real_cuda_available() -> bool:
    try:
        from numba import cuda
        return bool(cuda.is_available())
    except Exception:
        return False


@pytest.mark.skipif(not _real_cuda_available(), reason="NVIDIA CUDA GPU not available")
def test_real_gpu_diagnostics():
    """Real GPU: all fields are populated with consistent, non-fabricated values."""
    info = get_cuda_diagnostics()
    assert info["available"] is True
    assert isinstance(info["device_name"], str) and len(info["device_name"]) > 0
    assert isinstance(info["free_bytes"], int) and info["free_bytes"] >= 0
    assert isinstance(info["total_bytes"], int) and info["total_bytes"] > 0
    assert isinstance(info["used_bytes"], int) and info["used_bytes"] >= 0
    assert info["free_bytes"] + info["used_bytes"] == info["total_bytes"]
    assert info["free_bytes"] <= info["total_bytes"]
