"""Unit tests for GPU environment detection in GeneWeaver."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path when running pytest directly without python -m
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from src.gpu_environment import get_gpu_info


def test_cuda_unavailable_when_device_not_detected():
    """Test that get_gpu_info reports unavailable when no CUDA device is present."""
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = False

    with patch.dict(sys.modules, {"numba": MagicMock(cuda=mock_cuda), "numba.cuda": mock_cuda}):
        info = get_gpu_info()

    assert info["available"] is False
    assert info["device_name"] is None
    assert info["compute_capability"] is None
    assert "unavailable" in info["message"].lower()


def test_cuda_available_mocked_device():
    """Test successful detection of a mocked CUDA device."""
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True

    mock_device = MagicMock()
    mock_device.name = "NVIDIA GeForce RTX 4090"
    mock_device.compute_capability = (8, 9)
    mock_cuda.get_current_device.return_value = mock_device

    with patch.dict(sys.modules, {"numba": MagicMock(cuda=mock_cuda), "numba.cuda": mock_cuda}):
        info = get_gpu_info()

    assert info["available"] is True
    assert info["device_name"] == "NVIDIA GeForce RTX 4090"
    assert info["compute_capability"] == (8, 9)
    assert "available" in info["message"].lower()


def test_device_name_handling_bytes_and_string():
    """Test device name decoding when name is provided as bytes or string."""
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True

    # Bytes device name
    mock_device_bytes = MagicMock()
    mock_device_bytes.name = b"Tesla V100-PCIE-16GB"
    mock_device_bytes.compute_capability = (7, 0)
    mock_cuda.get_current_device.return_value = mock_device_bytes

    with patch.dict(sys.modules, {"numba": MagicMock(cuda=mock_cuda), "numba.cuda": mock_cuda}):
        info_bytes = get_gpu_info()

    assert info_bytes["available"] is True
    assert info_bytes["device_name"] == "Tesla V100-PCIE-16GB"
    assert isinstance(info_bytes["device_name"], str)

    # Standard string device name
    mock_device_str = MagicMock()
    mock_device_str.name = "NVIDIA A100-SXM4-40GB"
    mock_device_str.compute_capability = (8, 0)
    mock_cuda.get_current_device.return_value = mock_device_str

    with patch.dict(sys.modules, {"numba": MagicMock(cuda=mock_cuda), "numba.cuda": mock_cuda}):
        info_str = get_gpu_info()

    assert info_str["available"] is True
    assert info_str["device_name"] == "NVIDIA A100-SXM4-40GB"


def test_compute_capability_handling():
    """Test that compute capability is preserved as a tuple or handled when None."""
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True

    # Tuple compute capability
    mock_device_tuple = MagicMock()
    mock_device_tuple.name = "NVIDIA RTX 3080"
    mock_device_tuple.compute_capability = (8, 6)
    mock_cuda.get_current_device.return_value = mock_device_tuple

    with patch.dict(sys.modules, {"numba": MagicMock(cuda=mock_cuda), "numba.cuda": mock_cuda}):
        info_tuple = get_gpu_info()

    assert info_tuple["compute_capability"] == (8, 6)

    # List compute capability converted to tuple
    mock_device_list = MagicMock()
    mock_device_list.name = "NVIDIA RTX 3080"
    mock_device_list.compute_capability = [8, 6]
    mock_cuda.get_current_device.return_value = mock_device_list

    with patch.dict(sys.modules, {"numba": MagicMock(cuda=mock_cuda), "numba.cuda": mock_cuda}):
        info_list = get_gpu_info()

    assert info_list["compute_capability"] == (8, 6)

    # Missing compute capability
    mock_device_none = MagicMock(spec=["name"])
    mock_device_none.name = "Generic CUDA GPU"

    mock_cuda.get_current_device.return_value = mock_device_none

    with patch.dict(sys.modules, {"numba": MagicMock(cuda=mock_cuda), "numba.cuda": mock_cuda}):
        info_none = get_gpu_info()

    assert info_none["compute_capability"] is None


def test_missing_numba_package():
    """Test that missing or unimportable Numba package is handled gracefully."""
    # When numba is not importable
    with patch.dict(sys.modules, {"numba": None, "numba.cuda": None}):
        info = get_gpu_info()

    assert info["available"] is False
    assert info["device_name"] is None
    assert info["compute_capability"] is None
    assert "unavailable" in info["message"].lower()


def test_cuda_detection_error_on_is_available():
    """Test handling of driver errors raised during cuda.is_available() check."""
    mock_cuda = MagicMock()
    mock_cuda.is_available.side_effect = RuntimeError("Driver version insufficient for CUDA")

    with patch.dict(sys.modules, {"numba": MagicMock(cuda=mock_cuda), "numba.cuda": mock_cuda}):
        info = get_gpu_info()

    assert info["available"] is False
    assert info["device_name"] is None
    assert info["compute_capability"] is None
    assert "unavailable" in info["message"].lower()
    assert "Driver version insufficient" in info["message"]


def test_cuda_detection_error_on_get_device():
    """Test handling of runtime errors when accessing the current CUDA device."""
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True
    mock_cuda.get_current_device.side_effect = RuntimeError("CUDA device failed to respond")

    with patch.dict(sys.modules, {"numba": MagicMock(cuda=mock_cuda), "numba.cuda": mock_cuda}):
        info = get_gpu_info()

    assert info["available"] is False
    assert info["device_name"] is None
    assert info["compute_capability"] is None
    assert "unavailable" in info["message"].lower()
    assert "failed to respond" in info["message"]


def test_required_dictionary_fields():
    """Test that get_gpu_info returns all required dictionary fields in both scenarios."""
    required_keys = {"available", "device_name", "compute_capability", "message"}

    # Unavailable scenario
    mock_cuda_unavail = MagicMock()
    mock_cuda_unavail.is_available.return_value = False
    with patch.dict(sys.modules, {"numba": MagicMock(cuda=mock_cuda_unavail), "numba.cuda": mock_cuda_unavail}):
        info_unavail = get_gpu_info()
    assert set(info_unavail.keys()) == required_keys

    # Available scenario
    mock_cuda_avail = MagicMock()
    mock_cuda_avail.is_available.return_value = True
    mock_device = MagicMock()
    mock_device.name = "NVIDIA T4"
    mock_device.compute_capability = (7, 5)
    mock_cuda_avail.get_current_device.return_value = mock_device
    with patch.dict(sys.modules, {"numba": MagicMock(cuda=mock_cuda_avail), "numba.cuda": mock_cuda_avail}):
        info_avail = get_gpu_info()
    assert set(info_avail.keys()) == required_keys
