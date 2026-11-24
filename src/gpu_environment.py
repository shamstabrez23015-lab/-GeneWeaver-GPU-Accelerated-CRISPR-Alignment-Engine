"""GPU environment detection module for GeneWeaver."""

from typing import Any, Dict


def get_gpu_info() -> Dict[str, Any]:
    """Detect whether a compatible CUDA GPU environment is available.

    Checks for Numba CUDA support and attempts to query the active CUDA device.
    Gracefully handles missing packages, dynamic library errors, and driver/device
    detection failures without raising unhandled exceptions.

    Returns:
        dict: A dictionary containing:
            - available (bool): True if a functional CUDA device is detected, False otherwise.
            - device_name (str or None): Name of the detected CUDA device, or None.
            - compute_capability (tuple or None): Compute capability (major, minor), or None.
            - message (str): Status explanation describing availability or reason for failure.
    """
    try:
        from numba import cuda
    except Exception as err:
        return {
            "available": False,
            "device_name": None,
            "compute_capability": None,
            "message": f"CUDA is unavailable: Numba CUDA cannot be imported ({err}).",
        }

    try:
        is_avail = cuda.is_available()
    except Exception as err:
        return {
            "available": False,
            "device_name": None,
            "compute_capability": None,
            "message": f"CUDA is unavailable: error checking CUDA availability ({err}).",
        }

    if not is_avail:
        return {
            "available": False,
            "device_name": None,
            "compute_capability": None,
            "message": "CUDA is unavailable: no compatible CUDA device detected.",
        }

    try:
        device = cuda.get_current_device()
        device_name = getattr(device, "name", "Unknown CUDA Device")
        if isinstance(device_name, bytes):
            device_name = device_name.decode("utf-8", errors="replace")

        compute_capability = getattr(device, "compute_capability", None)
        if isinstance(compute_capability, list):
            compute_capability = tuple(compute_capability)

        return {
            "available": True,
            "device_name": device_name,
            "compute_capability": compute_capability,
            "message": f"CUDA is available on device: {device_name}.",
        }
    except Exception as err:
        return {
            "available": False,
            "device_name": None,
            "compute_capability": None,
            "message": f"CUDA is unavailable: error querying device properties ({err}).",
        }
