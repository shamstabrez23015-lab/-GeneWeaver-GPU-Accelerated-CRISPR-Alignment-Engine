"""Consolidated CUDA diagnostics module for GeneWeaver.

Combines GPU device detection, device naming, and memory information into a
single diagnostic snapshot.  All errors are caught internally — the function
never raises, making it safe to call from dashboards or logging code without
wrapping in a try/except.
"""

from __future__ import annotations


def get_cuda_diagnostics() -> dict:
    """Return a consolidated snapshot of CUDA device and memory status.

    Reuses :func:`src.cuda_alignment.get_gpu_memory_info` for memory data and
    queries the device name via the Numba CUDA API.  Handles API differences
    between Numba versions safely.

    Returns
    -------
    dict
        A dictionary with the following keys:

        ``available`` : bool
            True when a functional CUDA device was found and queried.
        ``device_name`` : str or None
            Human-readable name of the active CUDA device, or None when
            unavailable.
        ``free_bytes`` : int or None
            Free GPU memory in bytes, or None when unavailable.
        ``total_bytes`` : int or None
            Total GPU memory in bytes, or None when unavailable.
        ``used_bytes`` : int or None
            Used GPU memory (total − free) in bytes, or None when unavailable.
        ``message`` : str
            Human-readable diagnostic summary or error description.

    Notes
    -----
    - No GPU memory or device values are fabricated or hardcoded.
    - ``used_bytes`` is always ``total_bytes - free_bytes`` when available.
    - ``free_bytes`` is always ≤ ``total_bytes`` when available.
    - This function never raises; all failures surface through the dict.
    """
    _unavailable_base = {
        "available": False,
        "device_name": None,
        "free_bytes": None,
        "total_bytes": None,
        "used_bytes": None,
    }

    # ------------------------------------------------------------------
    # Step 1: import the cuda module (may fail if Numba isn't installed)
    # ------------------------------------------------------------------
    try:
        from numba import cuda as _cuda
    except Exception as exc:
        return {
            **_unavailable_base,
            "message": (
                f"CUDA diagnostics unavailable: "
                f"Numba CUDA package could not be imported ({exc})."
            ),
        }

    # ------------------------------------------------------------------
    # Step 2: check device availability
    # ------------------------------------------------------------------
    try:
        cuda_present = _cuda.is_available()
    except Exception as exc:
        return {
            **_unavailable_base,
            "message": (
                f"CUDA diagnostics unavailable: "
                f"CUDA availability check failed ({exc})."
            ),
        }

    if not cuda_present:
        return {
            **_unavailable_base,
            "message": (
                "CUDA diagnostics unavailable: "
                "no compatible CUDA device was detected."
            ),
        }

    # ------------------------------------------------------------------
    # Step 3: retrieve device name
    # ------------------------------------------------------------------
    device_name: str | None = None
    device_name_error: str | None = None

    try:
        device = _cuda.get_current_device()
        raw_name = getattr(device, "name", None)
        if raw_name is None:
            raw_name = "Unknown CUDA Device"
        if isinstance(raw_name, bytes):
            raw_name = raw_name.decode("utf-8", errors="replace")
        device_name = str(raw_name)
    except Exception as exc:
        device_name_error = str(exc)

    # ------------------------------------------------------------------
    # Step 4: retrieve memory info (reuse cuda_alignment logic directly)
    # ------------------------------------------------------------------
    free_bytes: int | None = None
    total_bytes: int | None = None
    used_bytes: int | None = None
    mem_error: str | None = None

    try:
        context = _cuda.current_context()
        mem_info = context.get_memory_info()

        # Numba ≥ 0.54 returns a named tuple (free, total);
        # older versions may return a plain tuple — handle both.
        if hasattr(mem_info, "free"):
            free_bytes = int(mem_info.free)
            total_bytes = int(mem_info.total)
        else:
            free_bytes = int(mem_info[0])
            total_bytes = int(mem_info[1])

        used_bytes = total_bytes - free_bytes

    except Exception as exc:
        mem_error = str(exc)

    # ------------------------------------------------------------------
    # Step 5: build result
    # ------------------------------------------------------------------
    # If both device name AND memory retrieval failed, treat as unavailable.
    if device_name is None and mem_error is not None:
        error_detail = device_name_error or mem_error
        return {
            **_unavailable_base,
            "message": (
                f"CUDA diagnostics unavailable: "
                f"failed to query device or memory information ({error_detail})."
            ),
        }

    # At least partial success — compose a combined message.
    if device_name is None:
        device_name = "Unknown CUDA Device"

    if mem_error is not None:
        # Memory failed but device name was retrieved.
        message = (
            f"CUDA device detected: {device_name}. "
            f"Memory information could not be retrieved ({mem_error})."
        )
        return {
            "available": True,
            "device_name": device_name,
            "free_bytes": None,
            "total_bytes": None,
            "used_bytes": None,
            "message": message,
        }

    # Full success.
    message = (
        f"CUDA device: {device_name} — "
        f"{free_bytes:,} bytes free of {total_bytes:,} bytes total "
        f"({used_bytes:,} bytes used)."
    )
    return {
        "available": True,
        "device_name": device_name,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "used_bytes": used_bytes,
        "message": message,
    }
