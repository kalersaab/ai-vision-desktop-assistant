from __future__ import annotations

import ctypes
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_native_dir = Path(__file__).resolve().parent
_build_dir = _native_dir / "build"

def _find_library() -> Optional[Path]:
    system = platform.system()
    lib_name = "libfast_vision.dylib" if system == "Darwin" else "libfast_vision.so"
    target = _build_dir / lib_name
    if target.is_file():
        return target
    return None


def compile_native_library() -> bool:
    """Attempt to compile the C++ shared library using the Makefile."""
    makefile = _native_dir / "Makefile"
    if not makefile.is_file():
        return False
    try:
        res = subprocess.run(["make", "-C", str(_native_dir)], capture_output=True, text=True, timeout=15)
        return res.returncode == 0
    except Exception as err:
        logger.debug("Failed to auto-compile native C++ library: %s", err)
        return False


_lib_path = _find_library()
if not _lib_path:
    if compile_native_library():
        _lib_path = _find_library()

_cpp_lib: Optional[ctypes.CDLL] = None
if _lib_path and _lib_path.is_file():
    try:
        _cpp_lib = ctypes.CDLL(str(_lib_path))
        
        # Configure compute_dhash64
        _cpp_lib.compute_dhash64.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        _cpp_lib.compute_dhash64.restype = ctypes.c_int

        # Configure detect_changed_roi
        _cpp_lib.detect_changed_roi.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        _cpp_lib.detect_changed_roi.restype = ctypes.c_int

        # Configure fast_downsample_bgr
        _cpp_lib.fast_downsample_bgr.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        _cpp_lib.fast_downsample_bgr.restype = ctypes.c_int

        # Configure fast_draw_box
        _cpp_lib.fast_draw_box.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint8,
            ctypes.c_uint8,
            ctypes.c_uint8,
            ctypes.c_int,
        ]
        _cpp_lib.fast_draw_box.restype = ctypes.c_int

        logger.info("Loaded native C++ fast vision engine from %s", _lib_path)
    except Exception as err:
        logger.warning("Failed loading native C++ library: %s. Using Python fallback.", err)
        _cpp_lib = None


def is_cpp_available() -> bool:
    """Return True if native C++ acceleration library is loaded and ready."""
    return _cpp_lib is not None


def compute_frame_hash(frame: np.ndarray) -> int:
    """
    Compute 64-bit perceptual difference hash of a BGR frame.
    Executes in <0.3ms in C++.
    """
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must be a 3-channel HxWx3 BGR NumPy array")

    h, w = frame.shape[:2]
    c_frame = np.ascontiguousarray(frame, dtype=np.uint8)

    if _cpp_lib is not None:
        out_hash = ctypes.c_uint64(0)
        ret = _cpp_lib.compute_dhash64(
            c_frame.ctypes.data_as(ctypes.c_char_p),
            ctypes.c_int(w),
            ctypes.c_int(h),
            ctypes.byref(out_hash),
        )
        if ret == 0:
            return out_hash.value

    # Pure Python / NumPy fallback
    # Grayscale conversion approx: 0.114*B + 0.587*G + 0.299*R
    gray = (frame[:, :, 0] * 0.114 + frame[:, :, 1] * 0.587 + frame[:, :, 2] * 0.299).astype(np.uint8)
    import cv2
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, :-1] > small[:, 1:]
    hash_val = 0
    for bit in diff.flatten():
        hash_val = (hash_val << 1) | int(bit)
    return hash_val


def detect_screen_changes(
    frame1: np.ndarray,
    frame2: np.ndarray,
    threshold: int = 15,
) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
    """
    Fast screen change comparison.
    Returns:
        (diff_ratio, bounding_box_or_none)
        where bounding_box is (min_x, min_y, max_x, max_y).
    """
    if frame1.shape != frame2.shape:
        raise ValueError("Frames must have identical dimensions for comparison")

    h, w = frame1.shape[:2]
    f1 = np.ascontiguousarray(frame1, dtype=np.uint8)
    f2 = np.ascontiguousarray(frame2, dtype=np.uint8)

    if _cpp_lib is not None:
        out_ratio = ctypes.c_float(0.0)
        min_x = ctypes.c_int(0)
        min_y = ctypes.c_int(0)
        max_x = ctypes.c_int(0)
        max_y = ctypes.c_int(0)

        ret = _cpp_lib.detect_changed_roi(
            f1.ctypes.data_as(ctypes.c_char_p),
            f2.ctypes.data_as(ctypes.c_char_p),
            ctypes.c_int(w),
            ctypes.c_int(h),
            ctypes.c_int(threshold),
            ctypes.byref(out_ratio),
            ctypes.byref(min_x),
            ctypes.byref(min_y),
            ctypes.byref(max_x),
            ctypes.byref(max_y),
        )
        if ret == 0:
            ratio = float(out_ratio.value)
            if ratio > 0.0001:
                return ratio, (min_x.value, min_y.value, max_x.value, max_y.value)
            return 0.0, None

    # NumPy Fallback
    diff = np.abs(f1.astype(np.int16) - f2.astype(np.int16))
    changed_mask = np.any(diff > threshold, axis=2)
    changed_count = np.count_nonzero(changed_mask)
    total = w * h
    ratio = float(changed_count / total)

    if ratio > 0.0001:
        y_indices, x_indices = np.where(changed_mask)
        return ratio, (int(x_indices.min()), int(y_indices.min()), int(x_indices.max()), int(y_indices.max()))
    return 0.0, None


def wait_for_screen_stabilize(
    capture_fn: Callable[[], np.ndarray],
    max_wait_sec: float = 1.2,
    poll_interval_sec: float = 0.04,
    hash_stability_count: int = 2,
) -> np.ndarray:
    """
    Sub-millisecond perceptual hash polling to detect when desktop window
    manager animations finish. As soon as the screen hash stays constant
    for consecutive samples, the stabilized screen is immediately returned.
    """
    start_t = time.time()
    prev_hash: Optional[int] = None
    streak = 0

    latest_frame = capture_fn()
    while (time.time() - start_t) < max_wait_sec:
        current_hash = compute_frame_hash(latest_frame)
        if prev_hash is not None and current_hash == prev_hash:
            streak += 1
            if streak >= hash_stability_count:
                logger.debug("Screen stabilized in %.3fs (hash=%x)", time.time() - start_t, current_hash)
                return latest_frame
        else:
            streak = 0
            prev_hash = current_hash

        time.sleep(poll_interval_sec)
        latest_frame = capture_fn()

    logger.debug("Screen stabilization reached timeout (%.2fs)", max_wait_sec)
    return latest_frame
