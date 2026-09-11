from __future__ import annotations

import logging
import time
from typing import Generator

import cv2
import mss
import mss.base
import numpy as np

__all__ = [
    "ScreenCapture",
    "get_screen_size",
    "capture_screen",
    "capture_region",
    "show_screen",
    "resize_frame",
    "compress_frame",
]

log = logging.getLogger(__name__)


def _bgra_to_bgr(screenshot: mss.base.ScreenShot) -> np.ndarray:
    """Convert an mss screenshot to an OpenCV BGR ndarray."""
    frame = np.array(screenshot)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def _validate_region(x: int, y: int, width: int, height: int) -> None:
    """Raise ValueError if region dimensions are invalid."""
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Region dimensions must be positive, got width={width}, height={height}."
        )
    if x < 0 or y < 0:
        raise ValueError(
            f"Region origin must be non-negative, got x={x}, y={y}."
        )


# ---------------------------------------------------------------------------
# Resize / Compress helpers
# ---------------------------------------------------------------------------

def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize *frame* to a fixed ``(width, height)`` resolution.

    Uses ``cv2.INTER_AREA`` interpolation, which is optimal when shrinking
    (i.e. the target resolution is smaller than the source).  For upscaling,
    ``cv2.INTER_LINEAR`` is used automatically.

    Parameters
    ----------
    frame:
        Input BGR image as a NumPy ndarray, shape ``(H, W, 3)``.
    width:
        Target width in pixels.
    height:
        Target height in pixels.

    Returns
    -------
    numpy.ndarray
        Resized BGR frame, shape ``(height, width, 3)``.
    """
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Target dimensions must be positive, got width={width}, height={height}."
        )
    src_h, src_w = frame.shape[:2]
    interpolation = (
        cv2.INTER_AREA
        if (width <= src_w and height <= src_h)
        else cv2.INTER_LINEAR
    )
    resized = cv2.resize(frame, (width, height), interpolation=interpolation)
    log.debug(
        "Frame resized from %dx%d → %dx%d.",
        src_w, src_h, width, height,
    )
    return resized


def compress_frame(frame: np.ndarray) -> bytes:
    """
    Encode *frame* to lossless PNG bytes.

    Parameters
    ----------
    frame:
        BGR image as a NumPy ndarray, shape ``(H, W, 3)``.

    Returns
    -------
    bytes
        PNG-encoded image data ready for file I/O, network transport, or
        passing directly to an AI vision API.

    Raises
    ------
    RuntimeError
        If OpenCV fails to encode the frame.
    """
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("cv2.imencode failed to encode frame as PNG.")
    data: bytes = buf.tobytes()
    log.debug("Frame compressed to PNG (%d bytes).", len(data))
    return data


class ScreenCapture:
    """
    A reusable screen capture session backed by a single ``mss`` instance.

    Use this class when you need to capture many frames in a loop, as it
    avoids the overhead of re-initialising ``mss`` on every call.

    Examples
    --------
    >>> with ScreenCapture() as sc:
    ...     frame = sc.capture()
    ...     region = sc.capture_region(0, 0, 800, 600)
    """

    def __init__(self, monitor_index: int = 1) -> None:
        """
        Parameters
        ----------
        monitor_index:
            Index into ``mss.monitors``.  ``0`` is the virtual full-desktop
            monitor; ``1`` (default) is the primary physical monitor.
        """
        if monitor_index < 0:
            raise ValueError(
                f"monitor_index must be >= 0, got {monitor_index}."
            )
        self._monitor_index = monitor_index
        self._sct: mss.base.MSSBase | None = None


    def __enter__(self) -> ScreenCapture:
        self._sct = mss.mss()
        log.debug("mss session opened (monitor_index=%d).", self._monitor_index)
        return self

    def __exit__(self, *_: object) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None
            log.debug("mss session closed.")

    @property
    def monitor(self) -> dict[str, int]:
        """Return the mss monitor dict for the configured monitor index."""
        self._ensure_open()
        return self._sct.monitors[self._monitor_index]  # type: ignore[index]

    @property
    def size(self) -> tuple[int, int]:
        """Return ``(width, height)`` of the configured monitor."""
        m = self.monitor
        return m["width"], m["height"]

    def capture(self) -> np.ndarray:
        """
        Capture the full configured monitor.

        Returns
        -------
        numpy.ndarray
            Screenshot in OpenCV BGR format, shape ``(H, W, 3)``.
        """
        self._ensure_open()
        screenshot = self._sct.grab(self.monitor)  # type: ignore[union-attr]
        log.debug("Full monitor captured (%dx%d).", *self.size)
        return _bgra_to_bgr(screenshot)

    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Capture a specific rectangular region of the screen.

        Parameters
        ----------
        x:      Left coordinate (pixels from left edge of monitor).
        y:      Top coordinate (pixels from top edge of monitor).
        width:  Region width in pixels.
        height: Region height in pixels.

        Returns
        -------
        numpy.ndarray
            Screenshot in OpenCV BGR format, shape ``(height, width, 3)``.
        """
        self._ensure_open()
        _validate_region(x, y, width, height)
        region = {"left": x, "top": y, "width": width, "height": height}
        screenshot = self._sct.grab(region)  # type: ignore[union-attr]
        log.debug("Region captured (x=%d, y=%d, %dx%d).", x, y, width, height)
        return _bgra_to_bgr(screenshot)

    def capture_resized(self, width: int, height: int) -> np.ndarray:
        """
        Capture the full monitor and resize to ``(width, height)``.

        Convenience wrapper for :meth:`capture` → :func:`resize_frame`.

        Parameters
        ----------
        width:
            Target width in pixels.
        height:
            Target height in pixels.

        Returns
        -------
        numpy.ndarray
            Resized BGR frame, shape ``(height, width, 3)``.
        """
        return resize_frame(self.capture(), width, height)

    def stream(self, fps: float = 30.0) -> Generator[np.ndarray, None, None]:
        """
        Yield frames from the monitor at up to *fps* frames per second.

        Parameters
        ----------
        fps:
            Target frame rate.  Use ``0`` or a negative value for uncapped.

        Yields
        ------
        numpy.ndarray
            Successive BGR frames.
        """
        delay = (1.0 / fps) if fps > 0 else 0.0
        while True:
            t0 = time.monotonic()
            yield self.capture()
            elapsed = time.monotonic() - t0
            remaining = delay - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def stream_resized(
        self,
        width: int,
        height: int,
        fps: float = 30.0,
    ) -> Generator[np.ndarray, None, None]:
        """
        Yield resized frames from the monitor at up to *fps* frames per second.

        Each frame is passed through :func:`resize_frame` before being yielded,
        so the full pipeline (capture → resize) runs inside the generator.

        Parameters
        ----------
        width:
            Target width in pixels.
        height:
            Target height in pixels.
        fps:
            Target frame rate.  Use ``0`` or a negative value for uncapped.

        Yields
        ------
        numpy.ndarray
            Successive resized BGR frames, shape ``(height, width, 3)``.
        """
        for frame in self.stream(fps=fps):
            yield resize_frame(frame, width, height)


    def _ensure_open(self) -> None:
        if self._sct is None:
            raise RuntimeError(
                "ScreenCapture must be used as a context manager "
                "(i.e. `with ScreenCapture() as sc: ...`)."
            )


def get_screen_size(monitor_index: int = 1) -> tuple[int, int]:
    """
    Return ``(width, height)`` of the specified monitor.

    Parameters
    ----------
    monitor_index:
        Index into ``mss.monitors``.  Defaults to ``1`` (primary monitor).
    """
    with ScreenCapture(monitor_index) as sc:
        return sc.size


def capture_screen(monitor_index: int = 1) -> np.ndarray:
    """
    Capture the full screen of the specified monitor.

    Parameters
    ----------
    monitor_index:
        Index into ``mss.monitors``.  Defaults to ``1`` (primary monitor).

    Returns
    -------
    numpy.ndarray
        Screenshot in OpenCV BGR format.
    """
    with ScreenCapture(monitor_index) as sc:
        return sc.capture()


def capture_region(
    x: int,
    y: int,
    width: int,
    height: int,
    monitor_index: int = 1,
) -> np.ndarray:
    """
    Capture a specific region of the screen.

    Parameters
    ----------
    x:             Left coordinate.
    y:             Top coordinate.
    width:         Region width in pixels.
    height:        Region height in pixels.
    monitor_index: Monitor to capture from (defaults to primary).

    Returns
    -------
    numpy.ndarray
        Screenshot in OpenCV BGR format.
    """
    with ScreenCapture(monitor_index) as sc:
        return sc.capture_region(x, y, width, height)


def show_screen(
    window_title: str = "AI Vision Desktop Assistant",
    fps: float = 30.0,
    monitor_index: int = 1,
) -> None:
    """
    Display a live preview of the screen in an OpenCV window.

    Parameters
    ----------
    window_title:  Title of the OpenCV display window.
    fps:           Target display frame rate (default 30).  Use ``0`` for
                   uncapped (high CPU usage).
    monitor_index: Monitor to preview (defaults to primary).

    Press **Q** to quit.
    """
    log.info("Starting live preview — press Q to quit.")
    with ScreenCapture(monitor_index) as sc:
        for frame in sc.stream(fps=fps):
            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    log.info("Live preview closed.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    frame = capture_region(x=0, y=0, width=800, height=600)
    cv2.imshow("Screen Region", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
