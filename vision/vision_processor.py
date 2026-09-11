from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

from vision.screen_capture import ScreenCapture, compress_frame, resize_frame

__all__ = [
    "VisionProcessor",
    "ProcessedFrame",
    "FrameCallback",
]

log = logging.getLogger(__name__)

FrameCallback = Callable[["ProcessedFrame"], None]

@dataclass(frozen=True)
class ProcessedFrame:
    """
    A single processed screen snapshot ready for AI model consumption.

    Attributes
    ----------
    frame_number:
        Monotonically increasing capture index (starts at 1).
    timestamp:
        Wall-clock time of the capture (``time.time()``).
    bgr_frame:
        Resized BGR NumPy array, shape ``(height, width, 3)``.
        Use this for further OpenCV processing.
    png_bytes:
        Lossless PNG encoding of *bgr_frame*.
        Pass directly to vision AI APIs (e.g. Gemini, GPT-4o).
    width:
        Frame width in pixels.
    height:
        Frame height in pixels.
    capture_ms:
        Time spent capturing + processing this frame (milliseconds).
    """

    frame_number: int
    timestamp: float
    bgr_frame: np.ndarray
    png_bytes: bytes
    width: int
    height: int
    capture_ms: float = field(default=0.0)

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(height, width)`` — same convention as ndarray.shape."""
        return self.height, self.width

    @property
    def size_kb(self) -> float:
        """Size of the PNG payload in kibibytes."""
        return len(self.png_bytes) / 1024

    def __hash__(self) -> int:
        return hash((self.frame_number, self.timestamp))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProcessedFrame):
            return NotImplemented
        return self.frame_number == other.frame_number and self.timestamp == other.timestamp

class VisionProcessor:
    """
    Periodically captures the Mac screen and prepares frames for an AI model.

    Runs a single background daemon thread that:
    1. Grabs the screen via ``ScreenCapture``.
    2. Resizes the frame to ``(width, height)``.
    3. Encodes the frame to lossless PNG bytes.
    4. Stores the result as the *latest frame* and optionally fires a callback.

    The processor is safe to use from multiple consumer threads: both
    :meth:`get_latest_frame` and the callback are called with a fully
    immutable :class:`ProcessedFrame`.

    Parameters
    ----------
    width:
        Target frame width in pixels (default 1280).
    height:
        Target frame height in pixels (default 720).
    fps:
        Capture rate in frames per second (default 1.0).
        Use a low value (e.g. 0.5-2.0) to stay within AI API rate limits.
        0 or negative = uncapped.
    monitor_index:
        mss monitor index. 1 (default) = primary monitor.
    on_frame:
        Optional callback invoked on the capture thread every time a new
        :class:`ProcessedFrame` is ready. Keep it short; offload heavy
        work to a separate thread.
    queue_size:
        Max number of frames kept in the internal pull queue (default 1).
        When the queue is full the oldest frame is dropped so consumers
        always receive the freshest data.
    """

    def __init__(
        self,
        *,
        width: int = 1280,
        height: int = 720,
        fps: float = 1.0,
        monitor_index: int = 1,
        on_frame: Optional[FrameCallback] = None,
        queue_size: int = 1,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(
                f"width and height must be positive, got {width}x{height}."
            )
        if queue_size < 1:
            raise ValueError(f"queue_size must be >= 1, got {queue_size}.")

        self.width = width
        self.height = height
        self.fps = fps
        self.monitor_index = monitor_index
        self.on_frame = on_frame

        self._queue: queue.Queue[ProcessedFrame] = queue.Queue(maxsize=queue_size)
        self._latest: Optional[ProcessedFrame] = None
        self._latest_lock = threading.Lock()
        self._frame_counter: int = 0
        self._counter_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "VisionProcessor":
        """
        Start the background capture thread.

        Returns *self* to allow chaining::

            proc = VisionProcessor(fps=2.0).start()
        """
        if self._thread is not None and self._thread.is_alive():
            log.warning("VisionProcessor is already running; ignoring start().")
            return self

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="VisionProcessor-capture",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "VisionProcessor started — %dx%d @ %.1f fps (monitor %d).",
            self.width, self.height, self.fps, self.monitor_index,
        )
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """
        Signal the capture thread to stop and wait for it to finish.

        Parameters
        ----------
        timeout:
            Seconds to wait for the thread to exit (default 5.0).
        """
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            log.warning("VisionProcessor thread did not exit within %.1fs.", timeout)
        else:
            log.info("VisionProcessor stopped.")
        self._thread = None

    def is_running(self) -> bool:
        """Return True if the capture thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> "VisionProcessor":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    def get_latest_frame(self, timeout: Optional[float] = None) -> Optional[ProcessedFrame]:
        """
        Return the most recently captured :class:`ProcessedFrame`.

        If no frame has been captured yet this blocks until one is
        available (or *timeout* seconds elapse).

        Parameters
        ----------
        timeout:
            Seconds to wait if no frame is available yet.
            ``None`` (default) returns immediately with ``None`` if
            no frame exists.

        Returns
        -------
        ProcessedFrame or None
        """
        if timeout is not None:
            try:
                self._queue.get(timeout=timeout)
            except queue.Empty:
                pass

        with self._latest_lock:
            return self._latest

    @property
    def frame_count(self) -> int:
        """Total number of frames captured since :meth:`start` was called."""
        with self._counter_lock:
            return self._frame_counter

    def _run(self) -> None:
        """Main loop executed on the background thread."""
        delay = (1.0 / self.fps) if self.fps > 0 else 0.0

        with ScreenCapture(self.monitor_index) as sc:
            while not self._stop_event.is_set():
                tick_start = time.monotonic()

                try:
                    pf = self._capture_one(sc)
                except Exception:
                    log.exception("Error during screen capture; retrying next tick.")
                else:
                    self._publish(pf)

                elapsed = time.monotonic() - tick_start
                sleep_for = max(0.0, delay - elapsed)
                if sleep_for > 0:
                    self._stop_event.wait(sleep_for)

    def _capture_one(self, sc: ScreenCapture) -> ProcessedFrame:
        """Execute the full pipeline for a single frame."""
        t0 = time.monotonic()
        ts = time.time()

        raw_frame: np.ndarray = sc.capture()

        resized: np.ndarray = resize_frame(raw_frame, self.width, self.height)

        png: bytes = compress_frame(resized)

        capture_ms = (time.monotonic() - t0) * 1000.0

        with self._counter_lock:
            self._frame_counter += 1
            frame_number = self._frame_counter

        log.debug(
            "Frame #%d captured in %.1f ms — %dx%d, %.1f KB.",
            frame_number, capture_ms, self.width, self.height, len(png) / 1024,
        )

        return ProcessedFrame(
            frame_number=frame_number,
            timestamp=ts,
            bgr_frame=resized,
            png_bytes=png,
            width=self.width,
            height=self.height,
            capture_ms=capture_ms,
        )

    def _publish(self, pf: ProcessedFrame) -> None:
        """Store the latest frame and notify consumers."""
        with self._latest_lock:
            self._latest = pf

        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(pf)
        except queue.Full:
            pass

        if self.on_frame is not None:
            try:
                self.on_frame(pf)
            except Exception:
                log.exception("Exception in on_frame callback.")

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    def _print_frame(pf: ProcessedFrame) -> None:
        print(
            f"  -> Frame #{pf.frame_number:>4}  |  "
            f"{pf.width}x{pf.height}  |  "
            f"{pf.size_kb:6.1f} KB  |  "
            f"{pf.capture_ms:5.1f} ms"
        )

    target_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"Capturing {target_frames} frames at 1 fps — 1280x720...\n")

    with VisionProcessor(
        width=1280,
        height=720,
        fps=1.0,
        on_frame=_print_frame,
    ) as proc:
        while proc.frame_count < target_frames:
            time.sleep(0.1)

    print(f"\nDone. Total frames captured: {proc.frame_count}")
