from __future__ import annotations

import base64
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generator, List, Mapping, Optional, Sequence, Union

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
import ollama

try:
    from vision.screen_capture import ScreenCapture, compress_frame
    from vision.vision_processor import ProcessedFrame
except ModuleNotFoundError:
    from screen_capture import ScreenCapture, compress_frame
    from vision_processor import ProcessedFrame

__all__ = [
    "VisionAnalyzer",
    "AnalysisResult",
    "ImageInput",
]

logger = logging.getLogger(__name__)

ImageInput = Union[bytes, ProcessedFrame, np.ndarray, str, Path]


@dataclass(frozen=True)
class AnalysisResult:
    content: str
    model: str
    duration_ms: float
    eval_count: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    thinking: Optional[str] = None

    @property
    def tokens_per_second(self) -> Optional[float]:
        if self.eval_count and self.duration_ms > 0:
            return self.eval_count / (self.duration_ms / 1000.0)
        return None



def _to_image_bytes(image: ImageInput) -> bytes:
    if isinstance(image, bytes):
        if not image:
            raise ValueError("Provided image bytes are empty.")
        return image

    if isinstance(image, ProcessedFrame):
        return image.png_bytes

    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Provided numpy array frame is empty.")
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Failed to encode ndarray frame to PNG.")
        return buf.tobytes()

    if isinstance(image, Path) or (isinstance(image, str) and Path(image).is_file()):
        return Path(image).read_bytes()

    if isinstance(image, str):
        try:
            return base64.b64decode(image)
        except Exception as err:
            raise ValueError("String input could not be decoded as base64 or file path.") from err

    raise TypeError(f"Unsupported image input type: {type(image).__name__}")


class VisionAnalyzer:
    """
    Sends screen frames to a local vision-language model through Ollama.

    Supports:
    - Multiple input types: raw PNG bytes, ProcessedFrame, OpenCV numpy frames, or file paths.
    - Optimized inference options for Apple Silicon GPU (num_ctx=4096 fits 100% in GPU).
    - Synchronous analysis, detailed metrics, and token streaming.
    - Multi-turn conversation context.
    """

    DEFAULT_PROMPT = "Describe what is currently visible on the screen."
    DEFAULT_SYSTEM_PROMPT = (
        "You are an AI desktop vision assistant. Analyze what is shown on the screen concisely, accurately, and helpfully."
    )

    def __init__(
        self,
        model: str = "qwen3-vl:4b",
        *,
        host: Optional[str] = None,
        timeout: float = 180.0,
        num_ctx: int = 8192,
        num_predict: int = 2048,
        temperature: float = 0.2,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.host = host
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.temperature = temperature
        self.system_prompt = system_prompt


        self.client = ollama.Client(host=self.host, timeout=self.timeout)
        self._history: List[dict[str, Any]] = []

    def check_connection(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception as err:
            logger.warning("Ollama connection check failed: %s", err)
            return False

    def is_model_available(self) -> bool:
        try:
            models_response = self.client.list()
            available = [m.model for m in models_response.models]
            return any(self.model == name or name.startswith(f"{self.model}:") for name in available)
        except Exception as err:
            logger.warning("Failed to check model availability: %s", err)
            return False

    def list_available_models(self) -> List[str]:
        try:
            res = self.client.list()
            return [m.model for m in res.models]
        except Exception:
            return []

    def _build_options(self, extra_options: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        options: dict[str, Any] = {
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "temperature": self.temperature,
        }
        if extra_options:
            options.update(extra_options)
        return options


    def _prepare_messages(
        self,
        png_bytes: bytes,
        prompt: str,
        include_history: bool = False,
    ) -> List[dict[str, Any]]:
        messages: List[dict[str, Any]] = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if include_history and self._history:
            messages.extend(self._history)

        messages.append({
            "role": "user",
            "content": prompt,
            "images": [png_bytes],
        })
        return messages

    def analyze(
        self,
        image: ImageInput,
        prompt: str = DEFAULT_PROMPT,
        *,
        remember: bool = False,
        options: Optional[Mapping[str, Any]] = None,
    ) -> str:
        res = self.analyze_detailed(image, prompt, remember=remember, options=options)
        return res.content

    def analyze_detailed(
        self,
        image: ImageInput,
        prompt: str = DEFAULT_PROMPT,
        *,
        remember: bool = False,
        options: Optional[Mapping[str, Any]] = None,
    ) -> AnalysisResult:
        png_bytes = _to_image_bytes(image)
        logger.debug("Sending image to %s (%.1f KB)", self.model, len(png_bytes) / 1024)

        messages = self._prepare_messages(png_bytes, prompt, include_history=remember)
        runtime_options = self._build_options(options)

        t0 = time.monotonic()
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=runtime_options,
            )
        except ollama.ResponseError as err:
            logger.error("Ollama response error for model '%s': %s", self.model, err)
            raise
        except Exception as err:
            logger.error("Failed to communicate with Ollama: %s", err)
            raise

        duration_ms = (time.monotonic() - t0) * 1000.0
        content = response.message.content.strip() if response.message and response.message.content else ""
        thinking = getattr(response.message, "thinking", None)
        if not content and thinking:
            logger.debug("Content was empty; using thinking trace as fallback (%d chars)", len(thinking))
            content = thinking.strip()

        if remember:
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": content})

        return AnalysisResult(
            content=content,
            model=self.model,
            duration_ms=duration_ms,
            eval_count=getattr(response, "eval_count", None),
            prompt_eval_count=getattr(response, "prompt_eval_count", None),
            thinking=thinking,
        )


    def analyze_stream(
        self,
        image: ImageInput,
        prompt: str = DEFAULT_PROMPT,
        *,
        remember: bool = False,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Generator[str, None, None]:
        png_bytes = _to_image_bytes(image)
        logger.debug("Streaming analysis from %s (%.1f KB)", self.model, len(png_bytes) / 1024)

        messages = self._prepare_messages(png_bytes, prompt, include_history=remember)
        runtime_options = self._build_options(options)

        full_content: list[str] = []
        try:
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                options=runtime_options,
                stream=True,
            )
            for chunk in stream:
                if chunk.message and chunk.message.content:
                    delta = chunk.message.content
                    full_content.append(delta)
                    yield delta
        except Exception as err:
            logger.error("Error during streaming analysis: %s", err)
            raise

        if remember and full_content:
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": "".join(full_content).strip()})

    def clear_history(self) -> None:
        self._history.clear()

    @property
    def history_depth(self) -> int:
        return len(self._history) // 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    analyzer = VisionAnalyzer(model="qwen3-vl:4b")

    print(f"Connecting to Ollama (model: {analyzer.model})...")
    if not analyzer.check_connection():
        print("ERROR: Ollama server is not running. Please start it with 'ollama serve'.")
        sys.exit(1)

    if not analyzer.is_model_available():
        print(f"WARNING: Model '{analyzer.model}' not found in local Ollama.")
        print(f"Available models: {analyzer.list_available_models()}")
        print(f"Run 'ollama pull {analyzer.model}' to download it.")
        sys.exit(1)

    print("Capturing live desktop screen...")
    with ScreenCapture() as sc:
        frame = sc.capture_resized(1280, 720)

    prompt = "Describe what is currently open on this screen in 2-3 concise sentences."
    print(f"\nPrompt: {prompt}\n")
    print("--- Streaming AI Analysis ---")

    t0 = time.monotonic()
    for token in analyzer.analyze_stream(frame, prompt=prompt):
        print(token, end="", flush=True)
    print()

    elapsed = time.monotonic() - t0
    print(f"\n--- Completed in {elapsed:.2f}s ---")
