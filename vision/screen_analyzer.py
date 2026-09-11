from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from vision.screen_capture import ScreenCapture, compress_frame, resize_frame
    from vision.vision_analyzer import AnalysisResult, ImageInput, VisionAnalyzer, _to_image_bytes
    from vision.vision_processor import ProcessedFrame
except ModuleNotFoundError:
    from screen_capture import ScreenCapture, compress_frame, resize_frame
    from vision_analyzer import AnalysisResult, ImageInput, VisionAnalyzer, _to_image_bytes
    from vision_processor import ProcessedFrame

__all__ = [
    "ScreenDimensions",
    "BoundingBox",
    "UIElement",
    "ScreenAnalysis",
    "ScreenAnalyzer",
    "annotate_frame",
    "display_analysis",
]

logger = logging.getLogger(__name__)

class ScreenDimensions(BaseModel):
    """Dimensions of the captured screen frame."""

    width: int = Field(gt=0, description="Width of the screen in pixels")
    height: int = Field(gt=0, description="Height of the screen in pixels")


class BoundingBox(BaseModel):
    """
    Axis-aligned 2D bounding box representing an element's screen coordinates.
    All values are relative to the analyzed frame's pixel dimensions.
    """

    x: int = Field(ge=0, description="X coordinate of top-left corner")
    y: int = Field(ge=0, description="Y coordinate of top-left corner")
    width: int = Field(gt=0, description="Width of the element bounding box")
    height: int = Field(gt=0, description="Height of the element bounding box")

    @field_validator("x", "y", mode="before")
    @classmethod
    def _coerce_non_negative(cls, v: Any) -> int:
        val = int(round(float(v)))
        return max(0, val)

    @field_validator("width", "height", mode="before")
    @classmethod
    def _coerce_positive_dim(cls, v: Any) -> int:
        val = int(round(float(v)))
        return max(1, val)

    @property
    def x2(self) -> int:
        """Right coordinate (exclusive)."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom coordinate (exclusive)."""
        return self.y + self.height

    @property
    def center_x(self) -> int:
        """Horizontal midpoint coordinate."""
        return self.x + (self.width // 2)

    @property
    def center_y(self) -> int:
        """Vertical midpoint coordinate."""
        return self.y + (self.height // 2)

    @property
    def center(self) -> Tuple[int, int]:
        """(center_x, center_y) tuple for pinpointing and target identification."""
        return self.center_x, self.center_y

    @property
    def area(self) -> int:
        """Total pixel area enclosed by the bounding box."""
        return self.width * self.height

    def clamp(self, max_width: int, max_height: int) -> BoundingBox:
        """Return a new BoundingBox constrained within the given screen bounds."""
        clamped_x = min(self.x, max(0, max_width - 1))
        clamped_y = min(self.y, max(0, max_height - 1))
        clamped_w = max(1, min(self.width, max_width - clamped_x))
        clamped_h = max(1, min(self.height, max_height - clamped_y))
        return BoundingBox(x=clamped_x, y=clamped_y, width=clamped_w, height=clamped_h)

    def scale(
        self,
        src_width: int,
        src_height: int,
        dst_width: int,
        dst_height: int,
    ) -> BoundingBox:
        """
        Project this bounding box from an analyzed resolution (e.g. 1280x720)
        to a physical screen resolution (e.g. 2560x1440).
        """
        if src_width <= 0 or src_height <= 0:
            raise ValueError("Source dimensions must be positive.")
        scale_x = dst_width / float(src_width)
        scale_y = dst_height / float(src_height)
        return BoundingBox(
            x=int(round(self.x * scale_x)),
            y=int(round(self.y * scale_y)),
            width=max(1, int(round(self.width * scale_x))),
            height=max(1, int(round(self.height * scale_y))),
        )


class UIElement(BaseModel):
    """A structured UI element identified by the vision model."""

    type: str = Field(
        default="element",
        description="Category: application, terminal, browser, editor, button, menu_bar, dock, input, etc.",
    )
    name: str = Field(
        default="Unknown",
        description="Human-readable title or label of the element",
    )
    description: Optional[str] = Field(
        default=None,
        description="Brief summary or content description of the element",
    )
    location: Optional[str] = Field(
        default=None,
        description="Qualitative region: center, top, bottom, left, right, top-left, etc.",
    )
    bbox: Optional[BoundingBox] = Field(
        default=None,
        description="Approximate bounding box coordinates (x, y, width, height)",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_keys(cls, data: Any) -> Any:
        """Support alternative key names produced by vision models."""
        if not isinstance(data, dict):
            return data

        if "bbox" not in data and "coordinates" in data:
            data["bbox"] = data["coordinates"]

        if "bbox" not in data and all(k in data for k in ("x", "y", "width", "height")):
            data["bbox"] = {
                "x": data.pop("x"),
                "y": data.pop("y"),
                "width": data.pop("width"),
                "height": data.pop("height"),
            }

        return data

    @property
    def has_coordinates(self) -> bool:
        """Check if precise bounding coordinates are available."""
        return self.bbox is not None


class ScreenAnalysis(BaseModel):

    screen: ScreenDimensions
    elements: List[UIElement] = Field(default_factory=list)
    query: Optional[str] = Field(default=None, description="Optional targeted query that produced this analysis")
    raw_response: Optional[str] = Field(default=None, description="Raw model text before parsing")
    inference_duration_ms: float = Field(default=0.0, description="Model inference duration in milliseconds")

    def find_element(self, query: str) -> Optional[UIElement]:
        """
        Case-insensitive fuzzy match across element name, type, and description.
        Returns the first matching UIElement, prioritizing name matches.
        """
        q = query.strip().lower()
        if not q:
            return None
        for elem in self.elements:
            if q in elem.name.lower():
                return elem
        for elem in self.elements:
            if q in elem.type.lower():
                return elem

        for elem in self.elements:
            if elem.description and q in elem.description.lower():
                return elem

        return None

    def filter_by_type(self, element_type: str) -> List[UIElement]:
        """Return all elements matching a specified type."""
        t = element_type.strip().lower()
        return [elem for elem in self.elements if t in elem.type.lower()]

    def get_element_at(self, x: int, y: int) -> Optional[UIElement]:
        """Find the smallest element enclosing the given (x, y) point."""
        matches: List[UIElement] = []
        for elem in self.elements:
            if elem.bbox and (elem.bbox.x <= x < elem.bbox.x2) and (elem.bbox.y <= y < elem.bbox.y2):
                matches.append(elem)

        if not matches:
            return None
        return min(matches, key=lambda e: e.bbox.area if e.bbox else sys.maxsize)

def extract_json_payload(text: str) -> dict[str, Any]:
    """
    Robustly extract a JSON dictionary from LLM output.
    Handles markdown code blocks (```json ... ```), preamble text,
    and trailing reasoning remarks.
    """
    cleaned = text.strip()
    code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Could not extract a valid JSON object from model response:\n{text[:300]}")

class ScreenAnalyzer:
    """
    High-level structured screen understanding engine.

    Pipeline:
        ImageInput → Vision Model → JSON Extraction → Pydantic Validation → Visual Overlay
    """

    STRUCTURED_DECOMPOSITION_PROMPT = """Analyze this {width}x{height} desktop screen. Identify the main UI elements/windows.
Output pure JSON strictly with this schema:
{{
  "screen": {{"width": {width}, "height": {height}}},
  "elements": [
    {{
      "type": "application",
      "name": "VS Code",
      "description": "code editor",
      "location": "center",
      "bbox": {{"x": 100, "y": 50, "width": 800, "height": 600}}
    }}
  ]
}}
"""

    TARGETED_ELEMENT_PROMPT = """Analyze this {width}x{height} desktop screen to locate: "{target}".
Output pure JSON strictly with this schema:
{{
  "screen": {{"width": {width}, "height": {height}}},
  "elements": [
    {{
      "type": "application",
      "name": "{target}",
      "description": "located target",
      "location": "center",
      "bbox": {{"x": 100, "y": 50, "width": 800, "height": 600}}
    }}
  ]
}}
"""

    def __init__(
        self,
        vision_analyzer: Optional[VisionAnalyzer] = None,
        model: str = "qwen3-vl:4b",
        timeout: float = 180.0,
        num_ctx: int = 8192,
        num_predict: int = 2048,
        default_resolution: Tuple[int, int] = (1280, 720),
    ) -> None:

        """
        Initialize the ScreenAnalyzer.

        Parameters
        ----------
        vision_analyzer:
            Existing VisionAnalyzer instance, or None to create one with default settings.
        model:
            Name of Ollama vision model (e.g. 'qwen3-vl:4b').
        timeout:
            Network/inference timeout in seconds.
        num_ctx:
            Context window token size for Ollama inference.
        num_predict:
            Maximum tokens to generate per inference call.
        default_resolution:
            Default (width, height) pixel dimensions for screen captures.
        """
        self.default_width, self.default_height = default_resolution
        if vision_analyzer is not None:
            self.analyzer = vision_analyzer
        else:
            self.analyzer = VisionAnalyzer(
                model=model,
                timeout=timeout,
                num_ctx=num_ctx,
                num_predict=num_predict,
                temperature=0.2,
            )




    def _resolve_dimensions(self, image: ImageInput) -> Tuple[int, int]:
        """Derive width and height from the provided image input."""
        if isinstance(image, ProcessedFrame):
            return image.width, image.height
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            return w, h
        return self.default_width, self.default_height

    def extract_and_validate(
        self,
        raw_text: str,
        screen_width: int,
        screen_height: int,
        query: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> ScreenAnalysis:
        """
        Parse raw model text into a validated ScreenAnalysis instance.
        Automatically fixes and clamps element bounding boxes to the frame bounds.
        """
        data = extract_json_payload(raw_text)

        if "screen" not in data or not isinstance(data["screen"], dict):
            data["screen"] = {"width": screen_width, "height": screen_height}
        else:
            data["screen"].setdefault("width", screen_width)
            data["screen"].setdefault("height", screen_height)

        analysis = ScreenAnalysis.model_validate(data)
        analysis.query = query
        analysis.raw_response = raw_text
        analysis.inference_duration_ms = duration_ms

        clamped_elements: List[UIElement] = []
        for elem in analysis.elements:
            if elem.bbox is not None:
                clamped_bbox = elem.bbox.clamp(analysis.screen.width, analysis.screen.height)
                clamped_elements.append(
                    elem.model_copy(update={"bbox": clamped_bbox})
                )
            else:
                clamped_elements.append(elem)

        return analysis.model_copy(update={"elements": clamped_elements})

    def analyze_screen(
        self,
        image: ImageInput,
        *,
        custom_prompt: Optional[str] = None,
    ) -> ScreenAnalysis:
        """
        Perform complete structured screen analysis on the given frame.

        Parameters
        ----------
        image:
            OpenCV numpy frame, ProcessedFrame, image bytes, or file path.
        custom_prompt:
            Optional custom prompt overriding the default schema prompt.

        Returns
        -------
        ScreenAnalysis
            Validated structured representation with detected UI elements and coordinates.
        """
        width, height = self._resolve_dimensions(image)
        prompt = custom_prompt or self.STRUCTURED_DECOMPOSITION_PROMPT.format(width=width, height=height)

        logger.info("Running structured screen analysis (%dx%d)...", width, height)
        t0 = time.monotonic()
        res: AnalysisResult = self.analyzer.analyze_detailed(
            image,
            prompt=prompt,
            remember=False,
        )
        duration_ms = (time.monotonic() - t0) * 1000.0

        return self.extract_and_validate(
            raw_text=res.content,
            screen_width=width,
            screen_height=height,
            query=None,
            duration_ms=duration_ms,
        )

    def locate_element(
        self,
        image: ImageInput,
        target: str,
    ) -> Optional[UIElement]:
        """
        Targeted visual query to find a specific UI element (e.g. "terminal", "search bar").
        Returns the UIElement with validated bounding box coordinates, or None if not located.

        Parameters
        ----------
        image:
            Screen image input.
        target:
            Name or description of the UI element to locate.

        Returns
        -------
        Optional[UIElement]
            Element with (x, y, width, height) coordinates, or None if not found.
        """
        width, height = self._resolve_dimensions(image)
        prompt = self.TARGETED_ELEMENT_PROMPT.format(
            width=width,
            height=height,
            target=target,
        )

        logger.info("Locating target '%s' in screen (%dx%d)...", target, width, height)
        t0 = time.monotonic()
        res: AnalysisResult = self.analyzer.analyze_detailed(
            image,
            prompt=prompt,
            remember=False,
        )
        duration_ms = (time.monotonic() - t0) * 1000.0

        analysis = self.extract_and_validate(
            raw_text=res.content,
            screen_width=width,
            screen_height=height,
            query=target,
            duration_ms=duration_ms,
        )

        return analysis.find_element(target) or (analysis.elements[0] if analysis.elements else None)

_PALETTE: dict[str, Tuple[int, int, int]] = {
    "application": (235, 130, 52),   # Sky blue (BGR)
    "terminal": (60, 220, 70),       # Vibrant emerald green
    "browser": (220, 80, 160),       # Magenta / Purple
    "editor": (255, 180, 50),        # Cyan / Blue
    "button": (50, 150, 255),        # Vibrant orange
    "menu_bar": (160, 160, 160),     # Neutral grey
    "dock": (180, 120, 70),          # Soft purple
    "input": (40, 200, 240),         # Warm amber
    "default": (120, 200, 120),      # Soft green
}


def _get_element_color(element_type: str) -> Tuple[int, int, int]:
    """Retrieve BGR color for element category."""
    t = element_type.lower()
    for key, color in _PALETTE.items():
        if key in t:
            return color
    return _PALETTE["default"]


def annotate_frame(
    frame: np.ndarray,
    analysis: ScreenAnalysis,
    highlight_target: Optional[str] = None,
) -> np.ndarray:
    """
    Draw clean, high-visibility visual annotations on an OpenCV BGR frame.

    Renders:
    - Bounding rectangle around each identified element.
    - Semi-transparent badge background with element name, category, and coordinates.
    - Center crosshair icon indicating exact target center point.

    Parameters
    ----------
    frame:
        Input BGR frame (numpy.ndarray).
    analysis:
        ScreenAnalysis object with detected elements.
    highlight_target:
        Optional name of an element to accentuate with a glowing border.

    Returns
    -------
    np.ndarray
        Annotated BGR frame copy.
    """
    annotated = frame.copy()
    overlay = frame.copy()

    for elem in analysis.elements:
        if not elem.bbox:
            continue

        box = elem.bbox
        is_highlighted = highlight_target and (
            highlight_target.lower() in elem.name.lower() or highlight_target.lower() in elem.type.lower()
        )
        base_color = (0, 255, 255) if is_highlighted else _get_element_color(elem.type)
        line_thickness = 3 if is_highlighted else 2

        cv2.rectangle(
            overlay,
            (box.x, box.y),
            (box.x2, box.y2),
            base_color,
            thickness=-1,
        )

        cv2.rectangle(
            annotated,
            (box.x, box.y),
            (box.x2, box.y2),
            base_color,
            thickness=line_thickness,
        )

        cx, cy = box.center
        arm = 6
        cv2.line(annotated, (cx - arm, cy), (cx + arm, cy), (255, 255, 255), 1)
        cv2.line(annotated, (cx, cy - arm), (cx, cy + arm), (255, 255, 255), 1)
        cv2.circle(annotated, (cx, cy), 2, base_color, -1)

        label = f"{elem.name} [{elem.type}] ({box.x},{box.y} {box.width}x{box.height})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        font_thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        label_y = max(label_h + 8, box.y - 4)
        label_x = box.x

        cv2.rectangle(
            annotated,
            (label_x, label_y - label_h - 4),
            (label_x + label_w + 6, label_y + baseline),
            (25, 25, 25),
            thickness=-1,
        )
        cv2.rectangle(
            annotated,
            (label_x, label_y - label_h - 4),
            (label_x + label_w + 6, label_y + baseline),
            base_color,
            thickness=1,
        )
        cv2.putText(
            annotated,
            label,
            (label_x + 3, label_y - 2),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.08, annotated, 0.92, 0, annotated)
    return annotated


def display_analysis(
    analysis: ScreenAnalysis,
    frame: Optional[np.ndarray] = None,
    *,
    window_name: str = "AI Screen Understanding",
    wait_key: bool = True,
) -> None:
    """
    Format and display screen analysis results:
    1. Outputs structured element table and coordinates to stdout.
    2. Opens an interactive OpenCV window with annotated bounding boxes if frame is provided.
    """
    print("\n" + "=" * 65)
    print(f"  AI STRUCTURED SCREEN UNDERSTANDING  ({analysis.screen.width}x{analysis.screen.height})")
    if analysis.query:
        print(f"  Query: \"{analysis.query}\"")
    if analysis.inference_duration_ms > 0:
        print(f"  Inference Time: {analysis.inference_duration_ms / 1000.0:.2f}s")
    print("=" * 65)

    if not analysis.elements:
        print("  No UI elements identified.")
    else:
        print(f"  {'TYPE':<14} {'NAME':<24} {'LOCATION':<10} {'COORDINATES (x, y, w, h)'}")
        print("  " + "-" * 63)
        for elem in analysis.elements:
            coord_str = (
                f"({elem.bbox.x}, {elem.bbox.y}, {elem.bbox.width}, {elem.bbox.height}) center=({elem.bbox.center_x}, {elem.bbox.center_y})"
                if elem.bbox
                else "N/A"
            )
            loc = elem.location or "-"
            print(f"  {elem.type:<14} {elem.name[:23]:<24} {loc:<10} {coord_str}")

    print("=" * 65 + "\n")

    if frame is not None:
        annotated = annotate_frame(frame, analysis)
        cv2.imshow(window_name, annotated)
        if wait_key:
            print("Displaying annotated screen window. Press any key or 'q' to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("Initializing ScreenAnalyzer...")
    screen_analyzer = ScreenAnalyzer()

    print("Capturing live desktop screen (1280x720)...")
    with ScreenCapture() as sc:
        frame = sc.capture_resized(1280, 720)

    print("Running structured screen understanding...")
    analysis = screen_analyzer.analyze_screen(frame)

    print("\n--- Validated JSON Output ---")
    print(json.dumps(analysis.model_dump(exclude={"raw_response"}), indent=2))

    print("\n--- Targeted Element Query: 'terminal' ---")
    terminal_elem = screen_analyzer.locate_element(frame, "terminal")
    if terminal_elem and terminal_elem.bbox:
        box = terminal_elem.bbox
        print(f"Located: {terminal_elem.name}")
        print(f"  x: {box.x}")
        print(f"  y: {box.y}")
        print(f"  width: {box.width}")
        print(f"  height: {box.height}")
        print(f"  center: ({box.center_x}, {box.center_y})")
    else:
        print("Terminal not currently located on screen.")

    display_analysis(analysis, frame, wait_key=False)
