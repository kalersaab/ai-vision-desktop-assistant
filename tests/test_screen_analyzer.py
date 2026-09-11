import unittest
import numpy as np

from vision.screen_analyzer import (
    BoundingBox,
    ScreenAnalysis,
    ScreenAnalyzer,
    ScreenDimensions,
    UIElement,
    annotate_frame,
    extract_json_payload,
)


class TestBoundingBox(unittest.TestCase):
    def test_basic_properties(self):
        box = BoundingBox(x=100, y=50, width=200, height=100)
        self.assertEqual(box.x2, 300)
        self.assertEqual(box.y2, 150)
        self.assertEqual(box.center_x, 200)
        self.assertEqual(box.center_y, 100)
        self.assertEqual(box.center, (200, 100))
        self.assertEqual(box.area, 20000)

    def test_clamping(self):
        box = BoundingBox(x=1200, y=700, width=300, height=200)
        clamped = box.clamp(max_width=1280, max_height=720)
        self.assertEqual(clamped.x, 1200)
        self.assertEqual(clamped.y, 700)
        self.assertEqual(clamped.width, 80)
        self.assertEqual(clamped.height, 20)
        self.assertLessEqual(clamped.x2, 1280)
        self.assertLessEqual(clamped.y2, 720)

    def test_scaling(self):
        box = BoundingBox(x=100, y=100, width=200, height=100)
        scaled = box.scale(src_width=1000, src_height=500, dst_width=2000, dst_height=1000)
        self.assertEqual(scaled.x, 200)
        self.assertEqual(scaled.y, 200)
        self.assertEqual(scaled.width, 400)
        self.assertEqual(scaled.height, 200)

    def test_coercion(self):
        box = BoundingBox(x="-5", y="10.7", width="50", height="20.2")
        self.assertEqual(box.x, 0)
        self.assertEqual(box.y, 11)
        self.assertEqual(box.width, 50)
        self.assertEqual(box.height, 20)


class TestUIElementAndScreenAnalysis(unittest.TestCase):
    def test_ui_element_alias_coordinates(self):
        data = {
            "type": "terminal",
            "name": "iTerm2",
            "coordinates": {"x": 50, "y": 100, "width": 600, "height": 400},
        }
        elem = UIElement.model_validate(data)
        self.assertIsNotNone(elem.bbox)
        self.assertEqual(elem.bbox.x, 50)
        self.assertEqual(elem.bbox.center, (350, 300))

    def test_ui_element_flat_coordinates(self):
        data = {
            "type": "terminal",
            "name": "Terminal",
            "x": 420,
            "y": 650,
            "width": 800,
            "height": 70,
        }
        elem = UIElement.model_validate(data)
        self.assertIsNotNone(elem.bbox)
        self.assertEqual(elem.bbox.x, 420)
        self.assertEqual(elem.bbox.y, 650)
        self.assertEqual(elem.bbox.width, 800)
        self.assertEqual(elem.bbox.height, 70)

    def test_screen_analysis_query_helpers(self):
        analysis = ScreenAnalysis(
            screen=ScreenDimensions(width=1280, height=720),
            elements=[
                UIElement(
                    type="application",
                    name="Visual Studio Code",
                    description="Code editor with Python script",
                    location="center",
                    bbox=BoundingBox(x=100, y=50, width=900, height=600),
                ),
                UIElement(
                    type="terminal",
                    name="Terminal",
                    description="Zsh shell prompt",
                    location="bottom",
                    bbox=BoundingBox(x=100, y=500, width=900, height=180),
                ),
            ],
        )

        # Name match
        term = analysis.find_element("terminal")
        self.assertIsNotNone(term)
        self.assertEqual(term.name, "Terminal")

        # Description match
        code = analysis.find_element("Python script")
        self.assertIsNotNone(code)
        self.assertEqual(code.name, "Visual Studio Code")

        # Type filter
        apps = analysis.filter_by_type("application")
        self.assertEqual(len(apps), 1)
        self.assertEqual(apps[0].name, "Visual Studio Code")

        # Point query (innermost / smallest)
        # Point inside both VS Code and Terminal (e.g. y=550) -> returns Terminal because area is smaller
        elem = analysis.get_element_at(150, 550)
        self.assertIsNotNone(elem)
        self.assertEqual(elem.name, "Terminal")


class TestJsonExtraction(unittest.TestCase):
    def test_markdown_code_fence(self):
        raw = """Here is the structured analysis:
```json
{
  "screen": {"width": 1280, "height": 720},
  "elements": [
    {
      "type": "editor",
      "name": "Sublime Text",
      "bbox": {"x": 50, "y": 50, "width": 500, "height": 400}
    }
  ]
}
```
Hope this helps!"""
        payload = extract_json_payload(raw)
        self.assertEqual(payload["screen"]["width"], 1280)
        self.assertEqual(len(payload["elements"]), 1)
        self.assertEqual(payload["elements"][0]["name"], "Sublime Text")

    def test_raw_json_with_trailing_comma(self):
        raw = """
        {
          "screen": {"width": 1280, "height": 720},
          "elements": [
            {
              "type": "browser",
              "name": "Chrome",
              "bbox": {"x": 10, "y": 20, "width": 300, "height": 400},
            },
          ]
        }
        """
        payload = extract_json_payload(raw)
        self.assertEqual(len(payload["elements"]), 1)
        self.assertEqual(payload["elements"][0]["name"], "Chrome")


class TestAnnotateFrame(unittest.TestCase):
    def test_annotate_frame_rendering(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        analysis = ScreenAnalysis(
            screen=ScreenDimensions(width=1280, height=720),
            elements=[
                UIElement(
                    type="terminal",
                    name="Terminal",
                    location="bottom",
                    bbox=BoundingBox(x=100, y=500, width=600, height=150),
                )
            ],
        )

        annotated = annotate_frame(frame, analysis, highlight_target="terminal")
        self.assertEqual(annotated.shape, (720, 1280, 3))
        # Verify that pixels inside the box are modified (non-zero)
        self.assertTrue(np.any(annotated[500:650, 100:700] > 0))


if __name__ == "__main__":
    unittest.main()
