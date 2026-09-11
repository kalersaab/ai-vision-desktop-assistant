from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

from agents.action_models import Action, ActionPlan, ActionType
from agents.action_executor import ActionExecutor, ExecutionResult
from agents.confirmation import ActionConfirmationHandler, ConfirmationResult
from agents.safety_validator import SafetyCheckResult, SafetyValidator
from vision.screen_analyzer import (
    ScreenAnalysis,
    ScreenAnalyzer,
    extract_json_payload,
)
from vision.screen_capture import ScreenCapture
from vision.vision_analyzer import AnalysisResult, VisionAnalyzer

logger = logging.getLogger(__name__)




class ActionPlanner:
    """
    Coordinates the full safety-guarded desktop assistant pipeline:

        LLM
         ↓
        validated JSON
         ↓
        Pydantic
         ↓
        Safety Validator
         ↓
        explicit confirmation
         ↓
        mouse / keyboard
    """

    ACTION_PLAN_PROMPT_TEMPLATE = """You are a desktop automation planner.
User Request: "{request}"

Screen Dimensions: {width}x{height} pixels.
Detected Screen Context:
{context}

Plan the sequence of actions to fulfill the user request.
Output pure JSON strictly with this schema:
{{
  "request": "{request}",
  "actions": [
    {{
      "type": "click | double_click | move | type | press | wait",
      "target": "Element or window name",
      "x": 100,
      "y": 50,
      "text": null,
      "key": null,
      "duration_ms": 500,
      "confidence": 0.95
    }}
  ],
  "requires_confirmation": true
}}

Rules:
- All (x, y) coordinates must be integers inside [0, {width}) and [0, {height}).
- Set confidence to a float between 0.0 and 1.0 based on how sure you are of the target location.
- Only output the JSON object.
"""

    def __init__(
        self,
        screen_analyzer: Optional[ScreenAnalyzer] = None,
        safety_validator: Optional[SafetyValidator] = None,
        confirmation_handler: Optional[ActionConfirmationHandler] = None,
        action_executor: Optional[ActionExecutor] = None,
    ) -> None:
        self.screen_analyzer = screen_analyzer or ScreenAnalyzer()
        self.safety_validator = safety_validator or SafetyValidator()
        self.confirmation_handler = confirmation_handler or ActionConfirmationHandler()
        self.action_executor = action_executor or ActionExecutor()

    def plan_from_request(
        self,
        request: str,
        frame: np.ndarray,
        screen_analysis: Optional[ScreenAnalysis] = None,
    ) -> ActionPlan:
        """
        Query the vision model to produce a candidate ActionPlan from user intent.
        Extracts and parses JSON into a Pydantic ActionPlan.
        """
        h, w = frame.shape[:2]

        if screen_analysis is None:
            screen_analysis = self.screen_analyzer.analyze_screen(frame)

        context_lines: List[str] = []
        for elem in screen_analysis.elements:
            coord_str = f"bbox=(x={elem.bbox.x}, y={elem.bbox.y}, w={elem.bbox.width}, h={elem.bbox.height}) center=({elem.bbox.center_x}, {elem.bbox.center_y})" if elem.bbox else "no coordinates"
            context_lines.append(f"- [{elem.type}] \"{elem.name}\" {coord_str}")

        context_str = "\n".join(context_lines) if context_lines else "No specific elements identified."

        prompt = self.ACTION_PLAN_PROMPT_TEMPLATE.format(
            request=request,
            width=w,
            height=h,
            context=context_str,
        )

        logger.info("Requesting action plan for '%s' from vision analyzer...", request)
        res: AnalysisResult = self.screen_analyzer.analyzer.analyze_detailed(
            frame,
            prompt=prompt,
            remember=False,
        )

        payload = extract_json_payload(res.content)
        plan = ActionPlan.model_validate(payload)
        logger.info("Parsed candidate ActionPlan with %d actions.", len(plan.actions))
        return plan

    def run_pipeline(
        self,
        request: str,
        *,
        frame: Optional[np.ndarray] =   None,
        auto_confirm: Optional[str] = None,
        dry_run: bool = False,
    ) -> Tuple[ActionPlan, Optional[ExecutionResult]]:
        """
        Execute the full guarded action pipeline:
        LLM → JSON → Pydantic → Safety Validator → Confirmation → Mouse/Keyboard
        """
        print("\n" + "=" * 65)
        print(f"  STARTING GUARDED ACTION PIPELINE: \"{request}\"")
        print("=" * 65)

        if frame is None:
            print("Step 0: Capturing live desktop screen...")
            with ScreenCapture() as sc:
                frame = sc.capture_resized(1280, 720)

        h, w = frame.shape[:2]

        print("Step 1: Analyzing screen and generating ActionPlan...")
        plan = self.plan_from_request(request, frame)

        print("Step 2: Passing plan through SafetyValidator...")
        safety_result: SafetyCheckResult = self.safety_validator.validate(
            plan,
            screen_width=w,
            screen_height=h,
        )

        if not safety_result.is_valid:
            print(f"\n[BLOCKED] SafetyValidator rejected plan:\n{safety_result.summary()}")
            return plan, None

        print("Step 3: Awaiting explicit human confirmation...")
        confirm_res: ConfirmationResult = self.confirmation_handler.request_confirmation(
            plan,
            safety_result=safety_result,
            frame=frame,
            auto_respond=auto_confirm,
        )

        if not confirm_res.confirmed:
            print(f"\n[ABORTED] Plan was not confirmed: {confirm_res.reason}")
            return plan, None

        effective_dry_run = dry_run or confirm_res.dry_run

        mode_str = "DRY-RUN (SIMULATION)" if effective_dry_run else "REAL HARDWARE"
        print(f"Step 4: Executing confirmed actions [{mode_str}]...")
        exec_result = self.action_executor.execute(
            plan,
            frame_size=(w, h),
            safety_result=safety_result,
            dry_run=effective_dry_run,
        )


        status_str = "SUCCESS" if exec_result.success else "FAILED"
        print(f"\nPipeline Finished: {status_str} ({exec_result.actions_executed} actions executed)")
        return plan, exec_result
