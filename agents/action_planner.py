from dataclasses import dataclass, field
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

from agents.action_models import Action, ActionPlan, ActionType, VerificationResult
from agents.action_executor import ActionExecutor, ExecutionResult
from agents.action_verifier import ActionVerifier
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


@dataclass
class ClosedLoopResult:
    """
    Result of a complete closed-loop action cycle:
    OBSERVE → UNDERSTAND → PLAN → CONFIRM → ACT → OBSERVE AGAIN → VERIFY
    """

    success: bool
    plan: ActionPlan
    execution_result: Optional[ExecutionResult] = None
    verification_result: Optional[VerificationResult] = None
    step_reached: str = "init"
    error_message: Optional[str] = None

    def __iter__(self):
        """Allow unpacking as (plan, execution_result, verification_result)."""
        return iter((self.plan, self.execution_result, self.verification_result))


class ActionPlanner:
    """
    Coordinates the full closed-loop desktop assistant cycle:

        OBSERVE (Capture Screen)
           ↓
        UNDERSTAND (Vision Scene Parsing)
           ↓
        PLAN (Candidate Actions + Expected Outcome)
           ↓
        CONFIRM (Safety Validator + Human Confirmation)
           ↓
        ACT (Physical Desktop Automation)
           ↓
        OBSERVE AGAIN (Settle UI + Post-Action Capture)
           ↓
        VERIFY (Visual Verification via Vision Model)
    """

    ACTION_PLAN_PROMPT_TEMPLATE = """You are a desktop automation planner.
User Request: "{request}"

Screen Dimensions: {width}x{height} pixels.
Detected Screen Context:
{context}

Plan the sequence of actions to fulfill the user request and declare the expected visual outcome.
Output pure JSON strictly with this schema:
{{
  "request": "{request}",
  "expected_outcome": "Visual condition expected after execution (e.g. 'VS Code is now focused')",
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
- Specify a concrete expected_outcome that can be visually verified on the screen.
- Only output the JSON object.
"""

    def __init__(
        self,
        screen_analyzer: Optional[ScreenAnalyzer] = None,
        safety_validator: Optional[SafetyValidator] = None,
        confirmation_handler: Optional[ActionConfirmationHandler] = None,
        action_executor: Optional[ActionExecutor] = None,
        action_verifier: Optional[ActionVerifier] = None,
    ) -> None:
        self.screen_analyzer = screen_analyzer or ScreenAnalyzer()
        self.safety_validator = safety_validator or SafetyValidator()
        self.confirmation_handler = confirmation_handler or ActionConfirmationHandler()
        self.action_executor = action_executor or ActionExecutor()
        self.action_verifier = action_verifier or ActionVerifier(
            vision_analyzer=self.screen_analyzer.analyzer
        )

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
            think=False,
            output_format="json",
        )

        try:
            payload = extract_json_payload(res.content)
        except ValueError:
            logger.warning("Planner response did not contain JSON; retrying with a strict JSON request.")
            retry_prompt = (
                f"{prompt}\n\nYour previous response was invalid. Return ONLY the JSON object, "
                "with no explanation, reasoning, markdown, or surrounding text."
            )
            retry_res: AnalysisResult = self.screen_analyzer.analyzer.analyze_detailed(
                frame,
                prompt=retry_prompt,
                remember=False,
                think=False,
                output_format="json",
                options={"temperature": 0.0, "num_predict": 1024},
            )
            payload = extract_json_payload(retry_res.content)
        if "expected_outcome" not in payload or not payload["expected_outcome"]:
            payload["expected_outcome"] = f"Fulfilled request: {request}"

        plan = ActionPlan.model_validate(payload)
        logger.info("Parsed candidate ActionPlan with %d actions. Expected: %s", len(plan.actions), plan.expected_outcome)
        return plan

    def run_pipeline(
        self,
        request: str,
        *,
        frame: Optional[np.ndarray] = None,
        post_frame: Optional[np.ndarray] = None,
        auto_confirm: Optional[str] = None,
        dry_run: bool = False,
        skip_verification: bool = False,
        interactive_on_failure: bool = False,
    ) -> ClosedLoopResult:
        """
        Execute the complete closed-loop action cycle:
        OBSERVE → UNDERSTAND → PLAN → CONFIRM → ACT → OBSERVE AGAIN → VERIFY

        Safety Policy:
        If verification fails, the cycle strictly DOES NOT retry automatically.
        It halts, marks status as 'verification_failed', and reports the failure.
        """
        print("\n" + "=" * 65)
        print(f"  CLOSED-LOOP ACTION CYCLE: \"{request}\"")
        print("  OBSERVE → UNDERSTAND → PLAN → CONFIRM → ACT → OBSERVE AGAIN → VERIFY")
        print("=" * 65)

        # 1. OBSERVE
        if frame is None:
            print("Step 1 (OBSERVE): Capturing desktop screen...")
            with ScreenCapture() as sc:
                frame = sc.capture_resized(1280, 720)
        else:
            print("Step 1 (OBSERVE): Using supplied screen frame...")

        h, w = frame.shape[:2]

        # 2. UNDERSTAND
        print("Step 2 (UNDERSTAND): Analyzing visual layout and UI elements...")
        screen_analysis = self.screen_analyzer.analyze_screen(frame)

        # 3. PLAN
        print("Step 3 (PLAN): Generating ActionPlan with expected post-action outcome...")
        plan = self.plan_from_request(request, frame, screen_analysis=screen_analysis)

        # 4. CONFIRM: Safety
        print("Step 4 (CONFIRM): Validating plan safety...")
        safety_result: SafetyCheckResult = self.safety_validator.validate(
            plan,
            screen_width=w,
            screen_height=h,
        )

        if not safety_result.is_valid:
            print(f"\n[BLOCKED] SafetyValidator rejected plan:\n{safety_result.summary()}")
            return ClosedLoopResult(
                success=False,
                plan=plan,
                step_reached="safety",
                error_message=safety_result.summary(),
            )

        # 4. CONFIRM: User Gate
        print("Step 4 (CONFIRM): Awaiting explicit human confirmation...")
        confirm_res: ConfirmationResult = self.confirmation_handler.request_confirmation(
            plan,
            safety_result=safety_result,
            frame=frame,
            auto_respond=auto_confirm,
        )

        if not confirm_res.confirmed:
            print(f"\n[ABORTED] Plan was not confirmed: {confirm_res.reason}")
            return ClosedLoopResult(
                success=False,
                plan=plan,
                step_reached="confirmation",
                error_message=confirm_res.reason or "Confirmation declined",
            )

        effective_dry_run = dry_run or confirm_res.dry_run
        mode_str = "DRY-RUN (SIMULATION)" if effective_dry_run else "REAL HARDWARE"

        # 5. ACT
        print(f"Step 5 (ACT): Executing confirmed actions [{mode_str}]...")
        exec_result = self.action_executor.execute(
            plan,
            frame_size=(w, h),
            safety_result=safety_result,
            dry_run=effective_dry_run,
        )

        if not exec_result.success:
            err_msg = "; ".join(exec_result.errors)
            print(f"\n[FAILED] Execution halted with error: {err_msg}")
            print(">>> NO automatic retry will be performed.")
            return ClosedLoopResult(
                success=False,
                plan=plan,
                execution_result=exec_result,
                step_reached="execution",
                error_message=err_msg,
            )

        if skip_verification:
            print("\n[NOTE] Post-action verification skipped by caller flag.")
            return ClosedLoopResult(
                success=True,
                plan=plan,
                execution_result=exec_result,
                step_reached="execution",
            )

        # 6. OBSERVE AGAIN
        print("Step 6 (OBSERVE AGAIN): Capturing post-action desktop frame...")
        if post_frame is None:
            if effective_dry_run:
                # In simulation dry-run, use the current frame unless live post-capture requested
                post_frame = frame
            else:
                post_frame = self.action_verifier.capture_post_action_screen()

        # 7. VERIFY
        print("Step 7 (VERIFY): Post-action verification via vision model...")
        verify_result = self.action_verifier.verify(
            plan,
            post_frame=post_frame,
            initial_frame=frame,
        )

        display_banner = self.action_verifier.format_verification_display(verify_result, plan)
        print(display_banner)

        if verify_result.verified:
            plan.status = "verified"
            print(f"\nClosed-Loop Action Cycle: SUCCESS (Confidence: {verify_result.confidence * 100:.0f}%)")
            return ClosedLoopResult(
                success=True,
                plan=plan,
                execution_result=exec_result,
                verification_result=verify_result,
                step_reached="verification",
            )
        else:
            plan.status = "verification_failed"
            plan.reason = verify_result.reason
            # Strict no-retry policy on verification failure
            self.action_verifier.handle_verification_failure(
                plan,
                verify_result,
                interactive=interactive_on_failure,
            )
            return ClosedLoopResult(
                success=False,
                plan=plan,
                execution_result=exec_result,
                verification_result=verify_result,
                step_reached="verification",
                error_message=f"Verification failed: {verify_result.reason}",
            )


