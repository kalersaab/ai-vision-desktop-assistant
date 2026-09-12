from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

from agents.action_models import Action, ActionPlan, VerificationResult
from vision.screen_analyzer import extract_json_payload
from vision.screen_capture import ScreenCapture
from vision.vision_analyzer import AnalysisResult, VisionAnalyzer

logger = logging.getLogger(__name__)


class ActionVerifier:
    """
    Post-action visual verification engine.

    Completes the closed-loop cycle:
        OBSERVE → UNDERSTAND → PLAN → CONFIRM → ACT → OBSERVE AGAIN → VERIFY

    Safety Policy:
        If verification fails, the assistant strictly DOES NOT automatically retry.
        It immediately reports the failure and requests user guidance.
    """

    VERIFICATION_PROMPT_TEMPLATE = """You are a desktop visual verifier for a closed-loop GUI automation agent.
An action has just been performed on the user's computer to fulfill the user request.

Goal / Request: "{request}"
Actions Executed:
{actions_summary}
Expected Outcome: "{expected_outcome}"

Inspect the provided screen capture taken immediately AFTER the actions were executed.
Determine whether the visual state of the screen confirms that the expected outcome has been achieved.

Output pure JSON strictly with this schema:
{{
  "verified": true,
  "confidence": 0.95,
  "observation": "Describe what is currently visible on the screen, noting window titles, focused app, active elements, or cursor/selection state.",
  "reason": "Clear explanation of why this is considered a success or failure according to the expected outcome."
}}

Verification Guidelines:
- If the requested app/window/element is visibly active, focused, opened, or changed as expected: set verified=true.
- If the target window is NOT focused, an error dialog is shown, the state did not change, or the click clearly missed: set verified=false.
- Be objective and ground your decision strictly on the visible screenshot.
- Only output the JSON object.
"""

    def __init__(
        self,
        vision_analyzer: Optional[VisionAnalyzer] = None,
        settle_time_sec: float = 0.5,
        default_resolution: Tuple[int, int] = (1280, 720),
    ) -> None:
        """
        Parameters
        ----------
        vision_analyzer:
            VisionAnalyzer instance (e.g. Qwen3-VL client). If None, defaults to new VisionAnalyzer.
        settle_time_sec:
            Seconds to pause post-action before capturing the screen to allow OS window focus
            and render transitions to stabilize.
        default_resolution:
            Target capture resolution (width, height) for screen captures.
        """
        self.vision_analyzer = vision_analyzer or VisionAnalyzer()
        self.settle_time_sec = max(0.0, settle_time_sec)
        self.default_resolution = default_resolution

    def capture_post_action_screen(self) -> np.ndarray:
        """
        Poll for UI stabilization using high-performance C++ perceptual dHash,
        then capture the current desktop screen.
        """
        with ScreenCapture() as sc:
            w, h = self.default_resolution
            try:
                from native.c_bridge import wait_for_screen_stabilize
                return wait_for_screen_stabilize(
                    lambda: sc.capture_resized(w, h),
                    max_wait_sec=max(0.6, self.settle_time_sec * 2),
                    poll_interval_sec=0.04,
                )
            except Exception as err:
                logger.debug("Fast C++ stabilization fallback to sleep: %s", err)
                if self.settle_time_sec > 0:
                    time.sleep(self.settle_time_sec)
                return sc.capture_resized(w, h)


    def verify(
        self,
        plan: ActionPlan,
        post_frame: Optional[np.ndarray] = None,
        initial_frame: Optional[np.ndarray] = None,
    ) -> VerificationResult:
        """
        Verify if the executed ActionPlan achieved its expected outcome.

        Parameters
        ----------
        plan:
            The executed ActionPlan containing request, actions, and expected_outcome.
        post_frame:
            Optional pre-captured post-action screenshot frame. If None, captures live screen.
        initial_frame:
            Optional screenshot captured before action execution for comparative reference.

        Returns
        -------
        VerificationResult
            Parsed verification verdict, confidence, observation, and explanation.
        """
        if post_frame is None:
            logger.info("Capturing post-action screen for verification...")
            post_frame = self.capture_post_action_screen()

        expected = plan.expected_outcome or f"Fulfill user request: {plan.request}"
        actions_summary = "\n".join(f"- {act.summary()}" for act in plan.actions) if plan.actions else "No specific actions."

        prompt = self.VERIFICATION_PROMPT_TEMPLATE.format(
            request=plan.request,
            actions_summary=actions_summary,
            expected_outcome=expected,
        )

        logger.info("Submitting post-action screen to vision model for verification...")
        try:
            res: AnalysisResult = self.vision_analyzer.analyze_detailed(
                post_frame,
                prompt=prompt,
                remember=False,
            )
            raw_content = res.content
        except Exception as err:
            logger.error("Vision model verification query failed: %s", err)
            result = VerificationResult(
                verified=False,
                confidence=0.0,
                observation="Vision model call failed during verification.",
                reason=f"Verification exception: {err}",
                expected_outcome=expected,
                raw_response=None,
            )
            plan.status = "verification_failed"
            plan.reason = result.reason
            return result

        # Parse JSON output
        try:
            payload = extract_json_payload(raw_content)
        except Exception as parse_err:
            logger.warning("extract_json_payload failed (%s), attempting regex recovery...", parse_err)
            payload = self._fallback_parse_payload(raw_content)

        if not payload:
            result = VerificationResult(
                verified=False,
                confidence=0.0,
                observation="Could not parse JSON verification response from vision model.",
                reason=f"Model output did not contain valid verification JSON:\n{raw_content[:200]}",
                expected_outcome=expected,
                raw_response=raw_content,
            )
            plan.status = "verification_failed"
            plan.reason = result.reason
            return result

        # Ensure expected_outcome and raw_response are preserved
        if "expected_outcome" not in payload or not payload["expected_outcome"]:
            payload["expected_outcome"] = expected
        payload["raw_response"] = raw_content

        try:
            result = VerificationResult.model_validate(payload)
        except Exception as val_err:
            result = VerificationResult(
                verified=bool(payload.get("verified", False)),
                confidence=float(payload.get("confidence", 0.0)),
                observation=str(payload.get("observation", "Schema validation failed")),
                reason=str(payload.get("reason", f"Pydantic validation error: {val_err}")),
                expected_outcome=expected,
                raw_response=raw_content,
            )

        if result.verified:
            plan.status = "verified"
            logger.info("ActionPlan verification SUCCEEDED (confidence: %.2f)", result.confidence)
        else:
            plan.status = "verification_failed"
            plan.reason = result.reason
            logger.warning("ActionPlan verification FAILED: %s", result.reason)

        return result

    def _fallback_parse_payload(self, text: str) -> Optional[dict]:
        """Attempt regex-based recovery if model output isn't strict JSON."""
        match_verified = re.search(r'"verified"\s*:\s*(true|false)', text, re.IGNORECASE)
        if not match_verified:
            return None

        is_verified = match_verified.group(1).lower() == "true"

        match_conf = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', text)
        confidence = float(match_conf.group(1)) if match_conf else 0.5

        match_obs = re.search(r'"observation"\s*:\s*"([^"]+)"', text)
        observation = match_obs.group(1) if match_obs else text[:150]

        match_reason = re.search(r'"reason"\s*:\s*"([^"]+)"', text)
        reason = match_reason.group(1) if match_reason else ("Action verified" if is_verified else "Verification failed")

        return {
            "verified": is_verified,
            "confidence": confidence,
            "observation": observation,
            "reason": reason,
        }

    def format_verification_display(
        self,
        result: VerificationResult,
        plan: ActionPlan,
    ) -> str:
        """Render a formatted, high-visibility summary of the verification verdict."""
        verdict = "[SUCCESS] VERIFIED" if result.verified else "[FAILED] VERIFICATION FAILED"
        lines = [
            "\n" + "=" * 65,
            f"       POST-ACTION VISUAL VERIFICATION: {verdict}",
            "=" * 65,
            f"  Request:          \"{plan.request}\"",
            f"  Expected Outcome: \"{result.expected_outcome or plan.expected_outcome}\"",
            f"  Verdict:          {'PASSED' if result.verified else 'FAILED'} (Confidence: {result.confidence * 100:.0f}%)",
            "-" * 65,
            f"  Observation:      {result.observation}",
            f"  Reason:           {result.reason}",
            "=" * 65,
        ]
        return "\n".join(lines)

    def handle_verification_failure(
        self,
        plan: ActionPlan,
        result: VerificationResult,
        interactive: bool = False,
    ) -> str:
        """
        Enforce strict failure handling:
        DO NOT automatically retry.
        Report failure clearly to user.
        """
        report = self.format_verification_display(result, plan)
        print(report)

        warning = (
            "\n>>> SAFETY POLICY ENFORCED: Automatic retries are DISABLED.\n"
            ">>> Desktop state did not match expected outcome. Halting closed-loop cycle.\n"
        )
        print(warning)

        if interactive:
            try:
                print("Options: [1] Acknowledge and abort (default)  [2] View debug info")
                choice = input("Select option (default 1): ").strip()
                if choice == "2":
                    print(f"\n--- Raw Model Output ---\n{result.raw_response}\n------------------------")
            except (EOFError, KeyboardInterrupt):
                pass

        return report
