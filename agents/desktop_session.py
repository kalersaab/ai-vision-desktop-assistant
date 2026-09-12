from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from agents.action_models import ActionPlan
from agents.action_planner import ActionPlanner
from native.c_bridge import wait_for_screen_stabilize
from vision.screen_capture import ScreenCapture

logger = logging.getLogger(__name__)

EventCallback = Callable[[str, Dict[str, Any]], None]
ConfirmationCallback = Callable[[ActionPlan], str]


@dataclass
class SessionResult:
    success: bool
    plan: Optional[ActionPlan] = None
    error: Optional[str] = None


class DesktopAgentSession:
    """Reusable closed-loop desktop workflow independent of any UI toolkit."""

    def __init__(self, planner: Optional[ActionPlanner] = None) -> None:
        self.planner = planner or ActionPlanner()

    def run(
        self,
        request: str,
        *,
        emit: Optional[EventCallback] = None,
        confirm: Optional[ConfirmationCallback] = None,
    ) -> SessionResult:
        def publish(event: str, **payload: Any) -> None:
            if emit:
                emit(event, payload)

        try:
            publish("status", message="Capturing desktop screen...")
            with ScreenCapture() as capture:
                frame = capture.capture_resized(1280, 720)
            height, width = frame.shape[:2]

            publish("status", message="Analyzing elements & generating action plan...")
            screen_analysis = self.planner.screen_analyzer.analyze_screen(frame)
            plan = self.planner.plan_from_request(request, frame, screen_analysis=screen_analysis)

            safety_result = self.planner.safety_validator.validate(
                plan,
                screen_width=width,
                screen_height=height,
            )
            if not safety_result.is_valid:
                message = safety_result.summary()
                publish("blocked", message=message, plan=plan.model_dump(mode="json"))
                return SessionResult(False, plan, message)

            publish("proposal", plan=plan.model_dump(mode="json"))
            choice = confirm(plan) if confirm else "n"
            if choice == "n":
                plan.status = "rejected"
                publish("cancelled", message="Action cancelled by user.")
                return SessionResult(False, plan, "Action cancelled by user.")

            dry_run = choice == "dry-run"
            mode = "DRY-RUN (SIMULATION)" if dry_run else "REAL HARDWARE"
            publish("status", message=f"Executing confirmed actions [{mode}]...")
            plan.status = "confirmed"
            execution_result = self.planner.action_executor.execute(
                plan,
                frame_size=(width, height),
                safety_result=safety_result,
                dry_run=dry_run,
            )
            if not execution_result.success:
                message = "; ".join(execution_result.errors)
                publish("failed", message=f"Execution failed: {message}")
                return SessionResult(False, plan, message)

            publish("status", message="Screen settling (C++ perceptual dHash)...")
            with ScreenCapture() as capture:
                post_frame = wait_for_screen_stabilize(lambda: capture.capture_resized(width, height))

            publish("status", message="Verifying post-action state with vision model...")
            verification = self.planner.action_verifier.verify(
                plan,
                post_frame=post_frame,
                initial_frame=frame,
            )
            if verification.verified:
                publish("success", message=f"Action completed and verified.\n({verification.reason})")
                return SessionResult(True, plan)

            message = (
                "Action verification failed.\n"
                f"Observation: {verification.observation}\n"
                f"Reason: {verification.reason}\n\n"
                "Policy: Automatic retries are DISABLED."
            )
            publish("failed", message=message)
            return SessionResult(False, plan, message)
        except Exception as err:
            logger.error("Error in desktop agent session: %s", err, exc_info=True)
            publish("error", message=str(err))
            return SessionResult(False, None, str(err))
        finally:
            publish("done")
