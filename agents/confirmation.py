from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np

try:
    from agents.action_models import Action, ActionPlan, ActionType
    from agents.safety_validator import SafetyCheckResult
except ModuleNotFoundError:
    from action_models import Action, ActionPlan, ActionType
    from safety_validator import SafetyCheckResult

logger = logging.getLogger(__name__)



@dataclass
class ConfirmationResult:
    confirmed: bool
    dry_run: bool = False
    reason: Optional[str] = None


class ActionConfirmationHandler:
    """
    Explicit confirmation gatekeeper.
    Ensures that humans always have final say before any desktop actions execute.
    """

    def __init__(self, default_to_reject: bool = True) -> None:
        self.default_to_reject = default_to_reject

    def format_plan_for_display(
        self,
        plan: ActionPlan,
        safety_result: Optional[SafetyCheckResult] = None,
    ) -> str:
        """Render a formatted, high-visibility summary of the plan."""
        lines = [
            "\n" + "=" * 65,
            "               ACTION PLAN CONFIRMATION REQUIRED",
            "=" * 65,
            f"  Request:    \"{plan.request}\"",
            f"  Status:     {plan.status.upper()}",
            f"  Risk Level: {plan.risk_level.upper()}",
        ]

        if safety_result:
            if safety_result.violations:
                lines.append("  [BLOCKED VIOLATIONS]:")
                for v in safety_result.violations:
                    lines.append(f"    - {v}")
            if safety_result.warnings:
                lines.append("  [SAFETY WARNINGS]:")
                for w in safety_result.warnings:
                    lines.append(f"    - {w}")

        lines.append("-" * 65)
        lines.append(f"  Planned Actions ({len(plan.actions)}):")
        for idx, act in enumerate(plan.actions, start=1):
            lines.append(f"    {idx:2d}. {act.summary()}")
        lines.append("=" * 65)
        return "\n".join(lines)

    def preview_on_frame(
        self,
        frame: np.ndarray,
        plan: ActionPlan,
    ) -> np.ndarray:
        """
        Draw visual target crosshairs and numbered markers on the screen
        where clicks/moves are planned to land.
        """
        annotated = frame.copy()
        step = 1

        for act in plan.actions:
            if act.is_mouse_action() and act.x is not None and act.y is not None:
                x, y = act.x, act.y
                cv2.circle(annotated, (x, y), 20, (0, 165, 255), 2)
                cv2.circle(annotated, (x, y), 6, (0, 255, 255), -1)
                cv2.line(annotated, (x - 26, y), (x + 26, y), (0, 165, 255), 1)
                cv2.line(annotated, (x, y - 26), (x, y + 26), (0, 165, 255), 1)

                badge_text = f"#{step}: {act.type.value.upper()}"
                cv2.putText(
                    annotated,
                    badge_text,
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                step += 1

        return annotated

    def request_confirmation(
        self,
        plan: ActionPlan,
        safety_result: Optional[SafetyCheckResult] = None,
        *,
        frame: Optional[np.ndarray] = None,
        auto_respond: Optional[str] = None,
        show_visual_preview: bool = False,
    ) -> ConfirmationResult:
        """
        Request human confirmation for the given ActionPlan.

        Options:
        - 'y' or 'yes': Confirmed for real execution.
        - 'd' or 'dry-run': Confirmed in simulation mode (logs actions without moving hardware).
        - 'n' or enter: Aborted / rejected.

        Parameters
        ----------
        plan:
            The validated ActionPlan to confirm.
        safety_result:
            Output from SafetyValidator.
        frame:
            Optional screen frame for visual target preview.
        auto_respond:
            Optional programmatic response for testing ('y', 'n', 'dry-run').
        show_visual_preview:
            If True and frame is given, opens an OpenCV window showing target clicks.
        """
        # Automatically block if safety check failed
        if safety_result and not safety_result.is_valid:
            print(self.format_plan_for_display(plan, safety_result))
            print("\n>>> EXECUTION BLOCKED: Safety check failed. Plan rejected automatically.\n")
            plan.status = "rejected"
            return ConfirmationResult(confirmed=False, reason="Blocked by safety validator")

        print(self.format_plan_for_display(plan, safety_result))

        if show_visual_preview and frame is not None:
            preview_img = self.preview_on_frame(frame, plan)
            cv2.imshow("Planned Action Targets Preview", preview_img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

        if auto_respond is not None:
            response = auto_respond.strip().lower()
            print(f">>> Auto-response: '{response}'")
        else:
            try:
                response = input("\nExecute this action plan? [y=Yes / d=Dry-Run / n=No] (default: n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                response = "n"

        if response in ("y", "yes"):
            plan.status = "confirmed"
            logger.info("Plan '%s' EXPLICITLY CONFIRMED by user for real execution.", plan.request)
            return ConfirmationResult(confirmed=True, dry_run=False)

        if response in ("d", "dry", "dry-run"):
            plan.status = "confirmed"
            logger.info("Plan '%s' confirmed in DRY-RUN simulation mode.", plan.request)
            return ConfirmationResult(confirmed=True, dry_run=True, reason="dry-run mode requested")

        plan.status = "rejected"
        plan.reason = "User declined confirmation"
        logger.info("Plan '%s' was DECLINED by user.", plan.request)
        return ConfirmationResult(confirmed=False, dry_run=False, reason="User declined confirmation")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    handler = ActionConfirmationHandler()

    test_plan = ActionPlan(
        request="Click the terminal and type ls",
        actions=[
            Action(type=ActionType.CLICK, x=420, y=650, target="Terminal", confidence=0.92),
            Action(type=ActionType.TYPE, text="ls -la"),
        ],
    )

    # Test auto-respond demonstration
    print("Simulating auto-response 'dry-run':")
    res = handler.request_confirmation(test_plan, auto_respond="dry-run")
    print(f"\nResult: confirmed={res.confirmed}, dry_run={res.dry_run}, status={test_plan.status}")

