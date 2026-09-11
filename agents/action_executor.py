from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from agents.action_models import Action, ActionPlan, ActionType
from agents.safety_validator import SafetyCheckResult
from automation.desktop_controller import DesktopController, ExecutionResult

logger = logging.getLogger(__name__)



class ActionExecutor:
    """
    Action executor bridge that delegates verified plans to DesktopController.

    Enforces strict gates:
    - REJECTS any plan where status != "confirmed".
    - REJECTS any plan if safety validation failed.
    - Delegates physical mouse & keyboard automation to DesktopController.
    """

    def __init__(
        self,
        desktop_controller: Optional[DesktopController] = None,
        pause_between_actions: float = 0.25,
        enable_failsafe: bool = True,
    ) -> None:
        self.controller = desktop_controller or DesktopController(
            pause_between_actions=pause_between_actions,
            enable_failsafe=enable_failsafe,
        )

    def execute(
        self,
        plan: ActionPlan,
        *,
        frame_size: Optional[Tuple[int, int]] = None,
        safety_result: Optional[SafetyCheckResult] = None,
        dry_run: bool = False,
        auto_confirm_real: Optional[bool] = None,
        grace_period_sec: float = 2.0,
    ) -> ExecutionResult:
        """
        Execute an ActionPlan via DesktopController.

        Raises
        ------
        PermissionError
            If plan is not confirmed or safety check failed.
        """
        return self.controller.execute_plan(
            plan,
            frame_size=frame_size,
            safety_result=safety_result,
            dry_run=dry_run,
            auto_confirm_real=auto_confirm_real,
            grace_period_sec=grace_period_sec,
        )



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("Testing ActionExecutor backed by DesktopController in DRY-RUN mode...")
    executor = ActionExecutor()

    sample_plan = ActionPlan(
        request="Focus code editor and type git status",
        actions=[
            Action(type=ActionType.MOVE, x=500, y=350, target="VS Code", confidence=0.95),
            Action(type=ActionType.CLICK, x=500, y=350, target="VS Code", confidence=0.95),
            Action(type=ActionType.WAIT, duration_ms=200),
            Action(type=ActionType.TYPE, text="git status"),
            Action(type=ActionType.PRESS, key="enter"),
        ],
        status="confirmed",
    )

    result = executor.execute(sample_plan, frame_size=(1280, 720), dry_run=True)
    print("\nExecution Result:", "SUCCESS" if result.success else "FAILED")
    print(f"Executed: {result.actions_executed}/{len(sample_plan.actions)} actions.")
    for entry in result.execution_log:
        print(" ", entry)
