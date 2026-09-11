from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pyautogui

from agents.action_models import Action, ActionPlan, ActionType
from agents.safety_validator import SafetyCheckResult



logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    success: bool
    actions_executed: int = 0
    errors: List[str] = field(default_factory=list)
    execution_log: List[str] = field(default_factory=list)
    dry_run: bool = False


class ActionExecutor:
    """
    Safe desktop action executor.

    Enforces strict gates:
    - REJECTS any plan where status != "confirmed".
    - REJECTS any plan if safety validation failed.
    - Enables PyAutoGUI FAILSAFE (moving mouse to corner 0,0 instantly halts execution).
    - Enforces pause delays between consecutive events for human intervention.
    - Provides safe dry-run simulation mode.
    """

    def __init__(
        self,
        pause_between_actions: float = 0.3,
        enable_failsafe: bool = True,
    ) -> None:
        self.pause_between_actions = pause_between_actions
        self.enable_failsafe = enable_failsafe

        # Configure PyAutoGUI global safety
        pyautogui.FAILSAFE = self.enable_failsafe
        pyautogui.PAUSE = self.pause_between_actions

    def execute(
        self,
        plan: ActionPlan,
        *,
        safety_result: Optional[SafetyCheckResult] = None,
        dry_run: bool = False,
    ) -> ExecutionResult:
        """
        Execute an ActionPlan.

        Raises
        ------
        PermissionError
            If plan is not confirmed or safety check failed.
        """
        if safety_result is not None and not safety_result.is_valid:
            raise PermissionError(
                f"Execution refused: ActionPlan failed safety check with violations: {safety_result.violations}"
            )

        if plan.status != "confirmed":
            raise PermissionError(
                f"Execution refused: ActionPlan status is '{plan.status}', but must be 'confirmed'."
            )

        logger.info(
            "Executing ActionPlan '%s' (dry_run=%s, actions=%d)",
            plan.request,
            dry_run,
            len(plan.actions),
        )

        logs: List[str] = []
        errors: List[str] = []
        executed_count = 0

        for idx, act in enumerate(plan.actions, start=1):
            log_prefix = f"[Action {idx}/{len(plan.actions)}]"
            try:
                if dry_run:
                    msg = f"{log_prefix} [SIMULATED] {act.summary()}"
                    print(msg)
                    logs.append(msg)
                    executed_count += 1
                    time.sleep(0.05)
                    continue

                if act.type == ActionType.MOVE:
                    msg = f"{log_prefix} Moving to ({act.x}, {act.y})"
                    print(msg)
                    logs.append(msg)
                    pyautogui.moveTo(act.x, act.y, duration=0.2)

                elif act.type == ActionType.CLICK:
                    msg = f"{log_prefix} Clicking at ({act.x}, {act.y})"
                    print(msg)
                    logs.append(msg)
                    pyautogui.click(x=act.x, y=act.y)

                elif act.type == ActionType.DOUBLE_CLICK:
                    msg = f"{log_prefix} Double-clicking at ({act.x}, {act.y})"
                    print(msg)
                    logs.append(msg)
                    pyautogui.doubleClick(x=act.x, y=act.y)

                elif act.type == ActionType.TYPE:
                    msg = f'{log_prefix} Typing "{act.text}"'
                    print(msg)
                    logs.append(msg)
                    pyautogui.write(act.text, interval=0.02)

                elif act.type == ActionType.PRESS:
                    msg = f'{log_prefix} Pressing key "{act.key}"'
                    print(msg)
                    logs.append(msg)
                    pyautogui.press(act.key)

                elif act.type == ActionType.WAIT:
                    wait_sec = (act.duration_ms or 500) / 1000.0
                    msg = f"{log_prefix} Waiting {wait_sec:.2f}s"
                    print(msg)
                    logs.append(msg)
                    time.sleep(wait_sec)

                executed_count += 1

            except pyautogui.FailSafeException as err:
                err_msg = f"{log_prefix} FAILSAFE TRIGGERED (mouse slammed to screen corner)! Aborting all remaining actions."
                logger.critical(err_msg)
                print(f"\n>>> CRITICAL: {err_msg}\n")
                errors.append(err_msg)
                break
            except Exception as err:
                err_msg = f"{log_prefix} Failed to execute: {err}"
                logger.error(err_msg)
                errors.append(err_msg)
                break

        success = len(errors) == 0
        if success:
            plan.status = "executed"
        else:
            plan.status = "failed"
            plan.reason = "; ".join(errors)

        return ExecutionResult(
            success=success,
            actions_executed=executed_count,
            errors=errors,
            execution_log=logs,
            dry_run=dry_run,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("Testing ActionExecutor in DRY-RUN mode...")
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

    result = executor.execute(sample_plan, dry_run=True)
    print("\nExecution Result:", "SUCCESS" if result.success else "FAILED")
    print(f"Executed: {result.actions_executed}/{len(sample_plan.actions)} actions.")
    for entry in result.execution_log:
        print(" ", entry)

