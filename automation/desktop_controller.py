from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union


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



class DesktopController:
    """
    Real OS desktop hardware controller for mouse and keyboard automation.

    Pipeline:
        ActionPlan → Safety Validator → User Confirmation → DesktopController → Mouse / Keyboard

    Features:
    - Automatic coordinate mapping from analyzed frame resolutions (e.g. 1280x720)
      to physical display points (e.g. 1710x1112 on Retina display).
    - Mouse actions: smooth movement with easing, single/double/right click, drag, scroll.
    - Keyboard actions: keystrokes, typing with human pacing, hotkey combos.
    - Fail-safe protection: PyAutoGUI FAILSAFE active (moving mouse to screen corner aborts).
    - Simulation mode (dry_run=True) for non-invasive testing and validation.
    """

    def __init__(
        self,
        pause_between_actions: float = 0.25,
        default_move_duration: float = 0.2,
        enable_failsafe: bool = True,
    ) -> None:
        self.pause_between_actions = pause_between_actions
        self.default_move_duration = default_move_duration
        self.enable_failsafe = enable_failsafe

        pyautogui.FAILSAFE = self.enable_failsafe
        pyautogui.PAUSE = self.pause_between_actions

    def get_screen_size(self) -> Tuple[int, int]:
        """Return the physical display (width, height) in OS points."""
        size = pyautogui.size()
        return size.width, size.height

    def map_coordinates(
        self,
        x: int,
        y: int,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        """
        Map coordinates from the analyzed image frame resolution to actual
        screen display coordinates.

        Parameters
        ----------
        x, y:
            Point in frame space.
        frame_size:
            (frame_width, frame_height). If None or equal to screen size,
            returns (x, y) unmodified.
        """
        if not frame_size:
            return x, y

        frame_w, frame_h = frame_size
        if frame_w <= 0 or frame_h <= 0:
            return x, y

        screen_w, screen_h = self.get_screen_size()
        if frame_w == screen_w and frame_h == screen_h:
            return x, y

        scale_x = screen_w / float(frame_w)
        scale_y = screen_h / float(frame_h)

        mapped_x = int(round(x * scale_x))
        mapped_y = int(round(y * scale_y))

        # Clamp to physical screen bounds
        mapped_x = max(0, min(mapped_x, screen_w - 1))
        mapped_y = max(0, min(mapped_y, screen_h - 1))

        return mapped_x, mapped_y

    # -----------------------------------------------------------------------
    # Emergency Stop & Confirmation Kill Switch
    # -----------------------------------------------------------------------

    def check_failsafe(self) -> None:
        """
        Check if the cursor is in an emergency kill-switch zone (any screen corner).
        Raises pyautogui.FailSafeException if triggered.
        """
        if not self.enable_failsafe:
            return

        cur_x, cur_y = pyautogui.position()
        screen_w, screen_h = self.get_screen_size()
        threshold = 5  # pixels from corner

        is_top_left = (cur_x <= threshold and cur_y <= threshold)
        is_top_right = (cur_x >= screen_w - threshold and cur_y <= threshold)
        is_bottom_left = (cur_x <= threshold and cur_y >= screen_h - threshold)
        is_bottom_right = (cur_x >= screen_w - threshold and cur_y >= screen_h - threshold)

        if is_top_left or is_top_right or is_bottom_left or is_bottom_right:
            raise pyautogui.FailSafeException(
                f"Emergency kill switch triggered! Cursor placed at corner ({cur_x}, {cur_y})."
            )

    def confirm_real_execution(
        self,
        plan: ActionPlan,
        first_action: Action,
        *,
        frame_size: Optional[Tuple[int, int]] = None,
        auto_confirm: Optional[bool] = None,
    ) -> bool:
        """
        Explicit final confirmation prompt specifically guarding the FIRST real hardware event.
        """
        mapped_target = ""
        if first_action.is_mouse_action() and first_action.x is not None and first_action.y is not None:
            mx, my = self.map_coordinates(first_action.x, first_action.y, frame_size)
            mapped_target = f"\n  Projected Target: screen point ({mx}, {my})"

        banner = [
            "\n" + "=" * 65,
            "         FINAL GATEWAY: REAL HARDWARE EXECUTION IMMINENT",
            "=" * 65,
            f"  Request:      \"{plan.request}\"",
            f"  First Action: {first_action.summary()}{mapped_target}",
            "-" * 65,
            "  KILL SWITCH (Emergency Stop):",
            "  PyAutoGUI fail-safe is ARMED.",
            "  Slam cursor into ANY CORNER of the screen (e.g. (0, 0)) to abort.",
            "=" * 65,
        ]
        print("\n".join(banner))

        if auto_confirm is not None:
            print(f">>> Auto-confirm real execution: {auto_confirm}")
            return auto_confirm

        try:
            resp = input("Allow REAL mouse/keyboard hardware execution? [y/N]: ").strip().lower()
            return resp in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def countdown_grace_period(self, seconds: float = 2.0) -> None:
        """
        Grace period allowing human user to slam cursor to corner or abort before hardware actuates.
        """
        if seconds <= 0:
            return

        print(f"\n[FAILSAFE ARMED] Starting hardware execution in {seconds:.0f}s... (Move mouse to corner to abort)")
        end_time = time.monotonic() + seconds
        while time.monotonic() < end_time:
            self.check_failsafe()
            time.sleep(0.05)

    # -----------------------------------------------------------------------
    # Mouse Operations
    # -----------------------------------------------------------------------


    def move_to(
        self,
        x: int,
        y: int,
        *,
        duration: Optional[float] = None,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        """Move cursor smoothly to target coordinates."""
        target_x, target_y = self.map_coordinates(x, y, frame_size)
        dur = duration if duration is not None else self.default_move_duration
        pyautogui.moveTo(target_x, target_y, duration=dur, tween=pyautogui.easeInOutQuad)
        return target_x, target_y

    def click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        *,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.0,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        """Click at (x, y) or current cursor position."""
        if x is not None and y is not None:
            target_x, target_y = self.map_coordinates(x, y, frame_size)
            pyautogui.click(x=target_x, y=target_y, button=button, clicks=clicks, interval=interval)
            return target_x, target_y
        else:
            cur_x, cur_y = pyautogui.position()
            pyautogui.click(button=button, clicks=clicks, interval=interval)
            return cur_x, cur_y

    def double_click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        *,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        """Double-click at target position."""
        return self.click(x, y, clicks=2, interval=0.1, frame_size=frame_size)

    def right_click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        *,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        """Right-click at target position."""
        return self.click(x, y, button="right", frame_size=frame_size)

    def drag_to(
        self,
        x: int,
        y: int,
        *,
        duration: float = 0.3,
        button: str = "left",
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        """Drag cursor to target position while holding button."""
        target_x, target_y = self.map_coordinates(x, y, frame_size)
        pyautogui.dragTo(target_x, target_y, duration=duration, button=button)
        return target_x, target_y

    def scroll(
        self,
        clicks: int,
        *,
        x: Optional[int] = None,
        y: Optional[int] = None,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Scroll vertical wheel."""
        if x is not None and y is not None:
            target_x, target_y = self.map_coordinates(x, y, frame_size)
            pyautogui.scroll(clicks, x=target_x, y=target_y)
        else:
            pyautogui.scroll(clicks)

    # -----------------------------------------------------------------------
    # Keyboard Operations
    # -----------------------------------------------------------------------

    def type_text(self, text: str, interval: float = 0.02) -> None:
        """Type string characters with human pacing."""
        pyautogui.write(text, interval=interval)

    def press_key(self, key: str) -> None:
        """Press a keyboard key (e.g. 'enter', 'esc', 'tab', 'backspace')."""
        pyautogui.press(key)

    def hotkey(self, *keys: str) -> None:
        """Press key combination sequentially and release (e.g. 'command', 'c')."""
        pyautogui.hotkey(*keys)

    def sleep(self, duration_ms: int) -> None:
        """Pause execution for specified milliseconds."""
        time.sleep(max(0, duration_ms) / 1000.0)

    # -----------------------------------------------------------------------
    # ActionPlan Execution Pipeline
    # -----------------------------------------------------------------------

    def execute_action(
        self,
        action: Action,
        *,
        frame_size: Optional[Tuple[int, int]] = None,
        dry_run: bool = False,
    ) -> str:
        """
        Execute a single validated Action.
        Returns a descriptive log entry string.
        """
        prefix = "[DRY-RUN]" if dry_run else "[EXEC]"

        if not dry_run:
            self.check_failsafe()

        if action.type == ActionType.MOVE:
            mapped_x, mapped_y = self.map_coordinates(action.x or 0, action.y or 0, frame_size)
            msg = f"{prefix} MOVE to ({mapped_x}, {mapped_y}) [frame: ({action.x}, {action.y})]"
            if not dry_run:
                self.move_to(action.x or 0, action.y or 0, frame_size=frame_size)
            return msg

        elif action.type == ActionType.CLICK:
            mapped_x, mapped_y = self.map_coordinates(action.x or 0, action.y or 0, frame_size)
            target_str = f" [{action.target}]" if action.target else ""
            msg = f"{prefix} CLICK at ({mapped_x}, {mapped_y}){target_str} [frame: ({action.x}, {action.y})]"
            if not dry_run:
                self.click(action.x, action.y, frame_size=frame_size)
            return msg

        elif action.type == ActionType.DOUBLE_CLICK:
            mapped_x, mapped_y = self.map_coordinates(action.x or 0, action.y or 0, frame_size)
            msg = f"{prefix} DOUBLE_CLICK at ({mapped_x}, {mapped_y})"
            if not dry_run:
                self.double_click(action.x, action.y, frame_size=frame_size)
            return msg

        elif action.type == ActionType.TYPE:
            msg = f'{prefix} TYPE "{action.text}"'
            if not dry_run and action.text:
                self.type_text(action.text)
            return msg

        elif action.type == ActionType.PRESS:
            msg = f'{prefix} PRESS key "{action.key}"'
            if not dry_run and action.key:
                self.press_key(action.key)
            return msg

        elif action.type == ActionType.WAIT:
            ms = action.duration_ms or 500
            msg = f"{prefix} WAIT {ms}ms"
            if not dry_run:
                self.sleep(ms)
            return msg

        return f"{prefix} UNKNOWN action type: {action.type}"

    def execute_plan(
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
        Execute an entire ActionPlan after verifying confirmation.

        Raises
        ------
        PermissionError
            If plan is not confirmed or safety validation failed.
        """
        # Strict gate 1: Safety validation
        if safety_result is not None and not safety_result.is_valid:
            raise PermissionError(
                f"Execution refused: ActionPlan failed safety validation: {safety_result.violations}"
            )

        # Strict gate 2: Plan confirmation
        if plan.status != "confirmed":
            raise PermissionError(
                f"Execution refused: ActionPlan status is '{plan.status}', but must be 'confirmed'."
            )

        # Strict gate 3: Emergency stop check & final confirmation before real hardware clicks
        if not dry_run:
            self.check_failsafe()

            first_real_act = next(
                (a for a in plan.actions if a.is_mouse_action() or a.is_keyboard_action()),
                None,
            )
            if first_real_act is not None:
                confirmed = self.confirm_real_execution(
                    plan,
                    first_real_act,
                    frame_size=frame_size,
                    auto_confirm=auto_confirm_real,
                )
                if not confirmed:
                    logger.warning("Real hardware execution declined at final gateway.")
                    plan.status = "rejected"
                    plan.reason = "Real hardware execution declined by user."
                    return ExecutionResult(
                        success=False,
                        errors=["Real hardware execution declined by user."],
                        dry_run=False,
                    )

                self.countdown_grace_period(seconds=grace_period_sec)

        logger.info(
            "DesktopController executing plan '%s' (actions=%d, dry_run=%s)",
            plan.request,
            len(plan.actions),
            dry_run,
        )


        logs: List[str] = []
        errors: List[str] = []
        executed_count = 0

        for idx, act in enumerate(plan.actions, start=1):
            log_prefix = f"Action {idx}/{len(plan.actions)}"
            try:
                log_entry = self.execute_action(act, frame_size=frame_size, dry_run=dry_run)
                formatted = f"[{log_prefix}] {log_entry}"
                print(formatted)
                logs.append(formatted)
                executed_count += 1
                if dry_run:
                    time.sleep(0.05)

            except pyautogui.FailSafeException:
                err_msg = f"[{log_prefix}] FAILSAFE TRIGGERED! Cursor moved to corner. Execution halted immediately."
                logger.critical(err_msg)
                print(f"\n>>> CRITICAL: {err_msg}\n")
                errors.append(err_msg)
                break
            except Exception as err:
                err_msg = f"[{log_prefix}] Failed executing {act.type}: {err}"
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

    controller = DesktopController()
    screen_w, screen_h = controller.get_screen_size()
    print(f"DesktopController initialized. Display Size: {screen_w}x{screen_h} points.")

    # Demonstration of coordinate mapping from 1280x720 frame space to actual screen points
    frame_coords = (500, 350)
    screen_coords = controller.map_coordinates(frame_coords[0], frame_coords[1], frame_size=(1280, 720))
    print(f"Coordinate Projection: frame {frame_coords} -> screen {screen_coords}")

    # Dry-run execution test
    sample_plan = ActionPlan(
        request="Focus terminal and check git branch",
        actions=[
            Action(type=ActionType.MOVE, x=420, y=650, target="Terminal", confidence=0.95),
            Action(type=ActionType.CLICK, x=420, y=650, target="Terminal", confidence=0.95),
            Action(type=ActionType.WAIT, duration_ms=200),
            Action(type=ActionType.TYPE, text="git branch"),
            Action(type=ActionType.PRESS, key="enter"),
        ],
        status="confirmed",
    )

    print("\nRunning DesktopController in DRY-RUN mode:")
    res = controller.execute_plan(sample_plan, frame_size=(1280, 720), dry_run=True)
    print(f"\nExecution Finished: {'SUCCESS' if res.success else 'FAILED'} ({res.actions_executed} actions)")
