import unittest

from agents.action_models import Action, ActionPlan, ActionType
from agents.safety_validator import SafetyCheckResult
from automation.desktop_controller import DesktopController


class TestDesktopController(unittest.TestCase):
    def setUp(self):
        self.controller = DesktopController()

    def test_screen_size_retrieval(self):
        w, h = self.controller.get_screen_size()
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)

    def test_coordinate_mapping(self):
        screen_w, screen_h = self.controller.get_screen_size()

        # Origin mapping
        x0, y0 = self.controller.map_coordinates(0, 0, frame_size=(1280, 720))
        self.assertEqual(x0, 0)
        self.assertEqual(y0, 0)

        # Center mapping
        xc, yc = self.controller.map_coordinates(640, 360, frame_size=(1280, 720))
        expected_xc = int(round(640 * (screen_w / 1280.0)))
        expected_yc = int(round(360 * (screen_h / 720.0)))
        self.assertEqual(xc, expected_xc)
        self.assertEqual(yc, expected_yc)

        # Max bound clamping
        xmax, ymax = self.controller.map_coordinates(1280, 720, frame_size=(1280, 720))
        self.assertLess(xmax, screen_w)
        self.assertLess(ymax, screen_h)

    def test_execute_action_dry_run(self):
        act_move = Action(type=ActionType.MOVE, x=100, y=100)
        msg_move = self.controller.execute_action(act_move, frame_size=(1280, 720), dry_run=True)
        self.assertIn("[DRY-RUN] MOVE", msg_move)

        act_click = Action(type=ActionType.CLICK, x=100, y=100, target="Terminal")
        msg_click = self.controller.execute_action(act_click, frame_size=(1280, 720), dry_run=True)
        self.assertIn("[DRY-RUN] CLICK", msg_click)
        self.assertIn("Terminal", msg_click)

        act_type = Action(type=ActionType.TYPE, text="git status")
        msg_type = self.controller.execute_action(act_type, dry_run=True)
        self.assertIn('[DRY-RUN] TYPE "git status"', msg_type)

        act_press = Action(type=ActionType.PRESS, key="enter")
        msg_press = self.controller.execute_action(act_press, dry_run=True)
        self.assertIn('[DRY-RUN] PRESS key "enter"', msg_press)

    def test_execute_plan_unconfirmed_rejected(self):
        plan = ActionPlan(
            request="Unconfirmed test",
            actions=[Action(type=ActionType.WAIT, duration_ms=100)],
            status="ready",
        )
        with self.assertRaises(PermissionError) as ctx:
            self.controller.execute_plan(plan, dry_run=True)
        self.assertIn("must be 'confirmed'", str(ctx.exception))

    def test_execute_plan_safety_failure_rejected(self):
        plan = ActionPlan(
            request="Unsafe test",
            actions=[Action(type=ActionType.WAIT, duration_ms=100)],
            status="confirmed",
        )
        safety_failure = SafetyCheckResult(is_valid=False, violations=["Destructive command"])
        with self.assertRaises(PermissionError) as ctx:
            self.controller.execute_plan(plan, safety_result=safety_failure, dry_run=True)
        self.assertIn("failed safety validation", str(ctx.exception))

    def test_execute_plan_confirmed_dry_run_success(self):
        plan = ActionPlan(
            request="Safe confirmed plan",
            actions=[
                Action(type=ActionType.MOVE, x=100, y=100),
                Action(type=ActionType.CLICK, x=100, y=100),
                Action(type=ActionType.WAIT, duration_ms=50),
            ],
            status="confirmed",
        )
        res = self.controller.execute_plan(plan, frame_size=(1280, 720), dry_run=True)
        self.assertTrue(res.success)
        self.assertEqual(res.actions_executed, 3)
        self.assertEqual(plan.status, "executed")

    def test_real_execution_confirmation_declined(self):
        plan = ActionPlan(
            request="Real action test",
            actions=[
                Action(type=ActionType.CLICK, x=100, y=100),
            ],
            status="confirmed",
        )
        # Final hardware confirmation declined
        res = self.controller.execute_plan(
            plan,
            frame_size=(1280, 720),
            dry_run=False,
            auto_confirm_real=False,
        )
        self.assertFalse(res.success)
        self.assertEqual(plan.status, "rejected")
        self.assertEqual(res.actions_executed, 0)
        self.assertIn("declined", res.errors[0])


if __name__ == "__main__":
    unittest.main()

