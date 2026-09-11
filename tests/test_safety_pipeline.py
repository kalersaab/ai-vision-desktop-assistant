import unittest
import numpy as np

from agents.action_models import Action, ActionPlan, ActionType
from agents.action_executor import ActionExecutor
from agents.confirmation import ActionConfirmationHandler
from agents.safety_validator import SafetyValidator


class TestSafetyValidator(unittest.TestCase):
    def setUp(self):
        self.validator = SafetyValidator(
            default_width=1280,
            default_height=720,
            max_actions_per_plan=5,
            restricted_zones=[(0, 0, 50, 30)],  # top-left system zone
        )

    def test_valid_safe_plan(self):
        plan = ActionPlan(
            request="Open terminal and check status",
            actions=[
                Action(type=ActionType.MOVE, x=420, y=650, confidence=0.9),
                Action(type=ActionType.CLICK, x=420, y=650, confidence=0.9),
                Action(type=ActionType.WAIT, duration_ms=500),
                Action(type=ActionType.TYPE, text="git status"),
                Action(type=ActionType.PRESS, key="enter"),
            ],
        )
        res = self.validator.validate(plan)
        self.assertTrue(res.is_valid)
        self.assertEqual(len(res.violations), 0)
        self.assertEqual(plan.status, "validated")

    def test_out_of_bounds_coordinates_rejected(self):
        plan = ActionPlan(
            request="Click off-screen",
            actions=[
                Action(type=ActionType.CLICK, x=1500, y=500, confidence=0.9),  # 1500 > 1280
            ],
        )
        res = self.validator.validate(plan)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("out of screen bounds" in v for v in res.violations))
        self.assertEqual(plan.status, "rejected")

    def test_restricted_zone_click_rejected(self):
        plan = ActionPlan(
            request="Click apple menu",
            actions=[
                Action(type=ActionType.CLICK, x=20, y=10, confidence=0.9),  # inside (0, 0, 50, 30)
            ],
        )
        res = self.validator.validate(plan)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("restricted zone" in v for v in res.violations))
        self.assertEqual(res.risk_level, "critical")
        self.assertEqual(plan.status, "rejected")

    def test_destructive_rm_rf_blocked(self):
        plan = ActionPlan(
            request="Run clean up",
            actions=[
                Action(type=ActionType.TYPE, text="rm -rf /tmp/test"),
            ],
        )
        res = self.validator.validate(plan)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("destructive command" in v for v in res.violations))
        self.assertEqual(plan.status, "rejected")

    def test_destructive_sudo_blocked(self):
        plan = ActionPlan(
            request="Run privileged command",
            actions=[
                Action(type=ActionType.TYPE, text="sudo apt-get update"),
            ],
        )
        res = self.validator.validate(plan)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("destructive command" in v for v in res.violations))
        self.assertEqual(plan.status, "rejected")

    def test_forbidden_key_blocked(self):
        plan = ActionPlan(
            request="Press power",
            actions=[
                Action(type=ActionType.PRESS, key="power"),
            ],
        )
        res = self.validator.validate(plan)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("strictly forbidden" in v for v in res.violations))
        self.assertEqual(plan.status, "rejected")

    def test_action_count_limit(self):
        plan = ActionPlan(
            request="Spam clicks",
            actions=[
                Action(type=ActionType.CLICK, x=100, y=100) for _ in range(10)
            ],
        )
        res = self.validator.validate(plan)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("exceeds maximum allowed" in v for v in res.violations))
        self.assertEqual(plan.status, "rejected")


class TestActionConfirmationHandler(unittest.TestCase):
    def setUp(self):
        self.handler = ActionConfirmationHandler()

    def test_auto_respond_yes(self):
        plan = ActionPlan(
            request="Test",
            actions=[Action(type=ActionType.WAIT, duration_ms=100)],
        )
        res = self.handler.request_confirmation(plan, auto_respond="y")
        self.assertTrue(res.confirmed)
        self.assertFalse(res.dry_run)
        self.assertEqual(plan.status, "confirmed")

    def test_auto_respond_dry_run(self):
        plan = ActionPlan(
            request="Test",
            actions=[Action(type=ActionType.WAIT, duration_ms=100)],
        )
        res = self.handler.request_confirmation(plan, auto_respond="dry-run")
        self.assertTrue(res.confirmed)
        self.assertTrue(res.dry_run)
        self.assertEqual(plan.status, "confirmed")

    def test_auto_respond_no_or_default(self):
        plan = ActionPlan(
            request="Test",
            actions=[Action(type=ActionType.WAIT, duration_ms=100)],
        )
        res = self.handler.request_confirmation(plan, auto_respond="n")
        self.assertFalse(res.confirmed)
        self.assertEqual(plan.status, "rejected")


class TestActionExecutorGates(unittest.TestCase):
    def setUp(self):
        self.executor = ActionExecutor()

    def test_unconfirmed_plan_rejected(self):
        plan = ActionPlan(
            request="Test unconfirmed",
            actions=[Action(type=ActionType.WAIT, duration_ms=100)],
            status="ready",  # Not confirmed
        )
        with self.assertRaises(PermissionError) as ctx:
            self.executor.execute(plan)
        self.assertIn("must be 'confirmed'", str(ctx.exception))

    def test_dry_run_execution(self):
        plan = ActionPlan(
            request="Test dry run",
            actions=[
                Action(type=ActionType.MOVE, x=100, y=100),
                Action(type=ActionType.CLICK, x=100, y=100),
                Action(type=ActionType.TYPE, text="hello"),
            ],
            status="confirmed",
        )
        res = self.executor.execute(plan, dry_run=True)
        self.assertTrue(res.success)
        self.assertEqual(res.actions_executed, 3)
        self.assertEqual(plan.status, "executed")
        self.assertTrue(res.dry_run)


if __name__ == "__main__":
    unittest.main()
