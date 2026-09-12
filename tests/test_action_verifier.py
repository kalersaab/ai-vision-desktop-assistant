import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from agents.action_models import Action, ActionPlan, ActionType, VerificationResult
from agents.action_executor import ActionExecutor, ExecutionResult
from agents.action_planner import ActionPlanner, ClosedLoopResult
from agents.action_verifier import ActionVerifier
from agents.confirmation import ActionConfirmationHandler, ConfirmationResult
from agents.safety_validator import SafetyCheckResult, SafetyValidator
from vision.screen_analyzer import ScreenAnalysis, ScreenDimensions
from vision.vision_analyzer import AnalysisResult, VisionAnalyzer


class TestVerificationResultModel(unittest.TestCase):
    def test_model_fields_and_summary(self):
        res = VerificationResult(
            verified=True,
            confidence=0.95,
            observation="VS Code editor window is focused and visible.",
            reason="VS Code is now in the foreground.",
            expected_outcome="VS Code is focused",
        )
        self.assertTrue(res.verified)
        self.assertEqual(res.confidence, 0.95)
        self.assertIn("PASSED", res.summary())
        self.assertIn("VS Code is now in the foreground", res.summary())
        self.assertIn("VS Code is focused", res.summary())

    def test_failed_summary(self):
        res = VerificationResult(
            verified=False,
            confidence=0.85,
            observation="Chrome browser is still the active window.",
            reason="VS Code was not brought to the foreground.",
            expected_outcome="VS Code is focused",
        )
        self.assertFalse(res.verified)
        self.assertIn("FAILED", res.summary())


class TestActionVerifier(unittest.TestCase):
    def setUp(self):
        self.mock_vision = MagicMock(spec=VisionAnalyzer)
        self.verifier = ActionVerifier(
            vision_analyzer=self.mock_vision,
            settle_time_sec=0.0,
        )
        self.dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_verify_success(self):
        self.mock_vision.analyze_detailed.return_value = AnalysisResult(
            content='''```json
            {
              "verified": true,
              "confidence": 0.96,
              "observation": "Visual Studio Code is open and active with blinking cursor.",
              "reason": "VS Code is now focused as requested."
            }
            ```''',
            model="qwen3-vl:4b",
            duration_ms=450.0,
        )

        plan = ActionPlan(
            request="Click the code editor",
            expected_outcome="VS Code is now focused",
            actions=[
                Action(type=ActionType.CLICK, x=500, y=350, target="VS Code", confidence=0.95),
            ],
            status="executed",
        )

        result = self.verifier.verify(plan, post_frame=self.dummy_frame)

        self.assertTrue(result.verified)
        self.assertEqual(result.confidence, 0.96)
        self.assertIn("blinking cursor", result.observation)
        self.assertEqual(plan.status, "verified")

    def test_verify_failure(self):
        self.mock_vision.analyze_detailed.return_value = AnalysisResult(
            content='''{
              "verified": false,
              "confidence": 0.90,
              "observation": "Desktop shows terminal window is active; VS Code remains in the background.",
              "reason": "VS Code was not focused."
            }''',
            model="qwen3-vl:4b",
            duration_ms=400.0,
        )

        plan = ActionPlan(
            request="Click the code editor",
            expected_outcome="VS Code is now focused",
            actions=[
                Action(type=ActionType.CLICK, x=500, y=350, target="VS Code", confidence=0.95),
            ],
            status="executed",
        )

        result = self.verifier.verify(plan, post_frame=self.dummy_frame)

        self.assertFalse(result.verified)
        self.assertEqual(plan.status, "verification_failed")
        self.assertEqual(plan.reason, "VS Code was not focused.")

    def test_verify_handles_vision_analyzer_exception(self):
        self.mock_vision.analyze_detailed.side_effect = RuntimeError("Connection timeout to Ollama")

        plan = ActionPlan(
            request="Click code editor",
            expected_outcome="VS Code is focused",
            actions=[],
        )

        result = self.verifier.verify(plan, post_frame=self.dummy_frame)
        self.assertFalse(result.verified)
        self.assertEqual(plan.status, "verification_failed")
        self.assertIn("Connection timeout to Ollama", result.reason)

    def test_verify_fallback_regex_extraction(self):
        # Model returns messy text without valid JSON fences
        self.mock_vision.analyze_detailed.return_value = AnalysisResult(
            content='I checked the screenshot: "verified": true, "confidence": 0.88, "observation": "editor visible", "reason": "window focused"',
            model="qwen3-vl:4b",
            duration_ms=300.0,
        )

        plan = ActionPlan(request="Focus editor", actions=[])
        result = self.verifier.verify(plan, post_frame=self.dummy_frame)
        self.assertTrue(result.verified)
        self.assertEqual(plan.status, "verified")

    def test_no_automatic_retry_policy_on_failure(self):
        plan = ActionPlan(
            request="Click button",
            expected_outcome="Button clicked",
            actions=[Action(type=ActionType.CLICK, x=10, y=10)],
            status="executed",
        )
        fail_result = VerificationResult(
            verified=False,
            confidence=0.9,
            observation="Screen did not change.",
            reason="Button was not clicked.",
        )

        # Ensure handle_verification_failure reports and does not loop/retry
        report = self.verifier.handle_verification_failure(plan, fail_result, interactive=False)
        self.assertIn("VERIFICATION FAILED", report)
        self.assertIn("Button was not clicked", report)


class TestClosedLoopActionPlanner(unittest.TestCase):
    def setUp(self):
        self.dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Mock ScreenAnalyzer
        self.mock_screen_analyzer = MagicMock()
        self.mock_screen_analyzer.analyze_screen.return_value = ScreenAnalysis(
            screen=ScreenDimensions(width=1280, height=720),
            elements=[],
            raw_response="",
        )

        # Mock SafetyValidator
        self.mock_safety_validator = MagicMock(spec=SafetyValidator)
        self.mock_safety_validator.validate.return_value = SafetyCheckResult(
            is_valid=True,
            violations=[],
            warnings=[],
            risk_level="low",
        )

        # Mock ConfirmationHandler
        self.mock_confirmation_handler = MagicMock(spec=ActionConfirmationHandler)
        self.mock_confirmation_handler.request_confirmation.return_value = ConfirmationResult(
            confirmed=True,
            dry_run=True,
        )

        # Mock ActionExecutor
        self.mock_action_executor = MagicMock(spec=ActionExecutor)
        self.mock_action_executor.execute.return_value = ExecutionResult(
            success=True,
            actions_executed=1,
            execution_log=["[Action 1/1] CLICK at (500, 350)"],
            dry_run=True,
        )

        # Mock ActionVerifier
        self.mock_action_verifier = MagicMock(spec=ActionVerifier)
        self.mock_action_verifier.format_verification_display.return_value = "[VERIFIED BANNER]"

        self.planner = ActionPlanner(
            screen_analyzer=self.mock_screen_analyzer,
            safety_validator=self.mock_safety_validator,
            confirmation_handler=self.mock_confirmation_handler,
            action_executor=self.mock_action_executor,
            action_verifier=self.mock_action_verifier,
        )

    def test_complete_closed_loop_cycle_success(self):
        # 1. Mock plan generation
        self.mock_screen_analyzer.analyzer.analyze_detailed.return_value = AnalysisResult(
            content='''{
              "request": "Click the code editor",
              "expected_outcome": "VS Code is now focused",
              "actions": [
                {"type": "click", "target": "VS Code", "x": 500, "y": 350, "confidence": 0.95}
              ],
              "requires_confirmation": true
            }''',
            model="qwen3-vl:4b",
            duration_ms=200.0,
        )

        # 2. Mock verification success
        self.mock_action_verifier.verify.return_value = VerificationResult(
            verified=True,
            confidence=0.98,
            observation="VS Code is now in the foreground.",
            reason="VS Code is now focused",
            expected_outcome="VS Code is now focused",
        )

        # Execute closed-loop cycle
        result: ClosedLoopResult = self.planner.run_pipeline(
            "Click the code editor",
            frame=self.dummy_frame,
            auto_confirm="dry-run",
            dry_run=True,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.step_reached, "verification")
        self.assertIsNotNone(result.verification_result)
        self.assertTrue(result.verification_result.verified)
        self.assertEqual(result.plan.status, "verified")

        # Check unpack interface
        plan, exec_res, verify_res = result
        self.assertEqual(plan.expected_outcome, "VS Code is now focused")
        self.assertTrue(exec_res.success)
        self.assertTrue(verify_res.verified)

    def test_complete_closed_loop_cycle_verification_failed_no_retry(self):
        # 1. Mock plan generation
        self.mock_screen_analyzer.analyzer.analyze_detailed.return_value = AnalysisResult(
            content='''{
              "request": "Click the code editor",
              "expected_outcome": "VS Code is now focused",
              "actions": [
                {"type": "click", "target": "VS Code", "x": 500, "y": 350, "confidence": 0.95}
              ],
              "requires_confirmation": true
            }''',
            model="qwen3-vl:4b",
            duration_ms=200.0,
        )

        # 2. Mock verification failure
        self.mock_action_verifier.verify.return_value = VerificationResult(
            verified=False,
            confidence=0.88,
            observation="Browser is still focused; click missed.",
            reason="VS Code was not brought to foreground.",
            expected_outcome="VS Code is now focused",
        )

        result: ClosedLoopResult = self.planner.run_pipeline(
            "Click the code editor",
            frame=self.dummy_frame,
            auto_confirm="dry-run",
            dry_run=True,
        )

        # Must fail and halt, NOT retry
        self.assertFalse(result.success)
        self.assertEqual(result.step_reached, "verification")
        self.assertIsNotNone(result.verification_result)
        self.assertFalse(result.verification_result.verified)
        self.assertEqual(result.plan.status, "verification_failed")
        self.mock_action_verifier.handle_verification_failure.assert_called_once()
        # Verify controller was only called once (NO automatic retry loop)
        self.assertEqual(self.mock_action_executor.execute.call_count, 1)

    def test_cycle_aborts_on_safety_violation(self):
        self.mock_screen_analyzer.analyzer.analyze_detailed.return_value = AnalysisResult(
            content='''{
              "request": "Delete system",
              "expected_outcome": "Deleted",
              "actions": [{"type": "type", "text": "rm -rf /"}],
              "requires_confirmation": true
            }''',
            model="qwen3-vl:4b",
            duration_ms=100.0,
        )
        self.mock_safety_validator.validate.return_value = SafetyCheckResult(
            is_valid=False,
            violations=["Destructive command blocked"],
            risk_level="critical",
        )

        result = self.planner.run_pipeline(
            "Delete system",
            frame=self.dummy_frame,
            auto_confirm="y",
        )

        self.assertFalse(result.success)
        self.assertEqual(result.step_reached, "safety")
        self.assertEqual(self.mock_action_executor.execute.call_count, 0)
        self.assertEqual(self.mock_action_verifier.verify.call_count, 0)


if __name__ == "__main__":
    unittest.main()
