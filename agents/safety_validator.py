from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from agents.action_models import Action, ActionPlan, ActionType


logger = logging.getLogger(__name__)



@dataclass
class SafetyCheckResult:

    is_valid: bool
    risk_level: str = "low"  
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0

    def summary(self) -> str:
        status_str = "PASSED" if self.is_valid else "REJECTED"
        lines = [f"Safety Check: {status_str} (Risk: {self.risk_level.upper()})"]
        if self.violations:
            lines.append("  Violations:")
            for v in self.violations:
                lines.append(f"    - [BLOCK] {v}")
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - [WARN] {w}")
        return "\n".join(lines)


class SafetyValidator:
    """
    Strict safety validation engine that inspects planned actions BEFORE
    any confirmation or execution.

    Safety Guards:
    1. Screen bounds: (x, y) coordinates must strictly lie within active display dimensions.
    2. Destructive command detection: blocks hazardous shell commands in TYPE actions.
    3. Forbidden system keys: blocks dangerous power/hardware keystrokes in PRESS actions.
    4. Restricted UI zones: blocks clicks on dangerous system regions (e.g. Apple menu reboot).
    5. Action count and duration rate limits: prevents runaway loops and spamming.
    6. Target confidence checks: flags low-confidence coordinates with warnings.
    """

    DEFAULT_FORBIDDEN_COMMANDS: Sequence[str] = (
        r"\brm\s+-[rfRF]{1,3}\b",         
        r"\bsudo\b",                        
        r"\bdd\s+if=",                     
        r":\(\)\s*\{\s*:\|:&\s*\};:",      
        r"\bmkfs\b",                        
        r"\bformat\s+[a-zA-Z]:",            
        r"\bshutdown\b",                    
        r"\breboot\b",                      
        r"\bkill\s+-9\s+-1\b",             
        r"\bchmod\s+(-R\s+)?777\s+/",       
        r"\bcurl\b.*\|\s*(ba)?sh\b",      
    )

    DEFAULT_FORBIDDEN_KEYS: Sequence[str] = (
        "power",
        "sleep",
        "sysrq",
    )

    def __init__(
        self,
        default_width: int = 1280,
        default_height: int = 720,
        max_actions_per_plan: int = 10,
        min_confidence: float = 0.4,
        forbidden_commands: Optional[Sequence[str]] = None,
        forbidden_keys: Optional[Sequence[str]] = None,
        restricted_zones: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> None:
        self.default_width = default_width
        self.default_height = default_height
        self.max_actions_per_plan = max_actions_per_plan
        self.min_confidence = min_confidence
        self.forbidden_commands = list(forbidden_commands or self.DEFAULT_FORBIDDEN_COMMANDS)
        self.forbidden_keys = [k.lower() for k in (forbidden_keys or self.DEFAULT_FORBIDDEN_KEYS)]
        self.restricted_zones = restricted_zones or []

    def validate(
        self,
        plan: ActionPlan,
        *,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None,
    ) -> SafetyCheckResult:
        """
        Validate an ActionPlan.
        Updates plan.status to 'validated' if safe, or 'rejected' if violations exist.
        """
        width = screen_width or self.default_width
        height = screen_height or self.default_height

        violations: List[str] = []
        warnings: List[str] = []
        highest_risk = "low"

        if len(plan.actions) == 0:
            warnings.append("Plan contains zero actions.")
        elif len(plan.actions) > self.max_actions_per_plan:
            violations.append(
                f"Action count ({len(plan.actions)}) exceeds maximum allowed ({self.max_actions_per_plan})."
            )
            highest_risk = "high"

        for idx, act in enumerate(plan.actions, start=1):
            act_num = f"Action #{idx} ({act.type.value})"

            if act.is_mouse_action():
                if act.x is None or act.y is None:
                    violations.append(f"{act_num}: Missing target coordinates (x={act.x}, y={act.y}).")
                    highest_risk = "high"
                else:
                    if act.x < 0 or act.x >= width or act.y < 0 or act.y >= height:
                        violations.append(
                            f"{act_num}: Coordinate ({act.x}, {act.y}) is out of screen bounds ({width}x{height})."
                        )
                        highest_risk = "high"

                    for zx, zy, zw, zh in self.restricted_zones:
                        if zx <= act.x < (zx + zw) and zy <= act.y < (zy + zh):
                            violations.append(
                                f"{act_num}: Coordinate ({act.x}, {act.y}) falls within restricted zone ({zx},{zy} {zw}x{zh})."
                            )
                            highest_risk = "critical"

                if act.confidence < self.min_confidence:
                    warnings.append(
                        f"{act_num}: Low confidence score ({act.confidence:.2f} < {self.min_confidence:.2f})."
                    )
                    if highest_risk == "low":
                        highest_risk = "medium"

            elif act.type == ActionType.TYPE:
                if not act.text:
                    violations.append(f"{act_num}: Type action has empty text.")
                else:
                    for pattern in self.forbidden_commands:
                        if re.search(pattern, act.text, re.IGNORECASE):
                            violations.append(
                                f"{act_num}: Blocked destructive command pattern '{pattern}' in text: \"{act.text}\"."
                            )
                            highest_risk = "critical"

            elif act.type == ActionType.PRESS:
                if not act.key:
                    violations.append(f"{act_num}: Press action has no key specified.")
                elif act.key.lower() in self.forbidden_keys:
                    violations.append(f"{act_num}: Key '{act.key}' is strictly forbidden.")
                    highest_risk = "critical"

            elif act.type == ActionType.WAIT:
                if act.duration_ms is not None and act.duration_ms > 30000:
                    warnings.append(f"{act_num}: Excessive wait duration ({act.duration_ms}ms).")

        is_valid = len(violations) == 0

        if is_valid:
            plan.status = "validated"
            plan.risk_level = highest_risk
            plan.reason = None if not warnings else "; ".join(warnings)
        else:
            plan.status = "rejected"
            plan.risk_level = highest_risk
            plan.reason = "; ".join(violations)

        logger.info(
            "Safety validation result for '%s': is_valid=%s, risk=%s (violations=%d, warnings=%d)",
            plan.request,
            is_valid,
            highest_risk,
            len(violations),
            len(warnings),
        )

        return SafetyCheckResult(
            is_valid=is_valid,
            risk_level=highest_risk,
            violations=violations,
            warnings=warnings,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    validator = SafetyValidator()

    safe_plan = ActionPlan(
        request="Open terminal",
        actions=[
            Action(type=ActionType.CLICK, x=500, y=350, confidence=0.9),
            Action(type=ActionType.TYPE, text="ls -la"),
        ],
    )
    res_safe = validator.validate(safe_plan)
    print("Safe Plan Validation:")
    print(res_safe.summary())

    unsafe_plan = ActionPlan(
        request="Destructive command test",
        actions=[
            Action(type=ActionType.CLICK, x=2000, y=500, confidence=0.9),  # Out of bounds
            Action(type=ActionType.TYPE, text="sudo rm -rf /"),            # Blacklisted
        ],
    )
    res_unsafe = validator.validate(unsafe_plan)
    print("\nUnsafe Plan Validation:")
    print(res_unsafe.summary())

