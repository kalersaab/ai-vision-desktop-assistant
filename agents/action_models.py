from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    MOVE = "move"
    TYPE = "type"
    PRESS = "press"
    WAIT = "wait"


class Action(BaseModel):
    type: ActionType

    target: Optional[str] = None

    x: Optional[int] = Field(default=None, ge=0)
    y: Optional[int] = Field(default=None, ge=0)

    text: Optional[str] = None

    key: Optional[str] = None

    duration_ms: Optional[int] = Field(
        default=None,
        ge=0,
        le=30000,
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    def is_mouse_action(self) -> bool:
        return self.type in (ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.MOVE)

    def is_keyboard_action(self) -> bool:
        return self.type in (ActionType.TYPE, ActionType.PRESS)

    def summary(self) -> str:
        if self.is_mouse_action():
            target_str = f" [{self.target}]" if self.target else ""
            return f"{self.type.value.upper()} at ({self.x}, {self.y}){target_str} (conf: {self.confidence:.2f})"
        if self.type == ActionType.TYPE:
            preview = (self.text[:25] + "...") if self.text and len(self.text) > 25 else self.text
            return f'TYPE "{preview}"'
        if self.type == ActionType.PRESS:
            return f'PRESS key "{self.key}"'
        if self.type == ActionType.WAIT:
            return f"WAIT {self.duration_ms or 500}ms"
        return f"{self.type.value}"


class ActionPlan(BaseModel):
    request: str

    actions: List[Action] = Field(
        default_factory=list
    )

    requires_confirmation: bool = True

    status: str = Field(
        default="ready",
        description="Lifecycle status: ready, validated, confirmed, rejected, executed",
    )

    risk_level: str = Field(
        default="low",
        description="Safety risk: low, medium, high, critical",
    )

    reason: Optional[str] = None

    def summary(self) -> str:
        lines = [
            f"ActionPlan (status: {self.status}, risk: {self.risk_level})",
            f"Request: {self.request}",
            f"Actions ({len(self.actions)}):",
        ]
        for idx, act in enumerate(self.actions, start=1):
            lines.append(f"  {idx}. {act.summary()}")
        if self.reason:
            lines.append(f"Reason: {self.reason}")
        return "\n".join(lines)