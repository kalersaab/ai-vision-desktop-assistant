from __future__ import annotations

import json
import logging
import sys
import threading
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, Dict, Optional

_project_root = __import__("pathlib").Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from agents.desktop_session import DesktopAgentSession
from agents.action_models import ActionPlan

logger = logging.getLogger(__name__)


@dataclass
class PendingConfirmation:
    event: threading.Event
    decision: Optional[str] = None


class AgentBridge:
    """JSON-lines adapter for launching the Python agent from Tauri."""

    def __init__(self) -> None:
        self.session = DesktopAgentSession()
        self.protocol_stream = sys.stdout
        self.output_lock = threading.Lock()
        self.confirmations: Dict[str, PendingConfirmation] = {}
        self.confirmations_lock = threading.Lock()

    def send(self, request_id: str, event: str, **payload: Any) -> None:
        message = {"id": request_id, "event": event, **payload}
        with self.output_lock:
            self.protocol_stream.write(json.dumps(message, separators=(",", ":")) + "\n")
            self.protocol_stream.flush()

    def confirm(self, request_id: str, plan: ActionPlan) -> str:
        pending = PendingConfirmation(threading.Event())
        with self.confirmations_lock:
            self.confirmations[request_id] = pending
        self.send(request_id, "proposal", plan=plan.model_dump(mode="json"))
        pending.event.wait()
        with self.confirmations_lock:
            self.confirmations.pop(request_id, None)
        return pending.decision or "n"

    def handle(self, message: Dict[str, Any]) -> None:
        request_id = str(message.get("id", ""))
        message_type = message.get("type")
        if not request_id:
            return

        if message_type == "request":
            request = str(message.get("request", "")).strip()
            if not request:
                self.send(request_id, "error", message="Request cannot be empty.")
                return
            threading.Thread(target=self._run_request, args=(request_id, request), daemon=True).start()
            return

        if message_type == "confirm":
            decision = str(message.get("decision", "n"))
            if decision not in {"y", "n", "dry-run"}:
                self.send(request_id, "error", message="Decision must be y, n, or dry-run.")
                return
            with self.confirmations_lock:
                pending = self.confirmations.get(request_id)
            if pending:
                pending.decision = decision
                pending.event.set()
            else:
                self.send(request_id, "error", message="No pending confirmation for request.")
            return

        if message_type == "cancel":
            with self.confirmations_lock:
                pending = self.confirmations.get(request_id)
            if pending:
                pending.decision = "n"
                pending.event.set()
            return

        self.send(request_id, "error", message=f"Unknown message type: {message_type}")

    def _run_request(self, request_id: str, request: str) -> None:
        self.send(request_id, "started", request=request)

        def emit(event: str, payload: Dict[str, Any]) -> None:
            self.send(request_id, event, **payload)

        # The underlying controller prints human-readable execution diagnostics.
        # Keep those off stdout so Tauri receives only JSON-lines protocol events.
        with redirect_stdout(sys.stderr):
            result = self.session.run(
                request,
                emit=emit,
                confirm=lambda plan: self.confirm(request_id, plan),
            )
        if result.plan is not None and result.error is not None:
            self.send(request_id, "result", success=result.success, error=result.error, plan=result.plan.model_dump(mode="json"))
        else:
            self.send(request_id, "result", success=result.success, plan=result.plan.model_dump(mode="json") if result.plan else None)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
    bridge = AgentBridge()
    for raw_line in sys.stdin:
        try:
            message = json.loads(raw_line)
            if isinstance(message, dict):
                bridge.handle(message)
        except json.JSONDecodeError as err:
            bridge.send("", "error", message=f"Invalid JSON: {err}")


if __name__ == "__main__":
    main()
