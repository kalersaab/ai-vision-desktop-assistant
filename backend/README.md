# Python Agent Bridge

`agent_bridge.py` exposes the desktop agent as a JSON-lines process for Tauri or another desktop UI.

Start it from the repository root:

```bash
.venv/bin/python -m backend.agent_bridge
```

Send a request:

```json
{"id":"request-1","type":"request","request":"Focus VS Code and type git status"}
```

The bridge emits status events and then a proposal event. Confirm the proposal with one of:

```json
{"id":"request-1","type":"confirm","decision":"y"}
{"id":"request-1","type":"confirm","decision":"dry-run"}
{"id":"request-1","type":"confirm","decision":"n"}
```

All protocol messages use the same request `id`. Logs go to stderr; stdout is reserved for JSON events.

## Tauri

The Tauri process manager starts this module automatically during `tauri:dev`.
It expects the project environment at `.venv/bin/python`. To use another Python
interpreter, set `AI_VISION_PYTHON` before launching Tauri:

```bash
AI_VISION_PYTHON=/absolute/path/to/python npm run tauri:dev
```
