#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, State};

struct BridgeProcess {
    child: Child,
    stdin: ChildStdin,
}

struct BridgeState(Mutex<Option<BridgeProcess>>);

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap_or_else(|_| PathBuf::from("."))
}

#[tauri::command]
fn start_agent(app: AppHandle, state: State<'_, BridgeState>) -> Result<(), String> {
    let mut process = state.0.lock().map_err(|_| "Bridge state lock failed".to_string())?;
    if process.is_some() {
        return Ok(());
    }

    let root = project_root();
    let python = std::env::var_os("AI_VISION_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join(".venv/bin/python"));
    eprintln!("Starting Python backend: {} (cwd: {})", python.display(), root.display());
    let mut child = Command::new(python)
        .current_dir(&root)
        .args(["-m", "backend.agent_bridge"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|err| format!("Could not start Python backend: {err}"))?;

    let stdin = child.stdin.take().ok_or_else(|| "Python backend stdin unavailable".to_string())?;
    let stdout = child.stdout.take().ok_or_else(|| "Python backend stdout unavailable".to_string())?;
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().flatten() {
            let event = serde_json::from_str::<Value>(&line).unwrap_or_else(|_| json!({ "event": "error", "message": line }));
            let _ = app.emit("agent-event", event);
        }
        let _ = app.emit("agent-event", json!({ "event": "bridge-stopped" }));
    });

    *process = Some(BridgeProcess { child, stdin });
    Ok(())
}

#[tauri::command]
fn send_agent_message(state: State<'_, BridgeState>, message: Value) -> Result<(), String> {
    let mut process = state.0.lock().map_err(|_| "Bridge state lock failed".to_string())?;
    let bridge = process.as_mut().ok_or_else(|| "Python backend is not running".to_string())?;
    serde_json::to_writer(&mut bridge.stdin, &message).map_err(|err| err.to_string())?;
    bridge.stdin.write_all(b"\n").map_err(|err| err.to_string())?;
    bridge.stdin.flush().map_err(|err| err.to_string())
}

#[tauri::command]
fn stop_agent(state: State<'_, BridgeState>) -> Result<(), String> {
    let mut process = state.0.lock().map_err(|_| "Bridge state lock failed".to_string())?;
    if let Some(mut bridge) = process.take() {
        bridge.child.kill().map_err(|err| err.to_string())?;
    }
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .manage(BridgeState(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![start_agent, send_agent_message, stop_agent])
        .run(tauri::generate_context!())
        .expect("error while running Vision Desk");
}
