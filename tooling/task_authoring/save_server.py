"""Local, repository-scoped save endpoint for the embedded authoring workspace."""

from __future__ import annotations

import json
import random
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

try:
    from .authoring import sample_opposite_camera_pair
except ImportError:  # Streamlit executes dashboard.py with this directory on sys.path.
    from authoring import sample_opposite_camera_pair


TASK_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def save_task_config(repo_root: Path, task_name: str, yaml_text: str) -> Path:
    """Validate and exclusively create a REALM_DROID10 task configuration."""
    if not TASK_NAME_PATTERN.fullmatch(task_name):
        raise ValueError("Task name may contain only lowercase letters, numbers, underscores, and hyphens.")
    document = yaml.safe_load(yaml_text)
    if not isinstance(document, dict) or not isinstance(document.get("task_type"), str):
        raise ValueError("Generated YAML is not a valid REALM task configuration.")

    task_directory = repo_root / "realm" / "config" / "tasks" / "REALM_DROID10" / task_name
    output_path = task_directory / "default.yaml"
    if output_path.exists():
        raise FileExistsError(f"Task {task_name!r} already exists.")
    task_directory.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        output.write(yaml_text.rstrip() + "\n")
    return output_path


def start_save_server(
    repo_root: Path, camera_poses: dict[str, dict[str, list[float]]]
) -> tuple[ThreadingHTTPServer, str, str]:
    """Start a loopback-only HTTP endpoint and return its server and URL."""

    class SaveHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/camera-pair":
                self.send_error(404)
                return
            body = json.dumps(
                sample_opposite_camera_pair(camera_poses, random.SystemRandom())
            ).encode("utf-8")
            self.send_response(200)
            self._cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors_headers()
            self.end_headers()

        def do_POST(self):
            if self.path != "/save":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 2_000_000:
                    raise ValueError("Save payload is empty or too large.")
                payload = json.loads(self.rfile.read(length))
                output_path = save_task_config(repo_root, payload["task_name"], payload["yaml"])
                response = {"ok": True, "path": str(output_path.relative_to(repo_root))}
                status = 200
            except (ValueError, KeyError, json.JSONDecodeError, FileExistsError) as error:
                response = {"ok": False, "error": str(error)}
                status = 400
            body = json.dumps(response).encode("utf-8")
            self.send_response(status)
            self._cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _cors_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), SaveHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    return server, f"{base_url}/save", f"{base_url}/camera-pair"
