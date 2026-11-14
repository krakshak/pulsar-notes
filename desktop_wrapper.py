"""Native macOS Desktop Wrapper (PyWebView) for Zero-Trust Local Meeting Assistant.

Launches the FastAPI backend strictly on 127.0.0.1:8000 in a background thread,
opens a native macOS PyWebView window ("Local Meeting Notes"), and binds
window.events.closed += purge_ephemeral_buffers so all RAM buffers and MLX caches
are scrubbed immediately when the window closes.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from typing import Optional


def _configure_offline_env() -> None:
    """Ensure offline and anti-telemetry flags are set before importing backend modules."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("OLLAMA_HOST", "127.0.0.1:11434")
    os.environ.setdefault("OLLAMA_ORIGINS", "http://127.0.0.1:8000")
    os.environ.setdefault("OLLAMA_NOHISTORY", "1")
    os.environ.setdefault("OLLAMA_DEBUG", "0")


_configure_offline_env()

import uvicorn  # noqa: E402
import webview  # noqa: E402

from backend.config import HOST, PORT  # noqa: E402
from backend.main import app, purge_ephemeral_buffers  # noqa: E402


def create_server(host: str = HOST, port: int = PORT) -> uvicorn.Server:
    """Create a Uvicorn server strictly bound to loopback with access logging disabled."""
    uv_config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        access_log=False,
        log_level="warning",
    )
    return uvicorn.Server(uv_config)


def wait_for_server(host: str = HOST, port: int = PORT, timeout: float = 10.0) -> bool:
    """Poll loopback TCP port until FastAPI is ready to accept connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def launch_desktop_app(
    host: str = HOST,
    port: int = PORT,
    server: Optional[uvicorn.Server] = None,
) -> None:
    """Start FastAPI in a background thread and open the native PyWebView window."""
    srv = server or create_server(host=host, port=port)
    server_thread = threading.Thread(target=srv.run, daemon=True, name="fastapi-loopback")
    server_thread.start()

    if not wait_for_server(host=host, port=port, timeout=10.0):
        srv.should_exit = True
        purge_ephemeral_buffers()
        raise RuntimeError(f"Timed out waiting for local server on http://{host}:{port}")

    window = webview.create_window(
        "Local Meeting Notes",
        f"http://{host}:{port}",
        width=1280,
        height=840,
        min_size=(960, 640),
    )

    def _on_window_closed() -> None:
        purge_ephemeral_buffers()
        srv.should_exit = True

    window.events.closed += purge_ephemeral_buffers
    window.events.closed += _on_window_closed

    try:
        webview.start()
    finally:
        purge_ephemeral_buffers()
        srv.should_exit = True


if __name__ == "__main__":
    launch_desktop_app()
