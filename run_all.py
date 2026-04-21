from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "react_dashboard_app"


def _stream_output(prefix: str, pipe) -> None:
    if pipe is None:
        return
    for line in iter(pipe.readline, ""):
        print(f"[{prefix}] {line.rstrip()}")


def _resolve_frontend_command() -> list[str]:
    frontend_npm = os.environ.get("FRONTEND_NPM")
    if frontend_npm:
        return [frontend_npm, "run", "dev"]

    npm_exe = shutil.which("npm.cmd") or shutil.which("npm")
    if npm_exe:
        return [npm_exe, "run", "dev"]

    vite_cmd = FRONTEND_DIR / "node_modules" / ".bin" / ("vite.cmd" if os.name == "nt" else "vite")
    if vite_cmd.exists():
        return [str(vite_cmd)]

    raise FileNotFoundError(
        "Unable to locate npm or local Vite binary. Install Node.js (npm) or run `npm install` in react_dashboard_app."
    )


def _check_backend_prereqs() -> str | None:
    try:
        subprocess.run(
            [sys.executable, "-c", "import flask"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return (
            "Backend dependency missing: Flask is not installed in the active Python environment. "
            "Run `python -m pip install -r requirement.txt`."
        )
    return None


def _check_frontend_prereqs(frontend_cmd: list[str]) -> str | None:
    # vite(.cmd) shells out to node, so make sure Node is available before launch.
    first = frontend_cmd[0].lower()
    if "vite" in first and shutil.which("node") is None:
        return (
            "Frontend dependency missing: Node.js runtime was not found in PATH. "
            "Install Node.js, then reopen terminal/IDE so PATH updates."
        )
    return None


def main() -> int:
    backend_cmd = [sys.executable, "web_enrollment_app.py"]
    try:
        frontend_cmd = _resolve_frontend_command()
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    preflight_errors = [
        err
        for err in (_check_backend_prereqs(), _check_frontend_prereqs(frontend_cmd))
        if err is not None
    ]
    if preflight_errors:
        print("Startup checks failed:")
        for err in preflight_errors:
            print(f"- {err}")
        return 1

    backend_env = os.environ.copy()
    backend_env.setdefault("PG_DEBUG", "0")
    backend_env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    backend_env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    backend_env.setdefault("ABSL_MIN_LOG_LEVEL", "2")
    backend_env.setdefault("GLOG_minloglevel", "2")

    backend = subprocess.Popen(
        backend_cmd,
        cwd=str(ROOT),
        env=backend_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        frontend = subprocess.Popen(
            frontend_cmd,
            cwd=str(FRONTEND_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=False,
        )
    except FileNotFoundError as exc:
        print(f"Failed to start frontend command: {' '.join(frontend_cmd)}")
        print(f"Reason: {exc}")
        print("Tip: install Node.js (which includes npm) and run `npm install` inside react_dashboard_app.")
        if backend.poll() is None:
            backend.terminate()
            try:
                backend.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend.kill()
        return 1

    threads = [
        threading.Thread(target=_stream_output, args=("BACKEND", backend.stdout), daemon=True),
        threading.Thread(target=_stream_output, args=("FRONTEND", frontend.stdout), daemon=True),
    ]
    for t in threads:
        t.start()

    print("Both services started.")
    print("Backend:  http://127.0.0.1:5000")
    print("Frontend: check Vite output (usually http://127.0.0.1:5173)")
    print("Press Ctrl+C to stop both.")

    try:
        while True:
            backend_rc = backend.poll()
            frontend_rc = frontend.poll()
            if backend_rc is not None:
                print(f"Backend exited with code {backend_rc}")
                break
            if frontend_rc is not None:
                print(f"Frontend exited with code {frontend_rc}")
                break
            signal.pause() if os.name != "nt" else threading.Event().wait(0.25)
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        for proc in (backend, frontend):
            if proc.poll() is None:
                proc.terminate()
        for proc in (backend, frontend):
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
