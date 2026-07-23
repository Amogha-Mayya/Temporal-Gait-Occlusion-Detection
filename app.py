#!/usr/bin/env python3
"""
app.py  –  Inference Frontend Server
Serves a premium web UI and runs inference_v2.py as a subprocess.
"""

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

BASE_DIR = Path(__file__).parent.resolve()
OUTPUTS_ROOT = BASE_DIR / "outputs" / "inference_v2"
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

# ── In-memory job store ──────────────────────────────────────────────────────
jobs: dict[str, dict] = {}


def _run_job(job_id: str, cmd: list[str], output_dir: Path):
    jobs[job_id]["status"] = "running"
    jobs[job_id]["log"] = []
    # Inherit the current environment so user site-packages (e.g. imageio)
    # are visible to the subprocess.
    env = os.environ.copy()
    user_site = os.path.expanduser("~/.local/lib/python{}.{}/site-packages".format(*sys.version_info[:2]))
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = user_site + (":" + existing_pp if existing_pp else "")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        jobs[job_id]["pid"] = proc.pid
        for line in proc.stdout:
            jobs[job_id]["log"].append(line.rstrip())
        proc.wait()
        if proc.returncode == 0:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["output_dir"] = str(output_dir)
        else:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = f"Process exited with code {proc.returncode}"
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/api/run", methods=["POST"])
def run_inference():
    data = request.json or {}

    sequence = data.get("sequence", "").strip()
    if not sequence or not os.path.isdir(sequence):
        return jsonify({"error": f"Invalid sequence folder: {sequence}"}), 400

    mode = data.get("mode", "inject")
    occlusion_type = data.get("occlusion_type", "multi_block")
    start_frame = int(data.get("start_frame", 10))
    end_frame = int(data.get("end_frame", 20))
    target_severity = float(data.get("target_severity", 0.5))
    checkpoint = data.get("checkpoint", "checkpoints/best_model_4heads.pth")

    # Build a unique output sub-dir from the sequence folder name + timestamp
    seq_name = Path(sequence).name
    run_id = f"{seq_name}_{int(time.time())}"
    output_dir = OUTPUTS_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "inference_v2.py",
        "--sequence", sequence,
        "--checkpoint", checkpoint,
        "--mode", mode,
        "--output", str(output_dir),
    ]
    if mode == "inject":
        cmd += [
            "--occlusion_type", occlusion_type,
            "--start_frame", str(start_frame),
            "--end_frame", str(end_frame),
            "--target_severity", str(target_severity),
        ]

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "log": [], "cmd": " ".join(cmd)}

    t = threading.Thread(target=_run_job, args=(job_id, cmd, output_dir), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)


@app.route("/api/results/<job_id>")
def job_results(job_id):
    job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Job not done or not found"}), 404

    out = Path(job["output_dir"])
    files = {}
    for fname in ["clean.gif", "occluded.gif", "paired.gif", "dashboard.png", "summary.txt"]:
        p = out / fname
        files[fname] = p.exists()

    summary = ""
    summary_path = out / "summary.txt"
    if summary_path.exists():
        summary = summary_path.read_text()

    return jsonify({
        "output_dir": str(out),
        "files": files,
        "summary": summary,
        "run_id": out.name,
    })


@app.route("/api/file/<run_id>/<filename>")
def serve_file(run_id, filename):
    """Serve output files (GIFs, PNGs) by run_id."""
    safe_name = Path(filename).name  # prevent path traversal
    file_path = OUTPUTS_ROOT / run_id / safe_name
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(str(file_path))


@app.route("/api/browse")
def browse_dirs():
    """Return direct child directories of a given path (for folder picker)."""
    root = request.args.get("path", str(Path.home()))
    try:
        root_p = Path(root).resolve()
        entries = []
        for child in sorted(root_p.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                entries.append({"name": child.name, "path": str(child)})
        return jsonify({
            "current": str(root_p),
            "parent": str(root_p.parent),
            "entries": entries,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🚀  Inference UI running at http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)
