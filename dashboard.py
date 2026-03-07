"""Unified training dashboard with version/backend selection.

Auto-discovers all v{N} and v{N}_{backend} directories and serves
a live-updating web UI with dropdowns for version, backend, and stage.

Usage:
    python dashboard.py
    python dashboard.py --port 5001
"""

import argparse
import json
import re
import time
from pathlib import Path

from flask import Flask, Response, render_template, jsonify, request

ROOT = Path(__file__).resolve().parent

app = Flask(__name__, template_folder=str(ROOT / "templates"))
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Auto-discovered versions: {"v1": {"mps": Path, "nvidia": Path, ...}, ...}
AVAILABLE = {}
DEFAULT_VERSION = "v1"
DEFAULT_BACKEND = "mps"


def discover():
    """Scan project root for v{N} and v{N}_{backend} directories."""
    global AVAILABLE, DEFAULT_VERSION, DEFAULT_BACKEND

    pattern = re.compile(r"^(v\d+)(?:_(\w+))?$")
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        # Must have a logs/ directory to be considered
        if not (d / "logs").is_dir():
            continue
        version = m.group(1)          # "v1"
        backend = m.group(2) or "mps"  # None -> "mps" (default)
        AVAILABLE.setdefault(version, {})[backend] = d

    if AVAILABLE:
        DEFAULT_VERSION = sorted(AVAILABLE.keys())[0]
        first_backends = AVAILABLE[DEFAULT_VERSION]
        DEFAULT_BACKEND = "mps" if "mps" in first_backends else sorted(first_backends.keys())[0]


def _get_params():
    """Extract version/backend/stage from request query params."""
    version = request.args.get("version", DEFAULT_VERSION)
    backend = request.args.get("backend", DEFAULT_BACKEND)
    stage = request.args.get("stage", "")
    return version, backend, stage


def _resolve_logs_dir(version: str, backend: str) -> Path | None:
    """Get the logs directory for a version/backend combo."""
    combo = AVAILABLE.get(version, {}).get(backend)
    return combo / "logs" if combo else None


def _resolve_path(version: str, backend: str, stage: str, filename: str) -> Path | None:
    """Resolve a file path in the logs directory.

    If stage is 'pretrain' or 'finetune', tries logs/{stage}/{filename} first,
    then falls back to logs/{filename} (backward compat for root-level logs).
    """
    logs_dir = _resolve_logs_dir(version, backend)
    if not logs_dir:
        return None
    if stage in ("pretrain", "finetune"):
        stage_path = logs_dir / stage / filename
        if stage_path.exists():
            return stage_path
        # Fallback to root-level (e.g., v1_nvidia/logs/metrics.jsonl)
        root_path = logs_dir / filename
        return root_path if root_path.exists() else stage_path
    return logs_dir / filename


def _read_jsonl(path: Path | None) -> list[dict]:
    """Read all records from a JSONL file."""
    if not path or not path.exists():
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


# --- Routes ---

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/available")
def get_available():
    """Return available version/backend combos with data availability."""
    result = {}
    for version, backends in sorted(AVAILABLE.items()):
        result[version] = {}
        for backend, path in sorted(backends.items()):
            logs = path / "logs"
            has_pretrain = (
                (logs / "pretrain" / "metrics.jsonl").exists()
                or (logs / "metrics.jsonl").exists()
            )
            has_finetune = (logs / "finetune" / "metrics.jsonl").exists()
            has_pretrain_diag = (logs / "pretrain" / "diagnostics.json").exists()
            has_finetune_diag = (logs / "finetune" / "diagnostics.json").exists()
            result[version][backend] = {
                "has_pretrain": has_pretrain,
                "has_finetune": has_finetune,
                "has_pretrain_diagnostics": has_pretrain_diag,
                "has_finetune_diagnostics": has_finetune_diag,
            }
    return jsonify(result)


@app.route("/api/metrics")
def get_all_metrics():
    """Return all metrics as JSON. ?version=...&backend=...&stage=..."""
    version, backend, stage = _get_params()
    path = _resolve_path(version, backend, stage, "metrics.jsonl")
    return jsonify(_read_jsonl(path))


@app.route("/api/weight_stats")
def get_weight_stats():
    """Return all weight stats as JSON. ?version=...&backend=...&stage=..."""
    version, backend, stage = _get_params()
    path = _resolve_path(version, backend, stage, "weight_stats.jsonl")
    return jsonify(_read_jsonl(path))


@app.route("/diagnostics")
def diagnostics_page():
    """Serve the model diagnostics page."""
    return render_template("diagnostics.html")


@app.route("/api/diagnostics")
def get_diagnostics():
    """Return pre-computed diagnostics as JSON. ?version=...&backend=...&stage=..."""
    version, backend, stage = _get_params()
    if stage not in ("pretrain", "finetune"):
        stage = "pretrain"
    logs_dir = _resolve_logs_dir(version, backend)
    if not logs_dir:
        return jsonify({"error": f"Invalid version/backend: {version}/{backend}"}), 404
    diag_path = logs_dir / stage / "diagnostics.json"
    if not diag_path.exists():
        return jsonify({
            "error": f"No {stage} diagnostics for {version}/{backend}. "
                     f"Run: python -m {version}.scripts.diagnose --stage {stage}"
        }), 404
    with open(diag_path, "r") as f:
        data = json.load(f)
    data["stage"] = stage
    data["version"] = version
    data["backend"] = backend
    return jsonify(data)


@app.route("/api/stream")
def stream():
    """SSE endpoint: streams new metrics. ?version=...&backend=...&stage=..."""
    version, backend, stage = _get_params()
    stream_path = _resolve_path(version, backend, stage, "metrics.jsonl")

    def event_stream():
        last_pos = stream_path.stat().st_size if (stream_path and stream_path.exists()) else 0
        while True:
            try:
                if stream_path and stream_path.exists():
                    with open(stream_path, "r") as f:
                        f.seek(last_pos)
                        new_lines = f.readlines()
                        last_pos = f.tell()

                    for line in new_lines:
                        line = line.strip()
                        if line:
                            yield f"data: {line}\n\n"
            except Exception:
                pass
            time.sleep(1)

    return Response(event_stream(), mimetype="text/event-stream")


def main():
    parser = argparse.ArgumentParser(description="GPT-2 Training Dashboard")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    discover()

    combos = [f"{v}/{b}" for v, bs in sorted(AVAILABLE.items()) for b in sorted(bs.keys())]
    print(f"Dashboard: http://localhost:{args.port}")
    print(f"Available: {', '.join(combos)}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
