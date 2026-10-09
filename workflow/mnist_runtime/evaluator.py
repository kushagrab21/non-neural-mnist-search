from __future__ import annotations

import ast
import json
import math
import os
import platform
import shutil
import statistics
import sys
from pathlib import Path

from .common import Blocked, canonical, digest, identifier, strict_json
from .processes import run_process

CHILD = Path(__file__).with_name("image_child.py")


def python_binary():
    # macOS's bin/python launcher spawns Python.app; bind the actual binary directly.
    app = Path(sys.base_prefix) / "Resources/Python.app/Contents/MacOS/Python"
    return app.resolve() if app.exists() else Path(sys.executable).resolve()


def restricted_profile(read_files, *, write_dirs=()):
    if platform.system() != "Darwin" or not shutil.which("sandbox-exec"):
        raise Blocked("tested macOS sandbox unavailable; no unsandboxed fallback")
    quote = lambda p: json.dumps(str(Path(p).resolve()))
    allowed = ["/System", "/usr/lib", sys.base_prefix]
    return "\n".join([
        "(version 1)", "(deny default)",
        "(allow process-exec (literal " + quote(python_binary()) + "))",
        "(allow file-read* (literal \"/\") (literal \"/dev/null\") (literal \"/dev/urandom\") " +
        " ".join("(subpath " + quote(p) + ")" for p in allowed) +
        " " + " ".join("(literal " + quote(p) + ")" for p in read_files) + ")",
        *["(allow file-read* file-write* (subpath " + quote(p) + "))" for p in write_dirs],
        "(allow sysctl-read)", "(allow mach-lookup)",
        "(allow process-info*)", "(allow signal (target self))", ""])


def sandbox_profile(candidate_path):
    return restricted_profile([CHILD, candidate_path])


def child_run(candidate_path, image, timeout):
    profile = sandbox_profile(candidate_path)
    raw = run_process([shutil.which("sandbox-exec"), "-p", profile, str(python_binary()),
                       "-I", "-B", str(CHILD), str(candidate_path)],
                      canonical({"image": image}), timeout=timeout, cwd="/",
                      env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"},
                      output_limit=65536)
    raw["profile_sha256"] = digest(profile.encode())
    parsed, error = None, None
    if raw["status"] != "exited" or raw["returncode"] != 0:
        error = raw["status"] + ": exit " + str(raw["returncode"])
    else:
        try:
            parsed = strict_json(raw["stdout"])
            if not isinstance(parsed, dict) or parsed.get("status") not in ("ok", "candidate_error"):
                raise ValueError("invalid child response")
            if parsed["status"] == "ok":
                if type(parsed.get("prediction")) is not int or not 0 <= parsed["prediction"] <= 9:
                    raise ValueError("invalid prediction")
                for key in ("import_ms", "prediction_ms"):
                    if type(parsed.get(key)) not in (int, float) or not math.isfinite(parsed[key]) or parsed[key] < 0:
                        raise ValueError("invalid child timing")
                if not isinstance(parsed.get("executed_lines"), list) or any(type(i) is not int for i in parsed["executed_lines"]):
                    raise ValueError("invalid trace")
        except (ValueError, TypeError) as exc:
            error = str(exc)
    return raw, parsed, error


def fixture_data():
    return [[i] + [0] * 783 for i in range(4)], [0, 1, 2, 3]


def evaluate(engine, candidate_id, evaluation_id, *, images=None, labels=None, timeout=2):
    """Synthetic harness only. No MNIST loader, baseline execution or sealed opener."""
    identifier(evaluation_id)
    if engine.config["mode"] != "synthetic":
        raise Blocked("synthetic evaluator cannot be promoted by a config flag")
    if evaluation_id in engine.state["evaluations"] or engine.store.find("RES-" + evaluation_id):
        raise Blocked("evaluation already admitted; reconcile instead of repeating")
    if images is None and labels is None:
        images, labels = fixture_data()
    if not isinstance(images, list) or not isinstance(labels, list) or not images or len(images) != len(labels):
        raise Blocked("nonempty matching image/label arrays required")
    if any(not isinstance(im, list) or len(im) != 784 or any(type(v) is not int or not 0 <= v <= 255 for v in im) for im in images):
        raise Blocked("synthetic image must contain 784 uint8-valued integers")
    if any(type(v) is not int or not 0 <= v <= 9 for v in labels):
        raise Blocked("invalid labels")
    candidate = engine.state["candidates"][candidate_id]
    if (candidate["batch_id"] != engine.state["current_batch"] or
        engine.state["batch_states"][candidate["batch_id"]] != "open"):
        raise Blocked("evaluation admission closed at the batch checkpoint")
    source = engine.store.read_artifact(candidate["source_sha256"])
    destination = engine.store.root / "invocations" / evaluation_id
    destination.mkdir(exist_ok=False)
    path = destination / "classifier.py"
    path.write_bytes(source)
    path.chmod(0o444)
    # Fail before reserving/starting if this platform cannot supply the boundary.
    profile = sandbox_profile(path)
    (destination / "candidate.sb").write_text(profile)
    reservation = "RES-" + evaluation_id
    engine.reserve(reservation, {"process_calls": len(images), "process_seconds": timeout * len(images)},
                   phase="synthetic", batch_id=candidate["batch_id"])
    engine.add("START-" + evaluation_id, "evaluation_started", {"evaluation_id": evaluation_id,
               "candidate_id": candidate_id, "reservation_id": reservation, "count": len(images)})
    observations, predictions, errors, timings, wall, traces = [], [], [], [], [], set()
    for index, image in enumerate(images):
        raw, child, error = child_run(path, image, timeout)
        observations.append({"index": index, "raw": raw, "parsed": child, "protocol_error": error})
        wall.append(raw["elapsed_seconds"])
        if error or child["status"] != "ok":
            errors.append({"index": index, "error": error or child["error_type"], "detail": child})
            predictions.append(None)
        else:
            predictions.append(child["prediction"])
            timings.append(child["prediction_ms"])
            traces.update(child["executed_lines"])
    raw_sha = engine.store.archive(canonical(observations))
    raw_ref = "RAW-" + evaluation_id
    engine.add(raw_ref, "evaluation_raw", {"evaluation_id": evaluation_id, "artifact_sha256": raw_sha})
    engine.settle("USAGE-" + evaluation_id, reservation,
                  {"process_calls": len(observations), "process_seconds": sum(wall)}, raw_ref)
    status = "ok" if not errors else "failed"
    n = len(images)
    confusion = [[0] * 10 for _ in range(10)]
    for truth, pred in zip(labels, predictions):
        if pred is not None:
            confusion[truth][pred] += 1
    changes = {}
    for parent in candidate["parents"]:
        matches = [e for e in engine.state["evaluations"].values()
                   if e["candidate_id"] == parent and e["versions"] == engine.config["versions"]
                   and e["data_sha256"] == digest({"images": images, "labels": labels}) and e["status"] == "ok"]
        if status != "ok" or not matches:
            changes[parent] = {"status": "not_comparable"}
        else:
            earlier = matches[-1]
            changes[parent] = {"status": "comparable",
                "prediction_change_count": sum(a != b for a, b in zip(earlier["predictions"], predictions)),
                "fixed_indices": [i for i, (a, b, y) in enumerate(zip(earlier["predictions"], predictions, labels)) if a != y and b == y],
                "broken_indices": [i for i, (a, b, y) in enumerate(zip(earlier["predictions"], predictions, labels)) if a == y and b != y],
                "source_bytes_delta": len(source) - earlier["source_bytes"]}
    try:
        executable = {node.lineno for node in ast.walk(ast.parse(source)) if isinstance(node, ast.stmt) and not isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    except (SyntaxError, ValueError):
        executable = set()
    result = {"evaluation_id": evaluation_id, "candidate_id": candidate_id,
              "candidate_sha256": candidate["source_sha256"], "versions": engine.config["versions"],
              "evaluator_scope": "synthetic_only_not_scientifically_qualified",
              "status": status, "n": n, "predictions": predictions,
              "accuracy": sum(y == pred for y, pred in zip(labels, predictions)) / n if status == "ok" else None,
              "confusion": confusion, "source_bytes": len(source), "errors": errors,
              "prediction_median_ms": statistics.median(timings) if status == "ok" else None,
              "parent_measured_process_median_ms": statistics.median(wall) * 1000,
              "timing_boundary": "one fresh process per image; child prediction time includes diagnostic tracing; not a warmed MNIST/CNN comparison",
              "child_telemetry_trust": "diagnostic only; hostile in-process Python could forge child timings/trace/protocol",
              "executed_lines": sorted(traces), "unexecuted_statement_lines": sorted(executable - traces),
              "parent_changes": changes, "data_sha256": digest({"images": images, "labels": labels}),
              "raw_artifact_sha256": raw_sha, "profile_sha256": digest(profile.encode()),
              "isolation": "macOS deny-default filesystem/network/process sandbox; fresh child receives one image; synthetic protected-path probes required",
              "memory_limit": "CPU and descriptors bounded; RSS/address-space limit not yet qualified"}
    engine.record_evaluation(result)
    engine.resolve_prediction(candidate["decision_id"], evaluation_id, "USAGE-" + evaluation_id)
    if status != "ok":
        engine.anomaly("ANOMALY-" + evaluation_id, evaluation_id, "Candidate failed; retain all missing prediction slots and diagnostics.")
    return result
