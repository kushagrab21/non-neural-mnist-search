from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from .common import Blocked, Conflict, canonical, digest, strict_json
from .processes import run_process
from .evaluator import python_binary, restricted_profile

STUB = Path(__file__).with_name("stub_actor.py")


def assert_live_ready(config, binding):
    """Fail closed. Merely enabling a flag cannot launch a historical session."""
    if config.get("live_inference_enabled") is not True:
        raise Blocked("live inference disabled")
    requirements = ("day_zero_approval_ref", "numeric_budget_ref", "isolation_acceptance_ref",
                    "scientific_acceptance_ref", "adapter_qualification_ref")
    if any(not config.get(key) for key in requirements):
        raise Blocked("live release gates incomplete")
    if binding.get("transport") in ("existing_claude_interactive", "codex_collaboration_task"):
        raise Blocked("native-session wake-up bridge is not implemented; do not clone or resume by guessed CLI ID")
    raise Blocked("no live adapter qualified in this preparation release")


def invoke_stub(engine, dispatch_id, role, task, payload, *, batch_id="B1", timeout=3,
                scenario="normal", ruling_ref=None):
    if engine.config["mode"] != "synthetic":
        raise Blocked("stub cannot masquerade as a live role")
    if dispatch_id in engine.state["dispatches"]:
        raise Conflict("dispatch already issued; inspect receipt or reconcile, never rerun blindly")
    if ruling_ref is not None and ruling_ref not in engine.state["rulings"]:
        raise Blocked("unknown repair ruling")
    actor = engine.actor(role)
    request = {"dispatch_id": dispatch_id, "actor": actor, "task": task, "payload": payload,
               "campaign_id": engine.config["campaign_id"], "batch_id": batch_id,
               "versions": engine.config["versions"], "ruling_ref": ruling_ref}
    request_bytes = canonical(request)
    request_sha = engine.store.archive(request_bytes)
    reservation_id = "RES-" + dispatch_id
    engine.reserve(reservation_id, {"process_calls": 1, "process_seconds": timeout}, phase="synthetic", batch_id=batch_id)
    outbox = engine.store.root / "workspaces" / role / "outbox"
    profile = restricted_profile([STUB], write_dirs=[outbox])
    argv = [shutil.which("sandbox-exec"), "-p", profile, str(python_binary()), "-I", "-B", str(STUB), scenario]
    engine.add(dispatch_id, "dispatch", {"dispatch_id": dispatch_id, "role": role, "actor_binding": actor,
               "transport": "synthetic_subprocess", "task": task, "request_sha256": request_sha,
               "argv": argv, "stub_sha256": digest(STUB.read_bytes()), "reservation_id": reservation_id,
               "profile_sha256": digest(profile.encode()),
               "ruling_ref": ruling_ref, "batch_id": batch_id})
    raw = run_process(argv, request_bytes, timeout=timeout,
                      env={"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"})
    raw_sha = engine.store.archive(canonical(raw))
    result, error = None, None
    if raw["status"] != "exited" or raw["returncode"] != 0:
        error = "process " + raw["status"] + ": " + str(raw["returncode"])
    else:
        try:
            result = strict_json(raw["stdout"])
            if not isinstance(result, dict) or set(result) != {"acknowledged_dispatch_id", "actor", "body", "usage"}:
                raise ValueError("missing return envelope")
            if result["acknowledged_dispatch_id"] != dispatch_id or result["actor"] != actor:
                raise ValueError("return identity differs from bound dispatch")
            if not isinstance(result["body"], dict):
                raise ValueError("non-object structured body")
        except (ValueError, TypeError) as exc:
            error = str(exc)
    status = "completed" if error is None else "operational_failure"
    receipt_id = "RECEIPT-" + dispatch_id
    engine.add(receipt_id, "invocation_receipt", {"dispatch_id": dispatch_id, "status": status,
                "raw_artifact_sha256": raw_sha, "error": error,
                "acknowledged": error is None, "bound_actor": actor,
                "started_at": raw["started_at"], "ended_at": raw["ended_at"],
                "result_sha256": engine.store.archive(canonical(result)) if result is not None else None})
    unknown = raw["status"] == "timeout" or (result is not None and result.get("usage") is None)
    engine.settle("USAGE-" + dispatch_id, reservation_id,
                  None if unknown else {"process_calls": 1, "process_seconds": raw["elapsed_seconds"]}, receipt_id)
    if error:
        engine.anomaly("ANOMALY-" + dispatch_id, receipt_id, error)
        raise Blocked(error + "; raw receipt retained")
    if ruling_ref is not None:
        engine.add("DELIVERED-" + dispatch_id, "ruling_delivered", {"ruling_ref": ruling_ref,
                    "dispatch_id": dispatch_id, "acknowledgment_ref": receipt_id})
    return result["body"]
