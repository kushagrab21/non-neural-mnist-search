"""Deterministic process fixture. These outputs are never live agent decisions."""
import json
import sys
import time
from pathlib import Path

request = json.load(sys.stdin)
scenario = sys.argv[1]
if scenario == "malformed":
    print('{"truncated":')
    raise SystemExit(0)
if scenario == "null":
    print("null")
    raise SystemExit(0)
if scenario == "timeout":
    time.sleep(60)
if scenario == "exit":
    print("fixture failure", file=sys.stderr)
    raise SystemExit(7)
if scenario == "noisy":
    print("x" * 2000000)
    raise SystemExit(0)

p = request["payload"]
task = request["task"]


def decision(decision_id, direction, choice, *, small=False):
    pred = {"kind": "qualitative", "statement": "Synthetic fixture prediction; no model was called.",
            "metric_id": None, "quantity": None, "operator": None, "threshold": None,
            "unit": None, "horizon": "next_candidate", "resolution_rule_id": None}
    if small:
        pred.update(kind="quantitative", statement="The compact source will be at most 100 bytes.",
                    metric_id="source_bytes", quantity="absolute", operator="le", threshold=100,
                    unit="bytes", resolution_rule_id="SYNTHETIC-BYTES-1")
    return {"schema_version": "0.1", "decision_id": decision_id,
            "campaign_id": request["campaign_id"], "batch_id": request["batch_id"],
            "direction_id": direction, "actor": request["actor"], "created_at": "2026-10-07T00:00:00Z",
            "evidence_refs": p.get("evidence_refs", ["CAMPAIGN"]), "choice": choice,
            "justification": "Exercise the declared preparation interface using a fixed fixture.",
            "prediction": pred, "outcome_ref": None, "effort_ref": None, "anomalies_raised": []}


if task == "plan":
    body = {"batch_id": request["batch_id"], "snapshot_id": "ACTIVE-A-B",
            "directions": ["DIR-A", "DIR-B"], "max_candidates": 3,
            "versions": request["versions"], "question": "Synthetic prediction, size and failure workflow."}
elif task == "hypothesis":
    body = decision(p["decision_id"], p["direction_id"], "candidate_change", small=p.get("small", False))
elif task == "build":
    variant = p["variant"]
    if variant == "large":
        source = "# synthetic padding " + "x" * 600 + "\ndef predict(image):\n    if image[0] < 0:\n        return 9\n    return int(image[0])\n"
    elif variant == "small":
        source = "def predict(image):\n    return int(image[0])\n"
    else:
        source = "def predict(image):\n    return 11\n"
    body = {"candidate": {"candidate_id": p["candidate_id"], "decision_id": p["decision_id"],
             "batch_id": request["batch_id"], "direction_id": p["direction_id"],
             "parents": p.get("parents", []), "build_recipe": "fixed synthetic source; no fitting",
             "retained_information": "no training data", "dependencies": ["python-stdlib"]}, "source": source}
elif task == "batch_decisions":
    body = {"schema_version": "0.1", "batch_id": request["batch_id"],
            "active_direction_snapshot_id": p["active_direction_snapshot_id"],
            "decisions": [decision("R-" + direction, direction, "stop" if direction == "DIR-A" else "diagnose")
                          for direction in p["active_directions"]], "new_directions": []}
elif task == "repair":
    body = {"received_ruling": request["ruling_ref"], "action": "synthetic diagnostic acknowledged",
            "evidence_refs": p["evidence_refs"], "no_candidate_or_model_run": True}
elif task == "echo":
    body = p
elif task == "filesystem_probe":
    body = {}
    for name, request_path in p.items():
        try:
            path = Path(request_path)
            if name.startswith("write_"):
                path.write_text("synthetic probe")
            else:
                path.read_bytes()
            body[name] = "allowed"
        except PermissionError:
            body[name] = "denied"
else:
    raise RuntimeError("unknown fixture task")

actor = dict(request["actor"])
if scenario == "spoof":
    actor["role"] = "researcher" if actor["role"] != "researcher" else "runner"
print(json.dumps({"acknowledged_dispatch_id": request["dispatch_id"], "actor": actor,
                  "body": body, "usage": None if scenario == "unknown_usage" else {"model_calls": 0}}))
