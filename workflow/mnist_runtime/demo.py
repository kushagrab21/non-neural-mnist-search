from __future__ import annotations

from pathlib import Path

from .adapters import invoke_stub
from .common import Blocked, atomic_write, canonical, digest, now
from .engine import Engine
from .evaluator import evaluate, fixture_data
from .store import Store


def fixture_config():
    modules = Path(__file__).parent
    code_hash = digest({p.name: digest(p.read_bytes()) for p in sorted(modules.glob("*.py"))})
    actors = {role: {"role": role, "agent_id": "synthetic-" + role, "session_id": "synthetic-no-live-session",
              "provider": "openai" if role == "researcher" else "anthropic",
              "model_id": "fixture-no-model-called"} for role in ("runner", "researcher", "controller")}
    actors["user"] = {"role": "user", "agent_id": "synthetic-user", "session_id": "synthetic-no-user-approval",
                      "provider": "user", "model_id": None}
    cap = {"process_calls": 60, "process_seconds": 180}
    return {"campaign_id": "SYNTHETIC-PREPARATION", "mode": "synthetic", "live_inference_enabled": False,
            "sealed_assessment_enabled": False, "actors": actors,
            "limits": {"campaign": cap, "phases": {"synthetic": dict(cap)}, "batches": {"B1": dict(cap)}},
            "max_elapsed_seconds": 3600, "max_batch_candidates": 3, "direction_allocations": {},
            "memory_policy": {"id": "fixture-memory-1", "recent_per_direction": 2},
            "versions": {"contract": "synthetic-contract-1", "evaluator": code_hash,
                         "data": digest(fixture_data()), "config": "synthetic-config-1"},
            "scientific_acceptance": "none; this configuration exists only to test infrastructure"}


def initialize_batch(engine):
    order = invoke_stub(engine, "PLAN", "researcher", "plan", {})
    engine.open_batch(order, engine.actor("researcher"))


def make_candidate(engine, number, variant, direction="DIR-A", parents=None):
    decision_id, candidate_id = "D" + str(number), "C" + str(number)
    payload = {"decision_id": decision_id, "direction_id": direction, "small": variant == "small",
               "evidence_refs": ["CAMPAIGN"] + (["E" + str(number - 1)] if number > 1 else [])}
    decision = invoke_stub(engine, "HYP-" + str(number), "runner", "hypothesis", payload)
    engine.freeze_decision(decision, engine.actor("runner"))
    result = invoke_stub(engine, "BUILD-" + str(number), "runner", "build",
                         {"candidate_id": candidate_id, "decision_id": decision_id,
                          "direction_id": direction, "variant": variant, "parents": parents or []})
    engine.submit(result["candidate"], result["source"].encode(), engine.actor("runner"))
    return evaluate(engine, candidate_id, "E" + str(number))


def run_demo(root):
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise Blocked("demo needs a new empty destination; previous evidence is never overwritten")
    with Store(root) as store:
        engine = Engine(store)
        engine.initialize(fixture_config())
        initialize_batch(engine)
        first = make_candidate(engine, 1, "large")
        second = make_candidate(engine, 2, "small", parents=["C1"])
        failed = make_candidate(engine, 3, "invalid", direction="DIR-B")
        if first["accuracy"] != 1 or second["accuracy"] != 1 or failed["status"] != "failed":
            raise Blocked("synthetic known-answer expectations failed")
        change = second["parent_changes"]["C1"]
        if change["prediction_change_count"] != 0 or change["source_bytes_delta"] >= 0:
            raise Blocked("same-prediction size improvement not recorded correctly")
        packet = engine.freeze_packet("PACKET-B1", "B1")
        if not packet["all_failures"] or not packet["outstanding_anomalies"]:
            raise Blocked("required negative evidence dropped from memory")
        response = invoke_stub(engine, "RESEARCH-BOUNDARY", "researcher", "batch_decisions",
                               dict(packet, evidence_refs=["E1", "E2", "E3"]))
        engine.accept_batch_response("RULING-B1", response, engine.actor("researcher"))
        repair = invoke_stub(engine, "REPAIR-DELIVERY", "runner", "repair",
                             {"evidence_refs": ["RULING-B1", "E3"]}, ruling_ref="RULING-B1")
        if repair["received_ruling"] != "RULING-B1":
            raise Blocked("repair ruling did not reach the stub actor")
        before = canonical(engine.state)
        state_hash = digest(before)
    # Close/reacquire writer lock and rebuild from the authoritative log, not the index.
    with Store(root) as store:
        engine = Engine(store)
        if canonical(engine.state) != before:
            raise Blocked("restart changed derived state")
        resources = engine.resources()
        if resources["unresolved"]:
            raise Blocked("completed demo left resource reservations unresolved")
        summary = {"status": "pass", "completed_at": now(), "scope": "synthetic_only",
                   "model_calls": 0, "real_training_or_search": False, "sealed_data_access": False,
                   "events": len(store.events), "state_sha256_before_and_after_restart": state_hash,
                   "candidate_results": {"C1": first["status"], "C2": second["status"], "C3": failed["status"]},
                   "same_predictions_size_improvement": change,
                   "all_active_directions_decided": [d["direction_id"] for d in response["decisions"]],
                   "remaining_active_directions": engine.state["active"],
                   "repair_ruling_delivered_and_acknowledged": True, "resources": resources,
                   "live_readiness": False,
                   "remaining_gate": "DIR-B is awaiting scientific follow-up; no automated acceptance or new work inferred"}
        atomic_write(root / "reports/SUMMARY.json", canonical(summary) + b"\n")
        transcript = ["# Synthetic workflow transcript", "", "No live agents, MNIST candidate runs or model calls.", "",
                      "| Sequence | Event | Kind | Source role |", "|---|---|---|---|"]
        transcript.extend(f"| {e['seq']} | {e['event_id']} | {e['kind']} | {e['actor']['role']} |" for e in store.events)
        transcript += ["", "State was rebuilt after releasing and reacquiring the writer lock.",
                       "The failed candidate remains in memory; its repair ruling was delivered to and acknowledged by a real stub process.",
                       "DIR-A was stopped. DIR-B remains awaiting the Researcher's next scoped action. No scientific acceptance is inferred."]
        atomic_write(root / "reports/TRANSCRIPT.md", ("\n".join(transcript) + "\n").encode())
        return summary
