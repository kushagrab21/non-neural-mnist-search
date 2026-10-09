from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path

from .common import Blocked, Conflict, amount, canonical, digest, identifier, strict_json, validate

SCHEMAS = Path(__file__).resolve().parents[2] / "contract"
RUNTIME = {"role": "controller_runtime", "agent_id": "deterministic-python", "session_id": "local",
           "provider": "local", "model_id": None}


class Engine:
    def __init__(self, store):
        self.store = store
        self.state = self.rebuild()

    def rebuild(self):
        state = {"config": None, "orders": {}, "active": [], "snapshot_id": None,
                 "current_batch": None, "batch_states": {}, "batch_snapshots": {},
                 "version_authorizations": {}, "guidance": {},
                 "reservations": {}, "dispatches": {}, "decisions": {}, "candidates": {},
                 "evaluations": {}, "anomalies": {}, "releases": {}, "consumed": [],
                 "rulings": {}, "outcomes": {}, "direction_definitions": {}, "event_refs": [], "last_event_hash": None}
        for event in self.store.events:
            # Derived state must never alias immutable event payloads.
            p, k = copy.deepcopy(event["payload"]), event["kind"]
            state["event_refs"].append(event["event_id"])
            state["last_event_hash"] = event["hash"]
            if k == "campaign":
                state["config"] = p
            elif k == "versions_changed":
                state["config"]["versions"] = p["versions"]
            elif k == "version_authorization":
                state["version_authorizations"][p["ruling_id"]] = p
            elif k == "guidance":
                state["guidance"][p["guidance_id"]] = p
            elif k == "order":
                state["orders"][p["batch_id"]] = p
                state["active"] = list(p["directions"])
                state["snapshot_id"] = p["snapshot_id"]
                state["current_batch"] = p["batch_id"]
                state["batch_states"][p["batch_id"]] = "open"
                state["batch_snapshots"][p["batch_id"]] = p["snapshot_id"]
                for direction in p["directions"]:
                    state["direction_definitions"].setdefault(direction, {"created_by_order": p["batch_id"]})
            elif k == "reservation":
                state["reservations"][p["reservation_id"]] = dict(p, actual=None, status="reserved")
            elif k == "settlement":
                state["reservations"][p["reservation_id"]].update(actual=p["actual"], status="settled")
            elif k == "usage_unknown":
                state["reservations"][p["reservation_id"]]["status"] = "unknown"
            elif k == "dispatch":
                state["dispatches"][p["dispatch_id"]] = dict(p, status="issued")
            elif k == "invocation_receipt":
                state["dispatches"][p["dispatch_id"]].update(status=p["status"], receipt=p)
            elif k == "decision":
                state["decisions"][p["decision_id"]] = p
            elif k == "checkpoint":
                state["batch_states"][p["batch_id"]] = "awaiting_researcher"
            elif k == "submission":
                state["candidates"][p["candidate_id"]] = p
            elif k == "evaluation":
                state["evaluations"][p["evaluation_id"]] = p
            elif k == "outcome":
                state["outcomes"][p["decision_id"]] = p
            elif k == "anomaly":
                state["anomalies"][p["anomaly_id"]] = p
            elif k == "anomaly_resolution":
                state["anomalies"][p["anomaly_id"]] = dict(state["anomalies"][p["anomaly_id"]], resolution=p)
            elif k == "batch_response":
                state["rulings"][p["ruling_id"]] = p
                state["batch_states"][p["response"]["batch_id"]] = "closed"
                for decision in p["response"]["decisions"]:
                    state["decisions"][decision["decision_id"]] = decision
                    if decision["choice"] == "stop":
                        state["active"].remove(decision["direction_id"])
                for direction in p["response"]["new_directions"]:
                    state["active"].append(direction["direction_id"])
                    state["direction_definitions"][direction["direction_id"]] = direction
                state["snapshot_id"] = "SNAP-" + digest({"ruling_id": p["ruling_id"], "active": state["active"]})[:20]
            elif k == "release":
                state["releases"][p["release_id"]] = p
            elif k == "release_consumed":
                state["consumed"].append(p["release_id"])
        self.state = state
        self.store.write_index(state)
        return state

    def add(self, event_id, kind, payload, actor=RUNTIME, **kwargs):
        result = self.store.append(event_id, kind, actor, payload, **kwargs)
        self.rebuild()
        return result

    def initialize(self, config, *, release=None):
        if config.get('mode') in ('bounded_pilot','bounded_search'):
            from .pilot_release import require_release
            require_release(release)
            if config.get('live_inference_enabled') is not True or config['actors'] != release['contract']['actors']:
                raise Blocked('pilot configuration differs from released actor bindings')
            if config.get('max_batch_candidates') != 1:
                raise Blocked('only one-candidate pilot implemented')
        elif config.get("mode") != "synthetic" or config.get("live_inference_enabled") is not False:
            raise Blocked("synthetic or explicitly released bounded pilot required")
        if config.get("sealed_assessment_enabled") is not False:
            raise Blocked("sealed data disabled in preparation release")
        identifier(config["campaign_id"])
        for cap in config["limits"]["campaign"].values():
            amount(cap)
        if not config["limits"]["campaign"] or amount(config["max_elapsed_seconds"]) == 0:
            raise Blocked("explicit finite preparation caps required")
        if release is not None:self.add('PILOT-RELEASE','pilot_release',release)
        return self.add("CAMPAIGN", "campaign", config)

    @property
    def config(self):
        if not self.state["config"]:
            raise Blocked("campaign unconfigured")
        return self.state["config"]

    def actor(self, role):
        try:
            return self.config["actors"][role]
        except KeyError as exc:
            raise Blocked("role not bound") from exc

    def resources(self):
        keys = self.config["limits"]["campaign"]
        charged = {k: 0 for k in keys}
        held = {k: 0 for k in keys}
        for reservation in self.state["reservations"].values():
            target = charged if reservation["status"] == "settled" else held
            for key, val in (reservation["actual"] if reservation["status"] == "settled" else reservation["reserved"]).items():
                target[key] += val
        return {"charged": charged, "held": held,
                "remaining": {k: keys[k] - charged[k] - held[k] for k in keys},
                "unresolved": [r["reservation_id"] for r in self.state["reservations"].values()
                               if r["status"] != "settled"]}

    def reserve(self, reservation_id, requested, *, phase, batch_id):
        identifier(reservation_id)
        payload = {"reservation_id": reservation_id, "reserved": requested, "phase": phase, "batch_id": batch_id}
        if self.store.find(reservation_id):
            return self.add(reservation_id, "reservation", payload)
        if set(requested) != set(self.config["limits"]["campaign"]):
            raise Blocked("resource dimensions mismatch")
        for value in requested.values():
            amount(value)
        origin = datetime.fromisoformat(self.store.find("CAMPAIGN")["receipt_time"])
        if (datetime.now(timezone.utc) - origin).total_seconds() > self.config["max_elapsed_seconds"]:
            raise Blocked("campaign elapsed-time cap reached")
        if any(r["status"] != "settled" for r in self.state["reservations"].values()):
            raise Blocked("unfinished reservation requires reconciliation before another invocation")
        if any(r["status"] == "settled" and any(r["actual"][k] > r["reserved"][k] for k in requested)
               for r in self.state["reservations"].values()):
            raise Blocked("observed invocation overrun requires a prospective allocation ruling")
        if any(d["status"] == "issued" for d in self.state["dispatches"].values()):
            raise Blocked("in-flight dispatch requires reconciliation, not another process")
        groups = [(self.config["limits"]["campaign"], lambda r: True)]
        for name, value in (("phases", phase), ("batches", batch_id)):
            cap = self.config["limits"].get(name, {}).get(value)
            if not cap:
                raise Blocked("phase/batch allocation absent")
            field = "phase" if name == "phases" else "batch_id"
            groups.append((cap, lambda r, f=field, v=value: r[f] == v))
        for caps, selects in groups:
            if set(caps) != set(requested):
                raise Blocked("limit dimensions mismatch")
            for key, val in requested.items():
                used = sum((r["actual"] if r["status"] == "settled" else r["reserved"])[key]
                           for r in self.state["reservations"].values() if selects(r))
                if used + val > amount(caps[key]):
                    raise Blocked("resource cap exceeded: " + key)
        return self.add(reservation_id, "reservation", payload)

    def settle(self, receipt_id, reservation_id, actual, evidence_ref):
        reservation = self.state["reservations"].get(reservation_id)
        if not reservation or evidence_ref not in self.state["event_refs"]:
            raise Blocked("settlement requires reservation and observed evidence")
        if actual is None:
            if reservation["status"] == "settled":
                raise Conflict("settled usage cannot be silently reset to unknown")
            return self.add(receipt_id, "usage_unknown", {"reservation_id": reservation_id, "evidence_ref": evidence_ref})
        if set(actual) != set(reservation["reserved"]):
            raise Blocked("actual usage dimensions mismatch")
        for value in actual.values():
            amount(value)
        if reservation["status"] == "settled" and not self.store.find(receipt_id):
            raise Conflict("reservation already reconciled")
        # Keep measured overruns, so subsequent admission sees the true spent amount.
        return self.add(receipt_id, "settlement", {"reservation_id": reservation_id, "actual": actual,
                        "evidence_ref": evidence_ref,
                        "overrun": any(actual[k] > reservation["reserved"][k] for k in actual)})

    def open_batch(self, order, actor):
        if actor != self.actor("researcher"):
            raise Blocked("only bound Researcher authors batch orders")
        required = {"batch_id", "snapshot_id", "directions", "max_candidates", "versions", "question"}
        if set(order) != required or not isinstance(order["directions"], list) or not order["directions"]:
            raise Blocked("invalid batch order")
        if self.store.find(order["batch_id"]):
            return self.add(order["batch_id"], "order", order, actor)
        for value in [order["batch_id"], order["snapshot_id"], *order["directions"]]:
            identifier(value)
        if len(set(order["directions"])) != len(order["directions"]):
            raise Blocked("duplicate direction")
        if type(order["max_candidates"]) is not int or not 0 < order["max_candidates"] <= self.config["max_batch_candidates"]:
            raise Blocked("invalid batch attempt cap")
        if self.state["orders"]:
            previous_batch = self.state["current_batch"]
            if self.state["batch_states"][previous_batch] != "closed":
                raise Blocked("previous batch awaits Researcher direction decisions")
            if set(order["directions"]) != set(self.state["active"]):
                raise Blocked("new order cannot silently drop or rename active directions")
        if order["snapshot_id"] in self.state["batch_snapshots"].values():
            raise Blocked("new batch requires a new direction snapshot identity")
        if order["versions"] != self.config["versions"] or order["batch_id"] not in self.config["limits"]["batches"]:
            raise Blocked("unbound version or batch allocation")
        return self.add(order["batch_id"], "order", order, actor)

    def check_decision(self, decision, actor, *, choices=None):
        validate(decision, strict_json((SCHEMAS / "decision.schema.json").read_bytes()), SCHEMAS)
        role = actor.get("role")
        if role not in ("researcher", "runner", "user") or actor != self.actor(role) or decision["actor"] != actor:
            raise Blocked("decision actor differs from dispatcher binding")
        if decision["campaign_id"] != self.config["campaign_id"]:
            raise Blocked("decision belongs to another campaign")
        if decision["batch_id"] != self.state["current_batch"] or decision["direction_id"] not in self.state["active"]:
            raise Blocked("decision outside active batch/direction")
        phase = self.state["batch_states"][decision["batch_id"]]
        if phase == "closed" or (role == "runner" and phase != "open"):
            raise Blocked("batch no longer admits this decision; a new scoped order is required")
        if choices and decision["choice"] not in choices:
            raise Blocked("decision action outside role's authority")
        if decision["outcome_ref"] is not None or decision["effort_ref"] is not None:
            raise Blocked("prospective decisions cannot contain retrospective outcomes")
        if not decision["evidence_refs"] or any(ref not in self.state["event_refs"] for ref in decision["evidence_refs"]):
            raise Blocked("missing or unknown decision evidence")
        if not decision["justification"].strip() or not decision["prediction"]["statement"].strip():
            raise Blocked("empty prospective rationale/prediction")
        if decision["prediction"]["kind"] == "quantitative":
            keys = ("metric_id", "quantity", "operator", "threshold", "unit", "resolution_rule_id")
            if any(decision["prediction"][key] is None for key in keys):
                raise Blocked("quantitative prediction missing its scoring rule")
            if decision["prediction"]["quantity"] == "delta_from_parent":
                raise Blocked("delta scoring disabled until an explicit comparator-reference contract exists")
        identifier(decision["decision_id"])

    def freeze_decision(self, decision, actor):
        if self.store.find(decision.get("decision_id")):
            return self.add(decision["decision_id"], "decision", decision, actor, event_time=decision["created_at"])
        self.check_decision(decision, actor, choices={"candidate_change"} if actor["role"] == "runner" else None)
        return self.add(decision["decision_id"], "decision", decision, actor,
                        event_time=decision["created_at"])

    def submit(self, candidate, source, actor):
        if actor != self.actor("runner"):
            raise Blocked("only bound Runner submits candidates")
        required = {"candidate_id", "decision_id", "batch_id", "direction_id", "parents", "build_recipe", "retained_information", "dependencies"}
        if set(candidate) != required:
            raise Blocked("invalid candidate submission")
        decision = self.state["decisions"].get(candidate["decision_id"])
        if not decision or decision["actor"] != actor or decision["choice"] != "candidate_change":
            raise Blocked("candidate lacks frozen Runner prediction")
        if (candidate["batch_id"], candidate["direction_id"]) != (decision["batch_id"], decision["direction_id"]):
            raise Blocked("candidate/decision scope mismatch")
        if any(p not in self.state["candidates"] for p in candidate["parents"]):
            raise Blocked("unknown parent")
        sha = digest(source)
        payload = dict(candidate, source_sha256=sha)
        if candidate["candidate_id"] in self.state["candidates"]:
            return self.add(candidate["candidate_id"], "submission", payload, actor)
        if (candidate["batch_id"] != self.state["current_batch"] or
            self.state["batch_states"][candidate["batch_id"]] != "open" or
            candidate["direction_id"] not in self.state["active"]):
            raise Blocked("candidate admission closed; require the next scoped batch")
        if any(c["decision_id"] == candidate["decision_id"] for c in self.state["candidates"].values()):
            raise Blocked("new candidate requires a new prospective decision")
        order = self.state["orders"][candidate["batch_id"]]
        if sum(c["batch_id"] == candidate["batch_id"] for c in self.state["candidates"].values()) >= order["max_candidates"]:
            raise Blocked("batch candidate cap reached")
        self.store.archive(source)
        return self.add(candidate["candidate_id"], "submission", payload, actor)

    def record_evaluation(self, result):
        candidate = self.state["candidates"].get(result["candidate_id"])
        if not candidate or result["candidate_sha256"] != candidate["source_sha256"]:
            raise Blocked("evaluation candidate identity mismatch")
        if result["versions"] != self.config["versions"]:
            raise Blocked("evaluation version mismatch")
        if type(result.get("n")) is not int or result["n"] <= 0 or len(result.get("predictions", [])) != result["n"]:
            raise Blocked("evaluation count or prediction slots invalid")
        if result.get("status") not in ("ok", "failed"):
            raise Blocked("unknown evaluation status")
        if result["status"] == "ok":
            if any(type(p) is not int or not 0 <= p <= 9 for p in result["predictions"]):
                raise Blocked("success has invalid predictions")
            if amount(result["accuracy"]) > 1 or result.get("errors"):
                raise Blocked("invalid accuracy or success with errors")
            amount(result["prediction_median_ms"])
        self.store.read_artifact(result["raw_artifact_sha256"])
        return self.add(result["evaluation_id"], "evaluation", result)

    def authorize_version_change(self, ruling_id, old_versions, new_versions, affected_evaluations, justification, actor):
        if actor != self.actor("researcher"):
            raise Blocked("only Researcher authorizes measurement semantics")
        if old_versions != self.config["versions"] or old_versions == new_versions:
            raise Blocked("version authorization does not name current and changed identities")
        if set(new_versions) != set(old_versions) or any(not isinstance(v, str) or not v for v in new_versions.values()):
            raise Blocked("new version identities incomplete")
        affected = {i for i, e in self.state["evaluations"].items() if e["versions"] == old_versions}
        if len(affected_evaluations) != len(set(affected_evaluations)) or set(affected_evaluations) != affected or not justification.strip():
            raise Blocked("version authorization must identify all affected evidence and its reason")
        return self.add(ruling_id, "version_authorization", {"ruling_id": ruling_id,
                        "old_versions": old_versions, "new_versions": new_versions,
                        "affected_evaluations": sorted(affected), "justification": justification}, actor)

    def change_versions(self, event_id, versions, ruling_ref, actor):
        ruling = self.state["version_authorizations"].get(ruling_ref)
        if actor != self.actor("researcher") or not ruling:
            raise Blocked("exact version-change authorization required; an unrelated ruling is insufficient")
        payload = {"versions": versions, "ruling_ref": ruling_ref,
                   "old_versions": ruling["old_versions"], "affected_evaluations": ruling["affected_evaluations"]}
        if self.store.find(event_id):
            return self.add(event_id, "versions_changed", payload, actor)
        if ruling["old_versions"] != self.config["versions"] or ruling["new_versions"] != versions:
            raise Blocked("requested versions differ from the specific authorization")
        affected = {i for i, e in self.state["evaluations"].items() if e["versions"] == self.config["versions"]}
        if set(ruling["affected_evaluations"]) != affected:
            raise Blocked("affected evidence changed after version authorization")
        if set(versions) != set(self.config["versions"]) or any(not isinstance(v, str) or not v for v in versions.values()):
            raise Blocked("version identities incomplete")
        if self.resources()["unresolved"]:
            raise Blocked("cannot change versions with unresolved work")
        return self.add(event_id, "versions_changed", payload, actor)

    def resolve_prediction(self, decision_id, evaluation_id, effort_ref):
        decision = self.state["decisions"][decision_id]
        result = self.state["evaluations"][evaluation_id]
        if self.state["candidates"][result["candidate_id"]]["decision_id"] != decision_id:
            raise Blocked("prediction outcome belongs to another candidate")
        effort = self.store.find(effort_ref)
        start = self.store.find("START-" + evaluation_id)
        raw = self.store.find(effort["payload"].get("evidence_ref")) if effort else None
        if (not effort or effort["kind"] != "settlement" or not start or start["kind"] != "evaluation_started" or
            effort["payload"]["reservation_id"] != start["payload"]["reservation_id"] or
            start["payload"]["candidate_id"] != result["candidate_id"] or
            not raw or raw["kind"] != "evaluation_raw" or raw["payload"]["evaluation_id"] != evaluation_id or
            raw["payload"]["artifact_sha256"] != result["raw_artifact_sha256"]):
            raise Blocked("prediction effort must be settled usage for this evaluation, with matching raw evidence")
        p = decision["prediction"]
        # Only this synthetic rule is bound here; free text is never self-scored.
        status, value, reason = "qualitative_unscored", None, "qualitative predictions are not numerically scored"
        if p["kind"] == "quantitative":
            status = "unresolved"
            reason = "unsupported rule, horizon, comparator or measurement context"
            if (p["resolution_rule_id"] == "SYNTHETIC-BYTES-1" and p["metric_id"] == "source_bytes" and
                p["quantity"] == "absolute" and p["unit"] == "bytes" and p["horizon"] == "next_candidate" and
                result.get("evaluator_scope") == "synthetic_only_not_scientifically_qualified"):
                value = result["source_bytes"]
                operations = {"gt": lambda: value > p["threshold"], "ge": lambda: value >= p["threshold"],
                              "lt": lambda: value < p["threshold"], "le": lambda: value <= p["threshold"], "eq": lambda: value == p["threshold"]}
                status = "supported" if operations[p["operator"]]() else "not_supported"
                reason = "bound synthetic source-byte rule at next-candidate horizon"
        return self.add("OUTCOME-" + decision_id, "outcome", {"decision_id": decision_id,
                        "evaluation_id": evaluation_id, "effort_ref": effort_ref,
                        "status": status, "observed_value": value, "resolution_reason": reason})

    def checkpoint(self, batch_id):
        if batch_id != self.state["current_batch"] or self.state["batch_states"][batch_id] == "closed":
            raise Blocked("checkpoint must identify the current nonclosed batch")
        if self.resources()["unresolved"]:
            raise Blocked("unresolved execution must be reconciled before checkpoint")
        measured = {e["candidate_id"] for e in self.state["evaluations"].values()}
        if any(c["batch_id"] == batch_id and cid not in measured for cid, c in self.state["candidates"].items()):
            raise Blocked("submitted candidate lacks evaluation/failure evidence")
        return self.add("CHECKPOINT-" + batch_id, "checkpoint", {"batch_id": batch_id,
                        "snapshot_id": self.state["batch_snapshots"][batch_id]})

    def anomaly(self, anomaly_id, evidence_ref, description):
        if evidence_ref not in self.state["event_refs"]:
            raise Blocked("anomaly evidence missing")
        return self.add(anomaly_id, "anomaly", {"anomaly_id": anomaly_id, "evidence_ref": evidence_ref,
                        "description": description, "resolution": None})

    def accept_batch_response(self, ruling_id, response, actor):
        if self.store.find(ruling_id):
            return self.add(ruling_id, "batch_response", {"ruling_id": ruling_id, "response": response}, actor)
        if actor != self.actor("researcher"):
            raise Blocked("only Researcher issues scientific direction decisions")
        validate(response, strict_json((SCHEMAS / "batch_decisions.schema.json").read_bytes()), SCHEMAS)
        if response["batch_id"] != self.state["current_batch"] or self.state["batch_states"].get(response["batch_id"]) == "closed":
            raise Blocked("response belongs to a closed or historical batch")
        if response["active_direction_snapshot_id"] != self.state["batch_snapshots"][response["batch_id"]]:
            raise Blocked("stale active-direction snapshot")
        ids = [d["direction_id"] for d in response["decisions"]]
        if len(ids) != len(set(ids)) or set(ids) != set(self.state["active"]):
            raise Blocked("exactly one decision per active direction required")
        decision_ids = [d["decision_id"] for d in response["decisions"]]
        if len(set(decision_ids)) != len(decision_ids) or any(i in self.state["decisions"] for i in decision_ids):
            raise Conflict("decision identity reused")
        for decision in response["decisions"]:
            self.check_decision(decision, actor, choices={"refine", "explore", "diagnose", "revise_guidance", "stop"})
            if decision["batch_id"] != response["batch_id"]:
                raise Blocked("response batch mismatch")
        new_ids = set()
        for direction in response["new_directions"]:
            identifier(direction["direction_id"])
            if direction["direction_id"] in self.state["direction_definitions"] or direction["direction_id"] in new_ids:
                raise Blocked("new direction must have a new unique identity")
            new_ids.add(direction["direction_id"])
            parent = next((d for d in response["decisions"] if d["decision_id"] == direction["parent_decision_id"]), None)
            allocation = self.config.get("direction_allocations", {}).get(direction["allocation_ref"])
            if not parent or parent["choice"] != "explore" or not allocation:
                raise Blocked("new direction requires an explore decision and bound allocation")
            if allocation["direction_id"] != direction["direction_id"] or allocation["batch_id"] not in self.config["limits"]["batches"]:
                raise Blocked("new direction allocation/scope mismatch")
            if not direction["definition"].strip() or not direction["question"].strip():
                raise Blocked("new direction definition/question absent")
        self.checkpoint(response["batch_id"])
        return self.add(ruling_id, "batch_response", {"ruling_id": ruling_id, "response": response}, actor)

    def _expected_packet(self, batch_id):
        if batch_id not in self.state["orders"]:
            raise Blocked("unknown batch")
        evaluations = list(self.state["evaluations"].values())
        comparable = [e for e in evaluations if e["versions"] == self.config["versions"] and e["status"] == "ok"]
        if self.config['mode'] in ('bounded_pilot','bounded_search'):
            accepted={e['payload']['source_sha256'] for e in self.store.events
                      if e['kind']=='admissibility_ruling' and e['payload']['verdict']=='accepted'
                      and e['actor']==self.actor('researcher')}
            comparable=[e for e in comparable if e['candidate_sha256'] in accepted]
        frontier = []
        for item in comparable:
            def dominates(other):
                if (other["data_sha256"], other["n"], other["timing_boundary"]) != (item["data_sha256"], item["n"], item["timing_boundary"]):
                    return False
                a = (other["accuracy"], -other["source_bytes"], -other["prediction_median_ms"])
                b = (item["accuracy"], -item["source_bytes"], -item["prediction_median_ms"])
                return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))
            if not any(dominates(e) for e in comparable if e is not item):
                frontier.append(item["evaluation_id"])
        by_direction = {}
        for direction in self.state["active"]:
            candidates = [c for c in self.state["candidates"].values() if c["direction_id"] == direction]
            by_direction[direction] = candidates[-self.config["memory_policy"]["recent_per_direction"]:]
        # Stable insertion order is append-log order; identifiers break no ties because
        # event IDs are unique. Keep mandatory negative evidence and ancestor closure.
        selected = {c["candidate_id"] for rows in by_direction.values() for c in rows}
        selected.update(e["candidate_id"] for e in evaluations if e["evaluation_id"] in frontier or e["status"] != "ok")
        pending = list(selected)
        while pending:
            for parent in self.state["candidates"][pending.pop()]["parents"]:
                if parent not in selected:
                    selected.add(parent)
                    pending.append(parent)
        candidates = {k: v for k, v in self.state["candidates"].items() if k in selected}
        measured = {e["evaluation_id"]: e for e in evaluations if e["candidate_id"] in selected}
        effort_links, effort_records = {}, {}
        for eid, evaluation in measured.items():
            start = self.store.find("START-" + eid)
            if not start or start["payload"].get("candidate_id") != evaluation["candidate_id"]:
                raise Blocked("packet evaluation lacks its scoped execution record")
            receipts = [e for e in self.store.events if e["kind"] in ("settlement", "usage_unknown") and
                        e["payload"]["reservation_id"] == start["payload"]["reservation_id"]]
            if not receipts:
                raise Blocked("packet evaluation lacks its scoped effort receipt")
            effort_links[eid] = [e["event_id"] for e in receipts]
            effort_records.update({e["event_id"]: e for e in receipts})
        packet = {"campaign_id": self.config["campaign_id"], "batch_id": batch_id,
                "contract": self.config, "active_direction_snapshot_id": self.state["snapshot_id"],
                "current_batch": self.state["current_batch"], "batch_states": self.state["batch_states"],
                "active_directions": self.state["active"], "recent_candidates": by_direction,
                "candidate_records": candidates, "evaluations": measured,
                "decisions": self.state["decisions"], "effort_records": effort_records, "effort_links": effort_links,
                "guidance": self.state["guidance"],
                "guidance_evidence": {ref: self.store.find(ref) for g in self.state["guidance"].values()
                                      for ref in g["evidence_refs"] + g["counterexample_refs"]},
                "selection": {"policy_version": "literal-joins-1", "order": "append_log_sequence",
                    "mandatory": ["recent_per_active_direction", "comparable_frontier", "all_failures", "ancestor_closure", "all_decisions", "all_guidance_and_counterexamples"],
                    "omitted_candidate_ids": [k for k in self.state["candidates"] if k not in candidates],
                    "omitted_evaluation_ids": [k for k in self.state["evaluations"] if k not in measured],
                    "overflow": "block_at_archive_size_cap_never_silently_truncate"},
                "direction_definitions": self.state["direction_definitions"],
                "lineage": {k: c["parents"] for k, c in self.state["candidates"].items()},
                "all_failures": [e for e in evaluations if e["status"] != "ok"],
                "invocation_failures": [d for d in self.state["dispatches"].values() if d["status"] != "completed"],
                "outstanding_anomalies": [a for a in self.state["anomalies"].values() if a["resolution"] is None],
                "rulings": self.state["rulings"], "outcomes": self.state["outcomes"],
                "provisional_synthetic_frontier": frontier,
                "frontier_comparison_groups": {e["evaluation_id"]: {"data_sha256": e["data_sha256"], "n": e["n"], "timing_boundary": e["timing_boundary"]} for e in comparable},
                "stale_evaluations": [e["evaluation_id"] for e in evaluations if e["versions"] != self.config["versions"]],
                "resources": self.resources(), "complete_archive_event_refs": self.state["event_refs"],
                "archive_head": self.state["last_event_hash"]}
        return packet

    def packet(self, batch_id):
        packet = self._expected_packet(batch_id)
        self.validate_packet(packet, batch_id=batch_id)
        return copy.deepcopy(packet)

    def record_guidance(self, guidance_id, direction_ids, statement, evidence_refs, counterexample_refs, actor):
        if actor != self.actor("researcher"):
            raise Blocked("only bound Researcher authors guidance")
        if (not statement.strip() or not direction_ids or not evidence_refs or
            any(d not in self.state["direction_definitions"] for d in direction_ids) or
            any(r not in self.state["event_refs"] for r in evidence_refs + counterexample_refs)):
            raise Blocked("guidance needs literal scoped evidence and known directions")
        return self.add(guidance_id, "guidance", {"guidance_id": guidance_id,
                        "direction_ids": direction_ids, "statement": statement,
                        "evidence_refs": evidence_refs, "counterexample_refs": counterexample_refs}, actor)

    def validate_packet(self, packet, *, batch_id=None):
        validate(packet, strict_json((SCHEMAS / "packet.schema.json").read_bytes()), SCHEMAS)
        # The requested batch comes from the caller's operation, never from the
        # untrusted packet. Historical inspection must name that batch explicitly.
        requested_batch = self.state["current_batch"] if batch_id is None else batch_id
        expected = self._expected_packet(requested_batch)
        # Shared pure construction derives frontier/selection and every join from
        # authoritative state. Canonical bytes preserve type distinctions such as
        # true versus 1 and detect nested alterations, not just missing keys.
        if canonical(packet) != canonical(expected):
            changed = [key for key in expected if canonical(packet.get(key)) != canonical(expected[key])]
            raise Blocked("packet differs from authoritative context/selection: " + ", ".join(changed))

    def freeze_packet(self, packet_id, batch_id):
        self.checkpoint(batch_id)
        packet = self.packet(batch_id)
        data = canonical(packet)
        markdown = ("# Synthetic evidence packet\n\nMachine-selected by the declared policy; no scientific summary is inferred.\n\n```json\n" +
                    data.decode() + "\n```\n").encode()
        sha = self.store.archive(markdown)
        self.store.write_outbox("controller", packet_id + ".md", markdown)
        self.add(packet_id, "packet", {"artifact_sha256": sha, "batch_id": batch_id})
        return packet

    def issue_release(self, release_id, manifest_hash, actor):
        if actor != self.actor("user"):
            raise Blocked("only bound user can authorize final release")
        if len(manifest_hash) != 64 or any(c not in "0123456789abcdef" for c in manifest_hash):
            raise Blocked("invalid finalist manifest hash")
        return self.add(release_id, "release", {"release_id": release_id, "manifest_sha256": manifest_hash,
                        "campaign_id": self.config["campaign_id"], "synthetic_only": True}, actor)

    def consume_release(self, release_id, manifest_hash):
        release = self.state["releases"].get(release_id)
        if not release or release_id in self.state["consumed"] or release["manifest_sha256"] != manifest_hash:
            raise Blocked("missing, reused or mismatched final release")
        # Write BEFORE any hypothetical opening: a crash cannot silently reset exposure.
        return self.add("CONSUMED-" + release_id, "release_consumed", {"release_id": release_id,
                        "manifest_sha256": manifest_hash, "exposure": "synthetic_only_no_sealed_access"})
