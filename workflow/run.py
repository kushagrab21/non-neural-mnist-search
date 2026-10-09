#!/usr/bin/env python3
"""Preparation commands only. No switch here enables live inference or sealed data."""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist_runtime.common import Blocked
from mnist_runtime.demo import run_demo
from mnist_runtime.engine import Engine
from mnist_runtime.store import Store

parser = argparse.ArgumentParser(description=__doc__)
sub = parser.add_subparsers(dest="command", required=True)
demo = sub.add_parser("demo", help="Execute a bounded synthetic workflow into a NEW directory")
demo.add_argument("--output", required=True, type=Path)
structured_demo = sub.add_parser("structured-demo", help="Local schema-bound three-role workflow; zero model calls")
structured_demo.add_argument("--output", required=True, type=Path)
pilot_check = sub.add_parser('pilot-check', help='Check a released pilot package; never dispatch')
pilot_check.add_argument('--release', required=True, type=Path)
pilot_run = sub.add_parser('pilot-run', help='Run one explicitly released pilot; refuses draft/missing approval')
pilot_run.add_argument('--release', required=True, type=Path)
pilot_run.add_argument('--output', required=True, type=Path)
replay = sub.add_parser("replay", help="Verify the event chain and rebuild the derived index without dispatch")
replay.add_argument("root", type=Path)
inspect = sub.add_parser("status", help="Show resources and pending work; never dispatch")
inspect.add_argument("root", type=Path)
qualification = sub.add_parser("qualify-synthetic", help="Run one planted control through the complete finite synthetic recipe")
qualification.add_argument("--control", choices=["stateless", "call_count", "previous_image", "delayed_512"], default="stateless")
qualification.add_argument("--output", required=True, type=Path)
queue_status = sub.add_parser("host-status", help="Verify and inspect a native-host queue; never dispatch or retry")
queue_status.add_argument("root", type=Path)
resource_status = sub.add_parser("resource-status", help="Inspect shared native/evaluator reservations; never dispatch")
resource_status.add_argument("root", type=Path)
for name,description in [("service-probe","Budgeted reduced synthetic metric probe"),("qualify-budgeted","Budgeted full synthetic planted-control qualification")]:
    command=sub.add_parser(name,help=description)
    command.add_argument("--budget-root",required=True,type=Path)
    command.add_argument("--output",required=True,type=Path)
    command.add_argument("--reservation",required=True)
    command.add_argument("--phase",required=True)
    command.add_argument("--batch",required=True)
    if name=="qualify-budgeted":command.add_argument("--control",choices=["stateless","call_count","previous_image","delayed_512"],default="stateless")
args = parser.parse_args()
try:
    if args.command in ('pilot-check','pilot-run'):
        from mnist_runtime.common import strict_json
        from mnist_runtime.pilot_release import require_release
        release=strict_json(args.release.read_bytes())
        sha=require_release(release)
        if args.command=='pilot-check':
            print(json.dumps({'release_sha256':sha,'valid':True,'model_calls':0},indent=2))
        else:
            from mnist_runtime.pilot import run
            print(json.dumps(run(release,args.output),indent=2))
    elif args.command == "structured-demo":
        from mnist_runtime.protocol_demo import run
        print(json.dumps(run(args.output), indent=2))
    elif args.command == "demo":
        print(json.dumps(run_demo(args.output), indent=2))
    elif args.command == "qualify-synthetic":
        import tempfile
        from mnist_runtime.canonical_fixtures import SOURCES
        from mnist_runtime.qualification import qualify
        with tempfile.TemporaryDirectory(prefix="mnist-control-") as temporary:
            path = Path(temporary) / "classifier.py"
            path.write_text(SOURCES[args.control])
            result = qualify(path, args.output)
        print(json.dumps({k:result[k] for k in ["scope","status","planned_calls","attempted_calls","completed_sessions",
              "order_history_evidence","primary_accuracy","campaign_qualified"]}, indent=2))
    elif args.command in ("resource-status","service-probe","qualify-budgeted"):
        from mnist_runtime.resources import Budget
        from mnist_runtime.admission import budgeted_service_probe,budgeted_qualify
        root=args.root if args.command=="resource-status" else args.budget_root
        if not (root/"coordination/events.jsonl").is_file():raise Blocked("no initialized resource authority")
        with Store(root) as store:
            budget=Budget(store)
            if args.command=="resource-status":
                print(json.dumps({"authority":budget.authority,"resources":budget.snapshot()},indent=2))
            else:
                host=store.find("RESOURCE-AUTHORITY")["actor"]
                options={"reservation_id":args.reservation,"phase":args.phase,"batch":args.batch,"host":host}
                if args.command=="service-probe":
                    result=budgeted_service_probe(store,args.output,**options)
                    summary={"scope":result["scope"],"requests":result["attempted_requests"],"objectives":{k:v["primary_value"] for k,v in result["objectives"].items()},"failures":result["failures"]}
                else:
                    import tempfile
                    from mnist_runtime.canonical_fixtures import SOURCES
                    with tempfile.TemporaryDirectory(prefix="mnist-control-") as temporary:
                        source=Path(temporary)/"classifier.py";source.write_text(SOURCES[args.control])
                        result=budgeted_qualify(store,source,args.output,**options)
                    summary={k:result[k] for k in ["scope","status","attempted_calls","completed_sessions","campaign_qualified"]}
                print(json.dumps({"result":summary,"resources":budget.snapshot()},indent=2))
    elif args.command == "host-status":
        from mnist_runtime.host_queue import HostQueue
        if not (args.root / "coordination/events.jsonl").is_file():
            raise Blocked("no initialized host queue")
        with Store(args.root) as store:
            queue = HostQueue(store)
            print(json.dumps({"authority":queue.authority,"jobs":queue.state(),"automatic_dispatch":False},indent=2))
    else:
        if not (args.root / "coordination/events.jsonl").is_file():
            raise Blocked("no initialized campaign event log at that path")
        with Store(args.root) as store:
            engine = Engine(store)
            print(json.dumps({"events": len(store.events), "archive_head": engine.state["last_event_hash"],
                              "active_directions": engine.state["active"], "resources": engine.resources(),
                              "dispatches": {k: v["status"] for k, v in engine.state["dispatches"].items()},
                              "mode": engine.config["mode"], "live_inference_enabled": False}, indent=2))
except (Blocked, OSError, ValueError) as exc:
    print(json.dumps({"status": "blocked", "reason": str(exc)}), file=sys.stderr)
    raise SystemExit(2)
