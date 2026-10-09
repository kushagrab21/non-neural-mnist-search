"""Budgeted Docker Claude transport into the structured workflow engine.

No automatic retry; incomplete remote accounting retains its reservation.
This module has no CLI entry point and cannot run without a released pilot.
"""
import json
import secrets
import subprocess
from pathlib import Path
from .admission import BudgetedHost
from .common import Blocked, canonical, digest, atomic_write
from .container_roles import command, cleanup, verify_container
from .docker_worker import DOCKER
from .pilot_release import require_store_release, require_store_readiness
from .processes import run_process
from .role_protocol import accept, request_for_claude, unwrap_claude
from .resources import units


def dispatch(engine, resource_store, prepared, *, host, wall_seconds=240):
    release = require_store_release(resource_store)
    contract = release['contract']
    request = prepared['request']; actor = request['actor']; role = actor['role']
    if contract['actors'][role] != actor:
        raise Blocked('dispatch actor differs from released binding')
    job = request['dispatch_id']; transport = BudgetedHost(resource_store)
    if contract['version'] == 'bounded-pilot/2':
        if job == 'PLAN':
            if request['task'] != 'pilot_plan' or role != 'researcher':
                raise Blocked('only the Researcher readiness/plan task may start the pilot')
        else:
            require_store_readiness(engine.store)
            require_store_readiness(resource_store)
    used = [e for e in resource_store.events if e['kind'] == 'native_model_receipt' and e['payload']['role'] == role]
    maximum = contract['cost_policy']['session_list_usd_caps'][role]
    client = request_for_claude(prepared, cli='claude', session_id=actor['session_id'],
                               model=actor['model_id'], max_usd=maximum, resume=bool(used))
    out = resource_store.root/'invocations'/job
    if out.exists(): raise Blocked('dispatch output exists; reconcile instead of retrying')
    transport.queue.prepare(job, role, canonical(prepared),
                            {'destination': str(out/'RETURN.json'), 'required_fields': ['dispatch_id', 'request_sha256', 'body']}, host)
    engine.reserve('RES-NATIVE-'+job,units(agent_tasks=1,wall_seconds=wall_seconds),phase='pilot',batch_id='B1')
    claim = transport.claim(job, phase='pilot', batch='B1', wall_seconds=wall_seconds, host=host)
    engine.add(job,'dispatch',{'dispatch_id':job,'role':role,'actor_binding':actor,'transport':'docker_claude_json',
                              'reservation_id':'RES-NATIVE-'+job,'request_sha256':prepared['request_sha256']})
    envelope = {k: claim[k] for k in ('dispatch_id', 'recipient', 'order_sha256', 'nonce')}
    out.mkdir(parents=True)
    name = 'mnist-native-'+secrets.token_hex(12)
    created = command(role, client['argv'], name=name, network=True, interactive=True)
    created[len(DOCKER)] = 'create'
    guard = transport.budget.execution_guard('TASK-'+job)
    try:
        guard()
        subprocess.run(created, capture_output=True, timeout=10, check=True)
        boundary = verify_container(name, role)
        atomic_write(out/'BOUNDARY.json', canonical(boundary)+b'\n')
        transport.queue.delivered(job, {'container': name, 'boundary_verified': True,
                                       'model_completion_inferred': False}, host)
        raw = run_process(DOCKER+['start', '--attach', '--interactive', name], client['stdin'].encode(),
                          timeout=wall_seconds-20, output_limit=2*1024*1024, guard=guard)
        atomic_write(out/'TRANSPORT.json', canonical(raw)+b'\n')
        if raw['status'] != 'exited' or raw['returncode'] != 0:
            raise Blocked('Claude transport incomplete; no retry')
        structured = unwrap_claude(prepared, raw['stdout'].encode())
        provider = json.loads(raw['stdout'])
        usage = provider.get('usage')
        clean = cleanup(name)
        if not clean['confirmed']: raise Blocked('role container cleanup unconfirmed')
        guard()
        receipt_sha = resource_store.archive(raw['stdout'].encode())
        prior = used[-1]['payload']['session_list_usd'] if used else 0
        total = provider.get('total_cost_usd')
        if type(total) not in (int, float) or total < prior:
            raise Blocked('cumulative provider cost cannot be reconciled')
        resource_store.append('MODEL-'+job, 'native_model_receipt', host,
                              {'role': role, 'session_id': actor['session_id'], 'receipt_sha256': receipt_sha,
                               'session_list_usd': total, 'incremental_list_usd': total-prior,
                               'actual_subscription_charge': None, 'num_turns': provider.get('num_turns')})
        transport.complete(job, envelope, json.loads(structured), usage, host)
        engine_receipt_sha = engine.store.archive(raw['stdout'].encode())
        engine.add('RECEIPT-'+job,'invocation_receipt',{'dispatch_id':job,'status':'completed',
                   'raw_artifact_sha256':engine_receipt_sha,'model_usage':provider['modelUsage'],'usage':usage})
        engine.settle('USAGE-'+job,'RES-NATIVE-'+job,transport.budget.state()['TASK-'+job]['actual'],'RECEIPT-'+job)
        result = accept(engine, job, structured,
                        transport_evidence={'kind': 'verified_host_queue', 'receipt_sha256': engine_receipt_sha,
                                            'host_queue_return': transport.queue.store.find('RETURN-'+job)['hash']})
        atomic_write(out/'RETURN.json', structured+b'\n')
        if total > maximum: raise Blocked('reported session spend exceeded its cap; stop pilot')
        return result
    except BaseException as exc:
        clean = cleanup(name)
        atomic_write(out/'FAILURE.json', canonical({'error': str(exc), 'cleanup': clean})+b'\n')
        if transport.queue.state()[job]['state'] in ('issued', 'delivered', 'acknowledged'):
            transport.uncertain(job, str(exc), host)
        if engine.state['dispatches'][job]['status']=='issued':
            engine.add('RECEIPT-'+job,'invocation_receipt',{'dispatch_id':job,'status':'operational_failure','error':str(exc)})
            engine.settle('USAGE-'+job,'RES-NATIVE-'+job,None,'RECEIPT-'+job)
        raise
