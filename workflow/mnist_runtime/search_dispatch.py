"""Adaptive-run transport with terminal failure accounting and bounded repairs."""
import math, re, secrets, subprocess
from .common import Blocked, canonical, digest, atomic_write, strict_json
from .admission import BudgetedHost
from .container_roles import command, cleanup, verify_container
from .docker_worker import DOCKER
from .pilot_release import require_store_release, require_store_readiness
from .role_protocol import accept, request_for_claude, unwrap_claude
from .processes import run_process
from .resources import units


class DispatchFailure(Blocked):
    def __init__(self, message, evidence):
        super().__init__(message);self.evidence=evidence


def dispatch(engine, resource_store, prepared, *, host, wall_seconds=240):
    release=require_store_release(resource_store);c=release['contract']
    if c['version']!='bounded-search/1':raise Blocked('adaptive dispatcher requires adaptive release')
    q=prepared['request'];actor=q['actor'];role=actor['role'];job=q['dispatch_id'];batch=q['batch_id']
    if c['actors'][role]!=actor:raise Blocked('bound actor changed')
    if job not in ('PLAN','PLAN-R1'):
        require_store_readiness(engine.store);require_store_readiness(resource_store)
    elif q['task']!='pilot_plan' or role!='researcher':raise Blocked('readiness must precede all data work')
    hostio=BudgetedHost(resource_store)
    used=[e for e in resource_store.events if e['kind']=='native_model_receipt' and e['payload']['role']==role]
    prior=used[-1]['payload']['session_list_usd'] if used else 0
    maximum=c['cost_policy']['session_list_usd_caps'][role]
    if prior>=maximum:raise Blocked('role accounting cap exhausted')
    client=request_for_claude(prepared,cli='claude',session_id=actor['session_id'],model=actor['model_id'],max_usd=maximum,resume=bool(used))
    out=resource_store.root/'invocations'/job
    if out.exists():raise Blocked('dispatch already exists; never replay')
    hostio.queue.prepare(job,role,canonical(prepared),{'destination':str(out/'RETURN.json'),'required_fields':['dispatch_id','request_sha256','body']},host)
    engine.reserve('RES-NATIVE-'+job,units(agent_tasks=1,wall_seconds=wall_seconds),phase='search',batch_id=batch)
    claim=hostio.claim(job,phase='search',batch=batch,wall_seconds=wall_seconds,host=host)
    engine.add(job,'dispatch',{'dispatch_id':job,'role':role,'actor_binding':actor,'transport':'docker_claude_json','reservation_id':'RES-NATIVE-'+job,'request_sha256':prepared['request_sha256']})
    envelope={k:claim[k] for k in ('dispatch_id','recipient','order_sha256','nonce')}
    out.mkdir(parents=True);name='mnist-search-'+secrets.token_hex(10)
    created=command(role,client['argv'],name=name,network=True,interactive=True);created[len(DOCKER)]='create'
    guard=hostio.budget.execution_guard('TASK-'+job)
    raw=None;provider=None;structured=None;problem=None;clean={'confirmed':False};receipt_sha=None
    failure={'kind':'unknown_completion','terminal_receipt':False,'usage_known':False,'cleanup_confirmed':False,
             'same_session_model':False,'within_caps':False,'provider_success':False,'safeguard_flag':False,'budget_exhausted':False}
    try:
        guard();subprocess.run(created,capture_output=True,timeout=10,check=True)
        atomic_write(out/'BOUNDARY.json',canonical(verify_container(name,role))+b'\n')
        hostio.queue.delivered(job,{'container':name,'boundary_verified':True,'model_completion_inferred':False},host)
        raw=run_process(DOCKER+['start','--attach','--interactive',name],client['stdin'].encode(),timeout=wall_seconds-20,output_limit=2*1024*1024,guard=guard)
        atomic_write(out/'TRANSPORT.json',canonical(raw)+b'\n')
        provider=strict_json(raw['stdout']);receipt_sha=resource_store.archive(raw['stdout'].encode())
        atomic_write(out/'PROVIDER.json',canonical(provider)+b'\n')
        terminal=(raw['status']=='exited' and provider.get('type')=='result' and isinstance(provider.get('subtype'),str))
        failure['terminal_receipt']=terminal
        failure['same_session_model']=(provider.get('session_id')==actor['session_id'] and set(provider.get('modelUsage',{}))=={actor['model_id']})
        total=provider.get('total_cost_usd')
        failure['usage_known']=(type(total) in (int,float) and math.isfinite(total) and total>=prior and isinstance(provider.get('usage'),dict))
        meta={k:provider.get(k) for k in ('stop_reason','terminal_reason','errors','safety_stops')}
        text=canonical(meta).decode().lower()
        failure['safeguard_flag']=(bool(provider.get('safety_stops')) or provider.get('stop_reason')=='refusal' or any(x in text for x in ('reasoning_extraction','safety_flag','safeguard','content_filter')))
        failure['budget_exhausted']=provider.get('subtype')=='error_max_budget_usd'
        failure['provider_success']=provider.get('subtype')=='success' and provider.get('is_error') is False and raw['returncode']==0
        if terminal and failure['usage_known']:
            resource_store.append('MODEL-'+job,'native_model_receipt',host,{'role':role,'session_id':actor['session_id'],'receipt_sha256':receipt_sha,'session_list_usd':total,'incremental_list_usd':total-prior,'actual_subscription_charge':None,'num_turns':provider.get('num_turns'),'provider_subtype':provider.get('subtype'),'model_ids':list(provider.get('modelUsage',{}))})
            failure['within_caps']=total<=maximum and sum(e['payload']['incremental_list_usd'] for e in resource_store.events if e['kind']=='native_model_receipt')<=c['cost_policy']['total_list_price_ceiling_usd']
        if not failure['provider_success']:
            errors=provider.get('errors',[])
            # Retry only explicit provider status fields / error records, never arbitrary prompt text.
            errtext=str(errors).lower();code=provider.get('http_status',provider.get('status_code'))
            if code is None:
                match=re.search(r'(?:http|status|api error)\s*[:=]?\s*(429|500|502|503|504)\b',errtext)
                code=int(match.group(1)) if match else None
            failure.update(kind='transient_provider_error' if code in c['retry_policy']['transient_codes'] else 'provider_failure',http_status=code)
            raise Blocked('provider did not return successful completion')
        if not all(failure[k] for k in ('terminal_receipt','same_session_model','usage_known','within_caps')) or failure['safeguard_flag']:
            raise Blocked('provider identity/accounting/safeguard validation failed')
        try:structured=unwrap_claude(prepared,raw['stdout'].encode())
        except (Blocked,ValueError,TypeError,KeyError) as exc:
            failure['kind']='malformed_structured_output';raise Blocked(str(exc)) from exc
    except BaseException as exc:problem=str(exc)
    finally:
        try:clean=cleanup(name)
        except BaseException as exc:clean={'confirmed':False,'error':str(exc)}
        failure['cleanup_confirmed']=clean['confirmed']
        if not clean['confirmed']:problem=problem or 'container cleanup unconfirmed'
        elapsed,clock=hostio.budget.elapsed_evidence('TASK-'+job)
        atomic_write(out/'COMPLETION.json',canonical({'cleanup':clean,'resource_clock':clock,'failure_classification':failure})+b'\n')
        known=failure['terminal_receipt'] and failure['usage_known'] and clean['confirmed'] and elapsed is not None
        if known:
            # This envelope acknowledges transport completion, never scientific acceptance.
            body=strict_json(structured) if structured is not None and problem is None else {'dispatch_id':job,'request_sha256':prepared['request_sha256'],'body':{'operational_failure':problem,'provider_receipt_sha256':receipt_sha}}
            hostio.queue.acknowledge(job,envelope,host)
            received=hostio.queue.receive(job,envelope,body,provider['usage'],host)
            actual=units(agent_tasks=1,wall_seconds=elapsed)
            hostio.budget.settle('TASK-'+job,actual,{'host_return_event':received['hash'],'resource_clock':clock,'cleanup_confirmed':True,'provider_receipt_sha256':receipt_sha},host,provider_usage=provider['usage'])
        else:
            hostio.uncertain(job,problem or 'accounting unavailable',host);actual=None
        status='completed' if problem is None else 'operational_failure'
        engine.add('RECEIPT-'+job,'invocation_receipt',{'dispatch_id':job,'status':status,'raw_artifact_sha256':receipt_sha,'error':problem})
        engine.settle('USAGE-'+job,'RES-NATIVE-'+job,actual,'RECEIPT-'+job)
        snap=hostio.budget.snapshot()
        if snap['unresolved'] or snap['overruns'] or snap['deadline_violations']:
            failure['within_caps']=False;problem=problem or 'resource completion unresolved or limit exceeded'
    if problem:
        failure['detail']=problem;failure['provider_receipt_sha256']=receipt_sha
        atomic_write(out/'FAILURE.json',canonical(failure)+b'\n')
        raise DispatchFailure(problem,failure)
    try:
        result=accept(engine,job,structured,transport_evidence={'kind':'verified_host_queue','receipt_sha256':engine.store.archive(raw['stdout'].encode()),'host_queue_return':resource_store.find('RETURN-'+job)['hash']})
    except Blocked as exc:
        failure.update(kind='semantic_failure',detail=str(exc));atomic_write(out/'FAILURE.json',canonical(failure)+b'\n');raise DispatchFailure(str(exc),failure) from exc
    atomic_write(out/'RETURN.json',structured+b'\n')
    return result
