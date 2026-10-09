"""Budgeted entry points; host still owns actual native API invocation."""
from __future__ import annotations
import time
from pathlib import Path
from .common import Blocked,canonical,digest,atomic_write,strict_json,identifier
from .resources import Budget,units,boot_identity
from .host_queue import HostQueue


class BudgetedHost:
    def __init__(self,store):
        self.store=store;self.queue=HostQueue(store);self.budget=Budget(store)

    def claim(self,dispatch_id,*,phase,batch,wall_seconds,host):
        job=self.queue.state()[dispatch_id]
        if job['state']!='prepared':raise Blocked('native job is already issued; no repeat')
        if any(v['state'] in ('issued','delivered','acknowledged','uncertain') for v in self.queue.state().values()):
            raise Blocked('native work unresolved')
        binding={k:job[k] for k in ('dispatch_id','recipient','order_sha256','nonce')}
        rid='TASK-'+dispatch_id
        self.budget.reserve(rid,'native_task',units(agent_tasks=1,wall_seconds=wall_seconds),phase=phase,batch=batch,binding=binding,host=host)
        self.budget.begin(rid,host)
        # If the process dies between begin and claim, retain the hold. Do not
        # infer non-execution or automatically retry on recovery.
        return self.queue.claim(dispatch_id,host)

    def complete(self,dispatch_id,envelope,body,provider_usage,host):
        job=self.queue.state()[dispatch_id]
        if job['state'] in ('issued','delivered'):self.queue.acknowledge(dispatch_id,envelope,host)
        event=self.queue.receive(dispatch_id,envelope,body,provider_usage,host)
        rid='TASK-'+dispatch_id;row=self.budget.state()[rid]
        if row['state']=='settled':return self.store.find('SETTLED-'+rid)
        elapsed,clock=self.budget.elapsed_evidence(rid)
        actual=units(agent_tasks=1,wall_seconds=elapsed) if elapsed is not None else None
        return self.budget.settle(rid,actual,{'host_return_event':event['hash'],'wall_boundary':'host claim admission to verified return intake; includes transport, host wait and system sleep','resource_clock':clock},host,provider_usage=provider_usage)

    def uncertain(self,dispatch_id,reason,host):
        self.queue.uncertain(dispatch_id,reason,host)
        return self.budget.settle('TASK-'+dispatch_id,None,{'reason':reason,'remote_completion':'unknown'},host)


def budgeted_qualify(store,source,output,*,reservation_id,phase,batch,host,evaluation_seconds=120):
    """Fixed synthetic recipe; complete reservation precedes child creation."""
    from .qualification import qualify
    budget=Budget(store);output=Path(output)
    if output.exists():raise Blocked('evaluation output already exists')
    source_bytes=Path(source).read_bytes()
    binding={'source_sha256':digest(source_bytes),'output':str(output.resolve()),'recipe':'synthetic-service-qualification/1'}
    # Setup/receipt writing has a separate ten-second headroom. This is charged
    # host wall time, never relabelled as predict duration or provider usage.
    budget.reserve(reservation_id,'synthetic_evaluation',units(evaluation_requests=6508,worker_sessions=22,wall_seconds=evaluation_seconds+10),phase=phase,batch=batch,binding=binding,host=host)
    budget.begin(reservation_id,host);started=time.monotonic()
    # Freeze the bound bytes in the archive and supply that immutable path, so a
    # changed caller source cannot silently change a reserved execution identity.
    source_sha=store.archive(source_bytes)
    try:
        result=qualify(store.root/'archive'/source_sha,output,evaluation_seconds=evaluation_seconds,resource_guard=budget.execution_guard(reservation_id))
        elapsed,clock=budget.elapsed_evidence(reservation_id)
        cleanup=[]
        for session in result['sessions']:
            name=identifier(session['name']);receipt=output/name/'RECEIPT.json'
            try:
                raw=receipt.read_bytes();observed=strict_json(raw)
                confirmed=(observed.get('cleanup') in ('worker_reaped','not_spawned') and observed.get('source_sha256')==source_sha)
                cleanup.append({'session':name,'confirmed':confirmed,'receipt_sha256':store.archive(raw),'cleanup':observed.get('cleanup'),'source_matches':observed.get('source_sha256')==source_sha})
            except (OSError,ValueError) as exc:
                cleanup.append({'session':name,'confirmed':False,'error':str(exc)})
        cleanup_confirmed=all(item['confirmed'] for item in cleanup)
        actual=units(evaluation_requests=result['attempted_calls'],worker_sessions=len(result['sessions']),wall_seconds=elapsed) if cleanup_confirmed and elapsed is not None else None
        evidence={'result_sha256':store.archive((output/'RESULT.json').read_bytes()),'recipe_sha256':store.archive((output/'RECIPE.json').read_bytes()),'source_sha256':source_sha,'status':result['status'],'resource_clock':clock,'cleanup_confirmed':cleanup_confirmed,'cleanup_evidence':cleanup,'wall_boundary':'qualify entry to returned result, including setup and cleanup'}
        budget.settle(reservation_id,actual,evidence,host)
        atomic_write(output/'RESOURCE_RECEIPT.json',canonical({'reservation_id':reservation_id,'charged':actual,'allocation_held':actual is None,'cleanup_confirmed':cleanup_confirmed,'budget_head':store.events[-1]['hash'],'provider_usage':None})+b'\n')
        return result
    except BaseException as exc:
        row=budget.state()[reservation_id]
        if row['state'] not in ('settled','unknown'):
            budget.settle(reservation_id,None,{'error_type':type(exc).__name__,'detail':str(exc),'reason':'complete evaluator resource evidence unavailable'},host)
        raise


def budgeted_service_probe(store,output,*,reservation_id,phase,batch,host):
    from .service_probe import run_probe
    budget=Budget(store);output=Path(output)
    if output.exists():raise Blocked('probe output already exists')
    binding={'protocol':'reduced-service-probe/2','output':str(output.resolve()),'probe_sha256':digest(Path(__file__).with_name('service_probe.py').read_bytes())}
    budget.reserve(reservation_id,'synthetic_evaluation',units(evaluation_requests=348,worker_sessions=6,wall_seconds=40),phase=phase,batch=batch,binding=binding,host=host)
    budget.begin(reservation_id,host);started=time.monotonic()
    try:
        result=run_probe(output,evaluation_deadline=started+30,resource_guard=budget.execution_guard(reservation_id))
        elapsed,clock=budget.elapsed_evidence(reservation_id)
        actual=units(evaluation_requests=result['attempted_requests'],worker_sessions=result['attempted_sessions'],wall_seconds=elapsed) if result['cleanup_confirmed'] and elapsed is not None else None
        budget.settle(reservation_id,actual,{'result_sha256':store.archive((output/'RESULT.json').read_bytes()),'recipe_sha256':store.archive((output/'RECIPE.json').read_bytes()),'scope':'reduced synthetic metric probe','resource_clock':clock,'cleanup_confirmed':result['cleanup_confirmed'],'uncertainty_reason':None if result['cleanup_confirmed'] else 'worker termination not confirmed'},host)
        atomic_write(output/'RESOURCE_RECEIPT.json',canonical({'reservation_id':reservation_id,'charged':actual,'allocation_held':actual is None,'budget_head':store.events[-1]['hash']})+b'\n')
        return result
    except BaseException as exc:
        row=budget.state()[reservation_id]
        if row['state'] not in ('settled','unknown'):budget.settle(reservation_id,None,{'error_type':type(exc).__name__,'detail':str(exc)},host)
        raise
