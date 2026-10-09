"""Reduced synthetic metric probe, explicitly not the full 6508-call recipe."""
from pathlib import Path
from dataclasses import asdict
import os,platform,sys,time,subprocess,json
import numpy as np
from .canonical import Worker,Limits,TIMING_BOUNDARY
from .common import atomic_write,canonical,digest,now
from .service_metric import METRIC_ID,aggregate,schedule
from .measurement_guard import MeasurementGuard,POLICY


def context():
    power=subprocess.run(['/usr/bin/pmset','-g','custom'],capture_output=True,text=True,timeout=2)
    return {'platform':platform.platform(),'machine':platform.machine(),'cpu_count':os.cpu_count(),'load_average':list(os.getloadavg()),
            'python':sys.version,'numpy':np.__version__,'power_policy_observed':power.stdout if power.returncode==0 else None,
            'power_policy_unknown_reason':None if power.returncode==0 else power.stderr,'concurrent_external_work':'not controlled; host load observed'}


def run_probe(output,*,evaluation_deadline=None,resource_guard=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    sources={
      'honest':'def predict(image): return int(image[0,0])%10\n',
      'forged_zero_delayed':'import time\ntime.perf_counter_ns=lambda: 0\ndef predict(image):\n time.sleep(.002)\n return int(image[0,0])%10\n'}
    data=np.random.Generator(np.random.PCG64(20261007)).integers(0,256,(58,28,28),dtype=np.uint8)
    order=schedule(list(sources),20261007);limits=Limits(total_seconds=10,per_call_seconds=1,max_calls=58)
    before=context();started_at=now();started=time.monotonic();deadline=min(started+30,evaluation_deadline) if evaluation_deadline else started+30
    identity={'protocol':'reduced-service-probe/2','measurement_guard_policy':POLICY,'measurement_guard_sha256':digest(Path(__file__).with_name('measurement_guard.py').read_bytes()),'resource_guard':'sleep-inclusive-checkpoints/1' if resource_guard else None,'metric_id':METRIC_ID,'parent_sha256':digest(Path(__file__).with_name('canonical.py').read_bytes()),
        'child_sha256':digest(Path(__file__).with_name('canonical_child.py').read_bytes()),'aggregator_sha256':digest(Path(__file__).with_name('service_metric.py').read_bytes()),'profile_generator_sha256':digest(Path(__file__).with_name('evaluator.py').read_bytes()),
        'warmup_hashes':[digest(x.tobytes()) for x in data[:50]],'timed_hashes':[digest(x.tobytes()) for x in data[50:]],
        'limits':asdict(limits),'numpy':np.__version__,'python':sys.version,'machine':platform.platform(),'schedule':order,
        'thread_policy':{'OMP_NUM_THREADS':1,'OPENBLAS_NUM_THREADS':1,'VECLIB_MAXIMUM_THREADS':1,'MKL_NUM_THREADS':1,'NUMEXPR_NUM_THREADS':1},
        'member_sources':{name:digest(source.encode()) for name,source in sources.items()},'power_policy':before['power_policy_observed']}
    recipe={'scope':'synthetic_proposed_metric','metric_id_at_start':METRIC_ID,'comparison_identity':identity,'timed_n':8,'warmups':50,'planned_requests':348,'planned_sessions':6}
    atomic_write(output/'RECIPE.json',canonical(recipe)+b'\n')
    records={name:dict(recipe,blocks=[]) for name in sources};attempted=0;sessions=0;failures=[];execution=[]
    for block,members in enumerate(order):
        for member in members:
            source=output/(member+'.py')
            if not source.exists():source.write_text(sources[member])
            row={'block':block,'parent_service_ms':[],'child_diagnostic_ms':[],'complete':False,'monitor_complete':False,'cleanup_confirmed':False}
            records[member]['blocks'].append(row);sessions+=1;run={'block':block,'member':member,'started_at':now(),'load_before':list(os.getloadavg())};execution.append(run)
            worker=None;guard=None
            try:
                if resource_guard:resource_guard()
                guard=MeasurementGuard()
                with Worker(source,output/(str(block)+'-'+member),limits=limits,evaluation_deadline=deadline) as worker:
                    for image in data[:50]:
                        if resource_guard:resource_guard()
                        guard.observe();attempted+=1;worker.predict(image);guard.observe()
                        if resource_guard:resource_guard()
                    for image in data[50:]:
                        if resource_guard:resource_guard()
                        guard.observe()
                        attempted+=1;receipt=worker.predict(image);row['parent_service_ms'].append(receipt['service_ms']);row['child_diagnostic_ms'].append(receipt['response']['child_predict_ms'])
                        guard.observe()
                        if resource_guard:resource_guard()
                    row['monitor_complete']=True
                    run['startup_ms']=worker.metadata['parent_startup_ms'];run['profile_sha256']=worker.metadata['profile_sha256'];run['normalized_profile_sha256']=digest(worker.profile.replace(str(worker.source),'{candidate_source}').encode())
                row['cleanup_confirmed']=True
                guard.observe()
                if resource_guard:resource_guard()
                row['complete']=True
            except Exception as exc:
                row['complete']=False
                cleanup={'termination_observed':False,'evidence':None}
                if worker is not None and getattr(worker,'proc',None) is not None:
                    try:
                        code=worker.proc.poll()
                        cleanup={'termination_observed':code is not None,'evidence':{'parent_poll_exit_code':code}}
                    except Exception as probe_error:
                        cleanup['probe_error']=str(probe_error)
                receipt_path=output/(str(block)+'-'+member)/'RECEIPT.json'
                if not cleanup['termination_observed'] and receipt_path.is_file():
                    try:
                        retained=json.loads(receipt_path.read_text())
                        if retained.get('cleanup') in ('worker_reaped','not_spawned'):
                            cleanup={'termination_observed':True,'evidence':{'receipt_sha256':digest(receipt_path.read_bytes()),'cleanup':retained['cleanup']}}
                    except (OSError,ValueError):pass
                row['cleanup_confirmed']=row['cleanup_confirmed'] or cleanup['termination_observed'];row['cleanup_evidence']=cleanup
                row['failure']=str(exc);failures.append({'block':block,'member':member,'detail':str(exc)})
            row['measurement_guard']=guard.evidence() if guard is not None else {'complete':False,'failure':'guard unavailable','policy':POLICY}
            if not row['measurement_guard']['complete']:row['monitor_complete']=False
            run['ended_at']=now();run['load_after']=list(os.getloadavg())
            if failures:break
        if failures:break
    normalized_profiles={r.get('normalized_profile_sha256') for r in execution if r.get('normalized_profile_sha256')}
    if len(normalized_profiles)>1:
        failures.append({'detail':'sandbox policy differs beyond opaque candidate source path'})
        for record in records.values():
            for block in record['blocks']:block['monitor_complete']=False
    result={'scope':'reduced_synthetic_metric_probe_only','full_6508_recipe':False,'campaign_qualified':False,'metric_id_at_start':METRIC_ID,
        'started_at':started_at,'ended_at':now(),'recipe_sha256':digest(recipe),'comparison_identity':identity,
        'records':records,'objectives':{name:aggregate(record) for name,record in records.items()},'execution_order':execution,
        'cleanup_confirmed':all(block['cleanup_confirmed'] for record in records.values() for block in record['blocks']),'attempted_requests':attempted,'attempted_sessions':sessions,'wall_seconds':time.monotonic()-started,'failures':failures,
        'machine_before':before,'machine_after':context(),'service_boundary':TIMING_BOUNDARY,'diagnostic_forgery_planted':True,'normalized_profile_consistency':len(normalized_profiles)==1,'normalized_profile_sha256':next(iter(normalized_profiles)) if len(normalized_profiles)==1 else None}
    atomic_write(output/'RESULT.json',canonical(result)+b'\n');return result
