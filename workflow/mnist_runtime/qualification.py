"""Researcher-defined synthetic-service-qualification/1; no real-data loader."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
import platform

import numpy as np

from .canonical import Worker, Limits, summary, TIMING_BOUNDARY, check_deadlines
from .common import Blocked, atomic_write, canonical, digest, now


def corpus(seed=20261007):
    rng=np.random.Generator(np.random.PCG64(seed))
    values=rng.integers(0,256,(1202,28,28),dtype=np.uint8)
    if len({digest(x.tobytes()) for x in values})!=len(values):
        raise Blocked("synthetic corpus collision")
    return {"images":values[:128],"warmups":values[128:178],"fillers":values[178:],"anchors":[0,1,2,3],
            "labels":(values[:128,0,0]%10).astype(np.int64),"seed":seed,"generator":"NumPy PCG64"}


def reason_code(error):
    if hasattr(error, "code"): return error.code
    message=str(error)
    if message.startswith('candidate failed:'): return 'CANDIDATE_ERROR'
    for fragment,reason in [("RSS cap","SAMPLED_RSS_LIMIT"),("RSS observation","MEMORY_MONITOR_FAILURE"),
        ("write deadline","REQUEST_WRITE_TIMEOUT"),("deadline","TIMEOUT"),("response nonce","NONCE_MISMATCH"),
        ("extra response","EXTRA_PROTOCOL_BYTES"),("malformed","MALFORMED_PROTOCOL"),
        ("output cap","OUTPUT_LIMIT"),("int in 0..9","INVALID_OUTPUT"),("invalid digit","INVALID_OUTPUT"),
        ("candidate failed","CANDIDATE_ERROR"),("exited","CHILD_EXIT")]:
        if fragment in message: return reason
    return "EXECUTION_OR_MONITOR_FAILURE"


def qualify(source, output, *, data=None, limits=Limits(), evaluation_seconds=120, aggregate_calls=8192, resource_guard=None):
    """Execute exactly the finite synthetic definition accepted by RESEARCH-003.

    Custom arrays must match this recipe's counts. It remains synthetic-only in
    reporting; this entry point is not an authority for a real dataset or CNN run.
    """
    started_at=now(); started=time.monotonic()
    data=corpus() if data is None else data
    required={"images":(128,28,28),"warmups":(50,28,28),"fillers":(1024,28,28)}
    for key,shape in required.items():
        value=data.get(key)
        if not isinstance(value,np.ndarray) or value.dtype!=np.uint8 or value.shape!=shape:
            raise Blocked("wrong corpus shape/dtype for "+key)
    if data.get('anchors')!=[0,1,2,3] or data['labels'].shape!=(128,) or data['labels'].dtype.kind not in 'iu' or np.any(data['labels']<0) or np.any(data['labels']>9):
        raise Blocked("anchor/label contract mismatch")
    identities={k:[digest(im.tobytes()) for im in data[k]] for k in required}
    if len(set(sum(identities.values(),[])))!=1202:
        raise Blocked("primary/warmup/filler arrays must be distinct by image identity")
    limits.validate()
    if type(evaluation_seconds) not in (int,float) or not 0<evaluation_seconds<=600 or type(aggregate_calls) is not int or aggregate_calls<1:
        raise Blocked("invalid finite evaluation allocation")
    deadline=started+evaluation_seconds
    order=list(range(128)); shuffled=np.random.Generator(np.random.PCG64(data['seed'])).permutation(128).tolist()
    jobs=[]
    for name,seq in [('forward',order),('reverse',order[::-1]),('shuffle',shuffled)]:
        jobs.append({'name':name,'kind':'order','sequence':[('images',i) for i in seq]})
    for anchor in data['anchors']:
        jobs.append({'name':'reference-'+str(anchor),'kind':'reference','anchor':anchor,'sequence':[('images',anchor)]})
        for length in (1,128,1024):
            jobs.append({'name':f'history-{anchor}-{length}','kind':'history','anchor':anchor,'length':length,
                         'sequence':[('images',anchor)]+[('fillers',i) for i in range(length)]+[('images',anchor)]})
    for block in range(3): jobs.append({'name':'timing-'+str(block),'kind':'timing','sequence':[('images',i) for i in order]})
    planned=sum(50+len(job['sequence']) for job in jobs)
    if planned>aggregate_calls or max(50+len(job['sequence']) for job in jobs)>limits.max_calls:
        raise Blocked("qualification recipe exceeds aggregate/session call allocation")
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    frozen=output/'classifier.py'; frozen.write_bytes(Path(source).read_bytes()); frozen.chmod(0o444)
    recipe={'version':'synthetic-service-qualification/1','seed':data['seed'],'generator':data['generator'],
            'numpy_version':np.__version__,'identities':identities,'anchors':data['anchors'],'jobs':jobs,
            'warmup_sequence':[('warmups',i) for i in range(50)],'limits':asdict(limits),
            'resource_guard':'sleep-inclusive-checkpoints/1' if resource_guard else None,'aggregate_call_cap':aggregate_calls,'planned_calls':planned,'evaluation_seconds':evaluation_seconds}
    atomic_write(output/'RECIPE.json',canonical(recipe)+b'\n')
    sessions=[]; failed=None; attempted=0
    for job in jobs:
        row={'name':job['name'],'kind':job['kind'],'expected':len(job['sequence']),'predictions':[None]*len(job['sequence']),
             'warmups_expected':50,'warmups_valid':0,'timed_receipts':[],'complete':False}
        sessions.append(row)
        try:
            if resource_guard:resource_guard()
            check_deadlines({'WHOLE_EVALUATION_TIMEOUT':deadline}, 'between_sessions')
            with Worker(frozen,output/job['name'],limits=limits,evaluation_deadline=deadline) as worker:
                row['process_start_to_ready_ms']=worker.metadata['parent_startup_ms']
                warm_start=time.monotonic()
                for i,image in enumerate(data['warmups']):
                    if resource_guard:resource_guard()
                    attempted+=1; receipt=worker.predict(image);row['warmups_valid']+=1
                    if resource_guard:resource_guard()
                    if i==0: row['first_request_service_ms']=receipt['service_ms']
                row['warmup_total_wall_ms']=(time.monotonic()-warm_start)*1000
                for i,(part,index) in enumerate(job['sequence']):
                    if resource_guard:resource_guard()
                    attempted+=1; receipt=worker.predict(data[part][index]);row['predictions'][i]=receipt['response']['digit']
                    if job['kind']=='timing': row['timed_receipts'].append(receipt)
                    if resource_guard:resource_guard()
            if resource_guard:resource_guard()
            row['complete']=True
        except Exception as exc:
            failed={'session':job['name'],'reason_code':reason_code(exc),'detail':str(exc),'deadline_evidence':getattr(exc,'evidence',None)}
            row['failure']=failed
            break
    by_name={r['name']:r for r in sessions}; dependence=[]; forward=[None]*128
    if 'forward' in by_name: forward=by_name['forward']['predictions']
    # Compare only observed valid digits, including prefixes of failed sessions.
    for job in jobs:
        row=by_name.get(job['name'])
        if not row: continue
        context={'session_complete':row['complete'], 'valid_predictions':sum(p is not None for p in row['predictions']),
                 'expected_predictions':row['expected']}
        if job['kind']=='order' and job['name']!='forward':
            for position,((part,index),pred) in enumerate(zip(job['sequence'],row['predictions'])):
                expected=forward[index]
                if pred is not None and expected is not None and pred!=expected:
                    dependence.append(dict(context,control=job['name'],image_sha256=identities[part][index],
                                           reference=expected,observed=pred,sequence_position=position,reference_position=index))
        if job['kind']=='history':
            reference=by_name.get('reference-'+str(job['anchor']))
            expected=reference['predictions'][0] if reference else None
            if expected is not None:
                for position,label in [(0,'first'),(len(row['predictions'])-1,'repeated')]:
                    observed=row['predictions'][position]
                    if observed is not None and observed!=expected:
                        dependence.append(dict(context,control=job['name'],image_sha256=identities['images'][job['anchor']],
                                               reference=expected,observed=observed,anchor_occurrence=label,
                                               sequence_position=position,reference_position=0))
    timings=[]
    for row in sessions:
        if row['kind']=='timing' and row['complete']:
            timings.append({'block':row['name'],'service_request_to_validated_digit_ms':summary([r['service_ms'] for r in row['timed_receipts']]),
                'predict_duration_ms_diagnostic':summary([r['response']['child_predict_ms'] for r in row['timed_receipts']])})
    result={'version':recipe['version'],'scope':'synthetic_only','started_at':started_at,'ended_at':now(),'status':'INCOMPLETE' if failed else 'COMPLETE_NO_DEFINED_CHECK_FAILURE',
        'execution':'incomplete' if failed else 'complete','required_check_coverage':'incomplete' if failed else 'complete_for_defined_synthetic_recipe',
        'observed_violations':dependence,'scientific_admissibility':'unadjudicated','campaign_qualified':False,
        'order_history_evidence':'dependence_observed' if dependence else ('incomplete' if failed else 'no_dependence_observed_within_defined_sequences'),
        'source_sha256':digest(frozen.read_bytes()),'source_bytes':frozen.stat().st_size,'recipe_sha256':digest(recipe),
        'numpy_version':np.__version__,'platform':platform.platform(),'worker_sha256':digest(Path(__file__).with_name('canonical_child.py').read_bytes()),
        'planned_calls':planned,'attempted_calls':attempted,'completed_sessions':sum(r['complete'] for r in sessions),'planned_sessions':len(jobs),
        'sessions':sessions,'failure':failed,'primary_n_expected':128,'primary_n_valid':sum(x is not None for x in forward),
        'primary_accuracy':float(np.mean(np.asarray(forward)==data['labels'])) if by_name.get('forward',{}).get('complete') else None,
        'timing_blocks':timings,'service_block_median_variability_ms':summary([r['service_request_to_validated_digit_ms']['median'] for r in timings]),
        'service_boundary':TIMING_BOUNDARY,'timing_trust':{'parent_service':'parent_observed_supplementary','predict_duration':'untrusted_child_diagnostic'},
        'memory_enforcement':'sampled_rss_kill','strict_peak_memory_enforced':False,
        'elapsed_seconds':time.monotonic()-started,'labels_sha256':digest(data['labels'].tobytes())}
    if dependence and not failed: result['status']='DEFINED_CHECK_FAILURE'
    atomic_write(output/'RESULT.json',canonical(result)+b'\n')
    return result
