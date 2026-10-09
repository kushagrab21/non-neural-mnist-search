"""Common Docker comparison executor for generated synthetic inputs only.
The recipe owns scheduling and accounting; no CLI/path accepts MNIST payloads.
"""
from pathlib import Path
from dataclasses import asdict
import json,os,platform,subprocess,sys,time
import numpy as np
from .common import Blocked,canonical,digest,atomic_write,now
from .canonical import Limits,TIMING_BOUNDARY
from .comparison_recipe import plan
from .docker_worker import DockerWorker,DOCKER,IMAGE_RECORD
from .measurement_guard import MeasurementGuard,POLICY
from .service_metric import METRIC_ID,aggregate


def comparison_limits():
    """Shared by the executor and the Runner's prospective instructions."""
    return Limits(startup_seconds=20,total_seconds=60,observed_rss_bytes=512*1024*1024,max_calls=10240)


def synthetic_data(n,seed):
    # Disjoint images are generated once in parent, never a batch sent to child.
    all_images=np.random.Generator(np.random.PCG64(seed)).integers(0,256,(n+1074,28,28),dtype=np.uint8)
    if len({digest(x.tobytes()) for x in all_images})!=len(all_images):raise Blocked('synthetic identity collision')
    return {'images':all_images[:n],'warmups':all_images[n:n+50],'fillers':all_images[n+50:],'labels':np.zeros(n,dtype=np.int64)}


def context():
    raw=subprocess.run(DOCKER+['info','--format','{{json .}}'],capture_output=True,text=True,timeout=10,check=True)
    d=json.loads(raw.stdout)
    return {'host_platform':platform.platform(),'host_python':sys.version,'host_numpy':np.__version__,'host_load':list(os.getloadavg()),'docker_vm':{k:d.get(k) for k in ['Architecture','OSType','KernelVersion','NCPU','MemTotal','ServerVersion','OperatingSystem','SecurityOptions']},'uncontrolled_contention':'host/VM competing work is observed indirectly; slow valid service calls are retained, no correction/subtraction','clock_boundary':'host parent observes whole service; guest clock is not used for objective'}


def execute(recipe,output,*,limits=None,evaluation_seconds=360,resource_guard=None,worker_factory=DockerWorker,fixture="constant",sensitivity_subset=False,_prepared=None):
    scope='synthetic_preparation_only' if _prepared is None else 'development_pilot'
    if _prepared is not None:
        from .pilot_release import require_release, require_readiness
        require_release(_prepared['release'])
        require_readiness(_prepared['release'], _prepared.get('readiness'))
    if recipe!=plan(recipe['primary_n'],seed=recipe['seed'],members=tuple(recipe['members']),timed_indices=recipe['timed_indices'],scope=scope):raise Blocked('recipe differs from canonical planner')
    if recipe['scope']!=scope:raise Blocked('comparison scope differs from admission')
    if fixture not in ('constant','stateful') or (sensitivity_subset and fixture!='stateful'):raise Blocked('only named public synthetic fixtures supported')
    if sensitivity_subset:
        wanted={'forward','reverse','reference-0','history-0-1'}
        selected=[j for j in recipe['jobs'] if j['member']=='candidate' and j['name'] in wanted]
        recipe=dict(recipe,jobs=selected,planned_requests=sum(j['requests'] for j in selected),planned_sessions=len(selected),defined_coverage='sensitivity_subset_only',per_member={'candidate':{'requests':sum(j['requests'] for j in selected),'sessions':len(selected)},'cnn':{'requests':0,'sessions':0}})
    limits=limits or comparison_limits()
    limits.validate()
    if recipe['largest_session_requests']>limits.max_calls:raise Blocked('recipe exceeds worker call cap')
    if type(evaluation_seconds) not in (int,float) or not 0<evaluation_seconds<=(1200 if _prepared else 600):raise Blocked('finite comparison wall cap required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);data=synthetic_data(recipe['primary_n'],recipe['seed']) if _prepared is None else _prepared['data'];machine=context()
    constant='def predict(image): return 0\n' if fixture=='constant' else 'calls=0\ndef predict(image):\n global calls\n calls+=1\n return calls%10\n';source=output/'constant.py';source.write_text(constant)
    cnn_source=IMAGE_RECORD.parent/'fixture_cnn.py'
    baseline_identity=None
    if _prepared is not None:
        from .baseline_stage import validate_stage
        if fixture!='constant' or sensitivity_subset:raise Blocked('development comparison requires the full recipe')
        constant=_prepared['candidate_source'].decode('utf-8');source.write_text(constant)
        cnn_source=Path(_prepared['baseline_stage'])/'architecture.py'
        baseline_identity=validate_stage(_prepared['baseline_stage'])
    image_record=json.loads(IMAGE_RECORD.read_text())
    identity={'recipe_sha256':digest(recipe),'runtime':'docker-host-service/1','metric_id':METRIC_ID,'guard_policy':POLICY,'guard_sha256':digest(Path(__file__).with_name('measurement_guard.py').read_bytes()),'executor_sha256':digest(Path(__file__).read_bytes()),'worker_sha256':digest(Path(__file__).with_name('docker_worker.py').read_bytes()),'canonical_parent_sha256':digest(Path(__file__).with_name('canonical.py').read_bytes()),'canonical_child_sha256':digest(Path(__file__).with_name('canonical_child.py').read_bytes()),'image_ids':{k:v['image_id'] for k,v in image_record.items()},'limits':asdict(limits),'machine':machine,'input_identities':{k:[digest(im.tobytes()) for im in data[k]] for k in ['images','warmups','fillers']},'labels_sha256':digest(data['labels'].tobytes()),'candidate_source_sha256':digest(constant.encode()),'cnn_fixture_source_sha256':digest(cnn_source.read_bytes()),'thread_policy':'OMP/MKL/OPENBLAS/NUMEXPR=1; torch intra/inter-op=1','fixture':fixture,'defined_coverage':recipe.get('defined_coverage','full_defined_recipe'),'source_scope':'named public candidate fixture and untrained zero-parameter CNN; no MNIST candidate/baseline performance claim'}
    if _prepared is not None:
        identity.update(source_scope='released development pilot; admissibility and scientific interpretation pending Researcher ruling',
                        fixture=None,baseline_stage=baseline_identity,development_identity=data['identity'],
                        release_sha256=digest(_prepared['release']))
    atomic_write(output/'RECIPE.json',canonical({'recipe':recipe,'comparison_identity':identity})+b'\n')
    started_at=now();started=time.monotonic();deadline=started+evaluation_seconds;rows=[];failures=[];attempted=0
    records={m:{'scope':'synthetic_proposed_metric','metric_id_at_start':METRIC_ID,'comparison_identity':identity,'timed_n':128,'blocks':[]} for m in recipe['members']}
    for job in recipe['jobs']:
        row={'member':job['member'],'name':job['name'],'kind':job['kind'],'predictions':[],'warmups_valid':0,'complete':False,'cleanup_confirmed':False,'timed_receipts':[]};rows.append(row)
        directory=output/(job['member']+'-'+job['name']);guard=None
        try:
            if resource_guard:resource_guard()
            if time.monotonic()>=deadline:raise Blocked('comparison deadline before session')
            guard=MeasurementGuard()
            mode='candidate' if job['member']=='candidate' else ('cnn_fixture' if _prepared is None else 'cnn_staged')
            extra={'baseline_stage':_prepared['baseline_stage']} if mode=='cnn_staged' else {}
            with worker_factory(source if mode=='candidate' else cnn_source,directory,mode=mode,limits=limits,evaluation_deadline=deadline,**extra) as worker:
                for image in data['warmups']:
                    if resource_guard:resource_guard()
                    guard.observe();attempted+=1;r=worker.predict(image);row['warmups_valid']+=1;guard.observe()
                    if row['warmups_valid']==1:row['first_request_service_ms']=r['service_ms']
                    if resource_guard:resource_guard()
                for part,index in job['sequence']:
                    if resource_guard:resource_guard()
                    guard.observe();attempted+=1;r=worker.predict(data[part][index]);row['predictions'].append(r['response']['digit'])
                    if job['kind']=='timing':row['timed_receipts'].append(r)
                    guard.observe()
                    if resource_guard:resource_guard()
                row['startup_ms']=worker.metadata['parent_startup_ms']
            row['cleanup_confirmed']=worker.cleanup_confirmed
            guard.observe()
            if resource_guard:resource_guard()
            if not row['cleanup_confirmed']:raise Blocked('worker cleanup unconfirmed')
            row['complete']=True
        except Exception as exc:
            row['failure']={'type':type(exc).__name__,'code':getattr(exc,'code',None),'detail':str(exc)};failures.append(dict(row['failure'],member=job['member'],session=job['name']))
        receipt=directory/'RECEIPT.json'
        if receipt.is_file():
            payload=receipt.read_bytes();v=json.loads(payload)
            row['receipt_sha256']=digest(payload)
            row['cleanup_confirmed']=v.get('cleanup')=='container_removed_and_client_reaped' and v.get('source_sha256')==digest((source if job['member']=='candidate' else cnn_source).read_bytes())
        row['measurement_guard']=guard.evidence() if guard else {'complete':False,'failure':'not constructed'}
        if job['kind']=='timing':records[job['member']]['blocks'].append({'block':job['block'],'parent_service_ms':[x['service_ms'] for x in row['timed_receipts']],'complete':row['complete'],'monitor_complete':row['measurement_guard']['complete'] and row['cleanup_confirmed']})
        if failures:break
    by={(x['member'],x['name']):x for x in rows};dependence=[];accuracy={}
    for member in recipe['members']:
        forward=by.get((member,'forward'),{});fp=forward.get('predictions',[])
        accuracy[member]=float(np.mean(np.asarray(fp)==data['labels'])) if forward.get('complete') else None
        for job in recipe['jobs']:
            if job['member']!=member:continue
            row=by.get((member,job['name']))
            if not row:continue
            if job['kind']=='order' and job['name']!='forward':
                for pos,((part,index),pred) in enumerate(zip(job['sequence'],row['predictions'])):
                    if index<len(fp) and pred!=fp[index]:dependence.append({'member':member,'session':job['name'],'position':pos,'reference':fp[index],'observed':pred,'session_complete':row['complete']})
            if job['kind']=='history':
                reference=by.get((member,'reference-'+str(job['anchor'])),{}).get('predictions',[])
                if reference:
                    for pos in [0,len(job['sequence'])-1]:
                        if pos<len(row['predictions']) and row['predictions'][pos]!=reference[0]:dependence.append({'member':member,'session':job['name'],'position':pos,'reference':reference[0],'observed':row['predictions'][pos],'session_complete':row['complete']})
    cleanup=all(x['cleanup_confirmed'] for x in rows)
    result={'scope':'generated_synthetic_docker_comparison_only','campaign_qualified':False,'started_at':started_at,'recipe_sha256':digest(recipe),'comparison_identity':identity,'status':'INCOMPLETE' if failures else ('DEFINED_CHECK_FAILURE' if dependence else 'COMPLETE_NO_DEFINED_CHECK_FAILURE'),'attempted_requests':attempted,'attempted_sessions':len(rows),'planned_requests':recipe['planned_requests'],'planned_sessions':recipe['planned_sessions'],'cleanup_confirmed':cleanup,'failures':failures,'dependence':dependence,'sessions':rows,'synthetic_known_zero_accuracy':accuracy,'objectives':{m:aggregate(r) for m,r in records.items()},'constant_transport_floor':'candidate objective reports full trivial-predict+conversion+protocol floor, not pure transport and never subtracted' if fixture=='constant' else 'not a floor fixture','defined_coverage':recipe.get('defined_coverage','full_defined_recipe'),'scientific_result':'no MNIST accuracy or trained-baseline performance claim','wall_seconds':time.monotonic()-started,'service_boundary':TIMING_BOUNDARY}
    if _prepared is not None:
        result.update(scope='development_pilot_observations',development_accuracy=result.pop('synthetic_known_zero_accuracy'),
                      constant_transport_floor=None,scientific_result='adaptive development measurements; no final-assessment or admissibility claim')
        for objective in result['objectives'].values():objective['state']='development_observation_pending_ruling'
    atomic_write(output/'RESULT.json',canonical(result)+b'\n');return result


def execute_development(recipe,output,*,data,candidate_source,baseline_stage,release,resource_guard,limits=None,evaluation_seconds=1180,readiness=None):
    from .pilot_release import require_release, require_readiness
    from .baseline_stage import MANIFEST_SHA
    require_release(release)
    require_readiness(release, readiness)
    if data['identity'].get('manifest_sha256')!=MANIFEST_SHA or data['identity']['feedback_n']!=10000:
        raise Blocked('identified full development fold required')
    if recipe!=plan(10000,scope='development_pilot'):raise Blocked('pilot recipe must match the frozen full-cohort plan')
    if len(data['images'])!=10000 or len(data['labels'])!=10000 or len(data['warmups'])!=50 or len(data['fillers'])!=1024:
        raise Blocked('development cohort/contexts differ from recipe')
    return execute(recipe,output,limits=limits,evaluation_seconds=evaluation_seconds,resource_guard=resource_guard,
                   _prepared={'data':data,'candidate_source':candidate_source,'baseline_stage':baseline_stage,'release':release,'readiness':readiness})
