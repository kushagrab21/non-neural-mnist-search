"""Bounded adaptive development: one candidate per Researcher checkpoint."""
import copy, time
from datetime import datetime,timedelta,timezone
from pathlib import Path
from .common import Blocked,atomic_write,canonical,digest,strict_json
from .store import Store
from .engine import Engine
from .resources import Budget,units
from .host_queue import HostQueue
from .experiment_release import require_release,consume_release
from .pilot_release import require_readiness
from .role_protocol import prepare
from .search_dispatch import dispatch,DispatchFailure
from .retry_policy import eligibility
from .development_data import load,export_fit
from .artifact_jobs import run as artifact_job
from .baseline_stage import validate_stage
from .comparison_recipe import plan
from .common_comparison import execute_development
from .pilot_handoff import runner_limits,preflight,freeze as freeze_handoff,HANDOFF_POLICY
from .pilot import archive_file,mirror,HOST,ROOT


def run(release,output):
    require_release(release);c=release['contract'];out=Path(output).resolve()
    if out.exists():raise Blocked('new experiment output required; issued work is never replayed')
    consumed=consume_release(release,out)
    cap={k:c['caps'][k] for k in ('agent_tasks','evaluation_requests','worker_sessions','wall_seconds')}
    deadline=(datetime.now(timezone.utc)+timedelta(seconds=cap['wall_seconds'])).isoformat()
    n=c['max_batches'];batchcaps={f'B{i}':units(agent_tasks=7,evaluation_requests=72248,worker_sessions=44,wall_seconds=cap['wall_seconds']//n) for i in range(1,n+1)}
    summary={'status':'starting','scope':'adaptive_development','final_data_opened':False,'candidates':[],
             'max_batches':n,'wall_limit_seconds':cap['wall_seconds'],'additional_charges_authorized':False,'retry_attempts':0}
    with Store(out/'workflow') as s,Store(out/'resources') as resources:
        engine=Engine(s);budget=Budget(resources)
        policy={'mode':'bounded_search','authorization_id':c['id'],'deadline':deadline,'campaign':cap,
                'phases':{'search':cap},'batches':{b:{'phase':'search','limits':v} for b,v in batchcaps.items()}}
        budget.initialize(policy,HOST,release=release)
        HostQueue(resources).initialize({'authorization_id':c['id'],'deadline':deadline,'max_tasks':cap['agent_tasks'],
                  'actors':{r:c['actors'][r] for r in ('runner','researcher','controller')},'scope':'bounded_adaptive_development'},HOST)
        config={'campaign_id':c['id'],'mode':'bounded_search','live_inference_enabled':True,'sealed_assessment_enabled':False,
                'actors':c['actors'],'limits':{'campaign':cap,'phases':{'search':cap},'batches':batchcaps},
                'max_elapsed_seconds':cap['wall_seconds'],'max_batch_candidates':1,
                'direction_allocations':{f'ALLOC-ND{i}':{'direction_id':f'ND{i}','batch_id':f'B{i}'} for i in range(2,n+1)},
                'memory_policy':{'id':'literal-joins-1','recent_per_direction':2},
                'versions':{'contract':digest(c),'evaluator':digest(c['measurement_files']),'data':c['development_manifest_sha256'],'config':'bounded-search/1'}}
        engine.initialize(config,release=release);engine.add('EXPERIMENT-CONSUMPTION','experiment_consumption',consumed)
        reference={'source':(ROOT/c['starting_reference']['source']).read_text(),
                   'results':strict_json((ROOT/c['starting_reference']['results']).read_bytes()),
                   'researcher_ruling':strict_json((ROOT/c['starting_reference']['ruling']).read_bytes()),
                   'scope':'Historical accepted development reference; earlier timing from a different comparison group. Not a new campaign result.'}
        engine.add('STARTING-REFERENCE','historical_reference',reference)
        def status(stage):
            summary['stage']=stage;summary['resources']={k:v for k,v in budget.snapshot().items() if k!='reservations'}
            summary['reported_list_price_usd']=sum(e['payload']['incremental_list_usd'] for e in resources.events if e['kind']=='native_model_receipt')
            summary['workflow_head_sha256']=s.events[-1]['hash'];summary['resources_head_sha256']=resources.events[-1]['hash']
            atomic_write(out/'STATUS.json',canonical(summary)+b'\n')
            print(stage,flush=True)
        def call(job,task,context,target,batch,seconds):
            if (out/'STOP').exists():raise Blocked('operator STOP present')
            for attempt in range(2):
                name=job if attempt==0 else job+'-R1'
                status('dispatch '+name)
                order=prepare(engine,name,task,context,target_id=target,batch_id=batch)
                try:return dispatch(engine,resources,order,host=HOST,wall_seconds=seconds)
                except DispatchFailure as exc:
                    retry,why=eligibility(exc.evidence,attempts=attempt,extras=summary['retry_attempts'],policy=c['retry_policy'])
                    engine.add('RETRY-DISPOSITION-'+name,'retry_disposition',{'dispatch_id':name,'eligible':retry,'reason':why,'failure':exc.evidence})
                    if not retry:raise
                    summary['retry_attempts']+=1
                    context=copy.deepcopy(context)
                    context['output_correction']={'failed_dispatch':name,'validation_error':str(exc)[:2000],
                        'instruction':'Previous attempt was retained and charged. Return the exact requested schema for this new dispatch and request hash. This is a format/transport correction; preserve the original scientific work and prospective decision.'}
                    status('bounded retry after '+name);time.sleep(c['retry_policy']['backoff_seconds'])
        def history():
            return {'starting_reference':reference,'prior_results':copy.deepcopy(summary['candidates']),
                    'prior_rulings':copy.deepcopy(engine.state['rulings']),
                    'failures_and_retries':[e for e in s.events if e['kind'] in ('structured_failure','retry_disposition','anomaly')],
                    'directions':copy.deepcopy(engine.state['direction_definitions']),
                    'active_directions':list(engine.state['active']),
                    'known_event_refs':list(engine.state['event_refs']),
                    'remaining_resources':{k:v for k,v in budget.snapshot().items() if k!='reservations'},
                    'required_direction_decisions':'At each checkpoint, exactly one action for EVERY active direction. A later batch must retain the resulting active roster; no silent rename.',
                    'new_direction_allocations':config['direction_allocations']}
        try:
            recipe=plan(10000,scope='development_pilot');limits=runner_limits(recipe)
            limits['comparison_execution_seconds']=280;limits['comparison_allocation_seconds']=300
            initial={'contract':c,'starting_reference':reference,'execution_limits':limits,
                     'review_evidence':[{'path':p,'sha256':c['files'][p],'content':(ROOT/p).read_text()} for p in c['readiness_evidence_files']],
                     'instruction':'Start the actual adaptive development experiment. First review the exact frozen contract and changed implementation evidence. Return readiness accepted/changes_required/unresolved with the contract hash from versions.contract and every review_evidence path. Acceptance covers this bounded run only. Then plan B1, max_candidates=1, choosing the scientific direction; no fixed family is required. Choose 1-3 active directions. Improve available accuracy/size/service-latency tradeoffs from the accepted historical reference. All roles use Claude. No final opening. Budget is up to two hours and six candidates with funded retry slots. All predictions are qualitative, not numerically scored. The next checkpoint can refine, explore, diagnose, revise guidance or stop. Make the visible plan concrete enough for the Runner to act without another clarification.'}
            call('PLAN','pilot_plan',initial,'B1','B1',240)
            readiness=s.find('PILOT-READINESS');require_readiness(release,readiness)
            resources.append('PILOT-READINESS','pilot_readiness',readiness['actor'],readiness['payload'])
            status('readiness accepted; preparing fit-only inputs')
            data=load(ROOT/'experiment_inputs');fit_dir=out/'fit_only';fit_manifest=export_fit(data,fit_dir)
            atomic_write(out/'FIT_PROVENANCE.json',canonical(fit_manifest)+b'\n')
            baseline=ROOT/c['baseline_stage']
            budget.reserve('STAGE','artifact_stage',units(wall_seconds=60),phase='search',batch='B1',binding={'scope':'development_pilot','baseline_stage':str(baseline)},host=HOST);budget.begin('STAGE',HOST)
            staged=validate_stage(baseline);elapsed,clock=budget.elapsed_evidence('STAGE')
            budget.settle('STAGE',units(wall_seconds=elapsed) if elapsed is not None else None,{'stage_sha256':digest(staged),'clock':clock},HOST)
            engine.add('STAGE-RECEIPT','stage_receipt',staged)
            for number in range(1,n+1):
                batch=f'B{number}';cid=f'C{number}';did=f'D{number}';eid=f'E{number}'
                if number>1:
                    if not engine.state['active']:
                        summary['stop_reason']='Researcher stopped all active directions';break
                    if budget.remaining_seconds()<900:
                        summary['stop_reason']='Remaining wall time cannot reserve a new build, comparison and review';break
                    call('PLAN-'+batch,'plan',{'history':history(),'previous_sources':{k:s.read_artifact(v['source_sha256']).decode() for k,v in engine.state['candidates'].items() if k==f'C{number-1}'},
                         'instruction':f'Plan {batch}, max_candidates=1, using the last Researcher ruling and all active directions. Keep exactly the current active roster; select the most informative next candidate through a concrete question. Use a fresh snapshot_id.'},batch,batch,180)
                order=engine.state['orders'][batch]
                call('PREDICT-'+cid,'hypothesis',{'decision_id':did,'candidate_id':cid,'history':history(),
                     'instruction':'Freeze one prospective candidate_change decision before construction. Choose an active direction; cite known event refs. Explain the proposed algorithm and expected tradeoff concretely. This is adaptive work: use prior measured results and Researcher direction decisions. Do not repeat the historical classifier unchanged.'},did,batch,180)
                direction=engine.state['decisions'][did]['direction_id']
                parents={k:s.read_artifact(v['source_sha256']).decode() for k,v in engine.state['candidates'].items() if k==f'C{number-1}'}
                call('CONSTRUCT-'+cid,'construct',{'decision_id':did,'candidate_id':cid,'direction_id':direction,'history':history(),'parent_sources':parents,
                     'build_interface':'Python/NumPy only. Read /fit/fit_images.npy and /fit/fit_labels.npy: 50,000 fit folds1-5 images. Write exactly /output/classifier.py and /output/provenance.json. No other output files, no subprocess/network/download. Emit predict(read-only uint8 (28,28))->Python int0..9. Memory 512MiB, build execution590 seconds on1CPU; avoid full large float64 matrices. Source at most65536 bytes, build_script32768 bytes. Embed learned coefficients; no external files or stored exemplars. Declare aggregate-template support counts if applicable. Fit-only internal validation is permitted; no fold0 access. Use accurate Python3.11/NumPy-compatible code with a small self-check before emitting. Respect exact ids and frozen decision.',
                     'source_bytes_cap':c['candidate_source_bytes_cap'],'execution_limits':limits,'admissibility':c['candidate_rules']},cid,batch,240)
                construction=s.find('CONSTRUCTION-'+cid)['payload'];directory=out/'candidates'/cid;directory.mkdir(parents=True)
                inputs=directory/'build_input';inputs.mkdir();(inputs/'build.py').write_text(construction['build_script'])
                status('building '+cid)
                engine.reserve('RES-BUILD-'+cid,units(wall_seconds=600),phase='search',batch_id=batch)
                built=artifact_job(resources,reservation_id='BUILD-'+cid,kind='artifact_build',input_dir=inputs,output_dir=directory/'build',script_name='build.py',wall_seconds=600,phase='search',batch=batch,host=HOST,scope='development_pilot',fit_dir=fit_dir,fit_manifest=fit_manifest)
                engine.add('BUILD-RECEIPT-'+cid,'build_receipt',{'receipt_sha256':s.archive(canonical(built))});mirror(engine,budget,'BUILD-'+cid,'BUILD-RECEIPT-'+cid)
                if built['status']!='completed' or set(built['artifacts'])!={'classifier.py','provenance.json'}:
                    raise Blocked('candidate build failed; exact failure retained, no blind rebuild')
                source=(directory/'build/emitted/classifier.py').read_bytes();pb=(directory/'build/emitted/provenance.json').read_bytes()
                if len(source)>c['candidate_source_bytes_cap'] or len(pb)>HANDOFF_POLICY['provenance_json_bytes']:raise Blocked('artifact exceeds evidence allocation')
                provenance=strict_json(pb);engine.submit(construction['candidate'],source,engine.actor('runner'))
                engine.add('BUILD-PROVENANCE-'+cid,'build_provenance',{'host_fit_manifest':fit_manifest,'runner_provenance_unverified':provenance,'build_receipt_ref':'BUILD-RECEIPT-'+cid})
                preflight(engine,source,construction,provenance,fit_manifest,built,staged,limits,history=history())
                requested=units(evaluation_requests=recipe['planned_requests'],worker_sessions=recipe['planned_sessions'],wall_seconds=300)
                engine.add('FEEDBACK-'+cid,'feedback_admission',{'cohort':data['identity'],'recipe_sha256':digest(recipe),'ordinal':number,'limit':n})
                engine.reserve('RES-COMPARE-'+cid,requested,phase='search',batch_id=batch)
                budget.reserve('COMPARE-'+cid,'development_evaluation',requested,phase='search',batch=batch,binding={'recipe_sha256':digest(recipe),'candidate_sha256':digest(source),'baseline':staged},host=HOST);budget.begin('COMPARE-'+cid,HOST)
                engine.add('START-'+eid,'evaluation_started',{'candidate_id':cid,'evaluation_id':eid,'reservation_id':'RES-COMPARE-'+cid})
                status('measuring '+cid+' against the CNN')
                measured=execute_development(recipe,directory/'comparison',data=data,candidate_source=source,baseline_stage=baseline,release=release,resource_guard=budget.execution_guard('COMPARE-'+cid),readiness=readiness,evaluation_seconds=280)
                rawsha=archive_file(s,directory/'comparison/RESULT.json');elapsed,clock=budget.elapsed_evidence('COMPARE-'+cid)
                actual=units(evaluation_requests=measured['attempted_requests'],worker_sessions=measured['attempted_sessions'],wall_seconds=elapsed) if measured['cleanup_confirmed'] and elapsed is not None else None
                budget.settle('COMPARE-'+cid,actual,{'raw_manifest_sha256':rawsha,'clock':clock,'cleanup':measured['cleanup_confirmed']},HOST)
                engine.add('COMPARE-RECEIPT-'+cid,'comparison_receipt',{'raw_manifest_sha256':rawsha});mirror(engine,budget,'COMPARE-'+cid,'COMPARE-RECEIPT-'+cid)
                forward=next((j for j in measured['sessions'] if j['member']=='candidate' and j['name']=='forward'),{})
                preds=forward.get('predictions',[]);preds=preds+[None]*(10000-len(preds))
                valid=measured['status']=='COMPLETE_NO_DEFINED_CHECK_FAILURE' and all(measured['objectives'][m]['complete'] for m in ('candidate','cnn'))
                changes={}
                for parent in construction['candidate']['parents']:
                    pe=next((v for v in engine.state['evaluations'].values() if v['candidate_id']==parent),None)
                    if pe and len(pe['predictions'])==len(preds):
                        changes[parent]={'prediction_change_count':sum(a!=b for a,b in zip(pe['predictions'],preds)),
                                         'fixed_count':sum(a!=int(y) and b==int(y) for a,b,y in zip(pe['predictions'],preds,data['labels'])),
                                         'broken_count':sum(a==int(y) and b!=int(y) for a,b,y in zip(pe['predictions'],preds,data['labels']))}
                evaluation={'evaluation_id':eid,'candidate_id':cid,'candidate_sha256':digest(source),'versions':engine.config['versions'],
                    'status':'ok' if valid else 'failed','n':10000,'predictions':preds,'accuracy':measured['development_accuracy']['candidate'] if valid else None,
                    'source_bytes':len(source),'prediction_median_ms':measured['objectives']['candidate']['primary_value'],'timing_boundary':measured['service_boundary'],
                    'data_sha256':digest(data['identity']),'raw_artifact_sha256':rawsha,'raw_artifact_format':'chunked-evidence/1','parent_changes':changes,
                    'errors':[] if valid else [measured['status']],'admissibility':'pending Researcher review',
                    'cnn_comparison':{'accuracy':measured['development_accuracy']['cnn'],'runtime':measured['objectives']['cnn'],'checkpoint_bytes':(baseline/'weights.pt').stat().st_size,'architecture_bytes':(baseline/'architecture.py').stat().st_size},
                    'candidate_runtime':measured['objectives']['candidate']}
                engine.record_evaluation(evaluation)
                if not valid:engine.anomaly('ANOMALY-'+eid,eid,'Comparison incomplete or defined history check failed; no performance acceptance.')
                engine.freeze_packet('PACKET-'+batch,batch)
                packet=freeze_handoff(engine,measured,evaluation,budget.snapshot())
                call('RECORD-'+batch,'record',packet,'CONTROLLER-NOTE-'+batch,batch,120)
                call('REVIEW-'+batch,'pilot_review',packet,'RULING-'+batch,batch,240)
                ruling=s.find('ADMISSIBILITY-RULING-'+batch)['payload']
                row={'candidate_id':cid,'direction_id':direction,'source_sha256':digest(source),'source_ref':str((directory/'build/emitted/classifier.py').relative_to(out)),'source_reference_base':'run_directory',
                     'source_bytes':len(source),'accuracy':evaluation['accuracy'],'service_latency_ms':evaluation['prediction_median_ms'],
                     'comparison_group_sha256':measured['objectives']['candidate'].get('comparison_group_sha256'),
                     'cnn_accuracy':measured['development_accuracy']['cnn'],'cnn_service_latency_ms':measured['objectives']['cnn']['primary_value'],
                     'parent_changes':changes,'admissibility':ruling,'ruling_ref':'RULING-'+batch,'comparison_status':measured['status']}
                summary['candidates'].append(row);summary['status']='running';atomic_write(directory/'SUMMARY.json',canonical(row)+b'\n')
                status('checkpoint '+batch+' saved: '+str(evaluation['accuracy'])+' accuracy; '+ruling['verdict'])
                if not valid:
                    summary['stop_reason']='Measurement failure requires review; no automatic evaluation repeat';break
            summary['status']='completed';summary.setdefault('stop_reason','Bounded candidate allocation completed')
        except BaseException as exc:
            summary['status']='stopped_on_failure';summary['error']=type(exc).__name__+': '+str(exc)
            atomic_write(out/'FAILURE.json',canonical({'error':summary['error'],'resources':budget.snapshot(),'engine_resources':engine.resources()})+b'\n')
        finally:
            status(summary['status']);atomic_write(out/'RESULT.json',canonical(summary)+b'\n')
        return summary
