"""One released all-Claude pilot. No adaptive search or final opening.

Preparation can inspect the proposed package without calling run(). Five native
tasks separate the Runner's prediction from its construction. The common Docker
recipe is executed once, and the resulting packet receives one Researcher ruling.
"""
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path

from .common import Blocked, atomic_write, canonical, digest, strict_json
from .store import Store
from .engine import Engine
from .resources import Budget, units
from .host_queue import HostQueue
from .pilot_release import require_release, consume_release, require_readiness
from .role_protocol import prepare
from .live_dispatch import dispatch
from .development_data import load, export_fit
from .artifact_jobs import run as artifact_job
from .baseline_stage import validate_stage
from .comparison_recipe import plan
from .common_comparison import execute_development
from .pilot_handoff import runner_limits, preflight, freeze as freeze_handoff, HANDOFF_POLICY

ROOT = Path(__file__).resolve().parents[3]
HOST = {'role': 'maintainer', 'session_id': '/root', 'provider': 'openai', 'model_id': None}


def archive_file(store, path):
    """Large raw measurements are chunked; no evidence is silently dropped."""
    sha = hashlib.sha256(); chunks = []; total = 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            sha.update(block); total += len(block)
            chunks.append({'sha256': store.archive(block), 'bytes': len(block)})
    return store.archive(canonical({'format': 'chunked-evidence/1', 'sha256': sha.hexdigest(),
                                    'bytes': total, 'chunks': chunks}))


def mirror(engine, budget, reservation_id, evidence_ref):
    row = budget.state()[reservation_id]
    engine.settle('USAGE-'+reservation_id, 'RES-'+reservation_id, row['actual'], evidence_ref)


def run(release, output):
    require_release(release)  # Before creating any workspace, reading arrays or launching a process.
    contract = release['contract']; out = Path(output).resolve()
    if out.exists(): raise Blocked('new pilot destination required; resume/retry is not automatic')
    consumed = consume_release(release, out)
    cap = {k: contract['caps'][k] for k in ('agent_tasks','evaluation_requests','worker_sessions','wall_seconds')}
    deadline = (datetime.now(timezone.utc)+timedelta(seconds=cap['wall_seconds'])).isoformat()
    with Store(out/'workflow') as s, Store(out/'resources') as resources:
        engine = Engine(s); budget = Budget(resources)
        policy = {'mode':'bounded_pilot','authorization_id':contract['id'],'deadline':deadline,
                  'campaign':cap,'phases':{'pilot':cap},'batches':{'B1':{'phase':'pilot','limits':cap}}}
        budget.initialize(policy, HOST, release=release)
        HostQueue(resources).initialize({'authorization_id':contract['id'],'deadline':deadline,
              'max_tasks':cap['agent_tasks'],'actors':{r:contract['actors'][r] for r in ('runner','researcher','controller')},
              'scope':'released_one_candidate_development_pilot'}, HOST)
        config = {'campaign_id':contract['id'],'mode':'bounded_pilot','live_inference_enabled':True,
                  'sealed_assessment_enabled':False,'actors':contract['actors'],
                  'limits':{'campaign':cap,'phases':{'pilot':cap},'batches':{'B1':cap}},
                  'max_elapsed_seconds':cap['wall_seconds'],'max_batch_candidates':1,'direction_allocations':{},
                  'memory_policy':{'id':'literal-joins-1','recent_per_direction':1},
                  'versions':{'contract':digest(contract),'evaluator':digest(contract['files']),
                              'data':contract['development_manifest_sha256'],'config':'bounded-pilot/1'}}
        engine.initialize(config,release=release)
        engine.add('PILOT-CONSUMPTION', 'pilot_consumption', consumed)

        def call(job, task, context, target, seconds):
            if (out/'STOP').exists(): raise Blocked('operator STOP present')
            order = prepare(engine,job,task,context,target_id=target)
            return dispatch(engine,resources,order,host=HOST,wall_seconds=seconds)

        try:
            recipe=plan(10000,scope='development_pilot')
            limits=runner_limits(recipe)
            inline = contract['version'] == 'bounded-pilot/2'
            context = {'contract': contract, 'instruction': 'One direction and one candidate; no new directions or search allocation.'}
            if inline:
                context.update(execution_limits=limits,
                    review_evidence=[{'path': name, 'sha256': contract['files'][name],
                                      'content': (ROOT/name).read_text()}
                                     for name in contract['readiness_evidence_files']])
                context['instruction'] += (' First review the exact scientific contract and supplied prior review/repair evidence. '
                    'Return accepted only if this one bounded attempt is scientifically ready; otherwise changes_required or unresolved. '
                    'Your readiness must identify the contract hash from versions.contract and cite every supplied evidence path. '
                    'No data has been loaded. Acceptance opens data access for this attempt, not main search or final assessment. '
                    'The complete live loop and Docker resume remain unqualified: this pilot is their bounded qualification. '
                    'Do not claim that local tests prove a live result. Include the one-candidate plan in the same response.')
            call('PLAN', 'pilot_plan' if inline else 'plan', context, 'B1', 240)
            readiness = s.find('PILOT-READINESS')
            require_readiness(release, readiness)
            if inline:
                resources.append('PILOT-READINESS', 'pilot_readiness', readiness['actor'], readiness['payload'])
            order = engine.state['orders']['B1']
            if len(order['directions']) != 1: raise Blocked('pilot requires exactly one direction')
            direction = order['directions'][0]
            data = load(ROOT/'experiment_inputs')
            fit_dir = out/'fit_only'; fit_manifest = export_fit(data,fit_dir)
            atomic_write(out/'FIT_PROVENANCE.json',canonical(fit_manifest)+b'\n')
            call('PREDICT','hypothesis',{'decision_id':'D1','direction_id':direction,
                   'instruction':'Freeze a prospective candidate_change decision before construction. Evidence: CAMPAIGN and B1.'},'D1',120)
            call('CONSTRUCT','construct',{'decision_id':'D1','candidate_id':'C1','direction_id':direction,
                 'build_interface':'Python/NumPy script reads /fit/fit_images.npy and /fit/fit_labels.npy (only folds1to5). Write exactly /output/classifier.py and /output/provenance.json, no other output files. No network or external dependencies. The classifier must implement predict(read-only uint8 (28,28)) -> Python int 0..9. Respect the execution_limits and handoff allocations; oversized returns stop this one-shot pilot before feedback.',
                 'source_bytes_cap':contract['candidate_source_bytes_cap'],
                 'execution_limits':limits,
                 'admissibility':contract['candidate_rules']},'C1',240)
            construction = s.find('CONSTRUCTION-C1')['payload']
            build_input = out/'build_input'; build_input.mkdir()
            (build_input/'build.py').write_text(construction['build_script'])
            engine.reserve('RES-BUILD',units(wall_seconds=600),phase='pilot',batch_id='B1')
            built = artifact_job(resources,reservation_id='BUILD',kind='artifact_build',input_dir=build_input,
                   output_dir=out/'build',script_name='build.py',wall_seconds=600,phase='pilot',batch='B1',host=HOST,
                   scope='development_pilot',fit_dir=fit_dir,fit_manifest=fit_manifest)
            engine.add('BUILD-RECEIPT','build_receipt',{'receipt_sha256':s.archive(canonical(built))})
            mirror(engine,budget,'BUILD','BUILD-RECEIPT')
            if built['status']!='completed': raise Blocked('candidate build failed; retain result and stop')
            if set(built['artifacts'])!={'classifier.py','provenance.json'}:raise Blocked('exact classifier/provenance output pair required')
            source=(out/'build/emitted/classifier.py').read_bytes()
            provenance_bytes=(out/'build/emitted/provenance.json').read_bytes()
            if len(provenance_bytes)>HANDOFF_POLICY['provenance_json_bytes']:
                raise Blocked('provenance output exceeds pilot handoff allocation')
            provenance=strict_json(provenance_bytes)
            if len(source)>contract['candidate_source_bytes_cap']:raise Blocked('source exceeds pilot context/source allocation')
            engine.submit(construction['candidate'],source,engine.actor('runner'))
            engine.add('BUILD-PROVENANCE','build_provenance',{'host_fit_manifest':fit_manifest,
                       'runner_provenance_unverified':provenance,'build_receipt_ref':'BUILD-RECEIPT'})

            baseline = ROOT/contract['baseline_stage']
            engine.reserve('RES-STAGE',units(wall_seconds=60),phase='pilot',batch_id='B1')
            budget.reserve('STAGE','artifact_stage',units(wall_seconds=60),phase='pilot',batch='B1',
                           binding={'scope':'development_pilot','baseline_stage':str(baseline)},host=HOST)
            budget.begin('STAGE',HOST)
            staged = validate_stage(baseline)
            elapsed,clock=budget.elapsed_evidence('STAGE')
            budget.settle('STAGE',units(wall_seconds=elapsed),{'stage_sha256':digest(staged),'clock':clock},HOST)
            engine.add('STAGE-RECEIPT','stage_receipt',staged);mirror(engine,budget,'STAGE','STAGE-RECEIPT')

            preflight(engine,source,construction,provenance,fit_manifest,built,staged,limits)
            requested=units(evaluation_requests=recipe['planned_requests'],worker_sessions=recipe['planned_sessions'],wall_seconds=1200)
            # This durable marker precedes feedback. A crash cannot grant a free repeat.
            engine.add('FEEDBACK-COHORT-1','feedback_admission',{'cohort':data['identity'],'recipe_sha256':digest(recipe),'limit':1})
            engine.reserve('RES-COMPARE',requested,phase='pilot',batch_id='B1')
            budget.reserve('COMPARE','development_evaluation',requested,phase='pilot',batch='B1',
                           binding={'recipe_sha256':digest(recipe),'candidate_sha256':digest(source),'baseline':staged},host=HOST)
            budget.begin('COMPARE',HOST)
            engine.add('START-E1','evaluation_started',{'candidate_id':'C1','evaluation_id':'E1','reservation_id':'RES-COMPARE'})
            measured=execute_development(recipe,out/'comparison',data=data,candidate_source=source,
                         baseline_stage=baseline,release=release,resource_guard=budget.execution_guard('COMPARE'),readiness=readiness)
            raw_sha=archive_file(s,out/'comparison/RESULT.json')
            elapsed,clock=budget.elapsed_evidence('COMPARE')
            actual=units(evaluation_requests=measured['attempted_requests'],worker_sessions=measured['attempted_sessions'],wall_seconds=elapsed) if measured['cleanup_confirmed'] and elapsed is not None else None
            budget.settle('COMPARE',actual,{'raw_manifest_sha256':raw_sha,'clock':clock,'cleanup':measured['cleanup_confirmed']},HOST)
            engine.add('COMPARE-RECEIPT','comparison_receipt',{'raw_manifest_sha256':raw_sha})
            mirror(engine,budget,'COMPARE','COMPARE-RECEIPT')
            forward=next((j for j in measured['sessions'] if j['member']=='candidate' and j['name']=='forward'),{})
            predictions=forward.get('predictions',[]);predictions=predictions+[None]*(10000-len(predictions))
            valid=(measured['status']=='COMPLETE_NO_DEFINED_CHECK_FAILURE' and measured['objectives']['candidate']['complete'] and measured['objectives']['cnn']['complete'])
            evaluation={'evaluation_id':'E1','candidate_id':'C1','candidate_sha256':digest(source),'versions':engine.config['versions'],
                  'status':'ok' if valid else 'failed','n':10000,'predictions':predictions,
                  'accuracy':measured['development_accuracy']['candidate'] if valid else None,'source_bytes':len(source),
                  'prediction_median_ms':measured['objectives']['candidate']['primary_value'],
                  'timing_boundary':measured['service_boundary'],'data_sha256':digest(data['identity']),
                  'raw_artifact_sha256':raw_sha,'raw_artifact_format':'chunked-evidence/1','parent_changes':{},
                  'errors':[] if valid else [measured['status']],'admissibility':'pending Researcher source/provenance review',
                  'cnn_comparison':{'accuracy':measured['development_accuracy']['cnn'],'runtime':measured['objectives']['cnn'],
                                    'checkpoint_bytes':(baseline/'weights.pt').stat().st_size,'architecture_bytes':(baseline/'architecture.py').stat().st_size},
                  'candidate_runtime':measured['objectives']['candidate']}
            engine.record_evaluation(evaluation)
            if not valid:engine.anomaly('ANOMALY-E1','E1','Comparison incomplete or a defined history check failed; no performance acceptance.')
            engine.freeze_packet('PACKET-B1','B1')
            packet=freeze_handoff(engine,measured,evaluation,budget.snapshot())
            call('RECORD','record',packet,'CONTROLLER-NOTE-1',120)
            call('REVIEW','pilot_review',packet,'RULING-B1',240)
            summary={'status':'pilot_returned_for_review','model_tasks':budget.snapshot()['charged']['agent_tasks'],
                     'resources':budget.snapshot(),'experiment_scope':'one adaptive-development cohort; no search or final opening',
                     'admissibility_not_inferred':True,'ruling':engine.state['rulings']['RULING-B1'],
                     'admissibility':s.find('ADMISSIBILITY-RULING-B1')['payload']}
            atomic_write(out/'RESULT.json',canonical(summary)+b'\n');return summary
        except BaseException as exc:
            # Any interrupted local job stays held; there is no automatic retry.
            atomic_write(out/'FAILURE.json',canonical({'reason':str(exc),'resources':budget.snapshot(),
                         'engine_resources':engine.resources(),'automatic_retry':False})+b'\n')
            raise
