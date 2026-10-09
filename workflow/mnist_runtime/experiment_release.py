"""Fresh finite adaptive-development authorization; final data never included."""
import math, os
from pathlib import Path
from .common import Blocked, canonical, digest
from .retry_policy import POLICY
ROOT = Path(__file__).resolve().parents[3]


def require_release(release, *, project=ROOT):
    if not isinstance(release, dict) or set(release) != {'contract','approval'}:
        raise Blocked('complete adaptive experiment release required')
    c=release['contract'];a=release['approval']
    if c.get('version')!='bounded-search/1' or c.get('scope')!='adaptive_development':
        raise Blocked('unknown adaptive contract')
    if a!={'actor_role':'user','release':'bounded_adaptive_experiment','contract_sha256':digest(c)}:
        raise Blocked('adaptive authorization differs from contract')
    if c.get('final_opening_enabled') is not False or c.get('retry_policy')!=POLICY:
        raise Blocked('final opening or unbounded retry is not authorized')
    caps=c.get('caps',{})
    if set(caps)!={'agent_tasks','evaluation_requests','worker_sessions','wall_seconds','feedback_cohorts'} or any(type(v) is not int or v<=0 for v in caps.values()):
        raise Blocked('finite complete adaptive caps required')
    if caps['feedback_cohorts']!=c.get('max_batches') or caps['evaluation_requests']!=72248*c['max_batches'] or caps['worker_sessions']!=44*c['max_batches']:
        raise Blocked('comparison resources must cover each allowed candidate')
    if c.get('max_batch_candidates')!=1 or caps['agent_tasks']!=5*c['max_batches']+POLICY['max_extra_tasks']:
        raise Blocked('one candidate per adaptive checkpoint and counted retries required')
    costs=c.get('cost_policy',{})
    if costs.get('additional_charges_authorized') is not False or costs.get('included_usage_confirmed') is not True:
        raise Blocked('included subscription usage only required')
    sc=costs.get('session_list_usd_caps',{})
    if set(sc)!={'researcher','runner','controller'} or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in sc.values()):
        raise Blocked('finite role accounting ceilings required')
    if sum(sc.values())!=costs.get('total_list_price_ceiling_usd'):
        raise Blocked('role cost limits must sum to accounting ceiling')
    if set(c.get('actors',{}))!={'researcher','runner','controller','user'}:
        raise Blocked('separate three-role actor bindings required')
    if len({c['actors'][r]['session_id'] for r in sc})!=3:
        raise Blocked('roles must have distinct sessions')
    for r in sc:
        if c['actors'][r]['role']!=r or c['actors'][r]['provider']!='anthropic' or not c['actors'][r]['model_id']:
            raise Blocked('explicit Claude role identities required')
    if not 0<c.get('candidate_source_bytes_cap',0)<=65536:
        raise Blocked('source must fit allocated evidence context')
    files=c.get('files',{});evidence=c.get('readiness_evidence_files',[])
    if not files or not evidence or any(p not in files for p in evidence):
        raise Blocked('frozen implementation and scientific evidence required')
    project=Path(project).resolve()
    for name,sha in files.items():
        p=project/name
        if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or project not in p.resolve().parents or digest(p.read_bytes())!=sha:
            raise Blocked('adaptive frozen input changed: '+name)
    for name in ('STAGE.json','architecture.py','weights.pt'):
        if str(Path(c['baseline_stage'])/name) not in files:
            raise Blocked('baseline inputs must be frozen')
    return digest(release)


def consume_release(release, output):
    require_release(release)
    registry=ROOT/'mnist_search/coordination/experiment_releases';registry.mkdir(parents=True,exist_ok=True)
    record={'contract_sha256':digest(release['contract']),'release_sha256':digest(release),'output':str(Path(output).resolve()),'state':'consumed_before_first_dispatch'}
    p=registry/(record['contract_sha256']+'.json')
    try:fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError as exc:raise Blocked('experiment release already consumed; no blind relaunch') from exc
    with os.fdopen(fd,'wb') as f:f.write(canonical(record)+b'\n');f.flush();os.fsync(f.fileno())
    return record
