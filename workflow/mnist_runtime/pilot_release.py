"""Check an exact user-released pilot package before data execution or model use.

Release records come from the trusted host/user, never from model output. This
guards accidental execution; it does not defend against the host administrator.
"""
from pathlib import Path
import math
import os
from .common import Blocked, canonical, digest, strict_json

ROOT = Path(__file__).resolve().parents[3]
READINESS_POLICY = {'first_dispatch_id': 'PLAN', 'first_task': 'pilot_plan',
                    'before_data_load': True, 'scope': 'bounded_pilot_only'}


def consume_release(release, output):
    """Claim once across output directories; even an interrupted claim stays spent.

    This is a trusted-host crash/replay guard, not protection from an administrator
    deleting records. There is deliberately no reset or caller-selected registry.
    """
    require_release(release)
    registry = ROOT / 'mnist_search/coordination/pilot_releases'
    registry.mkdir(parents=True, exist_ok=True)
    contract_sha = digest(release['contract'])
    record = {'contract_sha256': contract_sha, 'release_sha256': digest(release),
              'output': str(Path(output).resolve()), 'state': 'consumed_before_data_load',
              'automatic_retry': False}
    path = registry / (contract_sha + '.json')
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise Blocked('pilot release already consumed; a new output directory cannot reuse it') from exc
    # Never delete an incomplete record on failure: its existence is the claim.
    with os.fdopen(fd, 'wb') as stream:
        stream.write(canonical(record) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(registry, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return record


def require_release(release, *, project=ROOT):
    if not isinstance(release, dict) or set(release) != {'contract', 'approval'}:
        raise Blocked('exact pilot contract and user approval required')
    contract, approval = release['contract'], release['approval']
    if contract.get('version') == 'bounded-search/1':
        from .experiment_release import require_release as require_experiment
        return require_experiment(release, project=project)
    if contract.get('version') not in ('bounded-pilot/1', 'bounded-pilot/2') or contract.get('scope') != 'development_pilot':
        raise Blocked('not a bounded development pilot')
    if approval != {'actor_role': 'user', 'release': 'bounded_real_pilot', 'contract_sha256': digest(contract)}:
        raise Blocked('pilot approval absent or applies to different bytes')
    expected_gates = {'role_container_boundary', 'final_key_custody', 'scientific_contract_review',
                      'native_structured_transport', 'candidate_admissibility_process'}
    inline = contract['version'] == 'bounded-pilot/2'
    expected = {k: (False if inline and k in ('scientific_contract_review', 'native_structured_transport') else True)
                for k in expected_gates}
    if set(contract.get('gates', {})) != expected_gates or any(contract['gates'][k] is not expected[k] for k in expected_gates):
        raise Blocked('pilot readiness gates incomplete')
    if inline and (contract.get('readiness_policy') != READINESS_POLICY or contract.get('scientific_review_ref') is not None):
        raise Blocked('pilot must obtain fresh Researcher readiness in its first task')
    if (not inline and not contract.get('scientific_review_ref')) or not contract.get('custody_ref'):
        raise Blocked('review and custody evidence must be identified')
    caps = contract.get('caps', {})
    if set(caps) != {'agent_tasks', 'evaluation_requests', 'worker_sessions', 'wall_seconds', 'feedback_cohorts'}:
        raise Blocked('complete pilot resource caps required')
    if any(type(v) is not int or v <= 0 for v in caps.values()) or caps['feedback_cohorts'] != 1:
        raise Blocked('one feedback cohort and positive finite caps required')
    if contract.get('automatic_retry') is not False or contract.get('main_search_tasks') != 0 or contract.get('final_opening_enabled') is not False:
        raise Blocked('pilot must not enable retries, search or final opening')
    if contract.get('cost_policy', {}).get('additional_charges_authorized') is not False:
        raise Blocked('this pilot must preserve the current no-extra-spending direction')
    if contract.get('cost_policy', {}).get('included_usage_confirmed') is not True:
        raise Blocked('subscription allowance/extra-usage setting not confirmed; do not assume free calls')
    costs=contract['cost_policy'].get('session_list_usd_caps',{})
    if set(costs)!={'runner','researcher','controller'} or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in costs.values()):
        raise Blocked('explicit finite session list-price caps required')
    if type(contract.get('candidate_source_bytes_cap')) is not int or not 0<contract['candidate_source_bytes_cap']<=131072:
        raise Blocked('pilot source must fit the allocated review context')
    from .pilot_handoff import HANDOFF_POLICY
    if contract.get('handoff_policy') != HANDOFF_POLICY:
        raise Blocked('pilot handoff allocation differs from the implemented policy')
    files = contract.get('files', {})
    if not files: raise Blocked('frozen implementation manifest required')
    if inline:
        evidence = contract.get('readiness_evidence_files')
        if (not isinstance(evidence, list) or not evidence or len(evidence) != len(set(evidence))
                or any(path not in files for path in evidence)):
            raise Blocked('first Researcher task requires frozen review evidence')
    project = Path(project).resolve()
    for relative, sha in files.items():
        path = project / relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or path.is_symlink() or project not in path.resolve().parents:
            raise Blocked('release file escapes project')
        if digest(path.read_bytes()) != sha:
            raise Blocked('released implementation changed: ' + relative)
    baseline=contract.get('baseline_stage','')
    if not baseline or any(str(Path(baseline)/name) not in files for name in ('STAGE.json','architecture.py','weights.pt')):
        raise Blocked('baseline source, checkpoint and stage manifest must all be frozen')
    return digest(release)


def require_store_release(store):
    event = store.find('PILOT-RELEASE')
    if not event: raise Blocked('pilot resource store lacks a released contract')
    require_release(event['payload'])
    return event['payload']


def require_readiness(release, event):
    """V2 releases admit only PLAN until its authenticated Researcher accepts.

    This accepts scientific readiness for this bounded attempt. Full live-loop
    qualification is an outcome of the pilot, never an assumed prerequisite.
    """
    contract = release['contract']
    if contract['version'] == 'bounded-pilot/1':
        return
    if not event or event.get('kind') != 'pilot_readiness' or event.get('actor') != contract['actors']['researcher']:
        raise Blocked('authenticated Researcher readiness is required before data or further work')
    p = event['payload']
    if (p.get('contract_sha256') != digest(contract) or p.get('verdict') != 'accepted'
            or p.get('dispatch_id') not in (('PLAN', 'PLAN-R1') if contract['version']=='bounded-search/1' else ('PLAN',)) or not p.get('justification', '').strip()
            or set(p.get('evidence_refs', [])) != set(contract['readiness_evidence_files'])):
        raise Blocked('Researcher readiness rejected, incomplete or bound to a different contract')
    for key in ('provider_receipt_sha256', 'request_sha256'):
        value = p.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise Blocked('Researcher readiness lacks identified native transport evidence')


def require_store_readiness(store):
    release = require_store_release(store)
    require_readiness(release, store.find('PILOT-READINESS'))
    return release
