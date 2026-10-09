"""A preallocated pilot review view; complete raw evidence stays in the archive.

Known source/provenance bytes are measured before feedback. Future measurements
and the single Controller note have explicit, separately enforced allocations.
These are pilot transport limits, not professor admissibility restrictions.
"""
import copy
from collections import Counter
from dataclasses import asdict

from .common import Blocked, canonical, digest, strict_json
from .artifact_jobs import build_limits
from .common_comparison import comparison_limits
from .docker_worker import IMAGE_RECORD

HANDOFF_POLICY = {
    'version': 'pilot-handoff/1', 'request_bytes': 256 * 1024,
    'build_script_bytes': 32 * 1024, 'construction_json_bytes': 64 * 1024,
    'provenance_json_bytes': 8 * 1024, 'feedback_json_bytes': 32 * 1024,
    'controller_notes_json_bytes': 8 * 1024,
    'scope': 'pilot transport allocation only; not a professor restriction',
}


def runner_limits(recipe):
    images = strict_json(IMAGE_RECORD.read_bytes())
    observed = strict_json((IMAGE_RECORD.parent / 'RUNTIME_SMOKE.json').read_bytes())
    runtime = {}
    for role in ('candidate', 'baseline'):
        if observed[role]['image_id'] != images[role]['image_id']:
            raise Blocked('runtime version evidence differs from the pinned image')
        runtime[role] = {'image_id': images[role]['image_id'],
                         'versions': {k: v for k, v in observed[role]['runtime'].items()
                                      if k != 'files'}}
    return {'build': build_limits(600), 'inference': asdict(comparison_limits()),
            'inference_container': {'cpus': 1, 'pids': 32, 'network': 'none',
                                    'memory_cap_applies_to': 'whole container including imports'},
            'largest_session_requests_including_warmups': recipe['largest_session_requests'],
            'comparison_execution_seconds': 1180, 'comparison_allocation_seconds': 1200,
            'runtime': runtime, 'handoff': copy.deepcopy(HANDOFF_POLICY)}


def check_construction(body):
    if (len(body['build_script'].encode()) > HANDOFF_POLICY['build_script_bytes'] or
            len(canonical(body)) > HANDOFF_POLICY['construction_json_bytes']):
        raise Blocked('construction exceeds the pilot handoff allocation')


def check_notes(notes):
    if len(notes) > 1 or len(canonical(notes)) > HANDOFF_POLICY['controller_notes_json_bytes']:
        raise Blocked('Controller note exceeds its reserved pilot handoff space')


def preflight(engine, source, construction, provenance, fit_manifest, built, staged, limits, *, history=None):
    """Freeze actual known evidence and reserve maximum future payloads, before feedback."""
    from .role_protocol import request_envelope
    search=engine.config['mode']=='bounded_search'
    candidate=construction['candidate'];cid=candidate['candidate_id'];batch=candidate['batch_id']
    build_ref='BUILD-RECEIPT-'+cid if search else 'BUILD-RECEIPT'
    provenance_ref='BUILD-PROVENANCE-'+cid if search else 'BUILD-PROVENANCE'
    feedback_ref='FEEDBACK-'+cid if search else 'FEEDBACK-COHORT-1'
    check_construction(construction)
    if len(canonical(provenance)) > HANDOFF_POLICY['provenance_json_bytes']:
        raise Blocked('provenance exceeds the pilot handoff allocation')
    if engine.store.find(feedback_ref):
        raise Blocked('handoff must be checked before feedback admission')
    # These values are loaded from host records, not attributed to the Runner.
    build_event = engine.store.find(build_ref)
    if (not build_event or strict_json(engine.store.read_artifact(build_event['payload']['receipt_sha256'])) != built
            or built.get('fit_export') != fit_manifest):
        raise Blocked('host build/fit evidence differs from the archived receipt')
    if engine.store.find('STAGE-RECEIPT')['payload'] != staged:
        raise Blocked('baseline provenance differs from the host stage record')
    static = {'candidate_source': source.decode(), 'construction': construction,
              'runner_provenance_unverified': provenance,
              'host_fit_manifest': fit_manifest, 'host_build_receipt': built,
              'baseline_provenance': staged, 'execution_limits': limits,
              'candidate_record': engine.state['candidates'][cid],
              'decisions': copy.deepcopy(engine.state['decisions']),
              'campaign_contract': engine.store.find('PILOT-RELEASE')['payload']['contract'],
              'active_direction_snapshot_id': engine.state['snapshot_id'],
              'active_directions': list(engine.state['active']),
              'evidence_refs': ['CAMPAIGN', batch, candidate['decision_id'], 'CONSTRUCTION-'+cid, cid,
                                provenance_ref, build_ref, 'STAGE-RECEIPT'],
              'view_policy': 'Bounded review view. Full packet, per-image predictions, timing observations and failure details remain losslessly archived. This view supplies every observed failure category and dependence-session count; previews are labelled.',
              'controller_instruction': 'Return exactly one public factual experiment note about the supplied observations, limits and status. The scratchpad field is this user-facing log, not private model reasoning. The canonical JSON list containing the entire note body must fit controller_notes_json_bytes; do not reproduce the source or raw predictions.'}
    if history is not None:static['adaptive_history']=history
    base = dict(static, packet_ref='PACKET-'+batch, packet_sha256='0' * 64)
    # A canonical JSON value of N bytes replaces a placeholder of N+2 bytes.
    # Nested strings are already counted with their actual JSON escaping.
    base['feedback_summary'] = 'x' * HANDOFF_POLICY['feedback_json_bytes']
    base['controller_notes'] = 'x' * HANDOFF_POLICY['controller_notes_json_bytes']
    sizes = {}
    for job, task, target in [('RECORD', 'record', 'CONTROLLER-NOTE-1'), ('REVIEW', 'pilot_review', 'RULING-B1')]:
        sizes[job] = len(canonical(request_envelope(engine, job, task, base, target, batch)))
    if max(sizes.values()) > HANDOFF_POLICY['request_bytes']:
        raise Blocked('pilot handoff cannot fit before feedback; source/build/provenance must be smaller')
    payload = {'static_sha256': engine.store.archive(canonical(static)),
               'policy': HANDOFF_POLICY, 'worst_case_request_bytes': sizes,
               'active_order_sha256': digest(engine.state['orders'][batch]),
               'versions': copy.deepcopy(engine.config['versions'])}
    engine.add('HANDOFF-PREFLIGHT-'+batch, 'handoff_preflight', payload)
    return payload


def _runtime_summary(objective):
    # Raw timing arrays stay in the archived comparison. All block statistics
    # and completeness flags are presented, including failed/partial blocks.
    return {k: copy.deepcopy(objective.get(k)) for k in
            ('metric_id', 'primary_axis', 'primary_value', 'complete', 'block_median_range',
             'range_semantics', 'comparison_group_sha256', 'overhead_subtracted')} | {
        'blocks': [{k: copy.deepcopy(b.get(k)) for k in ('block', 'complete', 'summary')}
                   for b in objective.get('blocks', [])]}


def feedback_summary(measured, evaluation, resources):
    """Bound variable diagnostics, retaining category counts and raw identities.

    The frozen comparison has 44 sessions and stops after its first exception.
    Neither source text, per-image arrays nor arbitrary exception text are copied
    unboundedly into this future slot.
    """
    counts = Counter((str(x['member']), str(x['session'])) for x in measured['dependence'])
    failures = measured['failures']
    if len(counts) > 44 or len(failures) > 1:
        raise Blocked('comparison diagnostics differ from the frozen pilot recipe')
    view = {'evaluation_id': evaluation['evaluation_id'], 'candidate_id': evaluation['candidate_id'], 'status': evaluation['status'],
            'comparison_status': measured['status'], 'n': evaluation['n'],
            'candidate_source_sha256': evaluation['candidate_sha256'],
            'source_bytes': evaluation['source_bytes'], 'accuracy': measured['development_accuracy'],
            'service_boundary': measured['service_boundary'],
            'runtime': {m: _runtime_summary(measured['objectives'][m]) for m in ('candidate', 'cnn')},
            'attempted_requests': measured['attempted_requests'],
            'attempted_sessions': measured['attempted_sessions'],
            'cleanup_confirmed': measured['cleanup_confirmed'],
            'dependence_count': len(measured['dependence']),
            'dependence_by_session': [{'member': m[:32], 'session': s[:64], 'count': n}
                                      for (m, s), n in sorted(counts.items())],
            'failures': [{'type': str(f.get('type'))[:64], 'code': str(f.get('code'))[:64],
                          'member': str(f.get('member'))[:32], 'session': str(f.get('session'))[:64],
                          'detail_preview': str(f.get('detail'))[:256], 'full_record_sha256': digest(f),
                          'preview_only': True} for f in failures],
            'raw_comparison_manifest_sha256': evaluation['raw_artifact_sha256'],
            'prediction_count': len(evaluation['predictions']),
            'predictions_sha256': digest(evaluation['predictions']),
            'raw_arrays_inlined': False,
            'resource_totals': {k: resources[k] for k in ('charged', 'held', 'remaining', 'unresolved', 'overruns', 'deadline_violations')},
            'evidence_refs': ['E1', 'START-E1', 'COMPARE-RECEIPT', 'USAGE-COMPARE', 'PACKET-B1'] +
                             ([] if evaluation['status'] == 'ok' else ['ANOMALY-E1'])}
    if len(canonical(view)) > HANDOFF_POLICY['feedback_json_bytes']:
        raise Blocked('feedback view violates its preallocated size bound')
    return view


def freeze(engine, measured, evaluation, resources):
    batch=engine.state['current_batch']
    pre = engine.store.find('HANDOFF-PREFLIGHT-'+batch)['payload']
    if pre['active_order_sha256'] != digest(engine.state['orders'][batch]) or pre['versions'] != engine.config['versions']:
        raise Blocked('handoff identity changed after preflight')
    context = strict_json(engine.store.read_artifact(pre['static_sha256']))
    context.update(packet_ref='PACKET-'+batch, packet_sha256=engine.store.find('PACKET-'+batch)['payload']['artifact_sha256'],
                   feedback_summary=feedback_summary(measured, evaluation, resources))
    if engine.config['mode']=='bounded_search':
        cid=evaluation['candidate_id'];eid=evaluation['evaluation_id']
        context['feedback_summary']['evidence_refs']=[eid,'START-'+eid,'COMPARE-RECEIPT-'+cid,'USAGE-COMPARE-'+cid,'PACKET-'+batch]+([] if evaluation['status']=='ok' else ['ANOMALY-'+eid])
    engine.add('PILOT-HANDOFF-'+batch, 'pilot_handoff', {'context_sha256': engine.store.archive(canonical(context))})
    return {'packet_ref': context['packet_ref'], 'packet_sha256': context['packet_sha256'],
            'pilot_handoff_ref': 'PILOT-HANDOFF-'+batch}


def context_for_role(engine, incoming):
    event = engine.store.find(incoming.get('pilot_handoff_ref'))
    if not event or event['kind'] != 'pilot_handoff':
        raise Blocked('pilot checkpoint requires its preflighted host evidence view')
    context = strict_json(engine.store.read_artifact(event['payload']['context_sha256']))
    if any(context[k] != incoming[k] for k in ('packet_ref', 'packet_sha256')):
        raise Blocked('pilot handoff packet identity changed')
    notes = [copy.deepcopy(e['payload']) for e in engine.store.events
             if e['kind'] == 'controller_note' and e['payload']['packet_ref'] == context['packet_ref']]
    check_notes(notes)
    context['controller_notes'] = notes
    return context
