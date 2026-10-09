"""Schema-bound role returns, mechanically applied to the existing event engine.

This module never launches a model or executes returned source/commands. A host
dispatcher still owns authorization, resource admission and authenticated CLI
receipts. Local fixtures exercise exactly the same schema and engine routing.
"""
import copy
from pathlib import Path

from .common import Blocked, Conflict, atomic_write, canonical, digest, identifier, strict_json, validate
from .engine import SCHEMAS

ROLES = {'plan': 'researcher', 'pilot_plan': 'researcher', 'hypothesis': 'runner', 'build': 'runner',
         'construct': 'runner', 'record': 'controller', 'batch_decisions': 'researcher', 'pilot_review':'researcher'}
MAX_RETURN_BYTES = 2 * 1024 * 1024
FLAT_CLAUDE_TASKS = ('construct', 'record', 'hypothesis')
PILOT_PREDICTION_POLICY = {
    'kind': 'qualitative',
    'numeric_scoring': 'No quantitative resolution rule is qualified for this pilot.',
    'instruction': ('Use kind=qualitative. Set metric_id, quantity, operator, threshold, unit and '
                    'resolution_rule_id to null. Numeric expectations may appear in statement '
                    'as unscored forecasts; do not invent a scoring rule. Keep outcome_ref and effort_ref null.'),
}


def request_envelope(engine, dispatch_id, task, context, target_id, batch_id):
    """Pure shape shared by preflight and actual dispatch."""
    context = copy.deepcopy(context)
    if engine.config['mode'] in ('bounded_pilot','bounded_search') and task in ('pilot_plan', 'hypothesis', 'batch_decisions', 'pilot_review'):
        context['prediction_policy'] = copy.deepcopy(PILOT_PREDICTION_POLICY)
    return {'protocol': 'role-return/1', 'dispatch_id': dispatch_id, 'task': task,
            'actor': engine.actor(ROLES[task]), 'campaign_id': engine.config['campaign_id'],
            'batch_id': batch_id, 'target_id': target_id, 'context': context,
            'active_order': copy.deepcopy(engine.state['orders'].get(batch_id)),
            'versions': copy.deepcopy(engine.config['versions'])}


def obj(properties):
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def expanded_schema(filename):
    """Inline the one project reference for CLI JSON Schema consumption."""
    schema = strict_json((SCHEMAS / filename).read_bytes())
    def expand(value):
        if isinstance(value, dict):
            if '$ref' in value:
                if value['$ref'] != 'decision.schema.json':
                    raise Blocked('unbound schema reference')
                return expanded_schema('decision.schema.json')
            return {k: expand(v) for k, v in value.items()}
        return [expand(v) for v in value] if isinstance(value, list) else value
    return expand(schema)


def body_schema(task):
    text = {'type': 'string'}
    strings = {'type': 'array', 'items': text}
    if task == 'plan':
        return obj({'batch_id': text, 'snapshot_id': text, 'directions': strings,
                    'max_candidates': {'type': 'number'}, 'question': text,
                    'versions': obj({k: text for k in ('contract', 'evaluator', 'data', 'config')})})
    if task == 'pilot_plan':
        return obj({'plan': body_schema('plan'),
                    'readiness': obj({'contract_sha256': text,
                                      'verdict': {'enum': ['accepted', 'changes_required', 'unresolved']},
                                      'justification': text, 'evidence_refs': strings})})
    if task == 'hypothesis':
        return expanded_schema('decision.schema.json')
    if task in ('build', 'construct'):
        return obj({'candidate': obj({**{k: text for k in ('candidate_id', 'decision_id', 'batch_id', 'direction_id',
                                                         'build_recipe', 'retained_information')},
                                      'parents': strings, 'dependencies': strings}), ('source' if task=='build' else 'build_script'): text})
    if task == 'record':
        return obj({'packet_ref': text, 'packet_sha256': text, 'evidence_refs': strings,
                    'scratchpad': {'type': 'string', 'description':
                        'Public factual experiment log for the user: observed measurements, limitations and status. '
                        'This field is not a request for private model reasoning.'}})
    if task == 'batch_decisions':
        return expanded_schema('batch_decisions.schema.json')
    if task == 'pilot_review':
        return obj({'ruling':expanded_schema('batch_decisions.schema.json'),
                    'admissibility':obj({'candidate_id':text,'source_sha256':text,
                                        'verdict':{'enum':['accepted','rejected','unresolved']},
                                        'justification':text,'evidence_refs':strings})})
    raise Blocked('unknown structured task')


def response_body_schema(task, *, bounded_pilot=False):
    schema = body_schema(task)
    if not bounded_pilot or task not in ('hypothesis', 'batch_decisions', 'pilot_review'):
        return schema
    decision = (schema if task == 'hypothesis' else
                schema['properties']['decisions']['items'] if task == 'batch_decisions' else
                schema['properties']['ruling']['properties']['decisions']['items'])
    prediction = decision['properties']['prediction']['properties']
    prediction['kind'] = {'const': 'qualitative'}
    for key in ('metric_id', 'quantity', 'operator', 'threshold', 'unit', 'resolution_rule_id'):
        prediction[key] = {'type': 'null'}
    for key in ('outcome_ref', 'effort_ref'):
        decision['properties'][key] = {'type': 'null'}
    return schema


def prepare(engine, dispatch_id, task, context, *, target_id, batch_id='B1'):
    identifier(dispatch_id); identifier(target_id)
    if task not in ROLES or not isinstance(context, dict):
        raise Blocked('named task and object context required')
    if engine.store.find('ORDER-' + dispatch_id):
        raise Conflict('structured order already exists; inspect it instead of dispatching again')
    context = copy.deepcopy(context)
    if task in ('build','construct'):
        decision = engine.state['decisions'].get(context.get('decision_id'))
        if not decision or decision['actor'] != engine.actor('runner') or decision['batch_id'] != batch_id:
            raise Blocked('build requires an already frozen Runner decision')
        context['frozen_decision'] = copy.deepcopy(decision)
    if task in ('record', 'batch_decisions', 'pilot_review'):
        packet = engine.store.find(context.get('packet_ref'))
        if not packet or packet['kind'] != 'packet' or packet['payload']['batch_id'] != batch_id:
            raise Blocked('checkpoint return requires a frozen batch packet')
        if context.get('packet_sha256') != packet['payload']['artifact_sha256']:
            raise Blocked('packet hash differs from frozen evidence')
        if engine.config['mode'] in ('bounded_pilot','bounded_search'):
            from .pilot_handoff import context_for_role
            context = context_for_role(engine, context)
        else:
            # Fixture/legacy transport retains the complete frozen packet.
            context['packet_markdown'] = engine.store.read_artifact(context['packet_sha256']).decode('utf-8')
            context['controller_notes'] = [copy.deepcopy(e['payload']) for e in engine.store.events
                                           if e['kind'] == 'controller_note' and e['payload']['packet_ref'] == context['packet_ref']]
    request = request_envelope(engine, dispatch_id, task, context, target_id, batch_id)
    if len(canonical(request)) > 256 * 1024:
        raise Blocked('role context exceeds 256 KiB cap; no silent evidence truncation')
    sha = engine.store.archive(canonical(request))
    schema = obj({'dispatch_id': {'const': dispatch_id}, 'request_sha256': {'const': sha},
                  'body': response_body_schema(task, bounded_pilot=engine.config['mode'] in ('bounded_pilot','bounded_search'))})
    engine.add('ORDER-' + dispatch_id, 'structured_order',
               {'request_sha256': sha, 'schema_sha256': engine.store.archive(canonical(schema)),
                'request': request})
    return {'request': request, 'request_sha256': sha, 'schema': schema}


def claude_output_schema(prepared):
    """Flatten construction, recording and prediction at the CLI boundary.

    Long, nested construction replies repeatedly arrived as malformed JSON text
    inside body. Send candidate and build_script as top-level fields instead of
    asking the CLI to encode source inside another body object. The same failure
    occurred for a resumed prediction. Flatten that task too, preserving every
    decision field and nested type. No malformed text is repaired or accepted;
    both wire and engine schemas remain strict.
    """
    schema = copy.deepcopy(prepared['schema'])
    if prepared['request']['task'] in FLAT_CLAUDE_TASKS:
        properties = schema['properties']
        body = properties.pop('body')
        schema = obj({**properties, **body['properties']})
        schema['description'] = ('Flat task reply. Return the requested task fields at the top level, '
                                 'alongside dispatch_id and request_sha256. '
                                 'Do not add a body wrapper or JSON-encode an object as text.')
    return schema


def request_for_claude(prepared, *, cli, session_id, model, max_usd, resume=False):
    """Return argv/stdin for review only. Never invokes the provider.

    No tools are needed: role context comes from the host and the role returns
    structured text. This avoids granting model clients Docker or file authority.
    CLI budget is a list-price guard, not a guarantee of subscription billing.
    """
    import math
    import uuid
    uuid.UUID(session_id)
    actor = prepared['request']['actor']
    if actor['session_id'] != session_id or actor['model_id'] != model or actor['provider'] != 'anthropic':
        raise Blocked('CLI identity differs from bound actor')
    if type(max_usd) not in (int, float) or not math.isfinite(max_usd) or max_usd <= 0:
        raise Blocked('explicit positive list-price cap required')
    argv = [str(cli), '--print', '--resume' if resume else '--session-id', session_id,
            '--model', model, '--safe-mode', '--restricted', '--tools', '',
            '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}', '--no-chrome',
            '--permission-prompts', 'none', '--output-format', 'json',
            '--json-schema', canonical(claude_output_schema(prepared)).decode(), '--max-budget-usd', str(max_usd)]
    encoding = ('For this construction response, return exactly four top-level fields: '
                'dispatch_id, request_sha256, candidate (an object), and build_script '
                '(the Python source string, with ordinary newlines). There is no body field. '
                'Do not JSON-encode candidate or double-encode build_script.\n'
                if prepared['request']['task'] == 'construct' else '')
    if prepared['request']['task'] == 'record':
        encoding = ('For this recording response, return exactly these top-level fields: dispatch_id, '
                    'request_sha256, packet_ref, packet_sha256, evidence_refs, scratchpad. '
                    'There is no body field. scratchpad means a public factual experiment log: '
                    'report supplied observations and limits, not private model reasoning.\n')
    if prepared['request']['task'] == 'hypothesis':
        encoding = ('For this prediction response, return every decision field at the top level, '
                    'alongside dispatch_id and request_sha256. There is no body field. '
                    'Keep actor and prediction as objects and evidence_refs as an array; '
                    'do not JSON-encode objects or arrays as text. Preserve the supplied '
                    'prediction policy and exact decision identity.\n')
    prompt = ('Return only the requested structured result. Use the supplied evidence and role authority. '
              'Source and build recipes are text; execution belongs to the bounded host. '
              'Do not claim a run or access that is absent from the supplied evidence.\n' + encoding +
              canonical({'request': prepared['request'], 'request_sha256': prepared['request_sha256']}).decode())
    return {'argv': argv, 'stdin': prompt, 'invoked': False}


def unwrap_claude(prepared, raw):
    """Verify provider receipt identity before accepting its structured output.

    The caller must obtain raw bytes directly from its admitted CLI subprocess;
    a file supplied by an agent is not an authenticated transport receipt.
    """
    if len(raw) > MAX_RETURN_BYTES:
        raise Blocked('provider return exceeds cap')
    value = strict_json(raw)
    actor = prepared['request']['actor']
    if not isinstance(value, dict) or value.get('type') != 'result' or value.get('subtype') != 'success' or value.get('is_error') is not False:
        raise Blocked('provider did not return a successful result')
    if value.get('session_id') != actor['session_id'] or actor['provider'] != 'anthropic':
        raise Blocked('provider session identity mismatch')
    if set(value.get('modelUsage', {})) != {actor['model_id']} or not isinstance(value.get('usage'), dict):
        raise Blocked('provider model/usage identity unavailable or different')
    result = value.get('structured_output')
    validate(result, claude_output_schema(prepared), SCHEMAS)
    if prepared['request']['task'] in FLAT_CLAUDE_TASKS:
        result = {'dispatch_id': result['dispatch_id'], 'request_sha256': result['request_sha256'],
                  'body': {key: result[key] for key in prepared['schema']['properties']['body']['properties']}}
    validate(result, prepared['schema'], SCHEMAS)
    return canonical(result)


def accept(engine, dispatch_id, raw, *, transport_evidence):
    """Apply an observed return once, retaining malformed or rejected attempts.

    raw is structured_output from an authenticated CLI result or explicit local
    fixture. Transport evidence is supplied by the trusted host, never the model.
    """
    order = engine.store.find('ORDER-' + dispatch_id)
    if not order:
        raise Blocked('return has no frozen order')
    prior = engine.store.find('APPLIED-' + dispatch_id)
    if prior:
        if prior['payload']['return_sha256'] != digest(raw):
            raise Conflict('different bytes for already applied return')
        return prior
    if engine.store.find('FAILED-' + dispatch_id):
        raise Blocked('rejected return retained; no automatic retry')
    if not isinstance(transport_evidence, dict) or transport_evidence.get('kind') not in ('local_fixture', 'verified_host_queue'):
        raise Blocked('trusted transport evidence required')
    request = order['payload']['request']
    if transport_evidence['kind'] == 'local_fixture' and request['actor']['model_id'] != 'fixture-no-model-called':
        raise Blocked('local fixture cannot stand in for a real role')
    if transport_evidence['kind'] == 'verified_host_queue' and not transport_evidence.get('receipt_sha256'):
        raise Blocked('verified host receipt reference missing')
    if len(raw) > MAX_RETURN_BYTES:
        raise Blocked('structured return exceeds cap')
    returned_sha = engine.store.archive(raw)
    engine.add('RETURN-' + dispatch_id, 'structured_return',
               {'return_sha256': returned_sha, 'transport_evidence': transport_evidence})
    try:
        value = strict_json(raw)
        schema = strict_json(engine.store.read_artifact(order['payload']['schema_sha256']))
        validate(value, schema, SCHEMAS)
        if request['versions'] != engine.config['versions']:
            raise Blocked('measurement versions changed since dispatch')
        body, task, actor = value['body'], request['task'], request['actor']
        if actor != engine.actor(ROLES[task]):
            raise Blocked('role binding changed since dispatch')
        if task in ('plan', 'pilot_plan'):
            if task == 'pilot_plan':
                from .pilot_release import require_readiness, require_store_release
                release = require_store_release(engine.store)
                if request['dispatch_id'] not in (('PLAN','PLAN-R1') if release['contract']['version']=='bounded-search/1' else ('PLAN',)) or transport_evidence['kind'] != 'verified_host_queue':
                    raise Blocked('pilot readiness requires the first native Researcher task')
                payload = dict(body['readiness'], dispatch_id=request['dispatch_id'], request_sha256=order['payload']['request_sha256'],
                               provider_receipt_sha256=transport_evidence['receipt_sha256'])
                readiness = {'kind': 'pilot_readiness', 'actor': actor, 'payload': payload}
                require_readiness(release, readiness)
                body = body['plan']
            if body['batch_id'] != request['batch_id'] or body['batch_id'] != request['target_id']:
                raise Blocked('batch differs from requested target')
            engine.open_batch(body, actor)
            if task == 'pilot_plan':
                engine.add('PILOT-READINESS', 'pilot_readiness', payload, actor)
        elif task == 'hypothesis':
            if body['decision_id'] != request['target_id'] or body['batch_id'] != request['batch_id']:
                raise Blocked('decision differs from requested target')
            engine.freeze_decision(body, actor)
        elif task in ('build','construct'):
            candidate = body['candidate']
            if candidate['candidate_id'] != request['target_id'] or candidate['decision_id'] != request['context']['decision_id'] or candidate['batch_id'] != request['batch_id']:
                raise Blocked('submission differs from frozen construction order')
            decision=engine.state['decisions'][candidate['decision_id']]
            if candidate['direction_id']!=decision['direction_id'] or any(p not in engine.state['candidates'] for p in candidate['parents']):
                raise Blocked('construction direction/parents outside frozen decision')
            source = body['source' if task=='build' else 'build_script'].encode('utf-8')
            if not source.strip() or len(source) > 1024 * 1024:
                raise Blocked('source empty or exceeds return source cap')
            if task == 'construct' and engine.config['mode'] in ('bounded_pilot','bounded_search'):
                from .pilot_handoff import check_construction
                check_construction(body)
            if task=='build':engine.submit(candidate, source, actor)
            else:engine.add('CONSTRUCTION-'+request['target_id'], 'construction_recipe', body, actor)
        elif task == 'record':
            context = request['context']
            if body['packet_ref'] != context['packet_ref'] or body['packet_sha256'] != context['packet_sha256']:
                raise Blocked('Controller changed the packet identity')
            if not body['evidence_refs'] or any(ref not in engine.state['event_refs'] for ref in body['evidence_refs']):
                raise Blocked('Controller note has unknown or absent evidence')
            if not body['scratchpad'].strip() or len(body['scratchpad'].encode()) > 65536:
                raise Blocked('Controller scratchpad empty or too large')
            if engine.config['mode'] in ('bounded_pilot','bounded_search'):
                from .pilot_handoff import check_notes
                check_notes(context['controller_notes'] + [body])
            engine.add(request['target_id'], 'controller_note', body, actor)
        elif task == 'batch_decisions':
            if body['batch_id'] != request['batch_id']:
                raise Blocked('ruling names a different batch')
            engine.accept_batch_response(request['target_id'], body, actor)
        elif task == 'pilot_review':
            ruling, assessment=body['ruling'],body['admissibility']
            candidate=engine.state['candidates'].get(assessment['candidate_id'])
            if not candidate or candidate['source_sha256']!=assessment['source_sha256'] or ruling['batch_id']!=request['batch_id']:
                raise Blocked('pilot review candidate/batch mismatch')
            if not assessment['justification'].strip() or not assessment['evidence_refs'] or any(x not in engine.state['event_refs'] for x in assessment['evidence_refs']):
                raise Blocked('admissibility ruling lacks identified evidence')
            engine.accept_batch_response(request['target_id'],ruling,actor)
            engine.add('ADMISSIBILITY-'+request['target_id'],'admissibility_ruling',assessment,actor)
    except (Blocked, ValueError, TypeError, KeyError) as exc:
        engine.add('FAILED-' + dispatch_id, 'structured_failure',
                   {'return_sha256': returned_sha, 'reason': str(exc)})
        raise Blocked(str(exc)) from exc
    event = engine.add('APPLIED-' + dispatch_id, 'structured_applied',
                       {'return_sha256': returned_sha, 'target_id': request['target_id'], 'task': task})
    refresh_scratchpad(engine)
    return event


def refresh_scratchpad(engine):
    """Disposable Markdown view; the Controller's attributed notes remain in the log."""
    lines = ['# Controller scratchpad', '', 'Rebuilt from recorded Controller notes. Scientific authority remains with the Researcher.', '']
    for event in engine.store.events:
        if event['kind'] == 'controller_note':
            p = event['payload']
            lines += ['## ' + event['event_id'], '', p['scratchpad'], '',
                      'Packet: ' + p['packet_ref'] + ' (`' + p['packet_sha256'] + '`).',
                      'Evidence: ' + ', '.join(p['evidence_refs']) + '.', '']
    path = engine.store.root / 'workspaces/controller/SCRATCHPAD.md'
    atomic_write(path, ('\n'.join(lines) + '\n').encode())
    return path
