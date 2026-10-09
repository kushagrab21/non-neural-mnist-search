"""End-to-end structured workflow with explicit local fixtures, zero LLM calls."""
from pathlib import Path

from .adapters import invoke_stub
from .common import Blocked, atomic_write, canonical, digest
from .demo import fixture_config
from .engine import Engine
from .evaluator import evaluate
from .role_protocol import accept, prepare, refresh_scratchpad
from .store import Store


def run(root):
    root = Path(root).resolve()
    if root.exists():
        raise Blocked('new demo destination required')
    with Store(root) as store:
        engine = Engine(store)
        config = fixture_config()
        # All three proposed roles are Claude; these are still labelled fixtures.
        for role in ('runner', 'researcher', 'controller'):
            config['actors'][role]['provider'] = 'anthropic'
        engine.initialize(config)

        def step(dispatch, task, context, target, fixture_task=None, fixture_payload=None):
            prepared = prepare(engine, dispatch, task, context, target_id=target)
            role = prepared['request']['actor']['role']
            body = invoke_stub(engine, 'FIXTURE-' + dispatch, role, fixture_task or task,
                               context if fixture_payload is None else fixture_payload)
            raw = canonical({'dispatch_id': dispatch, 'request_sha256': prepared['request_sha256'], 'body': body})
            accept(engine, dispatch, raw,
                   transport_evidence={'kind': 'local_fixture', 'receipt_ref': 'RECEIPT-FIXTURE-' + dispatch, 'model_calls': 0})

        step('PLAN', 'plan', {}, 'B1')
        for number, variant, direction in [(1, 'small', 'DIR-A'), (2, 'invalid', 'DIR-B')]:
            step('HYP-' + str(number), 'hypothesis',
                 {'decision_id': 'D'+str(number), 'direction_id': direction, 'evidence_refs': ['CAMPAIGN']}, 'D'+str(number))
            step('BUILD-' + str(number), 'build',
                 {'candidate_id': 'C'+str(number), 'decision_id': 'D'+str(number),
                  'direction_id': direction, 'variant': variant}, 'C'+str(number))
            evaluate(engine, 'C'+str(number), 'E'+str(number))
        packet = engine.freeze_packet('PACKET-B1', 'B1')
        context = {'packet_ref': 'PACKET-B1', 'packet_sha256': store.find('PACKET-B1')['payload']['artifact_sha256']}
        step('RECORD', 'record', context, 'CONTROLLER-NOTE-1', fixture_task='echo',
             fixture_payload=dict(context, evidence_refs=['E1', 'E2'],
                                  scratchpad='C1 completed. C2 failed with an invalid digit; failure remains in the packet. Researcher decision pending.'))
        step('REVIEW', 'batch_decisions', context, 'RULING-B1',
             fixture_payload=dict(packet, evidence_refs=['E1', 'E2']))
        if engine.state['evaluations']['E1']['accuracy'] != 1 or engine.state['evaluations']['E2']['status'] != 'failed':
            raise Blocked('known fixture evaluation outcomes differ')
        if engine.state['batch_states']['B1'] != 'closed' or engine.resources()['unresolved']:
            raise Blocked('workflow did not close cleanly')
        state_hash = digest(engine.state)
        scratchpad = refresh_scratchpad(engine).read_bytes()
    with Store(root) as store:
        engine = Engine(store)
        if digest(engine.state) != state_hash or refresh_scratchpad(engine).read_bytes() != scratchpad:
            raise Blocked('restart changed state or Controller notes')
        result = {'status': 'PASS', 'scope': 'local_structured_role_fixtures', 'model_calls': 0,
                  'live_cli_integration_qualified': False, 'real_data_inference': False,
                  'sealed_data_access': False, 'events': len(store.events),
                  'state_sha256_before_and_after_restart': state_hash,
                  'candidates': {k: v['status'] for k, v in engine.state['evaluations'].items()},
                  'batch_state': engine.state['batch_states']['B1'],
                  'remaining_active_directions': engine.state['active'],
                  'scratchpad': str(root/'workspaces/controller/SCRATCHPAD.md'),
                  'resources': engine.resources()}
        atomic_write(root/'reports/RESULT.json', canonical(result)+b'\n')
    return result
