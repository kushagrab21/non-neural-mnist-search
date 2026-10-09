"""Parent-owned aggregate admission for native tasks and synthetic evaluations.

No provider tokens or subscription dollars are inferred from task counts. The
caller holding the Store writer capability is trusted; this is not OS isolation.
"""
from __future__ import annotations
import copy
import time
import subprocess
import sys
from datetime import datetime, timezone
from .common import Blocked, Conflict, amount, canonical, digest, identifier
from .elapsed import observation, duration, remaining

DIMENSIONS={'agent_tasks','evaluation_requests','worker_sessions','wall_seconds'}
COUNT_DIMENSIONS=DIMENSIONS-{'wall_seconds'}


def quantities(values):
    if not isinstance(values,dict) or set(values)!=DIMENSIONS:
        raise Blocked('resource dimensions must be exact')
    for key,value in values.items():
        amount(value)
        if key in COUNT_DIMENSIONS and type(value) is not int:
            raise Blocked('resource counts must be integers')
    return values


def units(**kwargs):
    return {key:kwargs.get(key,0) for key in sorted(DIMENSIONS)}


def boot_identity():
    if sys.platform!='darwin':return None
    try:
        result=subprocess.run(['/usr/sbin/sysctl','-n','kern.boottime'],capture_output=True,timeout=2)
    except (OSError,subprocess.TimeoutExpired):return None
    return digest(result.stdout) if result.returncode==0 and result.stdout.strip() else None


class Budget:
    def __init__(self,store): self.store=store

    @property
    def authority(self):
        event=self.store.find('RESOURCE-AUTHORITY')
        if not event:raise Blocked('resource authority absent')
        return event['payload']

    def _host(self,actor):
        if actor!=self.store.find('RESOURCE-AUTHORITY')['actor']:raise Blocked('resource host mismatch')

    def initialize(self,policy,host,*,release=None):
        if set(policy)!={'mode','authorization_id','deadline','campaign','phases','batches'} or policy['mode'] not in ('synthetic_preparation','bounded_pilot','bounded_search'):
            raise Blocked('explicit preparation or released pilot resource policy required')
        if policy['mode'] in ('bounded_pilot','bounded_search'):
            from .pilot_release import require_release
            require_release(release)
            caps=release['contract']['caps']
            if any(policy['campaign'][k]>caps[k] for k in DIMENSIONS):raise Blocked('resource authority exceeds released pilot')
            self.store.append('PILOT-RELEASE','pilot_release',host,release)
        elif release is not None:raise Blocked('synthetic policy cannot use a real release')
        identifier(policy['authorization_id']);deadline=datetime.fromisoformat(policy['deadline'])
        if deadline.tzinfo is None:raise Blocked('deadline needs timezone')
        quantities(policy['campaign'])
        if not policy['phases'] or not policy['batches']:raise Blocked('phase and batch allocations required')
        for name,cap in policy['phases'].items():identifier(name);quantities(cap)
        for key,cap in policy['campaign'].items():
            if sum(v[key] for v in policy['phases'].values())>cap:raise Blocked('phase allocations exceed campaign')
        for name,batch in policy['batches'].items():
            identifier(name)
            if set(batch)!={'phase','limits'} or batch['phase'] not in policy['phases']:raise Blocked('unbound batch phase')
            quantities(batch['limits'])
        for phase,cap in policy['phases'].items():
            for key,value in cap.items():
                if sum(b['limits'][key] for b in policy['batches'].values() if b['phase']==phase)>value:
                    raise Blocked('batch allocations exceed phase')
        prior=self.store.find('RESOURCE-AUTHORITY')
        if prior:
            if {k:v for k,v in prior['payload'].items() if k!='clock_origin'}!=policy or prior['actor']!=host:raise Conflict('resource authority differs')
            return prior
        origin=observation(boot_identity())
        if not origin['boot_identity']:raise Blocked('resource boot identity unavailable')
        return self.store.append('RESOURCE-AUTHORITY','resource_authority',host,dict(policy,clock_origin=origin))

    def state(self):
        rows={}
        for event in self.store.events:
            p=event['payload'];kind=event['kind']
            if kind=='resource_reserved':rows[p['reservation_id']]=dict(copy.deepcopy(p),state='reserved',actual=None)
            elif kind in ('resource_begun','resource_unknown','resource_settled','resource_cancelled'):
                row=rows[p['reservation_id']];row.update(copy.deepcopy(p))
                row['state']={'resource_begun':'begun','resource_unknown':'unknown','resource_settled':'settled','resource_cancelled':'cancelled'}[kind]
        return rows

    def remaining_seconds(self):
        return remaining(self.authority.get('clock_origin'),observation(boot_identity()),self.authority['deadline'])

    def _record_stop(self,reason,evidence):
        prior=self.store.find('RESOURCE-DEADLINE-STOP')
        if prior:return prior
        return self.store.append('RESOURCE-DEADLINE-STOP','resource_deadline_stop',self.store.find('RESOURCE-AUTHORITY')['actor'],{'reason':reason,'evidence':evidence,'resumption':'new prospective authority required; no clock correction or fresh reservation ID resumes this authority'})

    def _live(self):
        if self.authority['mode'] in ('bounded_pilot','bounded_search'):
            from .pilot_release import require_store_release
            require_store_release(self.store)
        if self.store.find('RESOURCE-DEADLINE-STOP') or self.snapshot()['deadline_violations']:
            raise Blocked('resource authority stopped by recorded deadline expiry')
        current=observation(boot_identity())
        left=remaining(self.authority.get('clock_origin'),current,self.authority['deadline'])
        if left<=0:
            self._record_stop('authority_deadline_at_admission',{'clock':current,'remaining_seconds':left})
            raise Blocked('resource deadline exhausted')
        return left

    def elapsed_evidence(self,reservation_id):
        row=self.state()[reservation_id]
        try:end=observation(boot_identity())
        except (Blocked,OSError) as exc:return None,{'error':str(exc),'clock_status':'unavailable'}
        elapsed=duration(row.get('started_clock'),end)
        return elapsed,{'start':row.get('started_clock'),'end':end,'elapsed_seconds':elapsed,'clock_status':'known' if elapsed is not None else 'unknown'}

    def execution_guard(self,reservation_id):
        row=self.state()[reservation_id]
        start=row.get('started_clock');authority=self.authority
        def guard():
            # Executing process cannot survive an OS reboot. Re-check boot at
            # final intake; avoid a host subprocess at every image checkpoint.
            current=observation(start.get('boot_identity') if isinstance(start,dict) else None)
            elapsed=duration(start,current)
            left=remaining(authority.get('clock_origin'),current,authority['deadline'])
            if elapsed is None:raise Blocked('resource clock evidence lost during execution')
            if left<=0 or elapsed>=row['requested']['wall_seconds']:
                self._record_stop('deadline_during_evaluation',{'resource_clock':current,'elapsed_seconds':elapsed,'authority_seconds_left':left,'task_seconds':row['requested']['wall_seconds']})
                exc=Blocked('sleep-inclusive resource deadline exhausted during evaluation')
                exc.code='RESOURCE_AUTHORITY_TIMEOUT' if left<=0 else 'RESOURCE_TASK_TIMEOUT'
                exc.evidence={'resource_clock':current,'elapsed_seconds':elapsed,'authority_seconds_left':left,'task_seconds':row['requested']['wall_seconds']}
                raise exc
        return guard

    def snapshot(self):
        rows=self.state();charged=units();held=units()
        for row in rows.values():
            if row['state']=='cancelled':continue
            target=charged if row['state']=='settled' else held
            for key,val in (row['actual'] if row['state']=='settled' else row['requested']).items():target[key]+=val
        return {'charged':charged,'held':held,'remaining':{k:self.authority['campaign'][k]-charged[k]-held[k] for k in DIMENSIONS},
                'unresolved':[name for name,row in rows.items() if row['state'] not in ('settled','cancelled')],
                'overruns':[name for name,row in rows.items() if row.get('overrun')],
                'deadline_violations':[name for name,row in rows.items() if row.get('deadline_status')=='exceeded' or row.get('work_deadline_status')=='exceeded'],
                'admission_stopped':self.store.find('RESOURCE-DEADLINE-STOP') is not None or any(row.get('deadline_status')=='exceeded' or row.get('work_deadline_status')=='exceeded' for row in rows.values()), 'reservations':rows}

    def reserve(self,reservation_id,kind,requested,*,phase,batch,binding,host):
        self._host(host);identifier(reservation_id);quantities(requested)
        if kind not in ('native_task','synthetic_evaluation','development_evaluation','artifact_build','artifact_stage'):raise Blocked('unknown resource work kind')
        if not isinstance(binding,dict) or not binding:raise Blocked('exact work identity required')
        if self.authority['mode'] in ('bounded_pilot','bounded_search'):
            from .pilot_release import require_store_release, require_store_readiness
            release = require_store_release(self.store)
            if release['contract']['version'] in ('bounded-pilot/2','bounded-search/1'):
                first = (kind == 'native_task' and reservation_id in (('TASK-PLAN','TASK-PLAN-R1') if release['contract']['version']=='bounded-search/1' else ('TASK-PLAN',))
                         and binding.get('dispatch_id') == reservation_id[5:]
                         and binding.get('recipient') == release['contract']['actors']['researcher'])
                if not first: require_store_readiness(self.store)
        if kind=='native_task' and (requested['agent_tasks']!=1 or requested['evaluation_requests'] or requested['worker_sessions']):raise Blocked('native task allocation mismatch')
        if kind in ('synthetic_evaluation','development_evaluation') and (requested['agent_tasks'] or requested['evaluation_requests']<1 or requested['worker_sessions']<1):raise Blocked('evaluation allocation mismatch')
        if kind=='development_evaluation' and self.authority['mode'] not in ('bounded_pilot','bounded_search'):raise Blocked('development evaluation requires released pilot resources')
        artifact_scope='development_pilot' if self.authority['mode'] in ('bounded_pilot','bounded_search') else 'synthetic_preparation_only'
        if kind in ('artifact_build','artifact_stage') and (any(requested[k] for k in COUNT_DIMENSIONS) or binding.get('scope')!=artifact_scope):raise Blocked('artifact work requires the authority scope and separate wall allocation')
        if requested['wall_seconds']<=0:raise Blocked('positive wall allocation required')
        payload={'reservation_id':reservation_id,'kind':kind,'requested':requested,'phase':phase,'batch':batch,'binding':binding}
        prior=self.store.find('RESERVED-'+reservation_id)
        if prior:
            if prior['payload']!=payload:raise Conflict('resource identity reused with different work')
            return prior
        left=self._live()
        if requested['wall_seconds']>left:raise Blocked('requested wall allocation exceeds remaining deadline')
        state=self.snapshot()
        if state['unresolved']:raise Blocked('unresolved resource work blocks admission')
        if state['overruns']:raise Blocked('observed overrun needs a new prospective allocation')
        spec=self.authority['batches'].get(batch)
        if not spec or spec['phase']!=phase:raise Blocked('batch/phase allocation absent')
        groups=[(self.authority['campaign'],lambda row:True),(self.authority['phases'][phase],lambda row:row['phase']==phase),(spec['limits'],lambda row:row['batch']==batch)]
        for cap,select in groups:
            for key,value in requested.items():
                used=sum((row['actual'] if row['state']=='settled' else row['requested'])[key] for row in state['reservations'].values() if row['state']!='cancelled' and select(row))
                if used+value>cap[key]:raise Blocked('aggregate resource cap exceeded: '+key)
        return self.store.append('RESERVED-'+reservation_id,'resource_reserved',host,payload)

    def begin(self,reservation_id,host):
        self._host(host);self._live();row=self.state()[reservation_id]
        if row['state']!='reserved':raise Conflict('begun/uncertain work cannot be retried')
        left=self._live()
        if row['requested']['wall_seconds']>left:raise Blocked('reserved wall allocation no longer fits deadline')
        return self.store.append('BEGUN-'+reservation_id,'resource_begun',host,{'reservation_id':reservation_id,'started_clock':observation(boot_identity()),'clock_boundary':'admission before dispatch/evaluator setup to parent-verified return after cleanup'})

    def cancel_unissued(self,reservation_id,evidence,host):
        self._host(host);row=self.state()[reservation_id]
        if row['state']!='reserved':raise Blocked('only never-begun reservation can be cancelled')
        sha=self.store.archive(canonical(evidence))
        return self.store.append('CANCELLED-'+reservation_id,'resource_cancelled',host,{'reservation_id':reservation_id,'evidence_sha256':sha})

    def settle(self,reservation_id,actual,evidence,host,*,provider_usage=None):
        self._host(host);row=self.state()[reservation_id]
        if actual is not None:quantities(actual)
        if not isinstance(evidence,dict) or not evidence:raise Blocked('observed evidence required')
        sha=self.store.archive(canonical({'evidence':evidence,'provider_usage':provider_usage,'actual':actual}))
        event_id=('UNKNOWN-' if actual is None else 'SETTLED-')+reservation_id
        prior=self.store.find(event_id)
        if prior:deadline_status=prior['payload'].get('deadline_status','unrecorded')
        else:
            try:deadline_status='exceeded' if self.remaining_seconds()<=0 else 'within'
            except (Blocked,OSError):deadline_status='unknown'
        payload={'reservation_id':reservation_id,'deadline_status':deadline_status,'work_deadline_status':'unknown' if actual is None else ('exceeded' if actual['wall_seconds']>=row['requested']['wall_seconds'] else 'within'),'actual':actual,'evidence_sha256':sha,'provider_usage':provider_usage,
                 'provider_usage_status':'unknown' if provider_usage is None else 'reported',
                 'overrun':actual is not None and any(actual[k]>row['requested'][k] for k in DIMENSIONS)}
        event_id=('UNKNOWN-' if actual is None else 'SETTLED-')+reservation_id
        prior=self.store.find(event_id)
        if prior:
            if prior['payload']!=payload:raise Conflict('conflicting resource receipt')
            return prior
        if row['state'] not in ('begun','unknown'):raise Blocked('settlement requires begun or uncertain work')
        result=self.store.append(event_id,'resource_unknown' if actual is None else 'resource_settled',host,payload)
        if payload['deadline_status']=='exceeded' or payload['work_deadline_status']=='exceeded':
            self._record_stop('deadline_at_settlement',{'settlement_event':result['hash'],'reservation_id':reservation_id})
        return result
