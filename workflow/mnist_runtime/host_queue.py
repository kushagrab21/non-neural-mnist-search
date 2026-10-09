"""Durable queue for host-owned native agent APIs; never dispatches by itself.

The host claims an exact immutable envelope before invoking its native API, then
appends observed transport/ack/return receipts. Restart never retries issued work.
Same-user administrators remain trusted; this is attribution, not cryptographic
authentication of an agent provider.
"""
from datetime import datetime,timezone
import secrets

from .common import Blocked,Conflict,canonical,digest,identifier


class HostQueue:
    def __init__(self,store): self.store=store

    @property
    def authority(self):
        event=self.store.find('HOST-AUTHORITY')
        if not event: raise Blocked('host queue unconfigured')
        return event['payload']

    def initialize(self,authority,host):
        if set(authority)!={'authorization_id','deadline','max_tasks','actors','scope'} or type(authority['max_tasks']) is not int or authority['max_tasks']<1:
            raise Blocked('explicit authority, deadline, actors, scope and task cap required')
        datetime.fromisoformat(authority['deadline'])
        return self.store.append('HOST-AUTHORITY','host_authority',host,authority)

    def state(self):
        jobs={}
        for event in self.store.events:
            p=event['payload']; kind=event['kind']
            if kind=='host_prepared': jobs[p['dispatch_id']]=dict(p,state='prepared',receipts=[])
            elif kind.startswith('host_') and 'dispatch_id' in p:
                job=jobs[p['dispatch_id']]
                job['receipts'].append(event['event_id'])
                job['state']={'host_issued':'issued','host_delivered':'delivered','host_acknowledged':'acknowledged',
                              'host_returned':'returned','host_uncertain':'uncertain'}.get(kind,job['state'])
        return jobs

    def _live(self):
        if datetime.now(timezone.utc)>=datetime.fromisoformat(self.authority['deadline']):
            raise Blocked('host preparation deadline exhausted')

    def _host(self,actor):
        if actor!=self.store.find('HOST-AUTHORITY')['actor']:
            raise Blocked('only bound host may append transport evidence')

    def prepare(self,dispatch_id,role,order,return_contract,host):
        self._host(host); self._live(); identifier(dispatch_id)
        jobs=self.state()
        if dispatch_id in jobs:
            job=jobs[dispatch_id]
            if (job['role']==role and job['order_sha256']==digest(order) and job['return_contract']==return_contract):
                return self.store.find('PREPARED-'+dispatch_id)
            raise Conflict('dispatch identity already prepared with different content')
        if len(jobs)>=self.authority['max_tasks']: raise Blocked('all host task reservations consumed')
        if role not in self.authority['actors'] or not isinstance(order,bytes) or not order:
            raise Blocked('unknown role or empty order')
        if set(return_contract)!={'required_fields','destination'} or not return_contract['required_fields']:
            raise Blocked('explicit structured return contract required')
        sha=self.store.archive(order)
        payload={'dispatch_id':dispatch_id,'role':role,'recipient':self.authority['actors'][role],
                 'order_sha256':sha,'nonce':secrets.token_hex(16),'return_contract':return_contract,
                 'authorization_id':self.authority['authorization_id']}
        return self.store.append('PREPARED-'+dispatch_id,'host_prepared',host,payload)

    def claim(self,dispatch_id,host):
        self._host(host); self._live(); jobs=self.state(); job=jobs[dispatch_id]
        if job['state']!='prepared': raise Conflict('issued work cannot be retried, including after restart')
        if any(j['state'] in ('issued','delivered','acknowledged','uncertain') for j in jobs.values()):
            raise Blocked('in-flight or uncertain task requires reconciliation before next issue')
        event=self.store.append('ISSUED-'+dispatch_id,'host_issued',host,
            {'dispatch_id':dispatch_id,'order_sha256':job['order_sha256'],'recipient':job['recipient'],'nonce':job['nonce']})
        return {**job,'order_utf8':self.store.read_artifact(job['order_sha256']).decode(), 'claim_event_hash':event['hash']}

    def delivered(self,dispatch_id,transport_receipt,host):
        self._host(host); job=self.state()[dispatch_id]
        if not isinstance(transport_receipt,dict) or not transport_receipt:
            raise Blocked('observed transport receipt required')
        sha=self.store.archive(canonical(transport_receipt))
        prior=self.store.find('DELIVERED-'+dispatch_id)
        if prior:
            if prior['payload']['raw_receipt_sha256']!=sha: raise Conflict('conflicting delivery receipt')
            return prior
        if job['state']!='issued': raise Blocked('delivery requires issued task')
        return self.store.append('DELIVERED-'+dispatch_id,'host_delivered',host,{'dispatch_id':dispatch_id,'raw_receipt_sha256':sha})

    def acknowledge(self,dispatch_id,envelope,host):
        self._host(host); job=self.state()[dispatch_id]
        expected={k:job[k] for k in ('dispatch_id','recipient','order_sha256','nonce')}
        if envelope!=expected: raise Blocked('acknowledgment differs from dispatcher binding')
        if self.store.find('ACK-'+dispatch_id): return self.store.find('ACK-'+dispatch_id)
        if job['state'] not in ('issued','delivered'): raise Blocked('acknowledgment outside issued task')
        return self.store.append('ACK-'+dispatch_id,'host_acknowledged',host,dict(dispatch_id=dispatch_id,envelope=envelope))

    def receive(self,dispatch_id,envelope,body,usage,host):
        self._host(host); job=self.state()[dispatch_id]
        raw=self.store.archive(canonical({'envelope':envelope,'body':body,'usage':usage}))
        prior=self.store.find('RETURN-'+dispatch_id)
        if prior:
            if prior['payload']['raw_artifact_sha256']!=raw: raise Conflict('conflicting completed return')
            return prior
        expected={k:job[k] for k in ('dispatch_id','recipient','order_sha256','nonce')}
        valid=(job['state'] in ('issued','delivered','acknowledged','uncertain') and envelope==expected and isinstance(body,dict)
               and all(k in body for k in job['return_contract']['required_fields']))
        if not valid:
            self.store.append('BAD-'+dispatch_id+'-'+raw[:12],'host_return_rejected',host,{'dispatch_id':dispatch_id,'raw_artifact_sha256':raw})
            raise Blocked('return identity/schema mismatch; raw evidence retained')
        return self.store.append('RETURN-'+dispatch_id,'host_returned',host,
              {'dispatch_id':dispatch_id,'raw_artifact_sha256':raw,'usage':usage,
               'usage_status':'unknown' if usage is None else 'provider_reported',
               'scientific_acceptance':'not inferred from receipt'})

    def uncertain(self,dispatch_id,reason,host):
        self._host(host); job=self.state()[dispatch_id]
        if job['state'] not in ('issued','delivered','acknowledged'): raise Blocked('no unresolved invocation to classify')
        return self.store.append('UNCERTAIN-'+dispatch_id,'host_uncertain',host,{'dispatch_id':dispatch_id,'reason':reason})
