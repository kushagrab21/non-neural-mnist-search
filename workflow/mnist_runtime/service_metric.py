"""Prospective service objective. Historical timing records cannot be relabelled."""
from __future__ import annotations
import math,statistics,random
from .common import Blocked,digest

METRIC_ID='runtime-objective/service-latency/1'
TRUST_ID='development-trust/cooperative-native-roles/1'


def stats(values):
    if not values or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in values):
        raise Blocked('invalid parent service duration')
    ordered=sorted(values);position=.9*(len(values)-1);lo=math.floor(position);hi=math.ceil(position)
    return {'n':len(values),'median':statistics.median(values),'p90':ordered[lo]+(position-lo)*(ordered[hi]-ordered[lo]),'min':ordered[0],'max':ordered[-1]}


def schedule(members,seed):
    if len(members)!=2 or len(set(members))!=2 or any(not isinstance(m,str) or not m for m in members) or type(seed) is not int:
        raise Blocked('this protocol requires exactly two identified members and integer seed')
    first=list(members);random.Random(seed).shuffle(first)
    last=list(members);random.Random(seed+1).shuffle(last)
    return [first,first[::-1],last]


def aggregate(record):
    if record.get('metric_id_at_start')!=METRIC_ID or record.get('scope')!='synthetic_proposed_metric':
        raise Blocked('no retrospective or campaign objective promotion')
    if type(record.get('timed_n')) is not int or record['timed_n']<1 or not record.get('comparison_identity'):
        raise Blocked('explicit cohort and comparison identity required')
    expected=record['timed_n'];blocks=record.get('blocks',[]);summaries=[]
    names=set();complete=len(blocks)==3
    for block in blocks:
        name=block.get('block')
        if type(name) is not int or name not in (0,1,2) or name in names:raise Blocked('duplicate/unknown timing block')
        names.add(name);values=block.get('parent_service_ms',[])
        if not isinstance(values,list) or len(values)>expected:raise Blocked('invalid timing coverage')
        summary=stats(values) if values else None
        valid=(block.get('complete') is True and block.get('monitor_complete') is True and len(values)==expected)
        complete=complete and valid
        summaries.append({'block':name,'complete':valid,'summary':summary,'raw_parent_service_ms':list(values)})
    complete=complete and names=={0,1,2}
    medians=[b['summary']['median'] for b in summaries if b['complete']]
    return {'metric_id':METRIC_ID,'state':'synthetic_proposal_only','campaign_approved':False,
            'primary_axis':'warmed_service_latency_ms','primary_value':statistics.median(medians) if complete else None,
            'complete':complete,'blocks':summaries,'block_median_range':[min(medians),max(medians)] if medians else None,
            'range_semantics':'descriptive, not confidence interval','overhead_subtracted':False,
            'child_diagnostics_used':False,'comparison_group_sha256':digest(record['comparison_identity']),
            'frontier_status':'observed_provisional_only'}


def comparable(left,right):
    return left['metric_id']==right['metric_id'] and left['comparison_group_sha256']==right['comparison_group_sha256'] and left['complete'] and right['complete']


def require_campaign_approval(contract,approval):
    """Fail-closed checker only; never enables a real executor or sealed opener."""
    if contract.get('metric_id')!=METRIC_ID or contract.get('trust_id')!=TRUST_ID:raise Blocked('unknown metric/trust version')
    if not isinstance(approval,dict) or approval.get('actor_role')!='user' or approval.get('contract_sha256')!=digest(contract) or approval.get('release')!='bounded_real_pilot':
        raise Blocked('exact user approval of complete contract required')
    required={'candidate_boundary','final_data_boundary','resource_allocation','common_measurement_protocol'}
    if set(contract.get('gates',{}))!=required or any(contract['gates'][k] is not True for k in required):
        raise Blocked('required qualified boundaries/allocation absent')
    raise Blocked('real executor is not implemented in this synthetic preparation release')
