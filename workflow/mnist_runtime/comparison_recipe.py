"""Frozen common candidate/baseline recipe. Pure planning; no data loader."""
from .common import Blocked,digest
from .service_metric import schedule
import random

def plan(primary_n, *, seed=20261007, members=('candidate','cnn'), timed_indices=None, scope='synthetic_preparation_only'):
    if scope not in ('synthetic_preparation_only','development_pilot'):raise Blocked('unknown comparison scope')
    if type(primary_n) is not int or not 128<=primary_n<=10000:raise Blocked('primary cohort must contain 128..10000 images')
    if timed_indices is None:timed_indices=sorted(random.Random(seed+2).sample(range(primary_n),128))
    if not isinstance(timed_indices,list) or len(timed_indices)!=128 or len(set(timed_indices))!=128 or any(type(i) is not int or not 0<=i<primary_n for i in timed_indices):raise Blocked('128 unique candidate-independent timing indices required')
    member_order=schedule(list(members),seed)
    forward=list(range(primary_n));shuffled=forward.copy();random.Random(seed).shuffle(shuffled)
    controls=[]
    for name,seq in [('forward',forward),('reverse',forward[::-1]),('shuffle',shuffled)]:controls.append({'name':name,'kind':'order','sequence':[['images',i] for i in seq]})
    for anchor in [0,1,2,3]:
        controls.append({'name':f'reference-{anchor}','kind':'reference','anchor':anchor,'sequence':[['images',anchor]]})
        for n in [1,128,1024]:controls.append({'name':f'history-{anchor}-{n}','kind':'history','anchor':anchor,'length':n,'sequence':[['images',anchor]]+[['fillers',i] for i in range(n)]+[['images',anchor]]})
    jobs=[]
    for member in members:
        jobs.extend(dict(job,member=member) for job in controls)
    for block,order in enumerate(member_order):
        jobs.extend({'name':f'timing-{block}','kind':'timing','block':block,'member':member,'sequence':[['images',i] for i in timed_indices]} for member in order)
    for job in jobs:job['requests']=50+len(job['sequence'])
    counts={m:{'requests':sum(j['requests'] for j in jobs if j['member']==m),'sessions':sum(j['member']==m for j in jobs)} for m in members}
    recipe={'version':'common-docker-comparison/1','scope':scope,'primary_n':primary_n,'seed':seed,'shuffle_algorithm':'Python random.Random (version-bound by executor)','anchors':[0,1,2,3],'timed_indices':timed_indices,'members':list(members),'warmups':50,'timing_member_order':member_order,'jobs':jobs,'per_member':counts,'planned_requests':sum(x['requests'] for x in counts.values()),'planned_sessions':len(jobs),'largest_session_requests':max(j['requests'] for j in jobs)}
    if any(c!={'requests':3*primary_n+6124,'sessions':22} for c in counts.values()):raise Blocked('common recipe accounting contradiction')
    return recipe
