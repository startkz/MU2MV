"""Live AirFogSim task/RB/CPU integration for matched-cost verification experiments.

This module MUST run against the upstream `airfogsim` package. It does not
simulate wireless packets or CPU completion outside that engine. Service-result
correctness is a separately disclosed, keyed Bernoulli injection applied only
after successful transport, CPU service, and return.

CAUTION: AirFogSim's task manager lacks a public API for injecting an externally
specified task; `_waiting_to_offload_tasks` is the sole private injection hook.
All transmission, computation, return and deadline transitions remain upstream.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core import Belief, cvar, draw


@dataclass(frozen=True)
class LiveConfig:
    tti_s: float = 0.1
    steps: int = 80
    slow_period: int = 20
    verify_interval: int = 20
    verify_reservation_steps: int = 15
    verify_rbs: int = 5
    verify_cpu_ghz: float = 1.0
    verify_task_cpu_gc: float = 0.35
    verify_request_mb: float = 0.025
    verify_reply_mb: float = 0.008
    verify_deadline_s: float = 1.5
    normal_interval: int = 5
    normal_task_cpu_gc: float = 0.95
    normal_task_mb: float = 0.18
    normal_reply_mb: float = 0.018
    normal_deadline_s: float = 2.7
    radio_contact_m: float = 260.0
    prior_strength: float = 3.0
    risk_beta: float = 0.8
    risk_budget: float = 10.0
    target_uav: str = 'UAV_0'
    vehicles: int = 4


@dataclass
class Reservation:
    task_id: str
    vehicle: str
    start_slot: int
    end_slot: int
    accounted_cpu_gc: float = 0.0
    verify_cpu_gc: float = 0.0
    rb_active: int = 0
    rb_reserved: int = 0
    upload_tti: int = 0
    compute_tti: int = 0
    return_tti: int = 0
    status: str = 'pending'


class AirFogSimLiveAdapter:
    """Inject and track real AirFogSim tasks; reserve equal RB/CPU envelopes.

    Reservation is a fixed-size opportunity-cost envelope, even after early
    completion. Actual radio packets, processor work and outcomes are tracked
    separately from reservations and never synthesized by this adapter.
    """
    def __init__(self,env: Any,config:LiveConfig):
        self.env,self.cfg=env,config
        from airfogsim.entities.task import Task
        from airfogsim.scheduler.task_sched import TaskScheduler
        from airfogsim.scheduler.communication_sched import CommunicationScheduler
        from airfogsim.scheduler.computation_sched import ComputationScheduler
        self.Task,self.tasksched,self.commsched,self.compsched=Task,TaskScheduler,CommunicationScheduler,ComputationScheduler
        self.reservations:dict[str,Reservation]={}
        self.active_reservation:Reservation|None=None
        self.tasks:dict[str,Any]={}
        self.kind:dict[str,str]={}
        self.task_target:dict[str,str]={}
        self.rb_capacity=self.commsched.getNumberOfRB(env)
        if config.verify_rbs>=self.rb_capacity:raise ValueError('challenge must leave RBs for normal workload')
        self.telemetry=[]
        self.total_reserved_rb_tti=0
        self.total_used_rb_tti=0
        self.total_reserved_cpu_gc=0.0
        self.total_used_cpu_gc=0.0
        self.total_verify_cpu_gc=0.0
        self._slot=-1
        self._allocated_cpu={}
        self.compsched.setComputingCallBack(env,self.allocate_cpu)

    def inject(self,task_id:str,node:str,target:str,kind:str,now:float,*,cpu:float,size:float,reply:float,deadline:float,priority:float=1.0):
        if node not in self.env.UAVs:raise ValueError(f'UAV not ready: {node}')
        if target not in self.env.vehicles and target!=node:raise ValueError(f'vehicle not ready: {target}')
        if task_id in self.tasks:raise ValueError('task id reused')
        task=self.Task(task_id=task_id,task_node_id=node,task_cpu=cpu,task_size=size,
            task_deadline=deadline,task_priority=priority,task_arrival_time=now,
            required_returned_size=reply,to_return_node_id=node,return_lazy_set=False)
        task.setGenerated()
        # Register a DAG node: upstream offloadTask assumes the source DAG exists.
        import networkx as nx
        self.env.task_manager._task_dependencies.setdefault(node,nx.DiGraph()).add_node(task_id)
        self.env.task_manager._waiting_to_offload_tasks.setdefault(node,[]).append(task)
        ok=self.tasksched.setTaskOffloading(self.env,node,task_id,target)
        if not ok:raise RuntimeError(f'AirFogSim rejected task admission {task_id}')
        self.tasks[task_id]=task
        self.kind[task_id]=kind
        self.task_target[task_id]=target
        return task

    def create_verification(self,vehicle:str,slot:int):
        if self.active_reservation is not None and slot<self.active_reservation.end_slot:
            raise RuntimeError('overlapping challenge reservations not permitted')
        c=self.cfg
        ident=f'VA_VERIFY_{slot}_{vehicle}'
        now=self.env.simulation_time
        self.inject(ident,c.target_uav,vehicle,'verification',now,cpu=c.verify_task_cpu_gc,
                    size=c.verify_request_mb,reply=c.verify_reply_mb,deadline=c.verify_deadline_s)
        r=Reservation(ident,vehicle,slot,slot+c.verify_reservation_steps)
        self.reservations[ident]=r
        self.active_reservation=r
        return r

    def _reservation(self):
        r=self.active_reservation
        return r if r is not None and r.start_slot<=self._slot<r.end_slot else None

    def prepare_returns(self):
        # Return paths MUST be assigned through AirFogSim before env.step().
        for node,items in list(self.env.task_manager._waiting_to_return_tasks.items()):
            for task in list(items):
                ident=task.getTaskId()
                if ident in self.tasks:
                    self.tasksched.setTaskReturnRoute(self.env,ident,[self.cfg.target_uav])

    def allocate_cpu(self,computing_tasks,**kwargs):
        reservation=self._reservation()
        out={}
        c=self.cfg
        for node,items in computing_tasks.items():
            if not items:continue
            entity=self.env.vehicles.get(node,self.env.UAVs.get(node))
            if entity is None:continue
            # AirFogSim's computation capacity is in Gcycles/s.
            cap=float(entity.getFogProfile().get('cpu',0))
            challenge=[task for task in items if reservation and task.getTaskId()==reservation.task_id]
            reserve=min(cap,c.verify_cpu_ghz) if reservation and node==reservation.vehicle else 0.0
            ordinary=[task for task in items if task not in challenge]
            if challenge:
                out[challenge[0].getTaskId()]=reserve
                actual=min(challenge[0].getTaskCPU()-challenge[0].getComputedSize(),reserve*c.tti_s)
                self.total_verify_cpu_gc+=max(0.,actual)
                reservation.verify_cpu_gc+=max(0.,actual)
                if actual > 0:
                    reservation.compute_tti += 1
            available=max(0.0,cap-reserve)
            for task in ordinary:
                out[task.getTaskId()]=available/len(ordinary)
        self._allocated_cpu=out
        return out

    def _all_transmitting(self):
        pending={}
        manager=self.env.task_manager
        for coll in (manager._offloading_tasks,manager._returning_tasks):
            for tasks in coll.values():
                for task in tasks:
                    if task.getTaskId() in self.tasks:
                        pending[task.getTaskId()]=task
        return pending

    def schedule_tti(self,slot:int):
        self._slot=slot
        # AirFogSim.clearDecisions resets alloc_cpu_callback every TTI.
        self.compsched.setComputingCallBack(self.env,self.allocate_cpu)
        self.prepare_returns()
        pending=self._all_transmitting()
        r=self._reservation()
        verify_id=r.task_id if r else None
        challenge=verify_id in pending if verify_id else False
        reserve_rb=self.cfg.verify_rbs if r else 0
        normal=[tid for tid in sorted(pending) if tid!=verify_id]
        remaining=self.rb_capacity-reserve_rb
        allocations={}
        if challenge:
            allocations[verify_id]=list(range(reserve_rb))
        if normal:
            for i,tid in enumerate(normal):
                start=reserve_rb+math.floor(i*remaining/len(normal))
                stop=reserve_rb+math.floor((i+1)*remaining/len(normal))
                if stop>start:allocations[tid]=list(range(start,stop))
        occupied=[rb for a in allocations.values() for rb in a]
        if len(occupied)!=len(set(occupied)):raise AssertionError('RB overlap')
        if occupied and max(occupied)>=self.rb_capacity:raise AssertionError('RB out of range')
        if r:
            self.total_reserved_rb_tti+=reserve_rb
            self.total_reserved_cpu_gc+=self.cfg.verify_cpu_ghz*self.cfg.tti_s
            r.rb_reserved+=reserve_rb
            if challenge:
                self.total_used_rb_tti+=reserve_rb
                r.rb_active+=reserve_rb
                if pending[verify_id].isReturning():
                    r.return_tti += 1
                else:
                    r.upload_tti += 1
        for tid,rbs in allocations.items():
            self.commsched.setCommunicationWithRB(self.env,tid,rbs)
        self.telemetry.append({'slot':slot,'sim_time':self.env.simulation_time,
            'verify_id':verify_id or '','verify_transmitting':int(challenge),
            'verify_rb_reserved':reserve_rb,'verify_rb_active':reserve_rb if challenge else 0,
            'other_rb_allocated':sum(len(v) for k,v in allocations.items() if k!=verify_id),
            'cpu_reserved_gc':self.cfg.verify_cpu_ghz*self.cfg.tti_s if r else 0.,
            'active_transmissions':len(pending),
            'computing_verification':int(any(verify_id==t.getTaskId() for a in self.env.task_manager._computing_tasks.values() for t in a)) if verify_id else 0})
        return allocations

    def advance(self,slot:int):
        # The env step performs transmission and computing; the adapter only
        # accounts for allocated CPU clock cycles and observes completed tasks.
        self.schedule_tti(slot)
        self.env.step()
        if self.active_reservation and slot+1>=self.active_reservation.end_slot:
            self.active_reservation=None

    def task_state(self,task_id:str):
        if task_id not in self.tasks:return 'unknown'
        m=self.env.task_manager
        for name,coll in (('done',m._done_tasks),('failed',m._out_of_ddl_tasks)):
            if any(task.getTaskId()==task_id for arr in coll.values() for task in arr):return name
        return 'pending'

    def completed_verifications(self,observed:set[str]):
        result=[]
        for rid,r in self.reservations.items():
            if rid not in observed and self.task_state(rid) in ('done','failed'):
                r.status=self.task_state(rid)
                result.append(r)
                observed.add(rid)
        return result


def hidden_probability(vehicle_index:int)->float:
    return (.45,.66,.83,.94)[vehicle_index%4]


def create_fixtures(out:Path,c:LiveConfig,seed:int,upstream:Path):
    """Real-mode AirFogSim uses the production channel/task engine with
    a generated trace instead of requiring a SUMO daemon for CI.
    """
    import yaml
    out.mkdir(parents=True,exist_ok=True)
    config=yaml.safe_load((upstream/'examples'/'config.yaml').read_text())
    # Minimal valid SUMO-compatible network read by sumolib and AirFogSim.
    net=out/'mini.net.xml'
    net.write_text('''<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16" junctionCornerDetail="5" limitTurnSpeed="5.50">
 <location netOffset="0.00,0.00" convBoundary="0.00,0.00,600.00,400.00" origBoundary="0.00,0.00,600.00,400.00" projParameter="!"/>
 <edge id="E0" from="J0" to="J1" priority="1">
  <lane id="E0_0" index="0" speed="13.89" length="600.00" shape="0.00,200.00 600.00,200.00"/>
 </edge>
 <junction id="J0" type="dead_end" x="0.00" y="200.00" incLanes="" intLanes="" shape="0,200"/>
 <junction id="J1" type="dead_end" x="600.00" y="200.00" incLanes="E0_0" intLanes="" shape="600,200"/>
</net>
''',encoding='utf8')
    csvpath=out/'vehicles.csv'
    with csvpath.open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f,delimiter=';')
        w.writerow(['vehicle_id','data_timestep','vehicle_x','vehicle_y','vehicle_speed','vehicle_angle','vehicle_route'])
        for k in range(c.steps+35):
            t=round(k*c.tti_s,5)
            for v in range(c.vehicles):
                x=190+v*42+math.sin(.43*t+v)*44+(.4-v*.2)*t*9
                y=200+(-1)**v*15
                w.writerow([f'vehicle_{v}',t,round(x,4),y,12+v*2,0,'E0'])
    config['simulation'].update(max_simulation_time=(c.steps+2)*c.tti_s,simulation_interval=c.tti_s,traffic_interval=c.tti_s)
    config['traffic'].update(traffic_mode='real',tripinfo=str(csvpath),traffic_interval=c.tti_s,
        max_n_UAVs=1,max_n_vehicles=c.vehicles,max_n_cloudServers=0,
        RSU_positions=[[100,100,0]],UAV_z_range=[70,70],UAV_speed_range=[0,0],arrival_lambda=0)
    config['sumo'].update(sumo_net=str(net),sumo_config='',sumo_osm='',export_tripinfo=False)
    config['task_profile'].update(task_node_gen_poss=0,task_node_profiles=[{'type':'UAV','max_node_num':1}])
    config['task']['task_generation_model']='None'
    config['task']['tti_threshold']=1.0
    config['task']['hard_ddl']=20
    config['task_profile']['uav']={'lambda':0,'dag_edge_prob':0.0}
    config['mission']['generation_model_args']['Poisson']['lambda']=0
    config['fog_profile']['vehicle']['cpu']=2.2
    config['fog_profile']['uav']['cpu']=2.3
    # Disable unrelated upstream authentication attack injection.
    # Service correctness is injected separately with hidden keyed draws.
    config['authentication']={'malicious_ratio':0.0}
    path=out/'config.yaml'
    path.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf8')
    return path


def _get_reliability_observation(seed:int,task_id:str,target:str):
    k=int(target.rsplit('_',1)[-1])
    return draw(seed,'service_truth',task_id,target)<hidden_probability(k)


def _observable_candidates(env,c:LiveConfig):
    try:pos=env.traffic_manager.getUAVTrafficInfos()[c.target_uav]['position']
    except KeyError:return []
    return sorted((vid,math.dist(pos,info['position'])) for vid,info in env.traffic_manager.getVehicleTrafficInfos().items()
                  if vid in env.vehicles and math.dist(pos,info['position'])<=c.radio_contact_m)


def choose_work_target(env,c:LiveConfig,beliefs,now:float,seed:int,task_id:str):
    """Myopic risk-constrained choice on OBSERVABLE posteriors only.

    This function is not DDP's exact multi-task optimizer; it is a controlled
    AirFogSim integration experiment to isolate verification policy effects.
    """
    cand=_observable_candidates(env,c)
    options=[]
    for vid,distance in cand:
        p=beliefs[vid].mean*max(.4,1-.4*(distance/c.radio_contact_m)**2)
        loss=cvar({0:p,10:1-p},c.risk_beta)
        if loss<=c.risk_budget:
            queue_size=len(env.task_manager._computing_tasks.get(vid,[]))
            score=10*p-.004*distance-.5*queue_size
            options.append((score,vid))
    # Never access hidden truth. Use local if no remote can beat its cost.
    if not options:return c.target_uav
    score,vid=max(options)
    return vid if score>=5.1 else c.target_uav


def _select_verification(method:str,env,c:LiveConfig,beliefs,seed:int,slot:int):
    cand=_observable_candidates(env,c)
    if not cand:return None
    if method=='random_verify':
        return cand[min(len(cand)-1,int(draw(seed,'random-node',slot)*len(cand)))][0]
    # Slow-tier priority: posterior uncertainty and disagreement with claimed
    # reputation; contact margin computed only from position.
    def rank(pair):
        vid,d=pair;b=beliefs[vid]
        return b.variance*(1+2*abs(b.mean-b.reported))*max(.05,1-d/c.radio_contact_m)
    return max(cand,key=rank)[0]



# ---- Joint slow-budget / fast distributional-PMF-DP extension ----
from dataclasses import replace
from collections import defaultdict
import subprocess
import time
from .core import Frontier, add_loss, dominates

@dataclass(frozen=True)
class JointConfig(LiveConfig):
    steps:int=129
    slow_period:int=32
    tokens_per_epoch:int=2
    task_batch:int=3
    normal_interval:int=6
    normal_task_cpu_gc:float=.72
    normal_deadline_s:float=2.3
    verify_reservation_steps:int=12
    verify_deadline_s:float=1.1
    risk_budget:float=21.
    max_local:int=1  # workload conservation: at most one new local job per arrival batch
    max_remote:int=1
    outage_probability:float=.08
    outage_burst:int=1
    outage_period:int=23
    stress_departures:bool=True
    sumo_mode:bool=False
    adaptive_timing:bool=False
    nominal_rate_rb_mbps:float=4.0

def contact_seconds(u,p,v,radius):
    delta=[p[i]-u[i] for i in range(3)]
    if sum(x*x for x in delta)>=radius*radius:return 0.
    a=v[0]*v[0]+v[1]*v[1]
    if a<1e-12:return 30.
    b=2*(delta[0]*v[0]+delta[1]*v[1])
    c=sum(x*x for x in delta)-radius*radius
    return max(0.,(-b+math.sqrt(max(0.,b*b-4*a*c)))/(2*a))

def observable_views(env,c,old):
    up=env.traffic_manager.getUAVTrafficInfos().get(c.target_uav)
    if not up:return {}
    views={}
    for vid,z in env.traffic_manager.getVehicleTrafficInfos().items():
        if vid not in env.vehicles:continue
        pos=z['position'];d=math.dist(up['position'],pos)
        if d>=c.radio_contact_m:continue
        if vid in old:
            vx=(pos[0]-old[vid][0])/c.tti_s
            vy=(pos[1]-old[vid][1])/c.tti_s
        else:
            angle=math.radians(z.get('angle',0))
            vx=z.get('speed',0)*math.sin(angle)
            vy=z.get('speed',0)*math.cos(angle)
        q=env.task_manager._computing_tasks.get(vid,[])
        cpu=float(env.vehicles[vid].getFogProfile().get('cpu',2.2))
        backlog=sum(max(0.,j.getTaskCPU()-j.getComputedSize()) for j in q)
        views[vid]={'d':d,'contact':contact_seconds(up['position'],pos,(vx,vy),c.radio_contact_m),
                    'wait':backlog/max(cpu,1e-6),'cpu':cpu}
    return views

class SlowBudget:
    def __init__(self,c,seed,method):
        self.c,self.seed,self.method=c,seed,method
        self.num_epochs=0;self.plans={};self.used=set();self.issued=0;self.records=[]
    def update(self,slot,views,beliefs):
        self.num_epochs+=1
        c=self.c; k=c.tokens_per_epoch
        if k*c.verify_reservation_steps>c.slow_period:raise ValueError('overcommitted verification epoch')
        scores={vid:beliefs[vid].variance*(1+2*abs(beliefs[vid].mean-beliefs[vid].reported))*
               (.15+.85*min(1.,v['contact']/(c.slow_period*c.tti_s))) for vid,v in views.items()}
        quotas=defaultdict(int);targets=[]
        for i in range(k):
            if not scores:targets.append(None);continue
            if self.method=='joint':
                target=max(scores,key=lambda x:(scores[x]/(1+quotas[x]),x))
            else:
                ids=sorted(scores)
                target=ids[min(len(ids)-1,int(draw(self.seed,'quota',slot,i)*len(ids)))]
            quotas[target]+=1;targets.append(target)
        last=slot
        for i,target in enumerate(targets):
            base=max(last,slot+i*c.verify_reservation_steps)
            remaining=k-i
            slack=max(0,slot+c.slow_period-remaining*c.verify_reservation_steps-base)
            if not c.adaptive_timing: shift=0
            elif self.method=='joint':
                win=views.get(target,{}).get('contact',0.)
                shift=min(slack,max(0,int(win/c.tti_s)-c.verify_reservation_steps-1))
            else:shift=int(draw(self.seed,'when',slot,i)*(slack+1))
            tick=base+shift;last=tick+c.verify_reservation_steps
            self.plans[tick]=target
        self.records.append({'epoch':self.num_epochs,'slot':slot,
          'quotas':str(dict(quotas)),'times':str(sorted(s for s in self.plans if s>=slot)),
          'scores':str(scores)})
    def choose(self,slot,views,beliefs):
        if self.method=='no_verify' or slot not in self.plans or slot in self.used:return None
        self.used.add(slot)
        if not views:return None
        preferred=self.plans[slot]
        if preferred in views:return preferred
        if self.method=='joint':
            return max(views,key=lambda v:(beliefs[v].variance*(1+abs(beliefs[v].mean-beliefs[v].reported))*views[v]['contact'],v))
        ids=sorted(views)
        return ids[min(len(ids)-1,int(draw(self.seed,'pick',slot)*len(ids)))]

class NativeJointAdapter(AirFogSimLiveAdapter):
    """Real AirFogSim RB scheduler: DDP demands, EDF, hard RB capacity, keyed outage."""
    def __init__(self,env,c,seed):
        super().__init__(env,c)
        self.seed=seed;self.rb_demand={};self.outage_slots=0;self.rb_shortfall_tti=0
    def schedule_tti(self,slot):
        # Delegate verification reservation and return routing to the tested
        # upstream-live adapter, then replace equal-share *normal* allocations
        # with the finite DDP's actual per-task RB decisions.
        super().schedule_tti(slot)
        rr=self._reservation();v_id=rr.task_id if rr else None
        pending=self._all_transmitting()
        blocked=0;ret={}
        self.env.activated_offloading_tasks_with_RB_Nos.clear()
        reserve=self.cfg.verify_rbs if rr else 0
        ordered=sorted((tid for tid in pending if tid!=v_id),
             key=lambda tid:(pending[tid].task_deadline-
                   (self.env.simulation_time-pending[tid].getTaskArrivalTime()),tid))
        if v_id in pending:
            ordered=[v_id]+ordered
        cursor=reserve
        for tid in ordered:
            target=self.task_target.get(tid,'')
            outage=(target!=self.cfg.target_uav and
                (self.cfg.outage_burst>0 and slot%self.cfg.outage_period<self.cfg.outage_burst or
                 draw(self.seed,'outage',slot,target)<self.cfg.outage_probability))
            if outage:
                blocked+=1
                if rr and tid==v_id:
                    self.total_used_rb_tti-=self.cfg.verify_rbs;rr.rb_active-=self.cfg.verify_rbs
                    if self.tasks[tid].isReturning():rr.return_tti-=1
                    else:rr.upload_tti-=1
                continue
            if tid==v_id:
                actual=list(range(reserve))
            else:
                need=max(1,int(self.rb_demand.get(tid,4)))
                take=min(need,max(0,self.rb_capacity-cursor))
                if take<need:self.rb_shortfall_tti+=1
                actual=list(range(cursor,cursor+take))
                cursor+=take
            if actual:
                self.commsched.setCommunicationWithRB(self.env,tid,actual)
                ret[tid]=actual
        if sum(map(len,ret.values()))>self.rb_capacity:
            raise AssertionError('radio blocks oversubscribed')
        self.outage_slots+=int(blocked>0)
        self.telemetry[-1]['outage_tasks']=blocked
        self.telemetry[-1]['other_rb_allocated']=sum(len(x) for tid,x in ret.items() if tid!=v_id)
        self.telemetry[-1]['verify_rb_active']=reserve if v_id in ret else 0
        self.telemetry[-1]['ddp_rb_shortfall_count']=self.rb_shortfall_tti
        return ret


def distributional_batch(env,c,beliefs,views,tasks,rb_available):
    """Exact enumeration over finite candidate actions, PMF convolution and CVaR.
    FSD pruning ONLY with identical RB/CPU/batch occupancy keys.
    Conditional independence and contact/PHY predictor are model assumptions.
    """
    actions={}
    for tid,cpu,size,return_mb,ddl,value in tasks:
        drop={'id':tid,'target':None,'rb':0,'p':0.,'score':0.,'loss':value,'cpu':cpu,'size':size,'ret':return_mb,'ddl':ddl}
        acts=[drop]
        local_cpu=float(env.UAVs[c.target_uav].getFogProfile().get('cpu',2.3))
        localq=sum(max(0.,t.getTaskCPU()-t.getComputedSize()) for t in env.task_manager._computing_tasks.get(c.target_uav,[]))
        local_latency=(localq+cpu)/local_cpu
        if local_latency<=ddl:
            acts.append(dict(drop,target=c.target_uav,p=1.,score=value-.45*local_latency-.8*cpu))
        for vid,z in views.items():
            b=beliefs[vid]
            for rb in (4,8,12,16):
                if rb>rb_available:continue
                rate=max(.6,c.nominal_rate_rb_mbps*(100/max(100.,z['d']))**2)/8
                delay=z['wait']+cpu/z['cpu']+(size+return_mb)/(rb*rate)
                if delay>=min(ddl,z['contact']):continue
                p=min(.999,b.mean*(1-c.outage_probability)*max(0.,min(1.,(z['contact']-delay)/1.)))
                acts.append(dict(drop,target=vid,rb=rb,p=p,score=value*p-.45*delay-.007*rb-.1*cpu))
        actions[tid]=acts
    dp={(0,0,()):[Frontier()]}
    for tid,*_ in tasks:
        nxt=defaultdict(list)
        for (used,loc,mask),lst in dp.items():
            for action in actions[tid]:
                total=used+action['rb']
                if total>rb_available:continue
                loc2=loc+int(action['target']==c.target_uav)
                if loc2>c.max_local:continue
                occupied=dict(mask)
                if action['target'] not in (None,c.target_uav):
                    occupied[action['target']]=occupied.get(action['target'],0)+1
                    if occupied[action['target']]>c.max_remote:continue
                key=(total,loc2,tuple(sorted(occupied.items())))
                for prev in lst:
                    candidate=Frontier(total,0,prev.utility+action['score'],
                            add_loss(prev.pmf,action['loss'],action['p']),prev.choices+(action,))
                    front=nxt[key]
                    if any(dominates(x,candidate) for x in front):continue
                    front[:]=[x for x in front if not dominates(candidate,x)]
                    front.append(candidate)
        dp=nxt
    states=[s for frontier in dp.values() for s in frontier]
    if not states:raise RuntimeError('drop should ensure candidate')
    considered=[(state,cvar(state.pmf,c.risk_beta)) for state in states]
    feasible=[(state,risk) for state,risk in considered if risk<=c.risk_budget+1e-8]
    if feasible:best,risk=max(feasible,key=lambda z:(z[0].utility,-z[1]))
    else:best,risk=min(considered,key=lambda z:(z[1],-z[0].utility))
    return best.choices,risk,bool(feasible),len(states)

def sumo_replay_trace(out,c,seed):
    """Real SUMO physics-generated trajectories, NOT observed traffic data."""
    import sumolib,traci
    out.mkdir(parents=True,exist_ok=True)
    net=out/'sumo.net.xml'
    subprocess.run(['netgenerate','--grid','--grid.x-number','3','--grid.y-number','2',
      '--grid.length','160','--default.speed','20','--no-turnarounds','-o',str(net)],check=True,capture_output=True)
    edges=[e.getID() for e in sorted(sumolib.net.readNet(str(net)).getEdges(),key=lambda e:e.getID()) if e.getLength()>90]
    if len(edges)<c.vehicles:raise RuntimeError('invalid SUMO edges')
    route=out/'vehicles.rou.xml'
    xml=['<routes><vType id="v" accel="2.6" decel="4.5" length="5" maxSpeed="25" sigma="0"/>']
    for i in range(c.vehicles):
        xml.append(f'<route id="r{i}" edges="{edges[(i+seed)%len(edges)]}"/>')
        xml.append(f'<vehicle id="vehicle_{i}" route="r{i}" type="v" depart="{i*.4:.1f}" departSpeed="max"/>')
    route.write_text(''.join(xml)+'</routes>')
    cfg=out/'sim.sumocfg'
    cfg.write_text(f'<configuration><input><net-file value="{net.name}"/><route-files value="{route.name}"/></input><time><step-length value="{c.tti_s}"/><end value="{(c.steps+10)*c.tti_s}"/></time><random_number><seed value="{seed}"/></random_number></configuration>')
    label=f'joined_{seed}'
    traci.start(['sumo','-c',str(cfg),'--no-step-log','true','--no-warnings','true'],label=label)
    rows=[]
    try:
        conn=traci.getConnection(label)
        for _ in range(c.steps+5):
            conn.simulationStep();t=round(conn.simulation.getTime(),5)
            for vid in sorted(conn.vehicle.getIDList()):
                x,y=conn.vehicle.getPosition(vid)
                rows.append([vid,t,x,y,conn.vehicle.getSpeed(vid),conn.vehicle.getAngle(vid),conn.vehicle.getRouteID(vid)])
    finally:
        traci.switch(label);traci.close()
    trace=out/'sumo_vehicles.csv'
    with trace.open('w',newline='') as f:
        w=csv.writer(f,delimiter=';')
        w.writerow(['vehicle_id','data_timestep','vehicle_x','vehicle_y','vehicle_speed','vehicle_angle','vehicle_route'])
        w.writerows(rows)
    return net,trace,len(rows)

def run_joint_case(method,seed,c,upstream,out,trace=None):
    import numpy as np,yaml
    from airfogsim import AirFogSimEnv
    random.seed(seed);np.random.seed(seed)
    folder=out/f'{method}_{seed}'
    path=create_fixtures(folder,c,seed,upstream)
    cfg=yaml.safe_load(path.read_text())
    if trace:
        cfg['sumo']['sumo_net']=str(trace[0].resolve())
        cfg['traffic']['tripinfo']=str(trace[1].resolve())
    elif c.stress_departures:
        tracepath=Path(cfg['traffic']['tripinfo'])
        with tracepath.open(newline='') as f:lines=list(csv.DictReader(f,delimiter=';'))
        lines=[r for r in lines if not (int(r['vehicle_id'].split('_')[-1])%2==0 and float(r['data_timestep'])>3.5+int(r['vehicle_id'].split('_')[-1])*1.2)]
        with tracepath.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(lines[0]),delimiter=';');w.writeheader();w.writerows(lines)
    cfg['task']['tti_threshold']=.65
    cfg['authentication']={'malicious_ratio':0.}
    path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    env=AirFogSimEnv(cfg,interactive_mode=None)
    if not cfg.get('storage',{}).get('capacity_bytes'):
        env._updateStorage=lambda:None # unrelated pinned-upstream storage.items defect
    adapter=NativeJointAdapter(env,c,seed)
    beliefs={}
    for i in range(c.vehicles):
        claimed=.96 if i<2 else .70+.06*(i%2)
        beliefs[f'vehicle_{i}']=Belief(1+c.prior_strength*claimed,1+c.prior_strength*(1-claimed),claimed)
    slow=SlowBudget(c,seed,method)
    observed=set();old={};planned=[];checks=[];outcomes=[];resource=[]
    try:
        adapter.advance(0)
        xx=cfg['traffic']['x_range']; yy=cfg['traffic']['y_range']
        u=env.traffic_manager._UAV_infos[c.target_uav]
        u['position']=[(xx[0]+xx[1])/2,(yy[0]+yy[1])/2,70.]
        u.update(speed=0,last_speed=0,angle=0,phi=0,acceleration=0)
        for slot in range(1,c.steps):
            views=observable_views(env,c,old)
            if (slot-1)%c.slow_period==0:slow.update(slot,views,beliefs)
            if not adapter.active_reservation:
                target=slow.choose(slot,views,beliefs)
                if target is not None:
                    r=adapter.create_verification(target,slot);slow.issued+=1
                    checks.append(dict(seed=seed,method=method,slot=slot,vid=target,task_id=r.task_id))
            if (slot-1)%c.normal_interval==0:
                tasks=[]
                for k in range(c.task_batch):
                    tid=f'JOB_{slot}_{k}'
                    tasks.append((tid,.57+.32*draw(seed,'cpu',slot,k),.12+.06*draw(seed,'size',slot,k),.015,c.normal_deadline_s*.9,10))
                avail=adapter.rb_capacity-(c.verify_rbs if adapter._reservation() else 0)
                clock=time.perf_counter()
                actions,risk,feasible,front=distributional_batch(env,c,beliefs,views,tasks,avail)
                elapsed=time.perf_counter()-clock
                resource.append(dict(seed=seed,method=method,slot=slot,risk=risk,feasible=int(feasible),frontier=front,wall_s=elapsed))
                for option in actions:
                    tid=option['id'];node=option['target']
                    planned.append(dict(seed=seed,method=method,id=tid,target=node or 'drop',p=option['p'],rb=option['rb']))
                    if node is None:continue
                    adapter.inject(tid,c.target_uav,node,'work',env.simulation_time,cpu=option['cpu'],size=option['size'],
                                   reply=option['ret'] if node!=c.target_uav else 0.,deadline=option['ddl'])
                    adapter.rb_demand[tid]=option['rb']
            prior={v:tuple(d['position']) for v,d in env.traffic_manager.getVehicleTrafficInfos().items()}
            adapter.advance(slot)
            for r in adapter.completed_verifications(observed):
                if r.status=='done':
                    beliefs[r.vehicle]=beliefs[r.vehicle].update(_get_reliability_observation(seed,r.task_id,r.vehicle))
            old=prior
        done=0
        for row in planned:
            tid=row['id'];target=row['target']
            status=adapter.task_state(tid) if target!='drop' else 'dropped'
            correct=status=='done' and (target==c.target_uav or _get_reliability_observation(seed,tid,target))
            done+=int(correct)
            outcomes.append(dict(seed=seed,method=method,task_id=tid,target=target,status=status,correct=int(correct)))
        verify=[dict(seed=seed,method=method,task_id=r.task_id,target=r.vehicle,
             status=adapter.task_state(r.task_id),up=r.upload_tti,cpu=r.compute_tti,down=r.return_tti,
             rb=r.rb_active,cpu_gc=r.verify_cpu_gc) for r in adapter.reservations.values()]
        return dict(seed=seed,method=method,jobs=len(planned),correct=done,rate=done/max(1,len(planned)),
            verifies=slow.issued,verify_success=sum(x['status']=='done' for x in verify),
            rb_reserved=adapter.total_reserved_rb_tti,rb_active=adapter.total_used_rb_tti,
            cpu_reserved=adapter.total_reserved_cpu_gc,cpu_executed=adapter.total_verify_cpu_gc,
            risk_infeasible=sum(x['feasible']==0 for x in resource),outage_slots=adapter.outage_slots,
            _tasks=outcomes,_checks=checks,_verify=verify,_dp=resource,
            _epochs=[dict(seed=seed,method=method,**x) for x in slow.records],
            _radio=[dict(seed=seed,method=method,**x) for x in adapter.telemetry])
    finally:
        env.close()

def joint_campaign(c,seeds,upstream,out):
    out.mkdir(parents=True,exist_ok=True)
    data=defaultdict(list);trace_map={}
    if c.sumo_mode:
        for seed in seeds:
            tup=sumo_replay_trace(out/f'sumo_seed_{seed}',c,seed)
            trace_map[seed]=tup
            data['sumo'].append({'seed':seed,'rows':tup[2]})
    for seed in seeds:
        for method in ('joint','random_verify','no_verify'):
            r=run_joint_case(method,seed,c,upstream,out,trace_map.get(seed))
            for kind in ('tasks','checks','verify','dp','epochs','radio'):
                data[kind].extend(r.pop('_'+kind))
            data['summary'].append(r)
    for name,records in data.items():
        if not records:continue
        with (out/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    pairs=[]
    for seed in seeds:
        a,b=[next(row for row in data['summary'] if row['seed']==seed and row['method']==m) for m in ('joint','random_verify')]
        if a['jobs']!=b['jobs']:raise AssertionError('unpaired workload')
        pairs.append(dict(seed=seed,delta=a['rate']-b['rate'],
             equal_attempts=a['verifies']==b['verifies'],
             equal_reserved_RB=a['rb_reserved']==b['rb_reserved'],
             equal_reserved_CPU=abs(a['cpu_reserved']-b['cpu_reserved'])<1e-8,
             equal_actual_cost=a['rb_active']==b['rb_active'] and abs(a['cpu_executed']-b['cpu_executed'])<1e-8))
    for row in data['verify']:
        if row['status']=='done' and any(float(row[x])<=0 for x in ('up','cpu','down','rb','cpu_gc')):
            raise AssertionError('completed verification lacked native AirFogSim RB/CPU stages')
    manifest={'status':'development, not unseen-seed publication evidence','config':vars(c),'seeds':seeds,
        'airfogsim':'pinned upstream native task/radio/cpu progression',
        'SUMO':'simulated trips replayed as trajectory CSV, not field GPS nor online TraCI coupling',
        'comparison':'matched potential token/envelope budget; actual successful challenges/cost may differ',
        'model':'finite loss-distribution DP, conditional independent failure approximation'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'paired.json').write_text(json.dumps(pairs,indent=2))
    print(json.dumps({'rows':len(data['summary']),
       'rates':{m:statistics.mean(row['rate'] for row in data['summary'] if row['method']==m)
            for m in ('joint','random_verify','no_verify')},
       'paired_delta':statistics.mean(x['delta'] for x in pairs),
       'matched_attempts':sum(x['equal_attempts'] for x in pairs),
       'matched_actual_cost':sum(x['equal_actual_cost'] for x in pairs)},indent=2))

def joint_main():
    p=argparse.ArgumentParser()
    p.add_argument('--airfogsim-root',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--seeds',type=int,default=2)
    p.add_argument('--steps',type=int,default=97)
    p.add_argument('--sumo-mode',action='store_true')
    p.add_argument('--adaptive-timing',action='store_true')
    p.add_argument('--outage',type=float,default=.08)
    p.add_argument('--burst',type=int,default=1)
    p.add_argument('--no-departures',action='store_true')
    args=p.parse_args()
    cfg=JointConfig(steps=args.steps,sumo_mode=args.sumo_mode,adaptive_timing=args.adaptive_timing,
        outage_probability=args.outage,outage_burst=args.burst,stress_departures=not args.no_departures)
    joint_campaign(cfg,list(range(args.seeds)),args.airfogsim_root,args.out)

if __name__=='__main__':joint_main()
