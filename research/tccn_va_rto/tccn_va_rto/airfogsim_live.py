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


def run_live_case(method:str,seed:int,c:LiveConfig,upstream:Path,out:Path):
    if method not in ('va_rto','random_verify'):raise ValueError(method)
    import numpy as np
    from airfogsim import AirFogSimEnv
    random.seed(seed);np.random.seed(seed)
    fixture=out/f'{method}_{seed}'
    config_file=create_fixtures(fixture,c,seed,upstream)
    import yaml
    env=AirFogSimEnv(yaml.safe_load(config_file.read_text()),interactive_mode=None)
    # Upstream 76c0edb4 _updateStorage iterates .items() on a list returned
    # by TaskManager.getDoneTasks(). Storage is unused in this experiment;
    # disable ONLY the unrelated, broken storage stage (radio/CPU stay native).
    if not env.config.get('storage',{}).get('capacity_bytes'):
        env._updateStorage=lambda: None
    adapter=AirFogSimLiveAdapter(env,c)
    beliefs={f'vehicle_{i}':Belief(1+c.prior_strength*(.96 if i<2 else hidden_probability(i)),
                 1+c.prior_strength*(1-(.96 if i<2 else hidden_probability(i))),.96 if i<2 else hidden_probability(i))
             for i in range(c.vehicles)}
    responses=set()
    outcome_records=[]
    checks=[]
    slow_epochs=0
    try:
        # First tick instantiates AirFogSim vehicles and UAV. No tasks injected.
        adapter.advance(0)
        info=env.traffic_manager._UAV_infos[c.target_uav]
        info['position']=[260.0,200.0,70.0]
        info.update(speed=0.,last_speed=0.,angle=0.,phi=0.,acceleration=0.)
        for slot in range(1,c.steps):
            now=env.simulation_time
            if slot%c.slow_period==1:slow_epochs+=1
            # Matched attempt times and identical per-verification resource envelope.
            if slot%c.verify_interval==1 and (not adapter.active_reservation):
                selected=_select_verification(method,env,c,beliefs,seed,slot)
                if selected is not None:
                    reservation=adapter.create_verification(selected,slot)
                    checks.append({'seed':seed,'method':method,'slot':slot,'target':selected,
                                   'task_id':reservation.task_id})
            if slot%c.normal_interval==1:
                taskid=f'VA_DATA_{slot}'
                node=choose_work_target(env,c,beliefs,now,seed,taskid)
                adapter.inject(taskid,c.target_uav,node,'work',now,cpu=c.normal_task_cpu_gc,
                    size=c.normal_task_mb,reply=c.normal_reply_mb if node!=c.target_uav else 0,
                    deadline=c.normal_deadline_s)
            adapter.advance(slot)
            for r in adapter.completed_verifications(responses):
                if r.status=='done':
                    # Observe correctness ONLY after physical return.
                    beliefs[r.vehicle]=beliefs[r.vehicle].update(_get_reliability_observation(seed,r.task_id,r.vehicle))
        for tid,task in adapter.tasks.items():
            state=adapter.task_state(tid)
            target=adapter.task_target[tid]
            success=(state=='done') and (target==c.target_uav or _get_reliability_observation(seed,tid,target))
            outcome_records.append({'seed':seed,'method':method,'task_id':tid,'kind':adapter.kind[tid],
                'target':target,'state':state,'on_time_correct':int(success),
                'latency_s':float(task.task_delay) if state=='done' else '',
                'deadline_s':task.task_deadline})
        return {
            'seed':seed,'method':method,'engine':'AirFogSimEnv (upstream task+radio+CPU)',
            'jobs':sum(r['kind']=='work' for r in outcome_records),
            'correct_on_time':sum(r['on_time_correct'] for r in outcome_records if r['kind']=='work'),
            'rate':statistics.mean(r['on_time_correct'] for r in outcome_records if r['kind']=='work'),
            'verifications_issued':len(checks),
            'verifications_completed':sum(r['state']=='done' for r in outcome_records if r['kind']=='verification'),
            'verify_rb_reserved_tti':adapter.total_reserved_rb_tti,
            'verify_rb_active_tti':adapter.total_used_rb_tti,
            'verify_cpu_reserved_gc':round(adapter.total_reserved_cpu_gc,8),
            'verify_cpu_executed_gc':round(adapter.total_verify_cpu_gc,8),
            'slow_epochs':slow_epochs,
            '_workload':outcome_records,'_radio':adapter.telemetry,'_verify':checks,
        }
    finally:
        env.close()


def paired_live_campaign(c:LiveConfig,seeds:list[int],upstream:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    summaries=[];work=[];radio=[];verification=[]
    for seed in seeds:
        for method in ('va_rto','random_verify'):
            row=run_live_case(method,seed,c,upstream,out)
            work.extend(row.pop('_workload'))
            radio.extend([dict(seed=seed,method=method,**x) for x in row.pop('_radio')])
            verification.extend(row.pop('_verify'))
            summaries.append(row)
    def write(name,rows):
        if not rows:return
        with (out/name).open('w',encoding='utf8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write('summary.csv',summaries);write('task_records.csv',work)
    write('radio_cpu_tti.csv',radio);write('verification_requests.csv',verification)
    # Invariant for strict matched offered resources; completed services may differ.
    for seed in seeds:
        a,b=[z for z in summaries if z['seed']==seed]
        for k in ('verifications_issued','verify_rb_reserved_tti','verify_cpu_reserved_gc'):
            if a[k]!=b[k]:raise AssertionError(f'paired verification budget mismatch: seed={seed}: {k}: {a[k]} != {b[k]}')
    meta={'engine':'real upstream AirFogSimEnv (traffic_mode=real, mobility CSV)',
          'basis':'full upstream radio and CPU/task progression; stochastic service-integrity injection',
          'upstream':str(upstream),'config':vars(c),'seeds':seeds,
          'method_comparison':'matched verification attempts and reserved RB/CPU budgets, NOT equal realized throughput',
          'scientific_status':'engineering integration; not final TCCN experiment'}
    (out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    for m in ('va_rto','random_verify'):
        x=[z for z in summaries if z['method']==m]
        print(m,'rate',round(statistics.mean(z['rate'] for z in x),4),
              'verify issued',statistics.mean(z['verifications_issued'] for z in x),
              'RB reserved',statistics.mean(z['verify_rb_reserved_tti'] for z in x),
              'CPU reserved',statistics.mean(z['verify_cpu_reserved_gc'] for z in x))
    return summaries


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--airfogsim-root',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('results/airfogsim_live'))
    p.add_argument('--seeds',type=int,default=2)
    p.add_argument('--steps',type=int,default=80)
    args=p.parse_args()
    if args.seeds<1 or args.steps<5:p.error('invalid seeds/steps')
    paired_live_campaign(LiveConfig(steps=args.steps),list(range(args.seeds)),args.airfogsim_root,args.out)

if __name__=='__main__':main()