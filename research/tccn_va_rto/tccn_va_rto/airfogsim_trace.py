"""AirFogSim source-compatible trace recording and SUMO-based mobility replay.

The bridge records real AirFogSim positions and task profiles; it does NOT alter
AirFogSim's task execution or claim full live scheduler integration. Service
correctness is a separate explicitly injected stochastic research model.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from .core import World, Vehicle, Task


def record_snapshot(env, time: float) -> dict:
    """Use the concrete public AirFogSim APIs inspected in its upstream repository."""
    from airfogsim.airfogsim_scheduler import AirFogSimScheduler
    scheduler=AirFogSimScheduler
    veh=env.traffic_manager.getVehicleTrafficInfos()
    uavs=scheduler.getUAVTrafficInfos(env)
    tasks=scheduler.getTaskScheduler().getAllToOffloadTaskInfos(env)
    def entities(d):
        return {str(k):{'position':list(map(float,v['position'])),
                        'speed':float(v.get('speed',0))} for k,v in d.items()}
    return {'time':float(time),'vehicles':entities(veh),'uavs':entities(uavs),
            'waiting_tasks':[{'task_id':str(t['task_id']),'task_node_id':str(t['task_node_id']),
                'task_cpu':float(t['task_cpu']),'task_size':float(t['task_size']),
                'required_returned_size':float(t.get('required_returned_size',0)),
                'task_deadline':float(t['task_deadline']),
                'task_priority':float(t['task_priority']),
                'task_arrival_time':float(t['task_arrival_time'])} for t in tasks]}


def record_airfogsim(config_file:Path,out:Path,max_time:float=60):
    """Execute AirFogSim (and SUMO) when installed. Never silently use a mock."""
    try:
        import yaml
        from airfogsim import AirFogSimEnv,BaseAlgorithmModule
    except ImportError as e:
        raise RuntimeError('Install upstream AirFogSim + SUMO and its requirements') from e
    config=yaml.safe_load(config_file.read_text(encoding='utf8'))
    env=AirFogSimEnv(config,interactive_mode=None)
    module=BaseAlgorithmModule();module.initialize(env)
    out.parent.mkdir(parents=True,exist_ok=True)
    try:
        with out.open('w',encoding='utf8') as f:
            while not env.isDone() and env.simulation_time<max_time:
                f.write(json.dumps(record_snapshot(env,env.simulation_time))+'\n')
                module.scheduleStep(env)
                env.step()
    finally:
        env.close()


class AirFogSimTraceWorld(World):
    """Replays logged AirFogSim traffic + task arrivals with explicit unit mapping.

    Upstream task_size interpreted as megabytes and task_cpu as giga-cycles:
    these units must be checked against the selected AirFogSim version/config.
    At least one UAV-produced task is needed to make a valid replay.
    """
    def __init__(self, config, seed, path:Path, *, data_mb_to_mbit=8.0):
        super().__init__(config,seed)
        frames=[json.loads(x) for x in path.read_text(encoding='utf8').splitlines() if x.strip()]
        if not frames:raise ValueError('empty AirFogSim trace')
        self.frames=sorted(frames,key=lambda x:x['time'])
        self.source_ids=sorted(set().union(*(set(f['vehicles']) for f in frames)))[:config.vehicles]
        if len(self.source_ids)<config.vehicles:
            raise ValueError('not enough vehicles in source AirFogSim trace')
        self.primary_uav=next(iter(self.frames[0]['uavs']),None)
        if self.primary_uav is None:raise ValueError('no UAV present in first frame')
        # Keep hidden service parameters from a separately declared injection model.
        self.vehicles=[Vehicle(i,self.vehicles[i].x0,self.vehicles[i].speed,
            self.vehicles[i].cpu_gcps,self.vehicles[i].hidden_reliability,
            self.vehicles[i].reported_reputation) for i in range(config.vehicles)]
        self.batches=[[] for _ in range(config.steps)]
        seen=set()
        for frame in self.frames:
            for t in frame['waiting_tasks']:
                if t['task_node_id'] not in frame['uavs'] or t['task_id'] in seen:continue
                seen.add(t['task_id'])
                slot=int(float(t['task_arrival_time'])/config.slot_s)
                if not 0<=slot<config.steps or len(self.batches[slot])>=config.tasks_per_slot:continue
                # Normalize to 1..10 reward units; not a platform accuracy metric.
                value=max(1,int(round(10*float(t['task_priority']))))
                self.batches[slot].append(Task(f'{slot}-{len(self.batches[slot])}',value,
                    float(t['task_size'])*data_mb_to_mbit,
                    float(t.get('required_returned_size',0))*data_mb_to_mbit,
                    float(t['task_cpu']),float(t['task_deadline'])))
        if not any(self.batches):raise ValueError('trace has no UAV-generated task arrivals')

    def _node_position(self,node_id:str,t:float):
        arr=self.frames
        if t<=arr[0]['time']:i=j=0
        elif t>=arr[-1]['time']:i=j=len(arr)-1
        else:
            lo,hi=0,len(arr)-1
            while lo+1<hi:
                mid=(lo+hi)//2
                if arr[mid]['time']<=t:lo=mid
                else:hi=mid
            i,j=lo,hi
        key='uavs' if node_id==self.primary_uav else 'vehicles'
        a=arr[i][key].get(node_id)
        b=arr[j][key].get(node_id)
        if not a or not b:return None
        if i==j:return a['position']
        weight=(t-arr[i]['time'])/(arr[j]['time']-arr[i]['time'])
        return [(1-weight)*x+weight*y for x,y in zip(a['position'],b['position'])]

    def distance(self,v:Vehicle,t:float):
        p=self._node_position(self.source_ids[v.vid],t)
        u=self._node_position(self.primary_uav,t)
        if p is None or u is None:return float('inf')
        return math.dist(p,u)