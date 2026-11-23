"""VA-RTO prototype; independent stochastic simulator, NOT an AirFogSim execution.
No hardcoded outcome curves. Model and solver are deliberately lightweight.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, statistics
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

@dataclass(frozen=True)
class Config:
    steps:int=40
    vehicles:int=5
    tasks_per_slot:int=3
    slow_period:int=8
    slot_s:float=1.0
    road_half_m:float=450.0
    contact_radius_m:float=280.0
    uav_height_m:float=50.0
    bandwidth_mhz:float=7.0
    snr_at_50m:float=75.0
    air_units:int=16
    local_gcps:float=2.8
    max_verifications_per_epoch:int=3
    verification_airtime_s:float=0.13
    verification_cpu_gc:float=0.25
    verification_cost:float=0.24
    prior_strength:float=5.0
    cvar_beta:float=0.9
    cvar_budget:float=20.0
    latency_weight:float=0.17
    energy_weight:float=0.06
    future_info_weight:float=1.4

@dataclass(frozen=True)
class Task:
    tid:str
    value:int
    data_mbit:float
    result_mbit:float
    cpu_gc:float
    deadline_s:float

@dataclass(frozen=True)
class Vehicle:
    vid:int
    x0:float
    speed:float
    cpu_gcps:float
    hidden_reliability:float
    reported_reputation:float

@dataclass(frozen=True)
class Belief:
    a:float
    b:float
    reported:float
    @property
    def mean(self):return self.a/(self.a+self.b)
    @property
    def variance(self):return self.a*self.b/((self.a+self.b)**2*(self.a+self.b+1))
    def update(self,valid:bool):return Belief(self.a+int(valid),self.b+int(not valid),self.reported)

@dataclass(frozen=True)
class Option:
    task:Task
    target:int  # -2 drop, -1 local, >=0 remote
    air:int
    latency:float
    p:float
    energy:float
    utility:float
    upload_s:float=0.0

@dataclass
class Frontier:
    air:int=0
    mask:int=0
    utility:float=0.0
    pmf:dict[int,float]=field(default_factory=lambda:{0:1.0})
    choices:tuple[Option,...]=()

@dataclass
class Solution:
    choices:tuple[Option,...]
    utility:float
    cvar:float
    feasible:bool
    frontier_size:int

def draw(seed,*parts):
    raw=(str(seed)+'|'+ '|'.join(map(str,parts))).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')/2**64

def cvar(pmf:dict[int,float],beta:float)->float:
    """Rockafellar-Uryasev discrete CVaR, handles probability atoms exactly."""
    if not 0<beta<1 or abs(sum(pmf.values())-1)>1e-7:raise ValueError('invalid PMF/beta')
    return min(z+sum(p*max(0,x-z) for x,p in pmf.items())/(1-beta) for z in pmf)

def add_loss(pmf,loss:int,p_success:float):
    out=defaultdict(float)
    for x,p in pmf.items():
        out[x]+=p*p_success
        out[x+loss]+=p*(1-p_success)
    return {x:v for x,v in out.items() if v>1e-13}

def dominates(a:Frontier,b:Frontier):
    if a.utility+1e-9<b.utility:return False
    fa=fb=0
    for x in sorted(set(a.pmf)|set(b.pmf)):
        fa+=a.pmf.get(x,0)
        fb+=b.pmf.get(x,0)
        if fa+1e-9<fb:return False
    return True

def reflect(x,speed,t,half):
    p=(x+half+speed*t)%(4*half)
    return -half+(p if p<=2*half else 4*half-p)

class World:
    def __init__(self,c:Config,seed:int):
        self.c,self.seed=c,seed
        rng=random.Random(seed)
        self.vehicles=[]
        for i in range(c.vehicles):
            p=[.47,.62,.84,.96,.91][i%5]
            rep=.98 if p<.7 else min(.98,max(.02,p+rng.uniform(-.05,.04)))
            self.vehicles.append(Vehicle(i,rng.uniform(-210,210),rng.choice((-1,1))*rng.uniform(10,22),rng.uniform(4.5,9),p,rep))
        self.batches=[]
        for t in range(c.steps):
            tasks=[]
            for k in range(c.tasks_per_slot):
                r=random.Random(int(draw(seed,'task',t,k)*2**32))
                tasks.append(Task(f'{t}-{k}',r.choice((6,8,10)),r.uniform(1.5,4.3),r.uniform(.04,.15),r.uniform(3.5,9.5),r.uniform(1.6,3.6)))
            self.batches.append(tasks)
    def distance(self,v,t):return math.hypot(reflect(v.x0,v.speed,t,self.c.road_half_m),self.c.uav_height_m)
    def rate(self,v,t):
        snr=self.c.snr_at_50m*(50/max(50,self.distance(v,t)))**2
        return self.c.bandwidth_mhz*math.log2(1+snr)
    def contact(self,v,t):
        if self.distance(v,t)>self.c.contact_radius_m:return 0.0
        for i in range(1,161):
            if self.distance(v,t+.05*i)>self.c.contact_radius_m:return (i-1)*.05
        return 8.0
    def p_link(self,v,t):
        d=self.distance(v,t)
        return 0.0 if d>self.c.contact_radius_m else max(.55,min(.985,.99-.23*(d/self.c.contact_radius_m)**2))

class DistributionalDP:
    """Loss-PMF knapsack, FSD pruning within identical consumption states.
    Small exact stage-wise solver; assumes conditional independence of task outcomes.
    """
    def __init__(self,world:World):self.world=world
    def options(self,task:Task,t:float,beliefs:dict[int,Belief],busy:dict[int,float],airmax:int,oracle=False):
        c=self.world.c
        arrival_slot=int(task.tid.split("-")[0])
        elapsed=t-arrival_slot*c.slot_s
        opts=[Option(task,-2,0,task.deadline_s,0,0,0)]
        ready=max(t,busy.get(-1,t))-t
        local=elapsed+ready+task.cpu_gc/c.local_gcps
        if local<=task.deadline_s:
            energy=.3*task.cpu_gc
            opts.append(Option(task,-1,0,local,.995,energy,.995*task.value-c.latency_weight*local-c.energy_weight*energy))
        for v in self.world.vehicles:
            win=self.world.contact(v,t)
            if win<=0:continue
            rate=self.world.rate(v,t)
            up=task.data_mbit/rate
            down=task.result_mbit/rate
            airtime=up+down
            units=math.ceil(airtime/c.slot_s*c.air_units-1e-12)
            if units>airmax or units<=0:continue
            remote_duration=max(up,max(t,busy.get(v.vid,t))-t)+task.cpu_gc/v.cpu_gcps+down
            completion=elapsed+remote_duration
            # Entire upload/compute/return must fit one contact window after verification.
            if completion>task.deadline_s or remote_duration>win:continue
            p=(v.hidden_reliability if oracle else beliefs[v.vid].mean)*self.world.p_link(v,t)
            energy=.09*airtime+.02*task.cpu_gc
            utility=p*task.value-c.latency_weight*completion-c.energy_weight*energy
            opts.append(Option(task,v.vid,units,completion,p,energy,utility,up))
        return opts
    def solve(self,tasks:list[Task],t:float,beliefs:dict[int,Belief],busy:dict[int,float],airmax:int,
              budget:float,neutral=False,oracle=False):
        states={(0,0):[Frontier()]}
        for task in tasks:
            next_states=defaultdict(list)
            opts=self.options(task,t,beliefs,busy,airmax,oracle)
            for (air,mask),frontier in states.items():
                for o in opts:
                    bit=0 if o.target==-2 else (1 if o.target==-1 else 1<<(o.target+1))
                    air2=air+o.air
                    if air2>airmax or mask&bit:continue
                    key=(air2,mask|bit)
                    for prev in frontier:
                        candidate=Frontier(air2,mask|bit,prev.utility+o.utility,
                                         add_loss(prev.pmf,task.value,o.p),prev.choices+(o,))
                        arr=next_states[key]
                        if any(dominates(z,candidate) for z in arr):continue
                        arr[:]=[z for z in arr if not dominates(candidate,z)]
                        arr.append(candidate)
            states=next_states
        candidates=[x for arr in states.values() for x in arr]
        if not candidates:raise RuntimeError('drop is always an available action')
        scored=[(x,cvar(x.pmf,self.world.c.cvar_beta)) for x in candidates]
        allowed=[(x,r) for x,r in scored if neutral or r<=budget+1e-8]
        if allowed:
            best,r=max(allowed,key=lambda x:(x[0].utility,-x[1]))
            return Solution(best.choices,best.utility,r,True,len(candidates))
        # No silent safety claim when original risk bound cannot be met.
        best,r=min(scored,key=lambda x:(x[1],-x[0].utility))
        return Solution(best.choices,best.utility,r,False,len(candidates))

class Experiment:
    def __init__(self,c:Config,seed:int,method:str,trace:Path|None=None):
        if method not in {'va_rto','ddp','neutral','oracle','random_verify'}:raise ValueError(method)
        self.c,self.seed,self.method=c,seed,method
        if trace is None:
            self.world=World(c,seed)
        else:
            from .airfogsim_trace import AirFogSimTraceWorld
            self.world=AirFogSimTraceWorld(c,seed,trace)
        self.dp=DistributionalDP(self.world)
        self.busy={}
        self.beliefs={v.vid:Belief(1+c.prior_strength*v.reported_reputation,1+c.prior_strength*(1-v.reported_reputation),v.reported_reputation) for v in self.world.vehicles}
        self.epoch_left=0
        self.epoch_quotas:dict[int,int]={}
        self.epoch_target_history:list[tuple[int,dict[int,int]]]=[]
        self.slow_updates=0
        self.verifications=0
        self.invalid_risk_slots=0
        self.total_frontier=0
        self.records=[]
    def slow_update(self,t:float):
        """Slow controller selects verifiable nodes and assigns epoch quotas.

        Uses observable posterior variance, disagreement with reported reputation,
        and predicted contact opportunities over the next L slots. It DOES NOT
        use hidden service probabilities. Quotas persist until the next epoch.
        """
        c=self.c
        self.slow_updates+=1
        self.epoch_left=c.max_verifications_per_epoch
        scores=[]
        for v in self.world.vehicles:
            b=self.beliefs[v.vid]
            opportunities=sum(self.world.contact(v,t+i*c.slot_s)>1.0
                              for i in range(c.slow_period))
            if opportunities==0:continue
            disagreement=abs(b.mean-v.reported_reputation)
            score=b.variance*(1+2*disagreement)*(0.25+0.75*opportunities/c.slow_period)
            scores.append((score,v.vid,opportunities))
        scores.sort(reverse=True)
        self.epoch_quotas={v.vid:0 for v in self.world.vehicles}
        # Give at most one quota at a time, allowing repeated challenges if
        # the epoch budget exceeds the number of eligible vehicles.
        for _ in range(c.max_verifications_per_epoch):
            if not scores:break
            _,vid,opportunities=min(scores,key=lambda z:(self.epoch_quotas[z[1]]/z[2]-z[0],z[1]))
            self.epoch_quotas[vid]+=1
        self.epoch_target_history.append((self.slow_updates,dict(self.epoch_quotas)))
    def verify(self,vid:int,t:float):
        v=self.world.vehicles[vid]
        self.epoch_left-=1
        self.epoch_quotas[vid]-=1
        self.verifications+=1
        self.busy[vid]=max(t,self.busy.get(vid,t))+self.c.verification_cpu_gc/v.cpu_gcps
        if draw(self.seed,'vlink',t,vid)>=self.world.p_link(v,t):return None
        valid=draw(self.seed,'vservice',t,vid)<v.hidden_reliability
        self.beliefs[vid]=self.beliefs[vid].update(valid)
        return valid
    def verification_choice(self,batch:list[Task],t:float):
        if self.method not in ('va_rto','random_verify') or self.epoch_left==0:return None
        candidates=[v for v in self.world.vehicles if self.epoch_quotas.get(v.vid,0)>0
                    and self.world.contact(v,t)>1]
        if not candidates:return None
        if self.method=='random_verify':
            return candidates[int(draw(self.seed,'random-verify',t)*len(candidates))].vid
        c=self.c
        remaining=c.air_units-math.ceil(c.verification_airtime_s/c.slot_s*c.air_units)
        if remaining<=0:return None
        base=self.dp.solve(batch,t,self.beliefs,self.busy,c.air_units,c.cvar_budget)
        base_objective=base.utility-.5*max(0,base.cvar-c.cvar_budget)
        best_vid,best_gain=None,0.0
        for v in candidates:
            b=self.beliefs[v.vid]
            busy=dict(self.busy)
            busy[v.vid]=max(t,busy.get(v.vid,t))+c.verification_cpu_gc/v.cpu_gcps
            vals=[]
            for posterior in (b.update(True),b.update(False),b):
                altered=dict(self.beliefs)
                altered[v.vid]=posterior
                sol=self.dp.solve(batch,t+c.verification_airtime_s,altered,busy,remaining,c.cvar_budget)
                vals.append(sol.utility-.5*max(0,sol.cvar-c.cvar_budget))
            p_link=self.world.p_link(v,t)
            expected=p_link*(b.mean*vals[0]+(1-b.mean)*vals[1])+(1-p_link)*vals[2]
            future_information=(b.variance-(b.mean*b.update(True).variance+(1-b.mean)*b.update(False).variance))*100*c.future_info_weight
            gain=expected-base_objective-c.verification_cost+future_information
            if gain>best_gain:best_gain,best_vid=gain,v.vid
        return best_vid
    def run(self):
        c=self.c
        for slot,batch in enumerate(self.world.batches):
            now=slot*c.slot_s
            if slot%c.slow_period==0:self.slow_update(now)
            vid=self.verification_choice(batch,now)
            signal=None
            if vid is not None:signal=self.verify(vid,now)
            air=c.air_units-(math.ceil(c.verification_airtime_s/c.slot_s*c.air_units) if vid is not None else 0)
            decision_now=now+(c.verification_airtime_s if vid is not None else 0.0)
            result=self.dp.solve(batch,decision_now,self.beliefs,self.busy,air,c.cvar_budget,
                                 neutral=self.method=='neutral',oracle=self.method=='oracle')
            self.total_frontier+=result.frontier_size
            self.invalid_risk_slots+=int(not result.feasible)
            for opt in result.choices:
                remote=opt.target>=0
                node=self.world.vehicles[opt.target] if remote else None
                if opt.target==-2:success=False
                elif opt.target==-1:success=draw(self.seed,'local',opt.task.tid)<.995
                else:
                    success=(draw(self.seed,'link',opt.task.tid,opt.target)<self.world.p_link(node,now)
                         and draw(self.seed,'work',opt.task.tid,opt.target)<node.hidden_reliability)
                delay=(opt.latency*(.9+.2*draw(self.seed,'jitter',opt.task.tid,opt.target))
                       if opt.target!=-2 else opt.task.deadline_s)
                success=bool(success and delay<=opt.task.deadline_s)
                if opt.target!=-2:
                    cpu=node.cpu_gcps if remote else c.local_gcps
                    self.busy[opt.target]=max(decision_now+opt.upload_s,self.busy.get(opt.target,decision_now))+opt.task.cpu_gc/cpu
                self.records.append({'seed':self.seed,'method':self.method,'slot':slot,'task':opt.task.tid,
                    'target':opt.target,'value':opt.task.value,'deadline_s':opt.task.deadline_s,
                    'delay_s':delay,'correct_on_time':int(success),'energy_j':opt.energy,
                    'predicted_success':opt.p,'predicted_cvar':result.cvar,
                    'risk_feasible':int(result.feasible),'verification_node':vid if vid is not None else '',
                    'verification_signal':signal if vid is not None else ''})
        return {'seed':self.seed,'method':self.method,'tasks':len(self.records),
            'on_time_correct_rate':statistics.mean(x['correct_on_time'] for x in self.records) if self.records else 0.0,
            'mean_delay_s':statistics.mean(x['delay_s'] for x in self.records) if self.records else 0.0,
            'energy_j':sum(x['energy_j'] for x in self.records)+self.verifications*0.08,
            'verification_energy_j':self.verifications*0.08,
            'completed_value':sum(x['value']*x['correct_on_time'] for x in self.records),
            'verifications':self.verifications,'verify_airtime_s':self.verifications*c.verification_airtime_s,
            'verify_cpu_gc':self.verifications*c.verification_cpu_gc,
            'risk_infeasible_slots':self.invalid_risk_slots,'slow_updates':self.slow_updates,
            'mean_frontier_size':self.total_frontier/c.steps,
            'slow_budget_targets':len({v for _,quota in self.epoch_target_history for v,n in quota.items() if n>0})}

def campaign(c:Config,seeds:list[int],methods:list[str],out:Path,trace:Path|None=None):
    out.mkdir(parents=True,exist_ok=True)
    summaries=[];traces=[]
    for seed in seeds:
        for method in methods:
            exp=Experiment(c,seed,method,trace=trace)
            summaries.append(exp.run())
            traces.extend(exp.records)
    for filename,rows in [('summary.csv',summaries),('task_records.csv',traces)]:
        with (out/filename).open('w',encoding='utf8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys())
            writer.writeheader();writer.writerows(rows)
    (out/'manifest.json').write_text(json.dumps({'config':vars(c),'seeds':seeds,'methods':methods,
        'engine':'AirFogSim trace replay + separate stochastic service model' if trace else 'standalone discrete-slot reference simulator, NOT AirFogSim',
        'trace':str(trace) if trace else None,
        'evidence_level':'engineering smoke test; not paper measurements'},indent=2),encoding='utf8')
    for m in methods:
        rows=[x for x in summaries if x['method']==m]
        print(f'{m:14s}: correct/on-time={statistics.mean(x["on_time_correct_rate"] for x in rows):.3f}, '
              f'verify={statistics.mean(x["verifications"] for x in rows):.1f}, '
              f'infeasible={statistics.mean(x["risk_infeasible_slots"] for x in rows):.1f}')

def main():
    parser=argparse.ArgumentParser(description='Small-scale risk-aware UAV/VFCN engineering benchmark')
    parser.add_argument('--seeds',type=int,default=2)
    parser.add_argument('--steps',type=int,default=16)
    parser.add_argument('--methods',nargs='+',default=['va_rto','ddp','neutral','oracle','random_verify'])
    parser.add_argument('--out',type=Path,default=Path('results/va_rto_smoke'))
    parser.add_argument('--airfogsim-trace',type=Path,default=None)
    args=parser.parse_args()
    if args.seeds<1 or args.steps<1:parser.error('seeds and steps must be positive')
    campaign(replace(Config(),steps=args.steps),list(range(args.seeds)),args.methods,args.out,args.airfogsim_trace)

if __name__=='__main__':main()