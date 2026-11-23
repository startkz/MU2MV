"""Tests target conservation, distributional risk, partial observability and clocks."""
import unittest
import json
import tempfile
from pathlib import Path
from .core import Config,Experiment,Frontier,cvar,add_loss,dominates,reflect

class CoreTests(unittest.TestCase):
    def test_discrete_cvar(self):
        self.assertAlmostEqual(cvar({0:.95,10:.05},.9),5)
        self.assertAlmostEqual(cvar({0:.5,10:.5},.9),10)
    def test_pmf(self):
        self.assertEqual(add_loss({0:1},7,.8),{0:.8,7:1-.8})
    def test_first_order_dominance(self):
        self.assertTrue(dominates(Frontier(utility=5,pmf={0:.8,10:.2}),Frontier(utility=4,pmf={0:.7,10:.3})))
        self.assertFalse(dominates(Frontier(utility=5,pmf={0:.7,10:.3}),Frontier(utility=4,pmf={0:.8,10:.2})))
    def test_road_no_teleportation(self):
        self.assertAlmostEqual(reflect(180,40,1,200),180)
        self.assertTrue(all(abs(reflect(0,17,t,200))<=200 for t in range(100)))
    def test_no_hidden_truth_in_observations(self):
        e=Experiment(Config(steps=1),0,'ddp')
        self.assertTrue(any(abs(e.beliefs[v.vid].mean-v.hidden_reliability)>.1 for v in e.world.vehicles))
    def test_capacity_and_deadlines(self):
        e=Experiment(Config(steps=1),0,'ddp');c=e.c
        sol=e.dp.solve(e.world.batches[0],0,e.beliefs,{},c.air_units,c.cvar_budget)
        self.assertEqual(len(sol.choices),c.tasks_per_slot)
        self.assertLessEqual(sum(x.air for x in sol.choices),c.air_units)
        self.assertEqual(len([x for x in sol.choices if x.target!=-2]),len(set(x.target for x in sol.choices if x.target!=-2)))
        self.assertTrue(all(x.latency<=x.task.deadline_s for x in sol.choices if x.target!=-2))
    def test_slow_period_and_verification_budget(self):
        e=Experiment(Config(steps=10,slow_period=4,max_verifications_per_epoch=2),2,'random_verify')
        m=e.run();self.assertEqual(m['slow_updates'],3);self.assertLessEqual(m['verifications'],6)
    def test_slow_epoch_vehicle_quotas(self):
        e=Experiment(Config(steps=9,slow_period=3,max_verifications_per_epoch=3),5,'va_rto')
        m=e.run()
        self.assertEqual(m['slow_updates'],3)
        self.assertEqual(len(e.epoch_target_history),3)
        for _,quota in e.epoch_target_history:
            self.assertLessEqual(sum(quota.values()),3)
            self.assertTrue(all(v>=0 for v in quota.values()))
        self.assertLessEqual(m['verifications'],9)
        self.assertTrue(all(v>=0 for v in e.epoch_quotas.values()))

    def test_airfogsim_trace_schema_and_replay(self):
        from .airfogsim_trace import AirFogSimTraceWorld
        frame0={'time':0.0,
                'vehicles':{'v0':{'position':[3,0,0],'speed':12},
                            'v1':{'position':[100,0,0],'speed':10}},
                'uavs':{'u0':{'position':[0,0,50],'speed':0}},
                'waiting_tasks':[{'task_id':'job1','task_node_id':'u0',
                    'task_arrival_time':0,'task_priority':0.8,'task_cpu':3.,
                    'task_size':0.25,'required_returned_size':0.02,'task_deadline':4.0}]}
        frame1={'time':1.0,'vehicles':frame0['vehicles'],
                'uavs':frame0['uavs'],'waiting_tasks':[]}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'trace.jsonl'
            path.write_text('\n'.join(json.dumps(f) for f in (frame0,frame1))+'\n')
            config=Config(steps=2,vehicles=2,tasks_per_slot=2)
            w=AirFogSimTraceWorld(config,3,path)
            self.assertEqual(len(w.batches[0]),1)
            self.assertAlmostEqual(w.batches[0][0].data_mbit,2.)
            self.assertGreater(w.distance(w.vehicles[0],0),50.)
            result=Experiment(config,3,'ddp',trace=path).run()
            self.assertEqual(result['tasks'],1)

    def test_determinism(self):
        c=Config(steps=4)
        self.assertEqual(Experiment(c,3,'ddp').run(),Experiment(c,3,'ddp').run())
    def test_risk_infeasible_is_not_hidden(self):
        e=Experiment(Config(steps=1,cvar_budget=0),4,'ddp')
        self.assertGreaterEqual(e.run()['risk_infeasible_slots'],1)
if __name__=='__main__':unittest.main()