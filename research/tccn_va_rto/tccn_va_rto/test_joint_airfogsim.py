"""Deterministic unit checks for the new DDP and slow-control boundaries."""
import unittest
from .airfogsim_joint import JointConfig, SlowBudget, contact_seconds, distributional_batch
from .core import Belief, cvar, add_loss

class StubNode:
    def getFogProfile(self): return {'cpu':2.3}
class StubTrafficManager:
    def getUAVTrafficInfos(self): return {'UAV_0':{'position':[0,0,70]}}
    def getVehicleTrafficInfos(self): return {'vehicle_0':{'position':[80,0,0],'speed':10,'angle':90}}
class StubManager:
    _computing_tasks={}
class StubEnv:
    UAVs={'UAV_0':StubNode()}
    vehicles={'vehicle_0':StubNode()}
    task_manager=StubManager()
    traffic_manager=StubTrafficManager()

class JointTests(unittest.TestCase):
    def test_cvar_exact_atom(self):
        p=add_loss({0:1.},10,.85)
        self.assertAlmostEqual(cvar(p,.8),7.5)
    def test_contact_exit_and_out_of_range(self):
        self.assertEqual(contact_seconds((0,0,70),(500,0,0),(0,0),260),0.)
        self.assertGreater(contact_seconds((0,0,70),(50,0,0),(-15,0),260),
                           contact_seconds((0,0,70),(50,0,0),(15,0),260))
    def test_slow_quota_respects_epoch(self):
        c=JointConfig(tokens_per_epoch=2,slow_period=32,verify_reservation_steps=12)
        a=SlowBudget(c,17,'joint')
        views={'vehicle_0':{'contact':3.,'d':100},'vehicle_1':{'contact':2.,'d':100}}
        b={k:Belief(2,2,.96) for k in views}
        a.update(1,views,b)
        chosen=[a.choose(t,views,b) for t in range(1,33)]
        self.assertEqual(sum(x is not None for x in chosen),2)
        self.assertEqual(len(set(x for x in chosen if x is not None)),1)
    def test_single_batch_dp_feasible(self):
        c=JointConfig(risk_budget=20.)
        belief={'vehicle_0':Belief(4,1,.9)}
        views={'vehicle_0':{'contact':5.,'d':80.,'wait':0.,'cpu':2.3}}
        opts,risk,ok,size=distributional_batch(StubEnv(),c,belief,views,
            [('taskA',.7,.1,.01,2.,10)],45)
        self.assertEqual(len(opts),1)
        self.assertTrue(ok)
        self.assertLessEqual(opts[0]['rb'],45)
        self.assertGreaterEqual(size,1)
    def test_risk_infeasible_fallback_explicit(self):
        c=JointConfig(risk_budget=0.,max_local=0)
        opts,risk,ok,size=distributional_batch(StubEnv(),c,{}, {},
            [('taskA',.7,.1,.01,2.,10)],45)
        self.assertFalse(ok)
        self.assertEqual(opts[0]['target'],None)
        self.assertAlmostEqual(risk,10.)

if __name__=='__main__':unittest.main()
