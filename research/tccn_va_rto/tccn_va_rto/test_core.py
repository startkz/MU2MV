"""Tests target conservation, distributional risk, partial observability and clocks."""
import unittest
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
    def test_determinism(self):
        c=Config(steps=4)
        self.assertEqual(Experiment(c,3,'ddp').run(),Experiment(c,3,'ddp').run())
    def test_risk_infeasible_is_not_hidden(self):
        e=Experiment(Config(steps=1,cvar_budget=0),4,'ddp')
        self.assertGreaterEqual(e.run()['risk_infeasible_slots'],1)
if __name__=='__main__':unittest.main()