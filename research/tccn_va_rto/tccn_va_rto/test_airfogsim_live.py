"""Adapter contract tests; separate from required upstream-live integration CI."""
import sys
import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .airfogsim_live import AirFogSimLiveAdapter, LiveConfig, Reservation, create_fixtures


class MockTask:
    def __init__(self, **kwargs):
        self.data=kwargs
        self.id=kwargs['task_id']
        self.computed=0
    def setGenerated(self):pass
    def getTaskId(self):return self.id
    def getTaskCPU(self):return self.data['task_cpu']
    def getComputedSize(self):return self.computed


class MockManager:
    def __init__(self):
        self._waiting_to_offload_tasks={}
        self._offloading_tasks={}
        self._returning_tasks={}
        self._waiting_to_return_tasks={}
        self._computing_tasks={}
        self._done_tasks={}
        self._out_of_ddl_tasks={}


class MockNode:
    def getFogProfile(self):return {'cpu':2.2}


class MockEnv:
    def __init__(self):
        self.simulation_time=0
        self.UAVs={'UAV_0':MockNode()}
        self.vehicles={'vehicle_0':MockNode()}
        self.task_manager=MockManager()
        self.activated_offloading_tasks_with_RB_Nos={}
        self.alloc_cpu_callback=None


class FakeTaskSched:
    @staticmethod
    def setTaskOffloading(env,src,tid,target):
        a=env.task_manager._waiting_to_offload_tasks[src]
        for task in a:
            if task.getTaskId()==tid:
                a.remove(task)
                env.task_manager._offloading_tasks.setdefault(src,[]).append(task)
                return True
        return False
    @staticmethod
    def setTaskReturnRoute(env,tid,route):pass


class FakeCommSched:
    @staticmethod
    def getNumberOfRB(env):return 16
    @staticmethod
    def setCommunicationWithRB(env,tid,ids):env.activated_offloading_tasks_with_RB_Nos[tid]=ids


class FakeCompSched:
    @staticmethod
    def setComputingCallBack(env,cb):env.alloc_cpu_callback=cb


class LiveAdapterTests(unittest.TestCase):
    def get_adapter(self):
        e=MockEnv()
        module_names=['airfogsim','airfogsim.entities','airfogsim.entities.task',
                      'airfogsim.scheduler','airfogsim.scheduler.task_sched',
                      'airfogsim.scheduler.communication_sched','airfogsim.scheduler.computation_sched']
        modules={m:types.ModuleType(m) for m in module_names}
        modules['airfogsim.entities.task'].Task=MockTask
        modules['airfogsim.scheduler.task_sched'].TaskScheduler=FakeTaskSched
        modules['airfogsim.scheduler.communication_sched'].CommunicationScheduler=FakeCommSched
        modules['airfogsim.scheduler.computation_sched'].ComputationScheduler=FakeCompSched
        with patch.dict(sys.modules,modules):
            adapter=AirFogSimLiveAdapter(e,LiveConfig(steps=10,verify_rbs=5,verify_reservation_steps=4))
        return adapter,e

    def test_verification_enters_same_transmission_tasks_as_work(self):
        a,e=self.get_adapter()
        a.create_verification('vehicle_0',0)
        a.inject('WORK_0','UAV_0','vehicle_0','work',0,cpu=0.8,size=.2,reply=.01,deadline=2)
        allocated=a.schedule_tti(0)
        self.assertIn('VA_VERIFY_0_vehicle_0',e.task_manager._offloading_tasks['UAV_0'][0].getTaskId())
        self.assertEqual(len(allocated['VA_VERIFY_0_vehicle_0']),5)
        self.assertEqual(len(allocated['WORK_0']),11)
        self.assertFalse(set(allocated['WORK_0']) & set(allocated['VA_VERIFY_0_vehicle_0']))
        self.assertEqual(a.total_reserved_rb_tti,5)
        self.assertEqual(e.activated_offloading_tasks_with_RB_Nos,allocated)

    def test_verification_reduces_cpu_for_other_jobs(self):
        a,e=self.get_adapter()
        a.create_verification('vehicle_0',0)
        task=a.tasks['VA_VERIFY_0_vehicle_0']
        data=MockTask(task_id='WORK_0',task_cpu=1.0)
        e.task_manager._computing_tasks={'vehicle_0':[task,data]}
        a._slot=0
        allocated=e.alloc_cpu_callback(e.task_manager._computing_tasks)
        self.assertAlmostEqual(allocated[task.getTaskId()],1.0)
        self.assertAlmostEqual(allocated['WORK_0'],1.2)
        self.assertAlmostEqual(a.total_verify_cpu_gc,.1)

    def test_fixed_reservation_even_when_challenge_finishes_early(self):
        a,e=self.get_adapter()
        a.create_verification('vehicle_0',0)
        ident='VA_VERIFY_0_vehicle_0'
        a.schedule_tti(0)
        e.task_manager._offloading_tasks['UAV_0'].clear()
        for s in range(1,4):a.schedule_tti(s)
        self.assertEqual(a.total_reserved_rb_tti,20)
        self.assertEqual(a.total_used_rb_tti,5)
        self.assertAlmostEqual(a.total_reserved_cpu_gc,.4)

    def test_no_forced_hidden_truth_in_policy(self):
        from .airfogsim_live import _select_verification
        class T:
            def getUAVTrafficInfos(self):return {'UAV_0':{'position':[0,0,60]}}
            def getVehicleTrafficInfos(self):return {'vehicle_0':{'position':[0,0,0]}}
        e=MockEnv();e.traffic_manager=T()
        from .core import Belief
        b={'vehicle_0':Belief(3,2,.96)}
        self.assertEqual(_select_verification('va_rto',e,LiveConfig(),b,0,1),'vehicle_0')

if __name__=='__main__':unittest.main()