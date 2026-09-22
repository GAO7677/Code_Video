"""Real Bullet regression tests: initialization may never step to resolve overlap."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from pybullet_initial_contact import InitialContactError, admit_initial_contact


class ContactGateTest(unittest.TestCase):
    def setUp(self):
        self.client = p.connect(p.DIRECT)
        self.floor = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[2,2,.05], physicsClientId=self.client), basePosition=[0,0,-.042238232], physicsClientId=self.client)
        self.ball = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE,radius=.11,physicsClientId=self.client), basePosition=[0,0,.10999],physicsClientId=self.client)
        p.resetBaseVelocity(self.ball,linearVelocity=[1,.2,0],angularVelocity=[0,9,0],physicsClientId=self.client)

    def tearDown(self):
        p.disconnect(self.client)

    def gate(self,policy='strict',confirmed=True):
        return admit_initial_contact(p,self.client,self.ball,{self.floor:'ground'},radius=.11,policy=policy,confirmed_support_names={'ground'} if confirmed else set())

    def test_missing_support_rejected_without_pose_change(self):
        p.resetBasePositionAndOrientation(self.ball,[3,0,.117751768],[0,0,0,1],physicsClientId=self.client)
        with self.assertRaises(InitialContactError) as err:
            self.gate()
        self.assertEqual(err.exception.audit['reason'],'missing_confirmed_initial_support')
        self.assertEqual(err.exception.audit['api_step_calls_before_decision'],0)
        self.assertEqual(p.getBasePositionAndOrientation(self.ball,physicsClientId=self.client)[0],(3.,0.,.117751768))

    def test_strict_rejects_without_time_advance_or_pose_change(self):
        before=p.getBasePositionAndOrientation(self.ball,physicsClientId=self.client)
        with self.assertRaises(InitialContactError) as err:
            self.gate()
        self.assertEqual(err.exception.audit['api_step_calls_before_decision'],0)
        self.assertEqual(before,p.getBasePositionAndOrientation(self.ball,physicsClientId=self.client))

    def test_aligned_contact_preserves_velocity_and_removes_launch(self):
        velocity=p.getBaseVelocity(self.ball,physicsClientId=self.client)
        audit=self.gate('support_aligned')
        self.assertEqual(audit['status'],'CONTACT_ALIGNED')
        self.assertLess(audit['after_penetration_m'],.001)
        np.testing.assert_allclose(velocity,p.getBaseVelocity(self.ball,physicsClientId=self.client))
        p.setGravity(0,0,-9.81,physicsClientId=self.client)
        heights=[]
        for _ in range(80):
            p.stepSimulation(physicsClientId=self.client)
            heights.append(p.getBasePositionAndOrientation(self.ball,physicsClientId=self.client)[0][2])
        self.assertLess(max(heights),.12)

    def test_unconfirmed_support_is_not_silently_corrected(self):
        with self.assertRaises(InitialContactError):
            self.gate('support_aligned',False)

    def test_large_correction_rejected(self):
        p.resetBasePositionAndOrientation(self.ball,[0,0,.05],[0,0,0,1],physicsClientId=self.client)
        with self.assertRaises(InitialContactError) as err:
            self.gate('support_aligned')
        self.assertEqual(err.exception.audit['reason'],'correction_exceeds_half_radius')

    def test_side_contact_is_not_support(self):
        wall=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.1,1,1],physicsClientId=self.client),basePosition=[.2,0,.5],physicsClientId=self.client)
        with self.assertRaises(InitialContactError):
            admit_initial_contact(p,self.client,self.ball,{self.floor:'ground',wall:'wall'},radius=.11,policy='support_aligned',confirmed_support_names={'ground'})

    def test_valid_input_is_unchanged_and_second_client_unaffected(self):
        other=p.connect(p.DIRECT)
        try:
            obj=p.createMultiBody(baseMass=0,basePosition=[3,4,5],physicsClientId=other)
            p.resetBasePositionAndOrientation(self.ball,[0,0,.117751768],[0,0,0,1],physicsClientId=self.client)
            before=p.getBasePositionAndOrientation(self.ball,physicsClientId=self.client)
            self.assertEqual(self.gate()['status'],'UNCHANGED')
            self.assertEqual(before,p.getBasePositionAndOrientation(self.ball,physicsClientId=self.client))
            self.assertEqual(p.getBasePositionAndOrientation(obj,physicsClientId=other)[0],(3.,4.,5.))
        finally:
            p.disconnect(other)

if __name__ == '__main__':
    unittest.main()

