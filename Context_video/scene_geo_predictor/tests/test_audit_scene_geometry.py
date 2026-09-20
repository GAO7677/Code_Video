import unittest
import numpy as np
from audit_scene_geometry import reference,coverage,interior_labels,open_volume


class GeometryAuditTests(unittest.TestCase):
    def test_box_and_gap_not_union(self):
        boxes=[dict(center=np.array([x,0,1]),half=np.array([.3,.5,.5]),rotation=np.eye(3)) for x in (-1,1)]
        origin=np.array([0.,0.,3.])
        rays=np.array([[[-1.,0,-2],[0,0,-2],[1,0,-2]]])
        points,labels,depth=reference(origin,rays,boxes)
        self.assertEqual(labels.tolist(),[[0,2,1]])
        self.assertAlmostEqual(points[0,1,2],0)
        self.assertLess(depth[0,0],depth[0,1])

    def test_rotated_box(self):
        angle=np.pi/4
        r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        box=dict(center=np.array([0.,0.,1.]),half=np.array([1.,.5,.5]),rotation=r)
        points,labels,_=reference(np.array([0.,-3,1]),np.array([[[0.,1,0]]]),[box])
        self.assertEqual(labels[0,0],0)
        self.assertAlmostEqual(points[0,0,1],-np.sqrt(.5))

    def test_coverage_and_boundary(self):
        points=np.array([[0.,0,0],[.1,0,0]])
        self.assertEqual(coverage(points,points)['within_2cm'],1)
        self.assertEqual(coverage(points,np.empty((0,3)))['within_5cm'],0)
        labels=np.zeros((7,7),dtype=int)
        self.assertEqual((interior_labels(labels,1)==0).sum(),9)

    def test_open_volume_does_not_include_platform(self):
        box=dict(name='left_platform',center=np.array([-.78,0,.41]),half=np.array([.72,.52,.07]))
        points=np.array([[-.10,0,.48],[.04,0,.48],[.3,0,.48],[.04,0,0]])
        self.assertEqual(open_volume('gap',points,[box],.2).tolist(),[False,True,False,False])


if __name__=='__main__':
    unittest.main()
