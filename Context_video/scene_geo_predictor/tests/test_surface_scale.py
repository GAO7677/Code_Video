import unittest
import numpy as np
from observed_surface_scale import ray_surface_depth,estimate_scale


class SurfaceTests(unittest.TestCase):
    def test_sphere(self):
        d,h=ray_surface_depth(np.array([0.,0,0]),np.array([[[0.,0,1],[1,0,1]]]),np.array([0.,0,3]),np.array([2.,2,2]))
        self.assertAlmostEqual(d[0,0],2);self.assertTrue(h[0,0]);self.assertFalse(h[0,1])

    def test_cylinder_caps_and_sides(self):
        size=np.array([2.,2,.2]);center=np.zeros(3)
        d,h=ray_surface_depth(np.array([0.,0,2]),np.array([[[0.,0,-1]]]),center,size)
        self.assertAlmostEqual(d[0,0],1.9);self.assertTrue(h[0,0])
        d,h=ray_surface_depth(np.array([0.,-3,0]),np.array([[[0.,1,0]]]),center,size)
        self.assertAlmostEqual(d[0,0],2);self.assertTrue(h[0,0])

    def test_known_scale(self):
        from align_observed_depth import world_rays
        k=np.array([[80.,0,16],[0,80,16],[0,0,1]])
        rt=np.column_stack((np.eye(3),np.zeros(3)))
        origin,rays=world_rays(k,rt,(33,33))
        center=np.array([0.,0,3]);size=np.array([1.,1,1])
        d,h=ray_surface_depth(origin,rays,center,size)
        depth=np.broadcast_to(np.where(h,d/7,1),(8,33,33)).copy()
        masks=np.broadcast_to(h,(8,33,33))
        g=dict(positions_world=np.broadcast_to(center,(8,1,3)),size_m=size[None],camera_world_to_view=rt)
        self.assertAlmostEqual(estimate_scale(depth,masks,g,k)['scale_midpoint'],7)
        with self.assertRaises(ValueError):estimate_scale(depth,np.zeros_like(masks),g,k)


if __name__=='__main__':unittest.main()
